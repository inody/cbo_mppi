from __future__ import annotations

import jax.numpy as jnp


SINUSOIDAL_MODES = ("half_full", "full_double")
_MODE_FREQUENCIES = {
    "half_full": (0.5, 1.0),
    "full_double": (1.0, 2.0),
}


def resonant_k(horizon: int, dt: float, g: float, l: float) -> int:
    """Integer swing-count whose template frequency matches the small-angle
    natural frequency sqrt(g/l) of the downward pendulum."""
    omega_n = (g / l) ** 0.5
    return max(1, int(round(omega_n * horizon * dt / (2.0 * jnp.pi))))


def _time_fraction(horizon: int) -> jnp.ndarray:
    return jnp.arange(horizon, dtype=jnp.float32) / horizon


def _mode_frequencies(mode: str) -> tuple[float, float]:
    if mode not in _MODE_FREQUENCIES:
        raise ValueError(f"unknown sinusoidal mode: {mode}")
    return _MODE_FREQUENCIES[mode]


def sinusoidal_template(
    a: float,
    b: float,
    horizon: int,
    *,
    mode: str | None = None,
    k1: float | None = None,
    k2: float | None = None,
) -> jnp.ndarray:
    """Single sinusoidal template control, shape (horizon, 1).

    Either ``mode`` (preset frequency pair) or explicit ``k1``, ``k2``
    (cycle counts over the horizon) must be given. The control is
    u_t = a sin(2*pi*k1*t) + b sin(2*pi*k2*t) with t in [0, 1).
    """
    t = _time_fraction(horizon)
    if k1 is None or k2 is None:
        k1, k2 = _mode_frequencies(mode or "full_double")
    u = a * jnp.sin(2.0 * jnp.pi * k1 * t) + b * jnp.sin(2.0 * jnp.pi * k2 * t)
    return u.reshape(horizon, 1)


def sinusoidal_grid(
    a_vals: jnp.ndarray,
    b_vals: jnp.ndarray,
    horizon: int,
    *,
    mode: str | None = None,
    k1: float | None = None,
    k2: float | None = None,
):
    """Vectorised sinusoidal templates over an (a, b) grid.

    Returns (templates (P, horizon, 1), params (P, 2)) with 'ij' ordering.
    """
    t = _time_fraction(horizon)
    if k1 is None or k2 is None:
        k1, k2 = _mode_frequencies(mode or "full_double")
    aa, bb = jnp.meshgrid(a_vals, b_vals, indexing="ij")
    a_flat = aa.reshape(-1, 1)
    b_flat = bb.reshape(-1, 1)
    basis_a = jnp.sin(2.0 * jnp.pi * k1 * t)
    basis_b = jnp.sin(2.0 * jnp.pi * k2 * t)
    u = a_flat * basis_a[None, :] + b_flat * basis_b[None, :]
    params = jnp.concatenate([a_flat, b_flat], axis=1)
    return u[..., None], params


def multi_swing_template(
    amplitude: float, k: int, phi: float, horizon: int
) -> jnp.ndarray:
    """k-swing template u_t = A sin(2*pi*k*t/H + phi), shape (horizon, 1)."""
    t = _time_fraction(horizon)
    u = amplitude * jnp.sin(2.0 * jnp.pi * k * t + phi)
    return u.reshape(horizon, 1)


def multi_swing_grid(
    amplitudes: jnp.ndarray,
    k_values,
    phi_values: jnp.ndarray,
    horizon: int,
):
    """Vectorised multi-swing templates over (k, A, phi).

    Returns (templates (P, horizon, 1), labels (P, 3)) with columns
    (amplitude, k, phi).
    """
    templates = []
    labels = []
    for k in k_values:
        for a in amplitudes:
            for phi in phi_values:
                templates.append(multi_swing_template(a, k, phi, horizon))
                labels.append(jnp.array([a, float(k), phi]))
    return jnp.stack(templates), jnp.stack(labels)


def template_population(
    key: jax.Array,
    n: int,
    horizon: int,
    u_max: float,
    ks=(2, 3, 4),
    a_min: float = 0.0,
):
    """Random multi-swing template control sequences for CBO initialization.

    Each of the ``n`` sequences draws a swing-count from ``ks``, an amplitude
    uniformly in [a_min, u_max], and a phase uniformly in [-pi, pi]. Returns a
    (n, horizon, 1) array clipped to +/-u_max. Seeding the CBO population here
    (rather than with isotropic noise around zero) keeps particles inside the
    energy-pumping swing-up basins so basin selection is the studied question.
    """
    import jax

    k_key, a_key, phi_key = jax.random.split(key, 3)
    ks_arr = jnp.asarray(ks, dtype=jnp.int32)
    k_idx = jax.random.randint(k_key, (n,), 0, len(ks_arr))
    kvals = ks_arr[k_idx]
    amplitudes = jax.random.uniform(a_key, (n, 1), minval=a_min, maxval=u_max)
    phis = jax.random.uniform(phi_key, (n, 1), minval=-jnp.pi, maxval=jnp.pi)
    t = _time_fraction(horizon)
    u = amplitudes * jnp.sin(
        2.0 * jnp.pi * kvals[:, None].astype(jnp.float32) * t[None, :] + phis
    )
    return jnp.clip(u[..., None], -u_max, u_max)
