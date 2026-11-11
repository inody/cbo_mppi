from __future__ import annotations

import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp


def free_energy(costs: jnp.ndarray, tau: float, axis: int = -1) -> jnp.ndarray:
    count = costs.shape[axis]
    return -tau * (logsumexp(-costs / tau, axis=axis) - jnp.log(count))


def normalized_exponential(logits: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    finite = jnp.isfinite(logits)
    has_finite = jnp.any(finite)
    safe_logits = jnp.where(finite, logits, -jnp.inf)
    weights = jnp.where(has_finite, jnp.exp(safe_logits - logsumexp(safe_logits)), 0.0)
    return weights, has_finite


def mppi_weighted_summaries(
    costs: jnp.ndarray, tau: float, axis: int = -1
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return MPPI weights and free energy from one logsumexp reduction.

    The weights are the normalized exponentials of ``-costs / tau`` along
    ``axis``. The free energy is ``-tau * log(mean(exp(-costs / tau)))`` along
    the same axis. All-infeasible slices return zero weights and infinite free
    energy, matching ``normalized_exponential`` and ``free_energy`` usage in the
    controllers.
    """
    logits = -costs / tau
    finite = jnp.isfinite(logits)
    has_finite = jnp.any(finite, axis=axis)
    safe_logits = jnp.where(finite, logits, -jnp.inf)
    log_normalizer = logsumexp(safe_logits, axis=axis, keepdims=True)
    weights = jnp.where(
        jnp.expand_dims(has_finite, axis),
        jnp.exp(safe_logits - log_normalizer),
        0.0,
    )
    count = costs.shape[axis]
    free_energy_values = -tau * (
        jnp.squeeze(log_normalizer, axis=axis) - jnp.log(count)
    )
    free_energy_values = jnp.where(has_finite, free_energy_values, jnp.inf)
    return weights, free_energy_values, has_finite


def weighted_or_fallback(
    values: jnp.ndarray, weights: jnp.ndarray, fallback: jnp.ndarray
) -> jnp.ndarray:
    weight_sum = jnp.sum(weights)
    weighted = jnp.tensordot(weights, values, axes=(0, 0))
    return jnp.where(weight_sum > 0.0, weighted, fallback)


def effective_sample_size(weights: jnp.ndarray) -> jnp.ndarray:
    denominator = jnp.sum(weights**2)
    return jnp.where(denominator > 0.0, 1.0 / denominator, 0.0)


def shift_control_sequence(controls: jnp.ndarray) -> jnp.ndarray:
    zero_tail = jnp.zeros_like(controls[..., :1, :])
    return jnp.concatenate((controls[..., 1:, :], zero_tail), axis=-2)


def control_sequence_jitter(
    key: jax.Array,
    shape: tuple[int, ...],
    sigma: float,
    kind: str,
    smooth_modes: int = 4,
) -> jnp.ndarray:
    """Generate full or low-frequency Gaussian control-sequence jitter."""
    if kind == "full":
        return sigma * jax.random.normal(key, shape)
    if kind != "smooth":
        raise ValueError(f"unknown jitter kind: {kind}")
    horizon, control_dimension = shape[-2:]
    mode_count = min(smooth_modes, horizon)
    time = (jnp.arange(horizon) + 0.5) / horizon
    frequencies = jnp.arange(mode_count)
    basis = jnp.cos(jnp.pi * time[:, None] * frequencies[None, :])
    row_norm = jnp.sqrt(jnp.sum(basis**2, axis=1, keepdims=True))
    basis = basis / jnp.maximum(row_norm, 1e-12)
    coefficients = jax.random.normal(
        key, shape[:-2] + (mode_count, control_dimension)
    )
    return sigma * jnp.einsum("tq,...qm->...tm", basis, coefficients)


def rejuvenate_control_population(
    key: jax.Array,
    shifted_population: jnp.ndarray,
    shifted_output: jnp.ndarray,
    *,
    mode: str,
    jitter_kind: str,
    sigma: float,
    refresh_rate: float,
    min_cov_trace: float,
    apply_to: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply an explicitly configured warm-start rejuvenation variant."""
    if mode == "shift":
        return shifted_population, jnp.asarray(0.0), jnp.asarray(0.0)
    if mode not in {"shift_jitter", "mixture_refresh"}:
        raise ValueError(f"unknown warm-start mode: {mode}")
    if sigma < 0.0 or not 0.0 <= refresh_rate <= 1.0:
        raise ValueError("invalid warm-start sigma or refresh rate")
    if apply_to not in {"all_particles", "subset"}:
        raise ValueError(f"unknown warm-start apply_to: {apply_to}")

    jitter_key, mask_key = jax.random.split(key)
    jitter = control_sequence_jitter(
        jitter_key, shifted_population.shape, sigma, jitter_kind
    )
    flat = shifted_population.reshape(shifted_population.shape[0], -1)
    centered = flat - jnp.mean(flat, axis=0, keepdims=True)
    covariance_trace = jnp.sum(centered**2) / max(
        shifted_population.shape[0] - 1, 1
    )
    covariance_gate = jnp.logical_or(
        min_cov_trace <= 0.0, covariance_trace < min_cov_trace
    )

    if mode == "mixture_refresh" or apply_to == "subset":
        mask = jax.random.bernoulli(
            mask_key, refresh_rate, (shifted_population.shape[0],)
        )
    else:
        mask = jnp.ones((shifted_population.shape[0],), dtype=jnp.bool_)
    mask = jnp.logical_and(mask, covariance_gate)
    mask_expanded = mask.reshape((-1,) + (1,) * (shifted_population.ndim - 1))
    if mode == "mixture_refresh":
        refreshed_base = jnp.broadcast_to(
            shifted_output, shifted_population.shape
        )
        candidate = refreshed_base + jitter
    else:
        candidate = shifted_population + jitter
    rejuvenated = jnp.where(mask_expanded, candidate, shifted_population)
    applied_noise = rejuvenated - shifted_population
    noise_rms = jnp.sqrt(jnp.mean(applied_noise**2))
    return rejuvenated, noise_rms, jnp.mean(mask)


def temporally_correlated_normal(
    key: jax.Array, shape: tuple[int, ...], correlation: float
) -> jnp.ndarray:
    """Stationary AR(1) Gaussian noise along the penultimate (time) axis."""
    white = jax.random.normal(key, shape)
    if correlation == 0.0:
        return white
    time_first = jnp.moveaxis(white, -2, 0)
    innovation_scale = jnp.sqrt(1.0 - correlation**2)

    def step(previous, innovation):
        current = correlation * previous + innovation_scale * innovation
        return current, current

    _, remainder = jax.lax.scan(step, time_first[0], time_first[1:])
    correlated = jnp.concatenate((time_first[:1], remainder), axis=0)
    return jnp.moveaxis(correlated, 0, -2)
