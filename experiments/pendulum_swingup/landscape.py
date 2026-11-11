from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from cbo_mppi.utils import mppi_weighted_summaries, temporally_correlated_normal

from .config import LandscapeConfig, PendulumConfig
from .env import initial_state, rollout_batch
from .costs import angle_error, is_success


@dataclass
class GridResult:
    a_vals: np.ndarray
    b_vals: np.ndarray
    params: np.ndarray
    nom_cost: np.ndarray
    nom_success: np.ndarray
    nom_terminal_angle: np.ndarray
    F_hat: np.ndarray
    pert_success: np.ndarray
    ess: np.ndarray
    mean_cost: np.ndarray
    q10_cost: np.ndarray
    tau: float
    sigma_local: float
    rho: float
    samples: int
    mode: str
    k1: float
    k2: float

    @property
    def shape(self):
        return (len(self.a_vals), len(self.b_vals))

    def field(self, name: str) -> np.ndarray:
        return getattr(self, name).reshape(self.shape)

    def argmin_flat(self, name: str) -> int:
        return int(np.argmin(getattr(self, name)))

    def point(self, flat_index: int) -> dict:
        return {
            "a": float(self.params[flat_index, 0]),
            "b": float(self.params[flat_index, 1]),
            "nom_cost": float(self.nom_cost[flat_index]),
            "nom_success": bool(self.nom_success[flat_index]),
            "nom_terminal_angle": float(self.nom_terminal_angle[flat_index]),
            "F_hat": float(self.F_hat[flat_index]),
            "pert_success": float(self.pert_success[flat_index]),
            "ess": float(self.ess[flat_index]),
            "mean_cost": float(self.mean_cost[flat_index]),
            "q10_cost": float(self.q10_cost[flat_index]),
        }


def evaluate_templates(
    templates: jnp.ndarray,
    cfg: PendulumConfig,
    lcfg: LandscapeConfig,
    *,
    basis: tuple[float, float] = (1.0, 2.0),
) -> GridResult:
    """Evaluate nominal J and MPPI free-energy diagnostics for a template batch.

    templates: (P, horizon, 1)
    """
    P = templates.shape[0]
    key = jax.random.PRNGKey(lcfg.seed)
    x0 = initial_state(cfg)

    nom_cost, nom_final = rollout_batch(
        x0, templates, cfg, clip_inputs=True, return_trajectory=False
    )
    nom_success = is_success(nom_final, cfg)
    nom_terminal_angle = jnp.abs(angle_error(nom_final[:, 0] - cfg.theta_goal))

    chunk = max(1, lcfg.target_batch // max(lcfg.samples, 1))
    F_parts, succ_parts, ess_parts = [], [], []
    mean_parts, q10_parts = [], []
    for start in range(0, P, chunk):
        end = min(P, start + chunk)
        M = templates[start:end]
        p = end - start
        key, noise_key = jax.random.split(key)
        noise = lcfg.sigma_local * temporally_correlated_normal(
            noise_key, (p, lcfg.samples, cfg.horizon, 1), lcfg.rho
        )
        samples = M[:, None, :, :] + noise
        flat = samples.reshape(p * lcfg.samples, cfg.horizon, 1)
        costs, finals = rollout_batch(
            x0, flat, cfg, clip_inputs=True, return_trajectory=False
        )
        costs = costs.reshape(p, lcfg.samples)
        finals = finals.reshape(p, lcfg.samples, 2)
        weights, F, _ = mppi_weighted_summaries(costs, lcfg.tau, axis=1)
        ess = 1.0 / jnp.sum(weights**2, axis=1)
        succ = jnp.mean(is_success(finals, cfg).astype(jnp.float32), axis=1)
        F_parts.append(F)
        succ_parts.append(succ)
        ess_parts.append(ess)
        mean_parts.append(jnp.mean(costs, axis=1))
        q10_parts.append(jnp.quantile(costs, 0.10, axis=1))

    grid = _build_axis(cfg, lcfg)
    return GridResult(
        a_vals=np.asarray(grid[0]),
        b_vals=np.asarray(grid[1]),
        params=np.asarray(_params_from_grid(grid[0], grid[1])),
        nom_cost=np.asarray(nom_cost),
        nom_success=np.asarray(nom_success),
        nom_terminal_angle=np.asarray(nom_terminal_angle),
        F_hat=np.asarray(jnp.concatenate(F_parts)),
        pert_success=np.asarray(jnp.concatenate(succ_parts)),
        ess=np.asarray(jnp.concatenate(ess_parts)),
        mean_cost=np.asarray(jnp.concatenate(mean_parts)),
        q10_cost=np.asarray(jnp.concatenate(q10_parts)),
        tau=lcfg.tau,
        sigma_local=lcfg.sigma_local,
        rho=lcfg.rho,
        samples=lcfg.samples,
        mode=lcfg.mode,
        k1=float(basis[0]),
        k2=float(basis[1]),
    )


def _build_axis(cfg: PendulumConfig, lcfg: LandscapeConfig):
    a_vals = jnp.linspace(-cfg.u_max, cfg.u_max, lcfg.grid_n, dtype=jnp.float32)
    b_vals = jnp.linspace(-cfg.u_max, cfg.u_max, lcfg.grid_n, dtype=jnp.float32)
    return a_vals, b_vals


def _params_from_grid(a_vals, b_vals):
    aa, bb = jnp.meshgrid(a_vals, b_vals, indexing="ij")
    return jnp.concatenate([aa.reshape(-1, 1), bb.reshape(-1, 1)], axis=1)


def build_grid_templates(cfg: PendulumConfig, lcfg: LandscapeConfig):
    from .templates import resonant_k, sinusoidal_grid

    a_vals, b_vals = _build_axis(cfg, lcfg)
    if lcfg.k1 is not None and lcfg.k2 is not None:
        k1, k2 = float(lcfg.k1), float(lcfg.k2)
    else:
        k_res = resonant_k(cfg.horizon, cfg.dt, cfg.g, cfg.l)
        k1, k2 = float(max(1, k_res - 1)), float(k_res)
    templates, params = sinusoidal_grid(
        a_vals, b_vals, cfg.horizon, k1=k1, k2=k2
    )
    return templates, params, a_vals, b_vals, (k1, k2)


def trajectory_for_params(
    a: float,
    b: float,
    cfg: PendulumConfig,
    *,
    k1: float | None = None,
    k2: float | None = None,
    mode: str | None = None,
):
    """Recompute the nominal trajectory + clipped control for one (a, b)."""
    from .templates import sinusoidal_template

    template = sinusoidal_template(a, b, cfg.horizon, mode=mode, k1=k1, k2=k2)
    clipped = jnp.clip(template, -cfg.u_max, cfg.u_max)
    total, final, trajectory = rollout_batch(
        initial_state(cfg), clipped[None], cfg, clip_inputs=False
    )
    return (
        np.asarray(trajectory[0]),
        np.asarray(clipped),
        float(total[0]),
        np.asarray(final[0]),
    )
