from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import PendulumConfig


def angle_error(theta: jnp.ndarray) -> jnp.ndarray:
    """Wrapped angle error to the nearest multiple of 2*pi."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


# --- regulation (honda/gym) cost: angle^2 + 0.1*omega^2, no control term ---
# Coefficients are hardcoded to the gym/honda values so this cost style is
# independent of the weighted-style q_* weights in PendulumConfig.
_REG_ANGLE = 1.0
_REG_OMEGA = 0.1


def regulation_stage_cost(next_state, u, prev_u, cfg: PendulumConfig):
    theta = next_state[..., 0]
    omega = next_state[..., 1]
    return _REG_ANGLE * angle_error(theta - cfg.theta_goal) ** 2 + _REG_OMEGA * omega**2


def regulation_terminal_cost(final_state, cfg: PendulumConfig):
    return regulation_stage_cost(
        final_state,
        jnp.zeros_like(final_state[..., :1]),
        jnp.zeros_like(final_state[..., :1]),
        cfg,
    )


# --- weighted cost: q_angle*(1-cos) + q_omega*omega^2 + q_u*u^2 + q_du*du^2 ---
def weighted_stage_cost(next_state, u, prev_u, cfg: PendulumConfig):
    theta = next_state[..., 0]
    omega = next_state[..., 1]
    uu = u[..., 0]
    pu = prev_u[..., 0]
    return (
        cfg.q_angle * (1.0 - jnp.cos(theta))
        + cfg.q_omega * omega**2
        + cfg.q_u * uu**2
        + cfg.q_du * (uu - pu) ** 2
    )


def weighted_terminal_cost(final_state, cfg: PendulumConfig):
    theta = final_state[..., 0]
    omega = final_state[..., 1]
    return cfg.Q_angle * (1.0 - jnp.cos(theta)) + cfg.Q_omega * omega**2


def stage_cost(next_state, u, prev_u, cfg: PendulumConfig, style: str = "regulation"):
    """Convenience dispatcher; only call where ``style`` is a concrete string."""
    sfn, _ = cost_functions(style)
    return sfn(next_state, u, prev_u, cfg)


def terminal_cost(final_state, cfg: PendulumConfig, style: str = "regulation"):
    _, tfn = cost_functions(style)
    return tfn(final_state, cfg)


def cost_functions(style: str):
    if style == "regulation":
        return regulation_stage_cost, regulation_terminal_cost
    return weighted_stage_cost, weighted_terminal_cost


def is_success(final_state: jnp.ndarray, cfg: PendulumConfig) -> jnp.ndarray:
    """Boolean success of a final state (batched over leading axis)."""
    theta = final_state[..., 0]
    omega = final_state[..., 1]
    return jnp.logical_and(
        jnp.abs(angle_error(theta - cfg.theta_goal)) < cfg.theta_tol,
        jnp.abs(omega - cfg.omega_goal) < cfg.omega_tol,
    )


def closed_loop_metrics(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    cfg: PendulumConfig,
    style: str = "regulation",
) -> dict:
    """Summary metrics for a realized closed-loop episode.

    states:      (T+1, 2)
    controls:    (T, 1)
    """
    states = jnp.asarray(states)
    controls = jnp.asarray(controls)
    theta = states[:, 0]
    omega = states[:, 1]
    uu = controls[:, 0]
    boundary = jnp.concatenate([jnp.zeros((1,), dtype=uu.dtype), uu], axis=0)
    diffs = jnp.diff(boundary)
    prev_u_seq = jnp.concatenate([jnp.zeros((1,), dtype=uu.dtype), uu[:-1]], axis=0)[
        :, None
    ]
    stage_fn, terminal_fn = cost_functions(style)
    stage = jax.vmap(lambda s, u, pu: stage_fn(s, u, pu, cfg))(
        states[1:], controls, prev_u_seq
    )
    total_cost = float(jnp.sum(stage) + terminal_fn(states[-1], cfg))
    return {
        "terminal_angle_error": float(jnp.abs(angle_error(theta[-1] - cfg.theta_goal))),
        "terminal_omega": float(omega[-1]),
        "success": bool(is_success(states[-1], cfg)),
        "total_cost": total_cost,
        "control_effort": float(jnp.sum(uu**2)),
        "control_variation": float(jnp.sum(diffs**2)),
        "max_abs_omega": float(jnp.max(jnp.abs(omega))),
        "saturation_steps": int(jnp.sum(jnp.abs(uu) >= 0.999 * cfg.u_max)),
    }
