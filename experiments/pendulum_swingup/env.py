from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import PendulumConfig
from .costs import cost_functions


def initial_state(cfg: PendulumConfig) -> jnp.ndarray:
    return jnp.array([cfg.theta0, cfg.omega0], dtype=jnp.float32)


def _deriv(state: jnp.ndarray, u_scalar: jnp.ndarray, cfg: PendulumConfig) -> jnp.ndarray:
    """Continuous-time RHS of the torque-limited damped pendulum.

    Angle convention: theta=0 is the upright (inverted) target, theta=pi is the
    downward rest position. With this convention the gravity term is
    +(g/l) sin(theta): it destabilises the upright equilibrium (theta=0) and
    stabilises the downward one (theta=pi), which is the desired swing-up
    structure. (The sign flips relative to the theta-from-downward textbook
    form because sin(pi+delta) = -sin(delta).)
    """
    theta = state[..., 0]
    omega = state[..., 1]
    theta_ddot = (
        (cfg.g / cfg.l) * jnp.sin(theta)
        - (cfg.c / (cfg.m * cfg.l**2)) * omega
        + u_scalar / (cfg.m * cfg.l**2)
    )
    return jnp.stack([omega, theta_ddot], axis=-1)


def rk4_step(
    state: jnp.ndarray, control: jnp.ndarray, cfg: PendulumConfig
) -> jnp.ndarray:
    """One zero-order-hold RK4 step. control shape (..., 1)."""
    u_scalar = control[..., 0]
    k1 = _deriv(state, u_scalar, cfg)
    k2 = _deriv(state + 0.5 * cfg.dt * k1, u_scalar, cfg)
    k3 = _deriv(state + 0.5 * cfg.dt * k2, u_scalar, cfg)
    k4 = _deriv(state + cfg.dt * k3, u_scalar, cfg)
    next_state = state + (cfg.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    omega = jnp.clip(next_state[..., 1], -cfg.omega_max, cfg.omega_max)
    return jnp.stack([next_state[..., 0], omega], axis=-1)


def _rollout_core(
    initial_state,
    controls,
    cfg,
    previous_control,
    *,
    clip_inputs: bool = True,
    return_trajectory: bool = True,
    stage_fn=None,
    terminal_fn=None,
):
    """Plain (non-jitted) batch rollout. ``cfg`` is a PendulumConfig.

    ``stage_fn`` / ``terminal_fn`` are passed in (and closed over by the
    controller adapter) so no string branch on a traced ``cfg`` is needed.
    """
    if stage_fn is None or terminal_fn is None:
        stage_fn, terminal_fn = cost_functions("weighted")
    if clip_inputs:
        controls = jnp.clip(controls, -cfg.u_max, cfg.u_max)
    batch = controls.shape[0]
    state_dim = initial_state.shape[-1]
    s0 = jnp.broadcast_to(initial_state, (batch, state_dim))
    if previous_control is None:
        previous_control = jnp.zeros((1,), dtype=controls.dtype)
    prev0 = jnp.broadcast_to(previous_control.reshape(-1), (batch, 1))
    controls_t = jnp.swapaxes(controls, 0, 1)

    def scan_step(carry, inp):
        state, prev_u = carry
        u = inp
        next_state = rk4_step(state, u, cfg)
        cost = stage_fn(next_state, u, prev_u, cfg)
        return (next_state, u), (next_state, cost)

    (final_state, _), (states_by_time, stage_costs) = jax.lax.scan(
        scan_step, (s0, prev0), controls_t
    )
    total = jnp.sum(stage_costs, axis=0) + terminal_fn(final_state, cfg)
    if return_trajectory:
        states = jnp.swapaxes(states_by_time, 0, 1)
        trajectory = jnp.concatenate([s0[:, None, :], states], axis=1)
        return total, final_state, trajectory
    return total, final_state


def make_rollout_batch(cost_style: str = "weighted"):
    """Jitted landscape rollout factory (cfg held static). Cost style baked in."""
    stage_fn, terminal_fn = cost_functions(cost_style)

    @partial(jax.jit, static_argnames=("cfg", "clip_inputs", "return_trajectory"))
    def rollout_batch(
        initial_state,
        controls,
        cfg: PendulumConfig,
        previous_control=None,
        *,
        clip_inputs: bool = True,
        return_trajectory: bool = True,
    ):
        return _rollout_core(
            initial_state,
            controls,
            cfg,
            previous_control,
            clip_inputs=clip_inputs,
            return_trajectory=return_trajectory,
            stage_fn=stage_fn,
            terminal_fn=terminal_fn,
        )

    return rollout_batch


# Default landscape rollout (weighted cost), for backward-compatible imports.
rollout_batch = make_rollout_batch("weighted")


def make_pendulum_rollout(cost_style: str = "regulation"):
    """Build a controller ``rollout_fn`` with the cost style baked in.

    The selected stage/terminal cost functions are closed over so that, inside
    a jitted controller step, no string branch on the traced ``problem``
    pytree is required.
    """
    stage_fn, terminal_fn = cost_functions(cost_style)

    def pendulum_rollout(current_state, samples, problem, previous_control, *,
                         clip_inputs: bool = True):
        total, _final, trajectory = _rollout_core(
            current_state,
            samples,
            problem,
            previous_control,
            clip_inputs=clip_inputs,
            return_trajectory=True,
            stage_fn=stage_fn,
            terminal_fn=terminal_fn,
        )
        return total, trajectory

    return pendulum_rollout


def step_dynamics(state, control, cfg: PendulumConfig) -> jnp.ndarray:
    """Single closed-loop plant step applying a scalar control."""
    return rk4_step(state, control.reshape(-1), cfg)
