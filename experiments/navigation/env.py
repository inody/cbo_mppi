"""Double-integrator dynamics, rollout, and nominal trajectory generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ProblemConfig


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def double_integrator_step(
    state: jnp.ndarray, control: jnp.ndarray, dt: float
) -> jnp.ndarray:
    position = state[..., :2]
    velocity = state[..., 2:]
    next_position = position + dt * velocity
    next_velocity = velocity + dt * control
    return jnp.concatenate((next_position, next_velocity), axis=-1)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_batch(
    initial_state: jnp.ndarray,
    controls: jnp.ndarray,
    problem: ProblemConfig,
    previous_control: jnp.ndarray | None = None,
    *,
    clip_inputs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll out double-integrator controls of shape (batch, horizon, 2)."""
    if clip_inputs:
        controls = jnp.clip(controls, -problem.u_max, problem.u_max)
    batch_size = controls.shape[0]
    initial_batch = jnp.broadcast_to(initial_state, (batch_size, initial_state.shape[-1]))

    def scan_step(state: jnp.ndarray, control: jnp.ndarray):
        next_state = double_integrator_step(state, control, problem.dt)
        position = next_state[:, :2]
        goal_cost = problem.q_goal_running * jnp.sum(
            (position - problem.goal) ** 2, axis=-1
        )
        velocity_cost = problem.q_velocity_running * jnp.sum(
            next_state[:, 2:] ** 2, axis=-1
        )
        distances = jnp.linalg.norm(
            position[:, None, :] - problem.obstacle_centers[None, :, :], axis=-1
        )
        safe_radii = problem.obstacle_radii + problem.safety_margin
        penetration = jnp.maximum(0.0, safe_radii[None, :] - distances)
        obstacle_cost = problem.q_obstacle * jnp.sum(penetration**2, axis=-1)
        collision_free = jnp.all(
            distances > problem.obstacle_radii[None, :], axis=-1
        )
        control_cost = problem.q_control * jnp.sum(control**2, axis=-1)
        stage_cost = goal_cost + velocity_cost + obstacle_cost + control_cost
        return next_state, (next_state, stage_cost, collision_free)

    final_state, (states_by_time, stage_costs, collision_free_by_time) = jax.lax.scan(
        scan_step, initial_batch, jnp.swapaxes(controls, 0, 1)
    )
    states = jnp.swapaxes(states_by_time, 0, 1)
    trajectory = jnp.concatenate((initial_batch[:, None, :], states), axis=1)
    terminal_cost = problem.q_goal_terminal * jnp.sum(
        (final_state[:, :2] - problem.goal) ** 2, axis=-1
    )
    terminal_cost += problem.q_velocity_terminal * jnp.sum(
        final_state[:, 2:] ** 2, axis=-1
    )
    if previous_control is None:
        previous_control = jnp.zeros(controls.shape[-1], dtype=controls.dtype)
    previous_batch = jnp.broadcast_to(
        previous_control, (batch_size, 1, controls.shape[-1])
    )
    controls_with_boundary = jnp.concatenate((previous_batch, controls), axis=1)
    control_differences = jnp.diff(controls_with_boundary, axis=1)
    rate_cost = problem.q_control_rate * jnp.sum(control_differences**2, axis=(1, 2))
    total_cost = jnp.sum(stage_costs, axis=0) + terminal_cost + rate_cost
    input_feasible = jnp.all(jnp.abs(controls) <= problem.u_max, axis=(1, 2))
    collision_free = jnp.all(collision_free_by_time, axis=0)
    obstacle_feasible = jnp.logical_or(
        jnp.logical_not(problem.hard_obstacles), collision_free
    )
    extended_cost = jnp.where(
        jnp.logical_and(input_feasible, obstacle_feasible), total_cost, jnp.inf
    )
    return extended_cost, trajectory


# ---------------------------------------------------------------------------
# Nominal trajectory generation
# ---------------------------------------------------------------------------

def bang_bang_nominal(
    initial_state: jnp.ndarray,
    problem: ProblemConfig,
    horizon: int,
    route_height: float = 0.0,
) -> jnp.ndarray:
    duration = horizon * problem.dt
    displacement = problem.goal - initial_state[:2]
    acceleration = jnp.clip(4.0 * displacement / duration**2, -problem.u_max, problem.u_max)
    nominal = jnp.concatenate(
        (
            jnp.broadcast_to(acceleration, (horizon // 2, 2)),
            jnp.broadcast_to(-acceleration, (horizon - horizon // 2, 2)),
        ),
        axis=0,
    )
    quarter = horizon // 4
    if route_height != 0.0 and quarter > 0:
        vertical_acceleration = route_height / (quarter * problem.dt) ** 2
        vertical_pattern = jnp.concatenate(
            (
                jnp.full((quarter,), vertical_acceleration),
                jnp.full((quarter,), -vertical_acceleration),
                jnp.full((quarter,), -vertical_acceleration),
                jnp.full((horizon - 3 * quarter,), vertical_acceleration),
            )
        )
        nominal = nominal.at[:, 1].set(vertical_pattern)
    return jnp.clip(nominal, -problem.u_max, problem.u_max)


def straight_line_nominal(
    initial_state: jnp.ndarray,
    problem: ProblemConfig,
    horizon: int,
    route_height: float = 0.0,
) -> jnp.ndarray:
    """Smooth nominal from current state to rest at goal.

    Accounts for current velocity via a linear-decay component so the
    generated controls are consistent with the actual system state.
    With zero initial velocity this reduces to the rest-to-rest profile.
    """
    time_index = jnp.arange(horizon, dtype=initial_state.dtype)
    phase = time_index / max(horizon - 1, 1)
    speed_shape = phase**2 * (1.0 - phase) ** 2
    displacement = problem.goal - initial_state[:2]
    initial_velocity = initial_state[2:]
    velocity_decay = initial_velocity[None, :] * (1.0 - phase)[:, None]
    velocity_decay_displacement = problem.dt * jnp.sum(velocity_decay, axis=0)
    remaining_displacement = displacement - velocity_decay_displacement
    bell_area = problem.dt * jnp.sum(speed_shape)
    velocity_scale = remaining_displacement / bell_area
    velocities = (
        speed_shape[:, None] * velocity_scale[None, :] + velocity_decay
    )

    if route_height != 0.0 and horizon > 1:
        route_velocity_shape = (
            phase**2 * (1.0 - phase) ** 2 * (1.0 - 2.0 * phase)
        )
        unscaled_heights = problem.dt * jnp.cumsum(route_velocity_shape)
        height_scale = route_height / jnp.max(unscaled_heights)
        velocities = velocities.at[:, 1].add(height_scale * route_velocity_shape)

    terminal_velocity = jnp.zeros((1, 2), dtype=velocities.dtype)
    velocity_sequence = jnp.concatenate((velocities, terminal_velocity), axis=0)
    controls = jnp.diff(velocity_sequence, axis=0) / problem.dt
    return jnp.clip(controls, -problem.u_max, problem.u_max)


def make_initial_nominal(
    initial_state: jnp.ndarray,
    problem: ProblemConfig,
    horizon: int,
    route_height: float,
    profile: str,
) -> jnp.ndarray:
    if profile == "zero":
        return jnp.zeros(
            (horizon, problem.goal.shape[-1]), dtype=initial_state.dtype
        )
    if profile == "minimum_jerk":
        return straight_line_nominal(
            initial_state, problem, horizon, route_height=route_height
        )
    if profile == "bang_bang":
        return bang_bang_nominal(
            initial_state, problem, horizon, route_height=route_height
        )
    raise ValueError(f"unknown nominal profile: {profile}")
