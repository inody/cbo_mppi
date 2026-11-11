from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .config import ArmConfig


def initial_state(cfg: ArmConfig) -> jnp.ndarray:
    return jnp.array(list(cfg.theta0) + list(cfg.omega0), dtype=jnp.float32)


def forward_kinematics(q: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """Joint positions of a planar serial arm. q: (..., n_links). Returns (..., n_links+1, 2)."""
    lengths = jnp.asarray(cfg.link_lengths, dtype=q.dtype)
    cum_q = jnp.cumsum(q, axis=-1)
    dx = jnp.cumsum(lengths * jnp.cos(cum_q), axis=-1)
    dy = jnp.cumsum(lengths * jnp.sin(cum_q), axis=-1)
    zeros = jnp.zeros_like(dx[..., :1])
    joints_x = jnp.concatenate([zeros, dx], axis=-1)
    joints_y = jnp.concatenate([zeros, dy], axis=-1)
    return jnp.stack([joints_x, joints_y], axis=-1)


def end_effector(q: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    return forward_kinematics(q, cfg)[..., -1, :]


def _segment_point_dist(p_a, p_b, c):
    """Min distance from point c to segment [p_a, p_b]. All (..., 2)."""
    d = p_b - p_a
    l2 = jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12)
    t = jnp.clip(jnp.sum((c - p_a) * d, axis=-1) / l2, 0.0, 1.0)
    proj = p_a + t[..., None] * d
    return jnp.linalg.norm(c - proj, axis=-1)


def obstacle_cost(q: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """Soft penalty for link-obstacle penetration. Returns (...) scalar."""
    joints = forward_kinematics(q, cfg)
    n_links = len(cfg.link_lengths)
    centers = jnp.asarray(cfg.obstacle_centers, dtype=q.dtype)
    radii = jnp.asarray(cfg.obstacle_radii, dtype=q.dtype)
    margin = cfg.link_radius + cfg.safety_margin
    total = jnp.zeros(q.shape[:-1], dtype=q.dtype)
    for i in range(n_links):
        seg_a = joints[..., i, :]
        seg_b = joints[..., i + 1, :]
        for o in range(centers.shape[0]):
            dist = _segment_point_dist(seg_a, seg_b, centers[o])
            penetration = jnp.maximum(0.0, radii[o] + margin - dist)
            total = total + penetration ** 2
    return total


def link_obstacle_clearance(q: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """Minimum link-obstacle clearance. Contact is zero; penetration is negative."""
    joints = forward_kinematics(q, cfg)
    centers = jnp.asarray(cfg.obstacle_centers, dtype=q.dtype)
    radii = jnp.asarray(cfg.obstacle_radii, dtype=q.dtype)
    clearance = jnp.full(q.shape[:-1], jnp.inf, dtype=q.dtype)
    for i in range(len(cfg.link_lengths)):
        seg_a = joints[..., i, :]
        seg_b = joints[..., i + 1, :]
        for o in range(centers.shape[0]):
            dist = _segment_point_dist(seg_a, seg_b, centers[o])
            clearance = jnp.minimum(clearance, dist - radii[o] - cfg.link_radius)
    return clearance


def trajectory_min_clearance(states: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """Minimum contact clearance over a trajectory or trajectory batch."""
    q = states[..., :3]
    clearance = link_obstacle_clearance(q, cfg)
    return jnp.min(clearance, axis=-1)


def collision_mask(states: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """True when any link penetrates any obstacle along the trajectory."""
    return trajectory_min_clearance(states, cfg) < 0.0


def joint_limit_cost(q: jnp.ndarray, cfg: ArmConfig) -> jnp.ndarray:
    """Soft penalty for joint-limit violation."""
    lo = cfg.q_min
    hi = cfg.q_max
    excess = jnp.maximum(0.0, q - hi) + jnp.maximum(0.0, lo - q)
    return jnp.sum(excess ** 2, axis=-1)


def stage_cost(state, u, prev_u, cfg: ArmConfig):
    q = state[..., :3]
    qd = state[..., 3:]
    ee = end_effector(q, cfg)
    goal = jnp.asarray(cfg.goal, dtype=state.dtype)
    ee_err = jnp.sum((ee - goal) ** 2, axis=-1)
    vel = jnp.sum(qd ** 2, axis=-1)
    uu = jnp.sum(u ** 2, axis=-1)
    du = jnp.sum((u - prev_u) ** 2, axis=-1)
    obs = obstacle_cost(q, cfg)
    lim = joint_limit_cost(q, cfg)
    return (
        cfg.q_ee * ee_err
        + cfg.q_v * vel
        + cfg.q_u * uu
        + cfg.q_du * du
        + cfg.q_obs * obs
        + cfg.q_lim * lim
    )


def stage_cost_without_obstacles(state, u, prev_u, cfg: ArmConfig):
    q = state[..., :3]
    qd = state[..., 3:]
    ee = end_effector(q, cfg)
    goal = jnp.asarray(cfg.goal, dtype=state.dtype)
    ee_err = jnp.sum((ee - goal) ** 2, axis=-1)
    vel = jnp.sum(qd ** 2, axis=-1)
    uu = jnp.sum(u ** 2, axis=-1)
    du = jnp.sum((u - prev_u) ** 2, axis=-1)
    return (
        cfg.q_ee * ee_err
        + cfg.q_v * vel
        + cfg.q_u * uu
        + cfg.q_du * du
    )


def terminal_cost(state, cfg: ArmConfig):
    q = state[..., :3]
    qd = state[..., 3:]
    ee = end_effector(q, cfg)
    goal = jnp.asarray(cfg.goal, dtype=state.dtype)
    ee_err = jnp.sum((ee - goal) ** 2, axis=-1)
    vel = jnp.sum(qd ** 2, axis=-1)
    return cfg.Q_ee * ee_err + cfg.Q_v * vel


def is_success(state, cfg: ArmConfig):
    q = state[..., :3]
    qd = state[..., 3:]
    ee = end_effector(q, cfg)
    goal = jnp.asarray(cfg.goal, dtype=state.dtype)
    ee_err = jnp.linalg.norm(ee - goal, axis=-1)
    vel_norm = jnp.linalg.norm(qd, axis=-1)
    return jnp.logical_and(ee_err < cfg.ee_tol, vel_norm < cfg.vel_tol)


def _step(state, u, cfg: ArmConfig):
    q = state[..., :3]
    qd = state[..., 3:]
    q_next = q + cfg.dt * qd
    qd_next = (1.0 - cfg.damping * cfg.dt) * qd + cfg.dt * u
    return jnp.concatenate([q_next, qd_next], axis=-1)


def _rollout_core(initial_state, controls, cfg, previous_control, *,
                  clip_inputs=True, return_trajectory=True):
    if clip_inputs:
        controls = jnp.clip(controls, -cfg.u_max, cfg.u_max)
    batch = controls.shape[0]
    state_dim = initial_state.shape[-1]
    s0 = jnp.broadcast_to(initial_state, (batch, state_dim))
    if previous_control is None:
        previous_control = jnp.zeros(3, dtype=controls.dtype)
    prev0 = jnp.broadcast_to(previous_control.reshape(-1), (batch, 3))
    controls_t = jnp.swapaxes(controls, 0, 1)

    def scan_step(carry, inp):
        state, prev_u = carry
        u = inp
        next_state = _step(state, u, cfg)
        cost = stage_cost(next_state, u, prev_u, cfg)
        return (next_state, u), (next_state, cost)

    (final_state, _), (states_by_time, stage_costs) = jax.lax.scan(
        scan_step, (s0, prev0), controls_t
    )
    total = jnp.sum(stage_costs, axis=0) + terminal_cost(final_state, cfg)
    if return_trajectory:
        states = jnp.swapaxes(states_by_time, 0, 1)
        trajectory = jnp.concatenate([s0[:, None, :], states], axis=1)
        return total, final_state, trajectory
    return total, final_state


def make_arm_rollout():
    """Controller adapter: returns (costs, trajectory)."""

    def arm_rollout(current_state, samples, problem, previous_control, *,
                    clip_inputs=True):
        total, _final, trajectory = _rollout_core(
            current_state, samples, problem, previous_control,
            clip_inputs=clip_inputs, return_trajectory=True,
        )
        return total, trajectory

    return arm_rollout


def make_arm_hard_rollout():
    """Controller adapter with hard link-obstacle constraints.

    The finite part excludes the soft obstacle penalty. Any rollout with
    negative contact clearance is assigned infinite cost.
    """

    def arm_hard_rollout(current_state, samples, problem, previous_control, *,
                         clip_inputs=True):
        if clip_inputs:
            samples = jnp.clip(samples, -problem.u_max, problem.u_max)
        batch = samples.shape[0]
        s0 = jnp.broadcast_to(current_state, (batch, current_state.shape[-1]))
        prev0 = jnp.broadcast_to(previous_control.reshape(-1), (batch, 3))
        controls_t = jnp.swapaxes(samples, 0, 1)

        def scan_step(carry, u):
            state, prev_u = carry
            next_state = _step(state, u, problem)
            cost = stage_cost_without_obstacles(next_state, u, prev_u, problem)
            return (next_state, u), (next_state, cost)

        (final_state, _), (states_by_time, stage_costs) = jax.lax.scan(
            scan_step, (s0, prev0), controls_t
        )
        states = jnp.swapaxes(states_by_time, 0, 1)
        trajectory = jnp.concatenate([s0[:, None, :], states], axis=1)
        total = jnp.sum(stage_costs, axis=0) + terminal_cost(final_state, problem)
        colliding = collision_mask(trajectory, problem)
        total = jnp.where(colliding, jnp.inf, total)
        return total, trajectory

    return arm_hard_rollout


@partial(jax.jit, static_argnames=("cfg",))
def rollout_batch(initial_state, controls, cfg: ArmConfig, previous_control=None):
    return _rollout_core(initial_state, controls, cfg, previous_control,
                         clip_inputs=True, return_trajectory=True)


def step_dynamics(state, control, cfg: ArmConfig):
    return _step(state, control.reshape(-1), cfg)
