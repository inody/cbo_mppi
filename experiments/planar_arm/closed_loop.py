"""Closed-loop runner for the planar arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import time

from experiments.episodic import termination_summary
from experiments.episodic_tasks import arm_monitor

from cbo_mppi.controllers import (
    CBOMPCState, ProposedState,
    init_cbo_mpc, init_iterated_mppi, init_mppi, init_proposed,
    make_cbo_mpc_step, make_iterated_mppi_step, make_mppi_step,
    make_proposed_step,
)

from .config import ArmConfig
from .env import (
    initial_state,
    make_arm_rollout,
    make_arm_hard_rollout,
    step_dynamics,
    is_success,
    end_effector,
    trajectory_min_clearance,
)


@dataclass
class ArmResult:
    controller: str
    seed: int
    budget: int
    states: np.ndarray
    controls: np.ndarray
    step_times: np.ndarray
    rollout_counts: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _build(controller, problem, cfg, base_nominal, init_key, *, hard_obstacles=False):
    rollout_fn = make_arm_hard_rollout() if hard_obstacles else make_arm_rollout()
    if controller == "mppi":
        return init_mppi(base_nominal), make_mppi_step(rollout_fn)
    if controller == "iterated_mppi":
        return (
            init_iterated_mppi(base_nominal),
            make_iterated_mppi_step(rollout_fn),
        )
    if controller == "cbo_mpc":
        state = init_cbo_mpc(init_key, base_nominal, problem, cfg, clip_initial=True)
        return state, make_cbo_mpc_step(rollout_fn)
    if controller == "proposed":
        state = init_proposed(init_key, base_nominal, problem, cfg, clip_initial=True)
        return state, make_proposed_step(rollout_fn)
    raise ValueError(controller)


def run_arm_closed_loop(controller, problem, cfg, *, seed, steps,
                        process_noise_sigma=0.0, hard_obstacles=False,
                        episodic=False):
    horizon = problem.horizon
    control_dim = len(problem.link_lengths)
    base_nominal = jnp.zeros((horizon, control_dim), dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    controller_state, step_fn = _build(
        controller, problem, cfg, base_nominal, init_key,
        hard_obstacles=hard_obstacles,
    )

    current_state = initial_state(problem)
    previous_control = jnp.zeros(control_dim, dtype=jnp.float32)

    key, controller_state, control, info = step_fn(
        key, controller_state, current_state, previous_control, problem, cfg)
    jax.block_until_ready(control)

    states = [np.asarray(current_state)]
    controls = []
    step_times = []
    rollout_counts = []
    diagnostics: dict[str, list] = {}
    episode_monitor = arm_monitor(problem) if episodic else None
    episode_termination = (
        episode_monitor.observe(np.asarray(current_state), step=0)
        if episode_monitor is not None
        else None
    )
    for step_index in range(steps):
        if episode_termination is not None:
            break
        t0 = time.perf_counter()
        key, controller_state, control, info = step_fn(
            key, controller_state, current_state, previous_control, problem, cfg)
        control = jnp.clip(control, -problem.u_max, problem.u_max)
        control.block_until_ready()
        step_times.append(time.perf_counter() - t0)
        if process_noise_sigma > 0.0:
            key, nk = jax.random.split(key)
            noise = jax.random.normal(nk, control.shape) * process_noise_sigma
            applied = jnp.clip(control + noise, -problem.u_max, problem.u_max)
        else:
            applied = control
        current_state = step_dynamics(current_state, applied, problem)
        previous_control = control
        states.append(np.asarray(current_state))
        controls.append(np.asarray(control))
        rollout_counts.append(int(np.asarray(info["rollout_count"])))
        for name, value in info.items():
            if name == "rollout_count" or name.startswith("prediction_"):
                continue
            diagnostics.setdefault(name, []).append(np.asarray(value))
        if episode_monitor is not None:
            episode_termination = episode_monitor.observe(
                np.asarray(current_state), step=step_index + 1
            )

    states_arr = np.asarray(states)
    controls_arr = np.asarray(controls).reshape((-1, control_dim))
    summary = _summary(states_arr, controls_arr, problem)
    summary["controller"] = controller
    summary["seed"] = seed
    summary["budget"] = _budget(controller, cfg)
    summary["horizon"] = horizon
    summary["hard_obstacles"] = bool(hard_obstacles)
    summary["mean_step_time"] = float(np.mean(step_times[1:])) if len(step_times) > 1 else 0.0
    summary["completion"] = summary["success"]
    summary["nominal_cost"] = summary["total_cost"]
    summary["stability"] = bool(
        np.all(np.isfinite(states_arr)) and np.all(np.isfinite(controls_arr))
    )
    if episode_monitor is not None:
        if episode_termination is None:
            episode_termination = episode_monitor.timeout(step=len(controls_arr))
        finite_episode_cost = _manuscript_episode_cost(states_arr, controls_arr, problem)
        summary.update(termination_summary(
            episode_termination, rollout_budget_per_step=summary["budget"]
        ))
        summary["J_CL_ep_finite"] = finite_episode_cost
        summary["J_CL_ep_extended"] = (
            float("inf") if episode_termination.collision else finite_episode_cost
        )
        summary["minimum_clearance"] = episode_termination.minimum_clearance
        summary["safe_success"] = episode_termination.success
        summary["terminal_speed"] = float(np.linalg.norm(states_arr[-1, 3:]))
    return ArmResult(
        controller=controller, seed=seed, budget=summary["budget"],
        states=states_arr, controls=controls_arr,
        step_times=np.asarray(step_times),
        rollout_counts=np.asarray(rollout_counts),
        diagnostics={k: np.asarray(v) for k, v in diagnostics.items()},
        summary=summary,
    )


def _manuscript_episode_cost(states, controls, problem):
    if states.shape[0] != controls.shape[0] + 1:
        raise ValueError("arm episode must have one more state than control")
    ee = np.asarray(end_effector(jnp.asarray(states[1:, :3]), problem), dtype=np.float64)
    velocity = np.asarray(states[1:, 3:], dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    goal = np.asarray(problem.goal, dtype=np.float64)
    ee_squared = np.sum((ee - goal) ** 2, axis=1)
    velocity_squared = np.sum(velocity ** 2, axis=1)
    delta_u = np.diff(
        np.vstack([np.zeros((1, control.shape[1]), dtype=np.float64), control]),
        axis=0,
    )
    running = (
        problem.q_ee * np.sum(ee_squared)
        + problem.q_v * np.sum(velocity_squared)
        + problem.q_u * np.sum(control**2)
        + problem.q_du * np.sum(delta_u**2)
    )
    terminal_ee = np.asarray(
        end_effector(jnp.asarray(states[-1, :3]), problem), dtype=np.float64
    )
    terminal = (
        problem.Q_ee * np.sum((terminal_ee - goal) ** 2)
        + problem.Q_v * np.sum(np.asarray(states[-1, 3:], dtype=np.float64) ** 2)
    )
    return float(running + terminal)


def _budget_of(controller, cfg):
    if controller == "mppi":
        return int(cfg.rollouts)
    if controller == "iterated_mppi":
        return int(cfg.rounds * cfg.rollouts_per_round)
    if controller == "cbo_mpc":
        return int(cfg.cycles * cfg.particles)
    return int(cfg.cycles * cfg.particles * cfg.local_samples)


_budget = _budget_of


def _summary(states, controls, problem):
    ee_traj = np.asarray([
        end_effector(jnp.asarray(s[:3]), problem) for s in states
    ])
    goal = np.asarray(problem.goal)
    ee_err = np.linalg.norm(ee_traj - goal, axis=-1)
    min_clearance = float(trajectory_min_clearance(jnp.asarray(states), problem))
    success = bool(is_success(jnp.asarray(states[-1]), problem))
    return {
        "ee_error_final": float(ee_err[-1]),
        "ee_error_min": float(np.min(ee_err)),
        "success": success,
        "minimum_clearance": min_clearance,
        "safe_success": bool(success and min_clearance >= 0.0),
        "total_cost": float(np.sum(controls ** 2) * problem.q_u
                            + np.sum(np.diff(np.vstack([np.zeros((1,3)), controls]), axis=0)**2) * problem.q_du
                            + problem.Q_ee * ee_err[-1]**2),
        "control_effort": float(np.sum(controls ** 2)),
        "control_variation": float(np.sum(np.diff(np.vstack([np.zeros((1,3)), controls]), axis=0)**2)),
        "max_joint_vel": float(np.max(np.abs(states[:, 3:]))),
    }
