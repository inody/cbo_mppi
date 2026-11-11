"""Generic closed-loop runner for the pendulum swing-up task.

Wires the pendulum rollout into the (factory-built) controllers and steps the
plant forward. Independent of the navigation-specific ``experiment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.episodic import termination_summary
from experiments.episodic_tasks import pendulum_monitor

from cbo_mppi.controllers import (
    CBOMPCState,
    ProposedState,
    init_cbo_mpc,
    init_iterated_mppi,
    init_mppi,
    init_proposed,
    make_cbo_mpc_step,
    make_iterated_mppi_step,
    make_mppi_step,
    make_proposed_step,
)

from .config import PendulumConfig
from .costs import angle_error, closed_loop_metrics, is_success
from .env import initial_state, make_pendulum_rollout, step_dynamics


@dataclass
class PendulumResult:
    controller: str
    seed: int
    budget: int
    states: np.ndarray
    controls: np.ndarray
    step_times: np.ndarray
    rollout_counts: np.ndarray
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _build(controller: str, problem: PendulumConfig, cfg, base_nominal, init_key,
           initial_population=None, cost_style: str = "regulation"):
    rollout_fn = make_pendulum_rollout(cost_style)
    if controller == "mppi":
        return init_mppi(base_nominal), make_mppi_step(rollout_fn)
    if controller == "iterated_mppi":
        return (
            init_iterated_mppi(base_nominal),
            make_iterated_mppi_step(rollout_fn),
        )
    if controller == "cbo_mpc":
        if initial_population is not None:
            state = init_cbo_mpc(
                init_key, base_nominal, problem, cfg, clip_initial=False
            )
            state = CBOMPCState(particles=initial_population)
        else:
            state = init_cbo_mpc(
                init_key, base_nominal, problem, cfg, clip_initial=True
            )
        return state, make_cbo_mpc_step(rollout_fn)
    if controller == "proposed":
        if initial_population is not None:
            state = init_proposed(
                init_key, base_nominal, problem, cfg, clip_initial=False
            )
            state = ProposedState(centers=initial_population)
        else:
            state = init_proposed(
                init_key, base_nominal, problem, cfg, clip_initial=True
            )
        return state, make_proposed_step(rollout_fn)
    raise ValueError(f"unknown controller: {controller}")


def run_pendulum_closed_loop(
    controller: str,
    problem: PendulumConfig,
    cfg,
    *,
    seed: int,
    steps: int,
    base_nominal: jnp.ndarray | None = None,
    initial_population: jnp.ndarray | None = None,
    use_template_init: bool = False,
    cost_style: str = "regulation",
    process_noise_sigma: float = 0.0,
    disturbance_sequence: np.ndarray | None = None,
    episodic: bool = False,
) -> PendulumResult:
    horizon = problem.horizon
    if base_nominal is None:
        base_nominal = jnp.zeros((horizon, 1), dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    key, init_key, pop_key = jax.random.split(key, 3)
    population = initial_population
    if population is None and use_template_init and controller in {"cbo_mpc", "proposed"}:
        n = cfg.particles
        from .templates import template_population

        population = template_population(pop_key, n, horizon, problem.u_max)
    controller_state, step_fn = _build(
        controller, problem, cfg, base_nominal, init_key, population, cost_style
    )

    current_state = initial_state(problem)
    previous_control = jnp.zeros((1,), dtype=jnp.float32)
    if process_noise_sigma < 0.0:
        raise ValueError("process_noise_sigma must be nonnegative")
    if disturbance_sequence is None:
        disturbance_rng = np.random.default_rng(seed + 20_000_033)
        disturbances = disturbance_rng.normal(
            scale=process_noise_sigma, size=(steps, 1)
        ).astype(np.float32)
    else:
        disturbances = np.asarray(disturbance_sequence, dtype=np.float32)
        if disturbances.shape != (steps, 1):
            raise ValueError(
                f"disturbance_sequence must have shape ({steps}, 1), "
                f"got {disturbances.shape}"
            )

    key, controller_state, control, info = step_fn(
        key, controller_state, current_state, previous_control, problem, cfg
    )
    jax.block_until_ready(control)

    states = [np.asarray(current_state)]
    controls = []
    step_times = []
    rollout_counts = []
    diagnostics: dict[str, list[np.ndarray]] = {
        "torque_disturbance": [],
    }
    episode_monitor = pendulum_monitor(problem) if episodic else None
    episode_termination = (
        episode_monitor.observe(np.asarray(current_state), step=0)
        if episode_monitor is not None
        else None
    )
    for step_index in range(steps):
        if episode_termination is not None:
            break
        start = time.perf_counter()
        key, controller_state, control, info = step_fn(
            key,
            controller_state,
            current_state,
            previous_control,
            problem,
            cfg,
        )
        control = jnp.clip(control, -problem.u_max, problem.u_max)
        control.block_until_ready()
        step_times.append(time.perf_counter() - start)
        torque_disturbance = jnp.asarray(disturbances[step_index])
        applied_control = jnp.clip(
            control + torque_disturbance, -problem.u_max, problem.u_max
        )
        current_state = step_dynamics(current_state, applied_control, problem)
        previous_control = control
        states.append(np.asarray(current_state))
        controls.append(np.asarray(control))
        rollout_counts.append(int(np.asarray(info["rollout_count"])))
        diagnostics["torque_disturbance"].append(
            np.asarray(torque_disturbance)
        )
        for name, value in info.items():
            if name == "rollout_count" or name.startswith("prediction_"):
                continue
            diagnostics.setdefault(name, []).append(np.asarray(value))
        if episode_monitor is not None:
            episode_termination = episode_monitor.observe(
                np.asarray(current_state), step=step_index + 1
            )

    states_arr = np.asarray(states)
    controls_arr = np.asarray(controls).reshape((-1, 1))
    summary = closed_loop_metrics(states_arr, controls_arr, problem, cost_style)
    summary["controller"] = controller
    summary["seed"] = seed
    summary["budget"] = _budget_of(controller, cfg)
    summary["horizon"] = horizon
    summary["u_max"] = problem.u_max
    summary["process_noise_sigma"] = process_noise_sigma
    summary["disturbance_sequence_pre_generated"] = True
    summary["completion"] = summary["success"]
    summary["nominal_cost"] = summary["total_cost"]
    summary["stability"] = bool(
        np.all(np.isfinite(states_arr))
        and np.all(np.isfinite(controls_arr))
        and np.max(np.abs(states_arr[:, 1])) <= problem.omega_max
    )
    if episode_monitor is not None:
        if episode_termination is None:
            episode_termination = episode_monitor.timeout(step=len(controls_arr))
        finite_episode_cost = float(summary["total_cost"])
        summary.update(termination_summary(
            episode_termination, rollout_budget_per_step=summary["budget"]
        ))
        summary["J_CL_ep_finite"] = finite_episode_cost
        summary["J_CL_ep_extended"] = finite_episode_cost
    return PendulumResult(
        controller=controller,
        seed=seed,
        budget=summary["budget"],
        states=states_arr,
        controls=controls_arr,
        step_times=np.asarray(step_times),
        rollout_counts=np.asarray(rollout_counts),
        diagnostics={
            name: np.asarray(values) for name, values in diagnostics.items()
        },
        summary=summary,
    )


def _budget_of(controller: str, cfg) -> int:
    if controller == "mppi":
        return int(cfg.rollouts)
    if controller == "iterated_mppi":
        return int(cfg.rounds * cfg.rollouts_per_round)
    if controller == "cbo_mpc":
        return int(cfg.cycles * cfg.particles)
    return int(cfg.cycles * cfg.particles * cfg.local_samples)


def aggregate_summaries(results: list[PendulumResult]) -> dict[str, dict]:
    grouped: dict[str, list[PendulumResult]] = {}
    for r in results:
        grouped.setdefault(r.controller, []).append(r)
    out: dict[str, dict] = {}
    for name, group in grouped.items():
        successes = np.array([r.summary["success"] for r in group], dtype=float)
        costs = np.array([r.summary["total_cost"] for r in group])
        terminal = np.array([r.summary["terminal_angle_error"] for r in group])
        out[name] = {
            "n": len(group),
            "success_rate": float(np.mean(successes)),
            "cost_median": float(np.median(costs)),
            "cost_iqr": float(np.percentile(costs, 75) - np.percentile(costs, 25)),
            "terminal_angle_median": float(np.median(terminal)),
        }
    return out
