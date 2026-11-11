"""Closed-loop runner and result persistence for the navigation experiment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cbo_mppi.config import (
    CBOConfig,
    IteratedMPPIConfig,
    MPPIConfig,
    ProposedConfig,
    validate_isotropic_cbo,
)
from cbo_mppi.controllers import (
    init_cbo_mpc,
    init_iterated_mppi,
    init_mppi,
    init_proposed,
    make_cbo_mpc_step,
    make_iterated_mppi_step,
    make_mppi_step,
    make_proposed_step,
)

from experiments.episodic import termination_summary
from experiments.episodic_tasks import navigation_monitor

from .config import ProblemConfig, load_experiment_config, make_problem  # noqa: F401
from .env import double_integrator_step, make_initial_nominal, rollout_batch


@dataclass(frozen=True)
class SimulationResult:
    controller: str
    seed: int
    budget: int
    states: np.ndarray
    controls: np.ndarray
    step_times: np.ndarray
    rollout_counts: np.ndarray
    diagnostics: dict[str, np.ndarray]
    summary: dict[str, Any]


def make_controller_config(
    controller: str,
    raw: dict[str, Any],
    budget: int,
    cbo_cycles: int,
    proposed_cycles: int,
    local_samples: int,
    record_history: bool = True,
    iterated_mppi_rounds: int = 1,
):
    if controller in {"mppi", "iterated_mppi"}:
        section = raw["mppi"]
        common = {
            "tau": float(section["tau"]),
            "noise_sigma": float(section["noise_sigma"]),
            "noise_correlation": float(section["noise_correlation"]),
            "clip_rollout_inputs": bool(section.get("clip_rollout_inputs", False)),
        }
        if controller == "mppi":
            return MPPIConfig(rollouts=budget, **common)
        if iterated_mppi_rounds <= 0 or budget % iterated_mppi_rounds:
            raise ValueError("budget must be divisible by positive MPPI rounds")
        return IteratedMPPIConfig(
            rollouts_per_round=budget // iterated_mppi_rounds,
            rounds=iterated_mppi_rounds,
            **common,
        )
    if controller == "cbo_mpc":
        if budget % cbo_cycles:
            raise ValueError("budget must be divisible by cbo_cycles")
        section = raw["cbo"]
        warm_start = section.get("warm_start", {})
        return CBOConfig(
            particles=budget // cbo_cycles,
            cycles=cbo_cycles,
            alpha=float(section["alpha"]),
            lambda_=float(section["lambda"]),
            sigma=float(section["sigma"]),
            delta_s=float(section["delta_s"]),
            init_sigma=float(section["init_sigma"]),
            init_noise_correlation=float(section["init_noise_correlation"]),
            warm_start_mode=str(warm_start.get("mode", "shift")),
            warm_start_jitter_kind=str(warm_start.get("jitter_kind", "smooth")),
            warm_start_sigma=float(warm_start.get("sigma", 0.0)),
            warm_start_refresh_rate=float(warm_start.get("refresh_rate", 0.0)),
            warm_start_min_cov_trace=float(warm_start.get("min_cov_trace", 0.0)),
            warm_start_apply_to=str(warm_start.get("apply_to", "all_particles")),
            record_history=record_history,
            clip_rollout_inputs=bool(section.get("clip_rollout_inputs", False)),
        )
    if controller == "proposed":
        divisor = proposed_cycles * local_samples
        if budget % divisor:
            raise ValueError("budget must be divisible by proposed_cycles*local_samples")
        section = raw["proposed"]
        warm_start = section.get("warm_start", {})
        eta_grad = float(section.get("eta_grad", 0.0))
        grad_clip = float(section.get("grad_clip", 0.0))
        if eta_grad < 0.0:
            raise ValueError("proposed eta_grad must be nonnegative")
        if grad_clip < 0.0:
            raise ValueError("proposed grad_clip must be nonnegative")
        return ProposedConfig(
            particles=budget // divisor,
            cycles=proposed_cycles,
            local_samples=local_samples,
            tau=float(section["tau"]),
            local_noise_sigma=float(section["local_noise_sigma"]),
            local_noise_correlation=float(section["local_noise_correlation"]),
            alpha=float(section["alpha"]),
            lambda_=float(section["lambda"]),
            sigma=float(section["sigma"]),
            delta_s=float(section["delta_s"]),
            init_sigma=float(section["init_sigma"]),
            init_noise_correlation=float(section["init_noise_correlation"]),
            output_rule=str(section.get("output_rule", "weighted")),
            shared_local_noise=bool(section.get("shared_local_noise", False)),
            warm_start_mode=str(warm_start.get("mode", "shift")),
            warm_start_jitter_kind=str(warm_start.get("jitter_kind", "smooth")),
            warm_start_sigma=float(warm_start.get("sigma", 0.0)),
            warm_start_refresh_rate=float(warm_start.get("refresh_rate", 0.0)),
            warm_start_min_cov_trace=float(warm_start.get("min_cov_trace", 0.0)),
            warm_start_apply_to=str(warm_start.get("apply_to", "all_particles")),
            record_history=record_history,
            eta_grad=eta_grad,
            grad_clip=grad_clip,
            clip_rollout_inputs=bool(section.get("clip_rollout_inputs", False)),
            center_update_rule=str(section.get("center_update_rule", "cbo")),
        )
    raise ValueError(f"unknown controller: {controller}")


def _block_tree(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _limited_prediction_population(
    sequences: jnp.ndarray,
    scores: jnp.ndarray,
    limit: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    population = scores.shape[0]
    if limit >= population:
        return sequences, scores
    elite_count = min(population, max(1, 3 * limit // 4))
    spread_count = limit - elite_count
    ranking_logits = jnp.where(jnp.isfinite(scores), -scores, -jnp.inf)
    _, elite_indices = jax.lax.top_k(ranking_logits, elite_count)
    if spread_count:
        spread_indices = jnp.linspace(
            0, population - 1, spread_count
        ).astype(jnp.int32)
        indices = jnp.concatenate((elite_indices, spread_indices))
    else:
        indices = elite_indices
    return sequences[indices], scores[indices]


def _closed_loop_summary(
    states: np.ndarray,
    controls: np.ndarray,
    problem: ProblemConfig,
    step_times: np.ndarray,
) -> dict[str, Any]:
    positions = states[:, :2]
    goal = np.asarray(problem.goal)
    obstacle_centers = np.asarray(problem.obstacle_centers)
    obstacle_radii = np.asarray(problem.obstacle_radii)
    distances = np.linalg.norm(
        positions[:, None, :] - obstacle_centers[None, :, :], axis=-1
    )
    collision = bool(np.any(distances < obstacle_radii[None, :]))
    goal_error = float(np.linalg.norm(positions[-1] - goal))
    running_goal = problem.q_goal_running * np.sum((positions[1:] - goal) ** 2)
    terminal_goal = problem.q_goal_terminal * goal_error**2
    running_velocity = problem.q_velocity_running * np.sum(states[1:, 2:] ** 2)
    terminal_velocity = problem.q_velocity_terminal * np.sum(states[-1, 2:] ** 2)
    control_effort = problem.q_control * np.sum(controls**2)
    controls_with_boundary = np.concatenate(
        (np.zeros((1, controls.shape[1]), dtype=controls.dtype), controls), axis=0
    )
    control_differences = np.diff(controls_with_boundary, axis=0)
    control_jump_norms = np.linalg.norm(control_differences, axis=1)
    control_rate = problem.q_control_rate * np.sum(control_differences**2)
    clearances = distances - obstacle_radii[None, :]
    minimum_safe_clearance = float(np.min(clearances))
    safety_violation = bool(minimum_safe_clearance <= problem.safety_margin)
    safe_radii = obstacle_radii + problem.safety_margin
    penetration = np.maximum(0.0, safe_radii[None, :] - distances[1:])
    obstacle_cost = problem.q_obstacle * np.sum(penetration**2)
    closest_to_obstacle = int(np.argmin(np.abs(positions[:, 0])))
    crossing_y = float(positions[closest_to_obstacle, 1])
    route_class = "upper" if crossing_y > 0.15 else "lower" if crossing_y < -0.15 else "center"
    return {
        "closed_loop_cost": float(
            running_goal
            + terminal_goal
            + running_velocity
            + terminal_velocity
            + control_effort
            + control_rate
            + obstacle_cost
        ),
        "success": bool(goal_error <= 0.6 and not collision),
        "safe_success": bool(
            goal_error <= 0.6 and not collision and not safety_violation
        ),
        "collision": collision,
        "safety_violation": safety_violation,
        "minimum_safe_clearance": minimum_safe_clearance,
        "goal_error": goal_error,
        "control_smoothness": float(np.sum(control_differences**2)),
        "input_variation": float(np.sum(control_jump_norms)),
        "max_control_jump": float(np.max(control_jump_norms, initial=0.0)),
        "route_class": route_class,
        "crossing_y": crossing_y,
        "mean_step_time_seconds": float(np.mean(step_times)) if len(step_times) else 0.0,
        "median_step_time_seconds": float(np.median(step_times)) if len(step_times) else 0.0,
    }


def run_closed_loop(
    controller: str,
    raw_config: dict[str, Any],
    *,
    budget: int,
    seed: int,
    simulation_steps: int | None = None,
    cbo_cycles: int = 8,
    proposed_cycles: int = 8,
    local_samples: int = 8,
    iterated_mppi_rounds: int = 1,
    initial_route_height: float = 0.0,
    initial_route_height_sigma: float = 0.0,
    matched_initial_distribution: bool = False,
    initial_u_noise_sigma: float = 1.0,
    process_acceleration_noise_sigma: float = 0.0,
    capture_predictions: bool = False,
    prediction_capture_limit: int | None = None,
    record_history: bool = True,
    episodic: bool = False,
) -> SimulationResult:
    problem = make_problem(raw_config)
    problem_raw = raw_config["problem"]
    horizon = int(problem_raw["horizon"])
    steps = int(simulation_steps or problem_raw["simulation_steps"])
    nominal_profile = str(problem_raw.get("nominal_profile", "minimum_jerk"))
    current_state = jnp.asarray(problem_raw["initial_state"], dtype=jnp.float32)
    if process_acceleration_noise_sigma < 0.0:
        raise ValueError("process_acceleration_noise_sigma must be nonnegative")
    if prediction_capture_limit is not None and prediction_capture_limit <= 0:
        raise ValueError("prediction_capture_limit must be positive")
    disturbance_rng = np.random.default_rng(seed + 10_000_019)
    acceleration_disturbances = disturbance_rng.normal(
        scale=process_acceleration_noise_sigma, size=(steps, 2)
    ).astype(np.float32)
    base_nominal = make_initial_nominal(
        current_state, problem, horizon,
        route_height=initial_route_height, profile=nominal_profile,
    )
    route_direction = make_initial_nominal(
        current_state, problem, horizon, route_height=1.0, profile=nominal_profile
    ) - make_initial_nominal(
        current_state, problem, horizon, route_height=0.0, profile=nominal_profile
    )
    cfg = make_controller_config(
        controller, raw_config, budget, cbo_cycles, proposed_cycles,
        local_samples, record_history=record_history,
        iterated_mppi_rounds=iterated_mppi_rounds,
    )
    if controller in {"cbo_mpc", "proposed"}:
        validate_isotropic_cbo(horizon * 2, cfg.lambda_, cfg.sigma)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    cbo_initial_noise_sigma = None
    proposed_initial_noise_sigma = None
    mppi_initial_noise_sigma = None
    mppi_initial_route_sigma = 0.0
    if matched_initial_distribution:
        if initial_u_noise_sigma <= 0.0:
            raise ValueError("initial_u_noise_sigma must be positive")
        cbo_initial_noise_sigma = initial_u_noise_sigma
        mppi_initial_noise_sigma = initial_u_noise_sigma
        mppi_initial_route_sigma = initial_route_height_sigma
        if controller == "proposed":
            residual_variance = initial_u_noise_sigma**2 - cfg.local_noise_sigma**2
            if residual_variance < 0.0:
                raise ValueError(
                    "matched initialization requires initial_u_noise_sigma "
                    ">= proposed local_noise_sigma"
                )
            proposed_initial_noise_sigma = residual_variance**0.5

    step_function = make_mppi_step(rollout_batch)
    if controller == "mppi":
        controller_state = init_mppi(
            base_nominal,
            route_direction=route_direction,
            initial_noise_sigma=mppi_initial_noise_sigma,
            initial_route_sigma=mppi_initial_route_sigma,
        )
    elif controller == "iterated_mppi":
        step_function = make_iterated_mppi_step(rollout_batch)
        controller_state = init_iterated_mppi(
            base_nominal,
            route_direction=route_direction,
            initial_noise_sigma=mppi_initial_noise_sigma,
            initial_route_sigma=mppi_initial_route_sigma,
        )
    elif controller == "cbo_mpc":
        step_function = make_cbo_mpc_step(rollout_batch)
        controller_state = init_cbo_mpc(
            init_key, base_nominal, problem, cfg,
            structured_direction=route_direction,
            structured_sigma=initial_route_height_sigma,
            noise_sigma=cbo_initial_noise_sigma,
            clip_initial=not matched_initial_distribution,
        )
    else:
        step_function = make_proposed_step(rollout_batch)
        controller_state = init_proposed(
            init_key, base_nominal, problem, cfg,
            structured_direction=route_direction,
            structured_sigma=initial_route_height_sigma,
            noise_sigma=proposed_initial_noise_sigma,
            clip_initial=not matched_initial_distribution,
        )

    previous_control = jnp.zeros(2, dtype=current_state.dtype)
    warm_result = step_function(
        key, controller_state, current_state, previous_control, problem, cfg
    )
    _block_tree(warm_result)

    states = [np.asarray(current_state)]
    controls = []
    step_times = []
    rollout_counts = []
    diagnostics: dict[str, list[np.ndarray]] = {}
    episode_monitor = navigation_monitor(problem) if episodic else None
    episode_termination = (
        episode_monitor.observe(np.asarray(current_state), step=0)
        if episode_monitor is not None
        else None
    )
    for step_index in range(steps):
        if episode_termination is not None:
            break
        start = time.perf_counter()
        key, controller_state, control, info = step_function(
            key, controller_state, current_state, previous_control, problem, cfg,
        )
        control = jnp.clip(control, -problem.u_max, problem.u_max)
        control.block_until_ready()
        step_times.append(time.perf_counter() - start)
        applied_acceleration = control + jnp.asarray(
            acceleration_disturbances[step_index]
        )
        current_state = double_integrator_step(
            current_state, applied_acceleration, problem.dt
        )
        previous_control = control
        states.append(np.asarray(current_state))
        controls.append(np.asarray(control))
        rollout_counts.append(int(np.asarray(info["rollout_count"])))
        prediction_overrides: dict[str, jax.Array] = {}
        if capture_predictions and prediction_capture_limit is not None:
            sequences, scores = _limited_prediction_population(
                info["prediction_sequences"],
                info["prediction_scores"],
                prediction_capture_limit,
            )
            prediction_overrides["prediction_sequences"] = sequences
            prediction_overrides["prediction_scores"] = scores
            if "prediction_local_sequences" in info:
                local_sequences, local_scores = _limited_prediction_population(
                    info["prediction_local_sequences"],
                    info["prediction_local_scores"],
                    prediction_capture_limit,
                )
                prediction_overrides["prediction_local_sequences"] = local_sequences
                prediction_overrides["prediction_local_scores"] = local_scores
        for name, value in info.items():
            if name == "rollout_count":
                continue
            if name.startswith("prediction_") and not capture_predictions:
                continue
            value = prediction_overrides.get(name, value)
            diagnostics.setdefault(name, []).append(np.asarray(value))
        if episode_monitor is not None:
            episode_termination = episode_monitor.observe(
                np.asarray(current_state), step=step_index + 1
            )

    states_array = np.asarray(states)
    controls_array = np.asarray(controls).reshape((-1, 2))
    times_array = np.asarray(step_times)
    summary = _closed_loop_summary(states_array, controls_array, problem, times_array)
    summary["compile_excluded"] = True
    summary["rollouts_per_step"] = budget
    summary["horizon"] = horizon
    summary["controller_config"] = dict(cfg._asdict())
    summary["initial_route_height"] = initial_route_height
    summary["initial_route_height_sigma"] = initial_route_height_sigma
    summary["matched_initial_distribution"] = matched_initial_distribution
    summary["initial_u_noise_sigma"] = initial_u_noise_sigma
    summary["nominal_profile"] = nominal_profile
    summary["process_acceleration_noise_sigma"] = process_acceleration_noise_sigma
    summary["capture_predictions"] = capture_predictions
    summary["prediction_capture_limit"] = prediction_capture_limit
    if episode_monitor is not None:
        if episode_termination is None:
            episode_termination = episode_monitor.timeout(step=len(controls_array))
        finite_episode_cost = float(summary["closed_loop_cost"])
        summary.update(termination_summary(
            episode_termination, rollout_budget_per_step=budget
        ))
        summary["J_CL_ep_finite"] = finite_episode_cost
        summary["J_CL_ep_extended"] = (
            float("inf") if episode_termination.collision else finite_episode_cost
        )
        summary["terminal_speed"] = float(np.linalg.norm(states_array[-1, 2:]))
        if np.min(states_array[:, 0]) > 0.0 or np.max(states_array[:, 0]) < 0.0:
            summary["route_class"] = "no_crossing"
            summary["crossing_y"] = None
    if controller == "mppi":
        summary["evaluation_cycles"] = 1
        summary["particles"] = 1
        summary["local_samples"] = cfg.rollouts
    elif controller == "iterated_mppi":
        summary["evaluation_cycles"] = cfg.rounds
        summary["particles"] = 1
        summary["local_samples"] = cfg.rollouts_per_round
    elif controller == "cbo_mpc":
        summary["evaluation_cycles"] = cfg.cycles
        summary["particles"] = cfg.particles
        summary["local_samples"] = 1
    else:
        summary["evaluation_cycles"] = cfg.cycles
        summary["particles"] = cfg.particles
        summary["local_samples"] = cfg.local_samples
    return SimulationResult(
        controller=controller,
        seed=seed,
        budget=budget,
        states=states_array,
        controls=controls_array,
        step_times=times_array,
        rollout_counts=np.asarray(rollout_counts),
        diagnostics={name: np.asarray(values) for name, values in diagnostics.items()},
        summary=summary,
    )


def save_result(result: SimulationResult, output_directory: str | Path) -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.controller}_budget{result.budget}_seed{result.seed}"
    data_path = output_directory / f"{stem}.npz"
    np.savez_compressed(
        data_path,
        states=result.states,
        controls=result.controls,
        step_times=result.step_times,
        rollout_counts=result.rollout_counts,
        **{f"diagnostic_{key}": value for key, value in result.diagnostics.items()},
    )
    metadata = {
        "controller": result.controller,
        "seed": result.seed,
        "budget": result.budget,
        "summary": result.summary,
    }
    (output_directory / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return data_path
