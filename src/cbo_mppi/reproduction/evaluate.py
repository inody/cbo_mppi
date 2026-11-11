"""Post-episode realized-cost and free-energy evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cbo_mppi.utils import temporally_correlated_normal
from experiments.episodic_tasks import arm_clearance, monitor_for_task, navigation_clearance
from experiments.navigation.closed_loop import _closed_loop_summary
from experiments.navigation.config import make_problem
from experiments.navigation.env import rollout_batch as navigation_rollout
from experiments.pendulum_swingup.costs import closed_loop_metrics
from experiments.pendulum_swingup.env import make_rollout_batch as make_pendulum_rollout
from experiments.planar_arm.closed_loop import _manuscript_episode_cost
from experiments.planar_arm.env import make_arm_hard_rollout

from .nominal import (
    CONTROL_DIMENSION,
    LOCAL_SIGMA,
    RHO,
    STEPS,
    TASKS,
    TAUS,
    Cell,
    arm_problem,
    cells,
    navigation_raw,
    pendulum_problem,
    select_cell,
)

ROOT = Path(__file__).resolve().parents[3]
EVALUATION_SAMPLES = {
    "navigation": 32_768,
    "many_obstacles": 8_192,
    "pendulum": 32_768,
    "planar_arm": 32_768,
}
EVALUATION_SEED = {
    "nominal": {
        "navigation": 2_026_1029,
        "many_obstacles": 2_026_1029,
        "pendulum": 2_026_1029,
        "planar_arm": 2_026_1101,
    },
    "sensitivity": {task: 2_026_1121 for task in TASKS},
    "horizon": {task: 2_026_1111 for task in TASKS},
}
TASK_INDEX = {task: index for index, task in enumerate(TASKS)}
CHUNK_SIZE = 512

_PENDULUM_ROLLOUT = make_pendulum_rollout("regulation")
_ARM_ROLLOUT = jax.jit(make_arm_hard_rollout(), static_argnames=("clip_inputs",))


@jax.jit
def _navigation_costs(initial_state, controls, noise, problem):
    perturbed = jnp.clip(
        controls[None, :, :] + noise, -problem.u_max, problem.u_max
    )
    costs, _ = navigation_rollout(initial_state, perturbed, problem, None)
    return costs


def task_key(task: str, campaign: str = "nominal"):
    # The corrected nominal Planar Arm campaign used a direct key. The nominal
    # three-task block and both auxiliary campaigns fold in the task index.
    key = jax.random.PRNGKey(EVALUATION_SEED[campaign][task])
    if campaign == "nominal" and task == "planar_arm":
        return key
    return jax.random.fold_in(key, TASK_INDEX[task])


def _paired_noise(key, samples: int, horizon: int, dimension: int, sigma: float, rho: float):
    if samples % 2:
        raise ValueError("paired antithetic samples must be even")
    base = sigma * temporally_correlated_normal(
        key, (samples // 2, horizon, dimension), rho
    )
    return jnp.concatenate((base, -base), axis=0)


def _navigation_noise(key, samples: int, horizon: int, sigma: float, rho: float):
    if samples % 4:
        raise ValueError("navigation antithetic samples must be divisible by four")
    base = sigma * temporally_correlated_normal(
        key, (samples // 4, horizon, 2), rho
    )
    x_sign = jnp.asarray([-1.0, 1.0])
    y_sign = jnp.asarray([1.0, -1.0])
    return jnp.concatenate((base, base * x_sign, base * y_sign, -base), axis=0)


def noise_paths(
    task: str, output_root: Path, campaign: str = "nominal"
) -> tuple[Path, Path]:
    root = output_root / "evaluation/common_noise" / campaign
    return root / f"{task}.npy", root / f"{task}.json"


def ensure_noise(
    task: str, output_root: Path, campaign: str = "nominal"
) -> tuple[Path, dict]:
    if campaign not in EVALUATION_SEED:
        raise ValueError(f"unknown evaluation campaign: {campaign}")
    npy, metadata_path = noise_paths(task, output_root, campaign)
    samples = EVALUATION_SAMPLES[task]
    shape = (samples, STEPS[task], CONTROL_DIMENSION[task])
    if npy.exists() or metadata_path.exists():
        if not npy.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"partial common-noise artifact for {task}")
        array = np.load(npy, mmap_mode="r", allow_pickle=False)
        metadata = json.loads(metadata_path.read_text())
        if tuple(array.shape) != shape:
            raise RuntimeError(f"common-noise shape mismatch for {task}: {array.shape}")
        data_hash = hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()
        if metadata.get("noise_data_sha256") != data_hash:
            raise RuntimeError(f"common-noise hash mismatch for {task}")
        return npy, metadata

    key = task_key(task, campaign)
    if task in {"navigation", "many_obstacles"}:
        generated = _navigation_noise(
            key, samples, STEPS[task], LOCAL_SIGMA[task], RHO[task]
        )
        construction = "base,x_sign,y_sign,negative_base"
    else:
        generated = _paired_noise(
            key, samples, STEPS[task], CONTROL_DIMENSION[task],
            LOCAL_SIGMA[task], RHO[task],
        )
        construction = "paired_base_and_negative_base"
    array = np.asarray(generated, dtype=np.float32)
    if array.shape != shape:
        raise RuntimeError(f"generated common-noise shape mismatch: {array.shape}")
    npy.parent.mkdir(parents=True, exist_ok=True)
    with npy.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "task": task,
        "campaign": campaign,
        "evaluation_seed": EVALUATION_SEED[campaign][task],
        "task_index": (
            None if campaign == "nominal" and task == "planar_arm"
            else TASK_INDEX[task]
        ),
        "shape": list(shape),
        "dtype": str(array.dtype),
        "samples": samples,
        "horizon": STEPS[task],
        "control_dimension": CONTROL_DIMENSION[task],
        "noise_sigma": LOCAL_SIGMA[task],
        "noise_correlation": RHO[task],
        "antithetic_construction": construction,
        "common_across_methods_seeds_temperatures": True,
        "noise_data_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "npy_sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return npy, metadata


def problem_for(cell: Cell):
    if cell.task in {"navigation", "many_obstacles"}:
        return make_problem(navigation_raw(cell))
    if cell.task == "pendulum":
        return pendulum_problem()
    return arm_problem()


def finite_free_energy(costs: np.ndarray, tau: float) -> float:
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("inf")
    logits = -finite / tau
    maximum = float(np.max(logits))
    log_sum = maximum + math.log(float(np.sum(np.exp(logits - maximum))))
    return float(-tau * (log_sum - math.log(len(values))))


def evaluate_costs(
    cell: Cell,
    initial_state: np.ndarray,
    controls: np.ndarray,
    noise_path: Path,
) -> np.ndarray:
    samples = EVALUATION_SAMPLES[cell.task]
    noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
    problem = problem_for(cell)
    chunks: list[np.ndarray] = []
    for start in range(0, samples, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, samples)
        chunk = np.asarray(noise[start:stop, : len(controls), :])
        if cell.task in {"navigation", "many_obstacles"}:
            costs = _navigation_costs(
                jnp.asarray(initial_state), jnp.asarray(controls),
                jnp.asarray(chunk), problem,
            )
        elif cell.task == "pendulum":
            perturbed = jnp.asarray(controls)[None, :, :] + jnp.asarray(chunk)
            costs, _ = _PENDULUM_ROLLOUT(
                jnp.asarray(initial_state), perturbed, problem,
                clip_inputs=True, return_trajectory=False,
            )
        else:
            perturbed = jnp.asarray(controls)[None, :, :] + jnp.asarray(chunk)
            costs, _ = _ARM_ROLLOUT(
                jnp.asarray(initial_state), perturbed, problem,
                jnp.zeros((CONTROL_DIMENSION[cell.task],), dtype=perturbed.dtype),
                clip_inputs=True,
            )
        chunks.append(np.asarray(costs))
    result = np.concatenate(chunks)
    if result.shape != (samples,):
        raise RuntimeError(f"unexpected evaluated-cost shape: {result.shape}")
    return result


def nominal_metrics(cell: Cell, states: np.ndarray, controls: np.ndarray) -> dict:
    problem = problem_for(cell)
    monitor_task = "two_obstacles" if cell.task == "navigation" else cell.task
    monitor = monitor_for_task(monitor_task, problem)
    termination = monitor.observe(states[0], step=0)
    for step in range(1, len(states)):
        if termination is None:
            termination = monitor.observe(states[step], step=step)
    if termination is None:
        termination = monitor.timeout(step=len(controls))

    if cell.task in {"navigation", "many_obstacles"}:
        summary = _closed_loop_summary(
            states, controls, problem, np.zeros((len(controls),))
        )
        finite_cost = float(summary["closed_loop_cost"])
        c_min = float(min(navigation_clearance(state, problem) for state in states))
        route_class = summary["route_class"]
        terminal_speed = float(np.linalg.norm(states[-1, 2:]))
    elif cell.task == "pendulum":
        summary = closed_loop_metrics(states, controls, problem, style="regulation")
        finite_cost = float(summary["total_cost"])
        c_min = None
        route_class = None
        terminal_speed = None
    else:
        finite_cost = float(_manuscript_episode_cost(states, controls, problem))
        c_min = float(min(arm_clearance(state, problem) for state in states))
        route_class = None
        terminal_speed = float(np.linalg.norm(states[-1, 3:]))
    collision = c_min is not None and c_min < 0.0
    return {
        "success": bool(termination.success),
        "termination": termination.reason,
        "success_step": termination.step if termination.success else None,
        "minimum_clearance": c_min,
        "collision": collision,
        "route_class": route_class,
        "terminal_speed": terminal_speed,
        "J_CL": finite_cost,
        "J_CL_extended": None if collision else finite_cost,
    }


def result_paths(cell: Cell, seed: int, output_root: Path) -> tuple[Path, Path]:
    root = output_root / "nominal" / cell.task / cell.cell_id
    return root / f"seed{seed}.npz", root / f"seed{seed}.json"


def metric_path(cell: Cell, seed: int, tau: float, output_root: Path) -> Path:
    tag = str(int(tau)) if float(tau).is_integer() else str(tau).replace(".", "p")
    return output_root / "evaluation/metrics" / cell.task / cell.cell_id / f"tau{tag}_seed{seed}.json"


def evaluate_one(cell: Cell, seed: int, tau: float, output_root: Path) -> Path:
    source_npz, source_json = result_paths(cell, seed, output_root)
    if not source_npz.is_file() or not source_json.is_file():
        raise FileNotFoundError(f"missing nominal trajectory for {cell.cell_id} seed {seed}")
    destination = metric_path(cell, seed, tau, output_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite metric: {destination}")
    with np.load(source_npz, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], dtype=np.float32)
        controls = np.asarray(archive["controls"], dtype=np.float32)
    expected = STEPS[cell.task]
    if states.shape[0] != expected + 1 or controls.shape[0] != expected:
        raise RuntimeError(
            f"trajectory length mismatch for {cell.task}: {states.shape}, {controls.shape}"
        )
    noise_path, noise_metadata = ensure_noise(cell.task, output_root)
    costs = evaluate_costs(cell, states[0], controls, noise_path)
    finite = costs[np.isfinite(costs)]
    metrics = nominal_metrics(cell, states, controls)
    free_energy = finite_free_energy(costs, tau)
    source_hash = hashlib.sha256(source_npz.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "task": cell.task,
        "method": cell.method,
        "cell_id": cell.cell_id,
        "seed": seed,
        "tau": tau,
        **metrics,
        "F_CL": None if not math.isfinite(free_energy) else free_energy,
        "F_CL_infinite": not math.isfinite(free_energy),
        "evaluation_samples": EVALUATION_SAMPLES[cell.task],
        "finite_cost_count": int(len(finite)),
        "feasible_fraction": float(len(finite) / len(costs)),
        "finite_cost_min": float(np.min(finite)) if len(finite) else None,
        "finite_cost_median": float(np.median(finite)) if len(finite) else None,
        "source_npz_sha256": source_hash,
        "noise_data_sha256": noise_metadata["noise_data_sha256"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def evaluation_taus(cell: Cell) -> tuple[float, ...]:
    return TAUS[cell.task] if cell.method == "cbo_mpc" else (float(cell.tau),)


def external_metric_path(
    stage: str, task: str, cell_id: str, seed: int, tau: float, output_root: Path
) -> Path:
    tag = str(int(tau)) if float(tau).is_integer() else str(tau).replace(".", "p")
    return output_root / "evaluation" / stage / task / cell_id / f"tau{tag}_seed{seed}.json"


def evaluate_external(
    stage: str,
    task: str,
    cell_id: str,
    seed: int,
    tau: float,
    source_npz: Path,
    output_root: Path,
    *,
    cell_metadata: dict,
) -> Path:
    destination = external_metric_path(stage, task, cell_id, seed, tau, output_root)
    if destination.exists():
        raise FileExistsError(destination)
    with np.load(source_npz, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], dtype=np.float32)
        controls = np.asarray(archive["controls"], dtype=np.float32)
    common = {
        "schema_version": 1,
        "stage": stage,
        "task": task,
        "cell_id": cell_id,
        "cell": cell_metadata,
        "seed": seed,
        "tau": tau,
        "source_npz_sha256": hashlib.sha256(source_npz.read_bytes()).hexdigest(),
    }
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        nonfinite_steps = np.flatnonzero(~np.isfinite(states).all(axis=1))
        payload = {
            **common,
            "metric_status": "completed_with_numerical_failure",
            "failure_status": "nonfinite_trajectory",
            "first_nonfinite_state_step": (
                int(nonfinite_steps[0]) if len(nonfinite_steps) else None
            ),
            "success": None,
            "termination": "nonfinite_trajectory",
            "success_step": None,
            "minimum_clearance": None,
            "collision": None,
            "route_class": None,
            "terminal_speed": None,
            "J_CL": None,
            "J_CL_extended": None,
            "F_CL": None,
            "F_CL_infinite": None,
            "evaluation_samples": EVALUATION_SAMPLES[task],
            "finite_cost_count": None,
            "feasible_fraction": None,
            "noise_data_sha256": None,
        }
    else:
        surrogate = Cell(task, "mppi", tau)
        noise_path, noise_metadata = ensure_noise(task, output_root, stage)
        costs = evaluate_costs(surrogate, states[0], controls, noise_path)
        finite = costs[np.isfinite(costs)]
        free_energy = finite_free_energy(costs, tau)
        payload = {
            **common,
            "metric_status": "evaluated" if len(finite) else "evaluated_all_infeasible",
            **nominal_metrics(surrogate, states, controls),
            "F_CL": None if not math.isfinite(free_energy) else free_energy,
            "F_CL_infinite": not math.isfinite(free_energy),
            "evaluation_samples": EVALUATION_SAMPLES[task],
            "finite_cost_count": int(len(finite)),
            "feasible_fraction": float(len(finite) / len(costs)),
            "noise_data_sha256": noise_metadata["noise_data_sha256"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=("mppi", "multistart_mppi", "cbo_mpc", "cbo_mppi"), required=True)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluation-tau", type=float)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/simulation")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cell = select_cell(args.task, args.method, args.tau)
    taus = (
        (args.evaluation_tau,) if args.evaluation_tau is not None
        else evaluation_taus(cell)
    )
    print(json.dumps({
        "task": cell.task,
        "cell": cell.manifest(),
        "seed": args.seed,
        "evaluation_taus": taus,
        "evaluation_samples": EVALUATION_SAMPLES[cell.task],
        "timing": "not measured",
    }, indent=2))
    if not args.execute:
        return
    for tau in taus:
        print(evaluate_one(cell, args.seed, float(tau), args.output_root.resolve()))


if __name__ == "__main__":
    main()
