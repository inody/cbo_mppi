"""Canonical nominal closed-loop campaign used by the manuscript."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from cbo_mppi.config import CBOConfig, MPPIConfig, ProposedConfig
from experiments.navigation.closed_loop import run_closed_loop
from experiments.navigation.config import load_experiment_config
from experiments.pendulum_swingup.closed_loop import run_pendulum_closed_loop
from experiments.pendulum_swingup.config import PendulumConfig
from experiments.planar_arm.closed_loop import run_arm_closed_loop
from experiments.planar_arm.config import ArmConfig

ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "configs/base"

Task = Literal["navigation", "many_obstacles", "pendulum", "planar_arm"]
Method = Literal["mppi", "multistart_mppi", "cbo_mpc", "cbo_mppi"]

TASKS: tuple[Task, ...] = (
    "navigation", "many_obstacles", "pendulum", "planar_arm"
)
METHODS: tuple[Method, ...] = (
    "mppi", "multistart_mppi", "cbo_mpc", "cbo_mppi"
)
HORIZON = {"navigation": 60, "many_obstacles": 50, "pendulum": 80, "planar_arm": 20}
STEPS = {"navigation": 80, "many_obstacles": 120, "pendulum": 170, "planar_arm": 35}
BUDGET = {"navigation": 131_072, "many_obstacles": 131_072, "pendulum": 16_384, "planar_arm": 131_072}
TAUS = {
    "navigation": (15.0, 300.0),
    "many_obstacles": (15.0, 3000.0),
    "pendulum": (1.0, 10.0),
    "planar_arm": (3.0, 300.0),
}
SEEDS = {
    "navigation": tuple(range(51_000, 51_100)),
    "many_obstacles": tuple(range(51_100, 51_200)),
    "pendulum": tuple(range(51_200, 51_300)),
    "planar_arm": tuple(range(60_000, 60_100)),
}
CONTROL_DIMENSION = {"navigation": 2, "many_obstacles": 2, "pendulum": 1, "planar_arm": 3}
LOCAL_SIGMA = {"navigation": 0.35, "many_obstacles": 0.45, "pendulum": 0.25, "planar_arm": 0.3}
RHO = {"navigation": 0.9, "many_obstacles": 0.9, "pendulum": 0.9, "planar_arm": 0.81}
ALPHA = {"navigation": 0.02, "many_obstacles": 0.02, "pendulum": 1.0, "planar_arm": 1.0}
CYCLES = {"navigation": 8, "many_obstacles": 8, "pendulum": 4, "planar_arm": 8}
LOCAL_SAMPLES = {"navigation": 64, "many_obstacles": 64, "pendulum": 16, "planar_arm": 64}
DELTA_S = {"navigation": 0.2, "many_obstacles": 0.2, "pendulum": 0.1, "planar_arm": 0.1}
INIT_SIGMA = {"navigation": 2.0, "many_obstacles": 2.5, "pendulum": 1.0, "planar_arm": 1.0}
WARM_START_SIGMA = {"navigation": 0.35, "many_obstacles": 0.50, "pendulum": 0.30, "planar_arm": 0.20}
SHARED_LOCAL_NOISE = {"navigation": True, "many_obstacles": True, "pendulum": False, "planar_arm": False}
PENDULUM_DISTURBANCE_SIGMA = 0.5
PENDULUM_DISTURBANCE_SEED_OFFSET = 30_000_091


@dataclass(frozen=True)
class Cell:
    task: Task
    method: Method
    tau: float | None

    @property
    def cell_id(self) -> str:
        if self.method == "cbo_mpc":
            return "cbo_mpc_canonical"
        assert self.tau is not None
        return f"{self.method}_tau{int(self.tau)}"

    @property
    def controller(self) -> str:
        return "proposed" if self.method in {"multistart_mppi", "cbo_mppi"} else self.method

    @property
    def cycles(self) -> int:
        return 1 if self.method == "mppi" else CYCLES[self.task]

    @property
    def local_samples(self) -> int:
        return 1 if self.method in {"mppi", "cbo_mpc"} else LOCAL_SAMPLES[self.task]

    @property
    def particles(self) -> int:
        if self.method == "mppi":
            return BUDGET[self.task]
        if self.method == "cbo_mpc":
            return BUDGET[self.task] // self.cycles
        return BUDGET[self.task] // (self.cycles * self.local_samples)

    @property
    def rollout_budget(self) -> int:
        return self.particles * self.cycles * self.local_samples

    @property
    def sigma_cbo(self) -> float | None:
        if self.method == "mppi":
            return None
        dimension = CONTROL_DIMENSION[self.task] * HORIZON[self.task]
        return math.sqrt(1.0 / dimension)

    def manifest(self) -> dict:
        return {
            **asdict(self),
            "cell_id": self.cell_id,
            "controller": self.controller,
            "particles_or_rollouts": self.particles,
            "cycles": self.cycles,
            "local_samples": self.local_samples,
            "rollout_budget": self.rollout_budget,
            "sigma_cbo": self.sigma_cbo,
        }


def cells(task: Task | None = None) -> tuple[Cell, ...]:
    selected = TASKS if task is None else (task,)
    result: list[Cell] = []
    for name in selected:
        low, high = TAUS[name]
        result.extend([
            Cell(name, "mppi", low),
            Cell(name, "multistart_mppi", low),
            Cell(name, "cbo_mpc", None),
            Cell(name, "cbo_mppi", low),
            Cell(name, "mppi", high),
            Cell(name, "multistart_mppi", high),
            Cell(name, "cbo_mppi", high),
        ])
    return tuple(result)


def navigation_raw(cell: Cell) -> dict:
    filename = "two_obstacles_case1.toml" if cell.task == "navigation" else "many_obstacles.toml"
    raw = copy.deepcopy(load_experiment_config(BASE_CONFIG / filename))
    raw["problem"]["horizon"] = HORIZON[cell.task]
    raw["problem"]["simulation_steps"] = STEPS[cell.task]
    if cell.task == "navigation":
        raw["problem"]["obstacle_radii"] = [0.5, 0.5]
    raw["mppi"].update({
        "tau": float(cell.tau if cell.tau is not None else TAUS[cell.task][0]),
        "noise_sigma": LOCAL_SIGMA[cell.task],
        "noise_correlation": RHO[cell.task],
        "clip_rollout_inputs": True,
    })
    warm = {
        "mode": "shift_jitter", "jitter_kind": "smooth",
        "sigma": WARM_START_SIGMA[cell.task], "apply_to": "all_particles",
        "refresh_rate": 0.0, "min_cov_trace": 0.0,
    }
    raw["cbo"].update({
        "alpha": ALPHA[cell.task], "lambda": 1.0,
        "sigma": float(cell.sigma_cbo or 0.0), "delta_s": DELTA_S[cell.task],
        "init_sigma": INIT_SIGMA[cell.task], "init_noise_correlation": RHO[cell.task],
        "clip_rollout_inputs": True, "warm_start": copy.deepcopy(warm),
    })
    raw["proposed"].update({
        "tau": float(cell.tau if cell.tau is not None else TAUS[cell.task][0]),
        "local_noise_sigma": LOCAL_SIGMA[cell.task],
        "local_noise_correlation": RHO[cell.task],
        "alpha": ALPHA[cell.task], "lambda": 1.0,
        "sigma": float(cell.sigma_cbo or 0.0), "delta_s": DELTA_S[cell.task],
        "init_sigma": INIT_SIGMA[cell.task], "init_noise_correlation": RHO[cell.task],
        "shared_local_noise": SHARED_LOCAL_NOISE[cell.task],
        "output_rule": "weighted", "eta_grad": 0.0, "grad_clip": 0.0,
        "clip_rollout_inputs": True, "warm_start": warm,
        "center_update_rule": "independent_mppi" if cell.method == "multistart_mppi" else "cbo",
    })
    return raw


def pendulum_problem() -> PendulumConfig:
    return PendulumConfig(horizon=HORIZON["pendulum"])


def arm_problem() -> ArmConfig:
    return ArmConfig(
        dt=0.1, horizon=HORIZON["planar_arm"],
        q_ee=10.0, q_v=0.2, q_u=0.002, q_du=0.01,
        q_obs=100.0, q_lim=20.0, Q_ee=200.0, Q_v=1.0,
    )


def controller_config(cell: Cell):
    if cell.method == "mppi":
        return MPPIConfig(
            rollouts=BUDGET[cell.task], tau=float(cell.tau),
            noise_sigma=LOCAL_SIGMA[cell.task], noise_correlation=RHO[cell.task],
            clip_rollout_inputs=True,
        )
    if cell.method == "cbo_mpc":
        return CBOConfig(
            particles=cell.particles, cycles=cell.cycles,
            alpha=ALPHA[cell.task], lambda_=1.0, sigma=float(cell.sigma_cbo),
            delta_s=DELTA_S[cell.task], init_sigma=INIT_SIGMA[cell.task],
            init_noise_correlation=RHO[cell.task], warm_start_mode="shift_jitter",
            warm_start_jitter_kind="smooth", warm_start_sigma=WARM_START_SIGMA[cell.task],
            warm_start_refresh_rate=0.0, warm_start_min_cov_trace=0.0,
            warm_start_apply_to="all_particles", record_history=False,
            clip_rollout_inputs=True,
        )
    return ProposedConfig(
        particles=cell.particles, cycles=cell.cycles, local_samples=cell.local_samples,
        tau=float(cell.tau), local_noise_sigma=LOCAL_SIGMA[cell.task],
        local_noise_correlation=RHO[cell.task], alpha=ALPHA[cell.task],
        lambda_=1.0, sigma=float(cell.sigma_cbo), delta_s=DELTA_S[cell.task],
        init_sigma=INIT_SIGMA[cell.task], init_noise_correlation=RHO[cell.task],
        output_rule="weighted", shared_local_noise=SHARED_LOCAL_NOISE[cell.task],
        warm_start_mode="shift_jitter", warm_start_jitter_kind="smooth",
        warm_start_sigma=WARM_START_SIGMA[cell.task], warm_start_refresh_rate=0.0,
        warm_start_min_cov_trace=0.0, warm_start_apply_to="all_particles",
        record_history=False, eta_grad=0.0, grad_clip=0.0,
        clip_rollout_inputs=True,
        center_update_rule="independent_mppi" if cell.method == "multistart_mppi" else "cbo",
    )


def pendulum_disturbances(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + PENDULUM_DISTURBANCE_SEED_OFFSET)
    return rng.normal(
        scale=PENDULUM_DISTURBANCE_SIGMA, size=(STEPS["pendulum"], 1)
    ).astype(np.float32)


def execute(cell: Cell, seed: int):
    if cell.rollout_budget != BUDGET[cell.task]:
        raise RuntimeError(f"rollout-budget mismatch: {cell.manifest()}")
    if cell.task in {"navigation", "many_obstacles"}:
        return run_closed_loop(
            cell.controller, navigation_raw(cell), budget=BUDGET[cell.task], seed=seed,
            simulation_steps=STEPS[cell.task], cbo_cycles=cell.cycles,
            proposed_cycles=cell.cycles, local_samples=cell.local_samples,
            initial_route_height=0.0, initial_route_height_sigma=0.0,
            matched_initial_distribution=False, process_acceleration_noise_sigma=0.0,
            capture_predictions=False, record_history=False, episodic=False,
        )
    cfg = controller_config(cell)
    if cell.task == "pendulum":
        return run_pendulum_closed_loop(
            cell.controller, pendulum_problem(), cfg, seed=seed,
            steps=STEPS[cell.task], cost_style="regulation",
            process_noise_sigma=PENDULUM_DISTURBANCE_SIGMA,
            disturbance_sequence=pendulum_disturbances(seed), episodic=False,
        )
    return run_arm_closed_loop(
        cell.controller, arm_problem(), cfg, seed=seed, steps=STEPS[cell.task],
        process_noise_sigma=0.0, hard_obstacles=True, episodic=False,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_result(result, cell: Cell, seed: int, output_root: Path) -> tuple[Path, Path]:
    directory = output_root / "nominal" / cell.task / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"seed{seed}.npz"
    json_path = directory / f"seed{seed}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result for {cell.cell_id} seed {seed}")
    disturbance = (
        pendulum_disturbances(seed) if cell.task == "pendulum"
        else np.empty((0,), dtype=np.float32)
    )
    np.savez_compressed(
        npz_path,
        states=np.asarray(result.states), controls=np.asarray(result.controls),
        rollout_counts=np.asarray(result.rollout_counts), disturbance=disturbance,
    )
    summary = {
        key: value for key, value in result.summary.items()
        if "time" not in key.lower()
    }
    payload = {
        "schema_version": 1,
        "task": cell.task,
        "method": cell.method,
        "cell": cell.manifest(),
        "seed": seed,
        "summary": _jsonable(summary),
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


def select_cell(task: Task, method: Method, tau: float | None) -> Cell:
    matches = [
        cell for cell in cells(task)
        if cell.method == method and (
            method == "cbo_mpc" or math.isclose(float(cell.tau), float(tau))
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"cell selection is ambiguous or absent: {task}, {method}, {tau}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/simulation")
    parser.add_argument("--list", action="store_true", help="print the canonical campaign plan")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.list or not args.execute:
        plan = {
            task: {
                "seeds": [SEEDS[task][0], SEEDS[task][-1]],
                "cells": [cell.manifest() for cell in cells(task)],
            }
            for task in TASKS
        }
        print(json.dumps(plan, indent=2))
        if not args.execute:
            return
    if None in (args.task, args.method, args.seed):
        parser.error("--execute requires --task, --method, and --seed")
    cell = select_cell(args.task, args.method, args.tau)
    if args.seed not in SEEDS[args.task]:
        parser.error(f"seed is outside the canonical range for {args.task}")
    result = execute(cell, args.seed)
    paths = save_result(result, cell, args.seed, args.output_root.resolve())
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
