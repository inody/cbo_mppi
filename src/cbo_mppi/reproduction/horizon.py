"""Canonical prediction-horizon campaign."""
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

from cbo_mppi.config import MPPIConfig, ProposedConfig
from experiments.navigation.closed_loop import run_closed_loop
from experiments.pendulum_swingup.closed_loop import run_pendulum_closed_loop
from experiments.pendulum_swingup.config import PendulumConfig
from experiments.planar_arm.closed_loop import run_arm_closed_loop
from experiments.planar_arm.config import ArmConfig

from .nominal import (
    ALPHA,
    BUDGET,
    CONTROL_DIMENSION,
    DELTA_S,
    INIT_SIGMA,
    LOCAL_SIGMA,
    PENDULUM_DISTURBANCE_SIGMA,
    RHO,
    SHARED_LOCAL_NOISE,
    STEPS,
    WARM_START_SIGMA,
    Cell as NominalCell,
    navigation_raw as nominal_navigation_raw,
    pendulum_disturbances,
)

ROOT = Path(__file__).resolve().parents[3]
TASKS = ("navigation", "pendulum", "planar_arm")
HORIZONS = {
    "navigation": (10, 20, 30, 40, 50, 60, 80, 120),
    "pendulum": (10, 20, 40, 60, 80, 160, 320),
    "planar_arm": (5, 10, 15, 20, 30, 40, 60),
}
TAU = {"navigation": 300.0, "pendulum": 10.0, "planar_arm": 300.0}
CYCLES = {"navigation": 8, "pendulum": 4, "planar_arm": 8}
LOCAL_SAMPLES = {"navigation": 64, "pendulum": 16, "planar_arm": 64}
SEEDS = tuple(range(70_000, 70_020))
METHODS = ("mppi", "multistart_mppi", "cbo_mppi")

Method = Literal["mppi", "multistart_mppi", "cbo_mppi"]


@dataclass(frozen=True)
class Cell:
    task: str
    horizon: int
    method: Method

    @property
    def tau(self) -> float:
        return TAU[self.task]

    @property
    def cell_id(self) -> str:
        return f"H{self.horizon}_{self.method}_tau{int(self.tau)}"

    @property
    def controller(self) -> str:
        return "mppi" if self.method == "mppi" else "proposed"

    @property
    def allocation(self) -> tuple[int, int, int]:
        if self.method == "mppi":
            return BUDGET[self.task], 1, 1
        cycles = CYCLES[self.task]
        local = LOCAL_SAMPLES[self.task]
        return BUDGET[self.task] // (cycles * local), cycles, local

    @property
    def sigma_cbo(self) -> float | None:
        if self.method == "mppi":
            return None
        return math.sqrt(1.0 / (CONTROL_DIMENSION[self.task] * self.horizon))

    @property
    def rollout_budget(self) -> int:
        n, cycles, local = self.allocation
        return n * cycles * local

    def manifest(self) -> dict:
        n, cycles, local = self.allocation
        dt = {"navigation": 0.1, "pendulum": 0.05, "planar_arm": 0.1}[self.task]
        return {
            **asdict(self),
            "cell_id": self.cell_id,
            "controller": self.controller,
            "tau": self.tau,
            "lookahead_seconds": self.horizon * dt,
            "particles_or_rollouts": n,
            "cycles": cycles,
            "local_samples": local,
            "rollout_budget": self.rollout_budget,
            "sigma_cbo": self.sigma_cbo,
        }


def cells(task: str | None = None) -> tuple[Cell, ...]:
    selected = TASKS if task is None else (task,)
    return tuple(
        Cell(name, horizon, method)
        for name in selected
        for horizon in HORIZONS[name]
        for method in METHODS
    )


def navigation_raw(cell: Cell) -> dict:
    nominal = NominalCell("navigation", cell.method, cell.tau)
    raw = copy.deepcopy(nominal_navigation_raw(nominal))
    raw["problem"]["horizon"] = cell.horizon
    if cell.method != "mppi":
        raw["proposed"]["sigma"] = float(cell.sigma_cbo)
        raw["proposed"]["center_update_rule"] = (
            "independent_mppi" if cell.method == "multistart_mppi" else "cbo"
        )
    return raw


def problem(cell: Cell):
    if cell.task == "pendulum":
        return PendulumConfig(horizon=cell.horizon)
    if cell.task == "planar_arm":
        return ArmConfig(
            dt=0.1, horizon=cell.horizon,
            q_ee=10.0, q_v=0.2, q_u=0.002, q_du=0.01,
            q_obs=100.0, q_lim=20.0, Q_ee=200.0, Q_v=1.0,
        )
    raise ValueError(cell.task)


def controller_config(cell: Cell):
    if cell.method == "mppi":
        return MPPIConfig(
            rollouts=BUDGET[cell.task], tau=cell.tau,
            noise_sigma=LOCAL_SIGMA[cell.task], noise_correlation=RHO[cell.task],
            clip_rollout_inputs=True,
        )
    n, cycles, local = cell.allocation
    return ProposedConfig(
        particles=n, cycles=cycles, local_samples=local, tau=cell.tau,
        local_noise_sigma=LOCAL_SIGMA[cell.task], local_noise_correlation=RHO[cell.task],
        alpha=ALPHA[cell.task], lambda_=1.0, sigma=float(cell.sigma_cbo),
        delta_s=DELTA_S[cell.task], init_sigma=INIT_SIGMA[cell.task],
        init_noise_correlation=RHO[cell.task], output_rule="weighted",
        shared_local_noise=SHARED_LOCAL_NOISE[cell.task],
        warm_start_mode="shift_jitter", warm_start_jitter_kind="smooth",
        warm_start_sigma=WARM_START_SIGMA[cell.task], warm_start_refresh_rate=0.0,
        warm_start_min_cov_trace=0.0, warm_start_apply_to="all_particles",
        record_history=False, eta_grad=0.0, grad_clip=0.0,
        clip_rollout_inputs=True,
        center_update_rule="independent_mppi" if cell.method == "multistart_mppi" else "cbo",
    )


def execute(cell: Cell, seed: int):
    if cell.rollout_budget != BUDGET[cell.task]:
        raise RuntimeError("horizon rollout-budget mismatch")
    n, cycles, local = cell.allocation
    if cell.task == "navigation":
        return run_closed_loop(
            cell.controller, navigation_raw(cell), budget=BUDGET[cell.task], seed=seed,
            simulation_steps=STEPS[cell.task], cbo_cycles=cycles,
            proposed_cycles=cycles, local_samples=local,
            initial_route_height=0.0, initial_route_height_sigma=0.0,
            matched_initial_distribution=False, process_acceleration_noise_sigma=0.0,
            capture_predictions=False, record_history=False, episodic=False,
        )
    cfg = controller_config(cell)
    if cell.task == "pendulum":
        return run_pendulum_closed_loop(
            cell.controller, problem(cell), cfg, seed=seed, steps=STEPS[cell.task],
            cost_style="regulation", process_noise_sigma=PENDULUM_DISTURBANCE_SIGMA,
            disturbance_sequence=pendulum_disturbances(seed), episodic=False,
        )
    return run_arm_closed_loop(
        cell.controller, problem(cell), cfg, seed=seed, steps=STEPS[cell.task],
        process_noise_sigma=0.0, hard_obstacles=True, episodic=False,
    )


def save_result(result, cell: Cell, seed: int, output_root: Path) -> tuple[Path, Path]:
    directory = output_root / "horizon" / cell.task / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"seed{seed}.npz"
    json_path = directory / f"seed{seed}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite {cell.cell_id} seed {seed}")
    np.savez_compressed(
        npz_path,
        states=np.asarray(result.states), controls=np.asarray(result.controls),
        disturbance=(pendulum_disturbances(seed) if cell.task == "pendulum" else np.empty((0,), dtype=np.float32)),
    )
    payload = {
        "schema_version": 1,
        "cell": cell.manifest(),
        "seed": seed,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


def select_cell(task: str, cell_id: str) -> Cell:
    matches = [cell for cell in cells(task) if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"unknown horizon cell: {task}/{cell_id}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--cell-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/simulation")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        task: {
            "seeds": [SEEDS[0], SEEDS[-1]],
            "cells": [cell.manifest() for cell in cells(task)],
        }
        for task in TASKS
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.task is None or args.cell_id is None or args.seed is None:
        parser.error("--execute requires --task, --cell-id, and --seed")
    if args.seed not in SEEDS:
        parser.error("seed is outside the canonical horizon range")
    cell = select_cell(args.task, args.cell_id)
    result = execute(cell, args.seed)
    print("\n".join(str(path) for path in save_result(result, cell, args.seed, args.output_root.resolve())))


if __name__ == "__main__":
    main()
