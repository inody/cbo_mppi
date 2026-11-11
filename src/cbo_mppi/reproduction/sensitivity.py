"""Canonical parameter-sensitivity campaign."""
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

from experiments.navigation.closed_loop import run_closed_loop
from experiments.pendulum_swingup.closed_loop import run_pendulum_closed_loop
from experiments.planar_arm.closed_loop import run_arm_closed_loop

from .nominal import (
    ALPHA,
    BUDGET,
    CONTROL_DIMENSION,
    CYCLES,
    HORIZON,
    LOCAL_SAMPLES,
    METHODS,
    PENDULUM_DISTURBANCE_SIGMA,
    SHARED_LOCAL_NOISE,
    STEPS,
    TASKS,
    TAUS,
    Cell as NominalCell,
    arm_problem,
    controller_config as nominal_controller_config,
    navigation_raw as nominal_navigation_raw,
    pendulum_disturbances,
    pendulum_problem,
)

ROOT = Path(__file__).resolve().parents[3]
ALPHA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
KAPPA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
KAPPA_NOMINAL = 0.5
SEEDS = tuple(range(71_000, 71_020))

Method = Literal["mppi", "cbo_mppi"]


def sigma_for(task: str, kappa: float) -> float:
    dimension = CONTROL_DIMENSION[task] * HORIZON[task]
    return math.sqrt(2.0 * float(kappa) / dimension)


def _tag(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Cell:
    task: str
    tau: float
    method: Method
    alpha: float | None = None
    kappa: float | None = None

    @property
    def cell_id(self) -> str:
        if self.method == "mppi":
            return f"mppi_tau{_tag(self.tau)}"
        identity = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
        return (
            f"cbo_mppi_tau{_tag(self.tau)}_alpha{_tag(float(self.alpha))}_"
            f"kappa{_tag(float(self.kappa))}_{digest}"
        )

    @property
    def alpha_ratio(self) -> float | None:
        return None if self.alpha is None else float(self.alpha) / ALPHA[self.task]

    @property
    def sigma_cbo(self) -> float | None:
        return None if self.kappa is None else sigma_for(self.task, self.kappa)

    @property
    def allocation(self) -> tuple[int, int, int]:
        if self.method == "mppi":
            return BUDGET[self.task], 1, 1
        cycles = CYCLES[self.task]
        local = LOCAL_SAMPLES[self.task]
        return BUDGET[self.task] // (cycles * local), cycles, local

    @property
    def rollout_budget(self) -> int:
        n, cycles, local = self.allocation
        return n * cycles * local

    def manifest(self) -> dict:
        n, cycles, local = self.allocation
        return {
            **asdict(self),
            "cell_id": self.cell_id,
            "alpha_ratio": self.alpha_ratio,
            "sigma_cbo": self.sigma_cbo,
            "particles_or_rollouts": n,
            "cycles": cycles,
            "local_samples": local,
            "rollout_budget": self.rollout_budget,
        }


def cells(task: str | None = None) -> tuple[Cell, ...]:
    selected = TASKS if task is None else (task,)
    output: list[Cell] = []
    for name in selected:
        for tau in TAUS[name]:
            output.append(Cell(name, tau, "mppi"))
            unique: dict[tuple[float, float], Cell] = {}
            for multiplier in ALPHA_MULTIPLIERS:
                cell = Cell(name, tau, "cbo_mppi", ALPHA[name] * multiplier, KAPPA_NOMINAL)
                unique[(float(cell.alpha), float(cell.kappa))] = cell
            for kappa in KAPPA_GRID:
                cell = Cell(name, tau, "cbo_mppi", ALPHA[name], kappa)
                unique.setdefault((float(cell.alpha), float(cell.kappa)), cell)
            output.extend(unique.values())
    return tuple(output)


def controller_config(cell: Cell):
    nominal = NominalCell(
        cell.task,
        "mppi" if cell.method == "mppi" else "cbo_mppi",
        cell.tau,
    )
    cfg = nominal_controller_config(nominal)
    if cell.method == "mppi":
        return cfg
    return cfg._replace(alpha=float(cell.alpha), sigma=float(cell.sigma_cbo))


def navigation_raw(cell: Cell) -> dict:
    nominal = NominalCell(
        cell.task,
        "mppi" if cell.method == "mppi" else "cbo_mppi",
        cell.tau,
    )
    raw = copy.deepcopy(nominal_navigation_raw(nominal))
    if cell.method == "cbo_mppi":
        raw["proposed"]["alpha"] = float(cell.alpha)
        raw["proposed"]["sigma"] = float(cell.sigma_cbo)
    return raw


def execute(cell: Cell, seed: int):
    if cell.rollout_budget != BUDGET[cell.task]:
        raise RuntimeError("sensitivity rollout-budget mismatch")
    controller = "mppi" if cell.method == "mppi" else "proposed"
    n, cycles, local = cell.allocation
    if cell.task in {"navigation", "many_obstacles"}:
        return run_closed_loop(
            controller, navigation_raw(cell), budget=BUDGET[cell.task], seed=seed,
            simulation_steps=STEPS[cell.task], cbo_cycles=cycles,
            proposed_cycles=cycles, local_samples=local,
            initial_route_height=0.0, initial_route_height_sigma=0.0,
            matched_initial_distribution=False, process_acceleration_noise_sigma=0.0,
            capture_predictions=False, record_history=False, episodic=False,
        )
    cfg = controller_config(cell)
    if cell.task == "pendulum":
        return run_pendulum_closed_loop(
            controller, pendulum_problem(), cfg, seed=seed, steps=STEPS[cell.task],
            cost_style="regulation", process_noise_sigma=PENDULUM_DISTURBANCE_SIGMA,
            disturbance_sequence=pendulum_disturbances(seed), episodic=False,
        )
    return run_arm_closed_loop(
        controller, arm_problem(), cfg, seed=seed, steps=STEPS[cell.task],
        process_noise_sigma=0.0, hard_obstacles=True, episodic=False,
    )


def save_result(result, cell: Cell, seed: int, output_root: Path) -> tuple[Path, Path]:
    directory = output_root / "sensitivity" / cell.task / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"seed{seed}.npz"
    json_path = directory / f"seed{seed}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite {cell.cell_id} seed {seed}")
    np.savez_compressed(
        npz_path,
        states=np.asarray(result.states),
        controls=np.asarray(result.controls),
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
        raise ValueError(f"unknown sensitivity cell: {task}/{cell_id}")
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
        parser.error("seed is outside the canonical sensitivity range")
    cell = select_cell(args.task, args.cell_id)
    result = execute(cell, args.seed)
    print("\n".join(str(path) for path in save_result(result, cell, args.seed, args.output_root.resolve())))


if __name__ == "__main__":
    main()
