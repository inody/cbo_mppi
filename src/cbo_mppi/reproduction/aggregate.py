"""Assemble fresh campaign outputs into figure inputs and summary tables."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from . import evaluate, horizon, nominal, sensitivity
from .nominal import arm_problem

TASK_LABELS = {
    "navigation": "Point-Mass Navigation",
    "many_obstacles": "Many-Obstacle Navigation",
    "pendulum": "Pendulum Swing-Up",
    "planar_arm": "Planar Arm",
}
METHOD_LABELS = {
    "mppi": "MPPI",
    "multistart_mppi": "multi-start MPPI",
    "cbo_mpc": "CBO-MPC",
    "cbo_mppi": "CBO-MPPI",
}
NOMINAL_HORIZON = {"navigation": 60, "many_obstacles": 60, "pendulum": 80, "planar_arm": 20}
DT = {"navigation": 0.1, "many_obstacles": 0.1, "pendulum": 0.05, "planar_arm": 0.1}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def closed_loop_bundle(output_root: Path, assembled: Path) -> Path:
    keys = {}
    for task in nominal.TASKS:
        low, high = nominal.TAUS[task]
        selections = {
            f"{task}_mppi_low": nominal.Cell(task, "mppi", low),
            f"{task}_mppi_high": nominal.Cell(task, "mppi", high),
            f"{task}_cbo_mppi_low": nominal.Cell(task, "cbo_mppi", low),
            f"{task}_cbo_mppi_high": nominal.Cell(task, "cbo_mppi", high),
            f"{task}_cbo_mpc": nominal.Cell(task, "cbo_mpc", None),
        }
        for key, cell in selections.items():
            states = []
            for seed in nominal.SEEDS[task]:
                path = output_root / "nominal" / task / cell.cell_id / f"seed{seed}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    states.append(np.asarray(archive["states"], dtype=np.float32))
            keys[key] = np.stack(states)
    destination = assembled / "closed_loop_figure_trajectories.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **keys)
    return destination


def nominal_table(output_root: Path, assembled: Path) -> Path:
    rows = []
    for cell in nominal.cells():
        for seed in nominal.SEEDS[cell.task]:
            for tau in evaluate.evaluation_taus(cell):
                payload = _load_json(evaluate.metric_path(cell, seed, tau, output_root))
                rows.append({
                    "task": cell.task,
                    "method": cell.method,
                    "cell_id": cell.cell_id,
                    "controller_tau": "" if cell.tau is None else cell.tau,
                    "evaluation_tau": tau,
                    "seed": seed,
                    "success": str(payload["success"]).lower(),
                    "collision": str(payload["collision"]).lower(),
                    "J_CL": payload["J_CL"],
                    "F_CL": payload["F_CL"],
                    "feasible_fraction": payload["feasible_fraction"],
                    "minimum_clearance": payload["minimum_clearance"],
                    "route_class": payload["route_class"],
                })
    destination = assembled / "tables/nominal_per_seed.csv"
    _write_csv(destination, rows)
    return destination


def sensitivity_table(output_root: Path, assembled: Path) -> Path:
    rows = []
    for cell in sensitivity.cells():
        for seed in sensitivity.SEEDS:
            path = evaluate.external_metric_path(
                "sensitivity", cell.task, cell.cell_id, seed, cell.tau, output_root
            )
            payload = _load_json(path)
            rows.append({
                "task": cell.task,
                "method": cell.method,
                "tau": cell.tau,
                "alpha": cell.alpha,
                "alpha_ratio": cell.alpha_ratio,
                "kappa": cell.kappa,
                "sigma_cbo": cell.sigma_cbo,
                "cell_id": cell.cell_id,
                "seed": seed,
                "metric_status": payload.get("metric_status", "evaluated"),
                "success": "" if payload["success"] is None else str(payload["success"]).lower(),
                "collision": "" if payload["collision"] is None else str(payload["collision"]).lower(),
                "J_CL": payload["J_CL"],
                "F_CL": payload["F_CL"],
                "feasible_fraction": payload["feasible_fraction"],
            })
    destination = assembled / "tables/sensitivity_per_seed.csv"
    _write_csv(destination, rows)
    return destination


def _summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(float(value) for value in np.quantile(array, [0.5, 0.25, 0.75]))


def horizon_tables(output_root: Path, assembled: Path) -> tuple[Path, Path]:
    per_seed = []
    grouped = []
    for cell in horizon.cells():
        payloads = []
        for seed in horizon.SEEDS:
            path = evaluate.external_metric_path(
                "horizon", cell.task, cell.cell_id, seed, cell.tau, output_root
            )
            payload = _load_json(path)
            payloads.append(payload)
            per_seed.append({
                "task": cell.task,
                "horizon": cell.horizon,
                "lookahead_seconds": cell.horizon * DT[cell.task],
                "method_id": cell.method,
                "method": METHOD_LABELS[cell.method],
                "tau": cell.tau,
                "seed": seed,
                "success": str(payload["success"]).lower(),
                "J_CL": payload["J_CL"],
                "F_CL": payload["F_CL"],
            })
        j_median, j_q25, j_q75 = _summary([row["J_CL"] for row in payloads])
        f_median, f_q25, f_q75 = _summary([row["F_CL"] for row in payloads])
        success_count = sum(bool(row["success"]) for row in payloads)
        grouped.append({
            "task": cell.task,
            "task_label": TASK_LABELS[cell.task],
            "horizon": cell.horizon,
            "lookahead_seconds": cell.horizon * DT[cell.task],
            "nominal_horizon": NOMINAL_HORIZON[cell.task],
            "nominal_lookahead_seconds": NOMINAL_HORIZON[cell.task] * DT[cell.task],
            "method_id": cell.method,
            "method": METHOD_LABELS[cell.method],
            "tau": cell.tau,
            "n": len(payloads),
            "J_CL_median": j_median,
            "J_CL_q25": j_q25,
            "J_CL_q75": j_q75,
            "F_CL_median": f_median,
            "F_CL_q25": f_q25,
            "F_CL_q75": f_q75,
            "success_count": success_count,
            "success_rate": success_count / len(payloads),
        })
    per_seed_path = assembled / "tables/horizon_per_seed.csv"
    figure_path = assembled / "horizon/figure_input.csv"
    _write_csv(per_seed_path, per_seed)
    _write_csv(figure_path, grouped)
    return per_seed_path, figure_path


def copy_landscapes(output_root: Path, assembled: Path) -> list[Path]:
    mapping = {
        output_root / "landscapes/navigation/route_heights.npz": assembled / "navigation/route_heights.npz",
        output_root / "landscapes/navigation/free_energy_landscape.npz": assembled / "navigation/free_energy_landscape.npz",
        output_root / "landscapes/pendulum/landscape_tau10.npz": assembled / "pendulum/landscape_tau10.npz",
        output_root / "landscapes/planar_arm/landscape_arrays.npz": assembled / "planar_arm/landscape_arrays.npz",
        output_root / "landscapes/planar_arm/base_trajectory.npz": assembled / "planar_arm/base_trajectory.npz",
    }
    outputs = []
    for source, destination in mapping.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        outputs.append(destination)
    design = assembled / "planar_arm/frozen_design.json"
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text(json.dumps({
        "authority": {
            "diagnostic_problem": arm_problem()._replace(horizon=60)._asdict()
        }
    }, indent=2, sort_keys=True) + "\n")
    outputs.append(design)
    return outputs


def run(output_root: Path) -> Path:
    assembled = output_root / "assembled/figures"
    if assembled.exists():
        raise FileExistsError(f"refusing to overwrite assembled data: {assembled}")
    outputs = [
        closed_loop_bundle(output_root, assembled),
        nominal_table(output_root, assembled),
        sensitivity_table(output_root, assembled),
        *horizon_tables(output_root, assembled),
        *copy_landscapes(output_root, assembled),
    ]
    manifest = output_root / "assembled/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "timing": "excluded",
        "files": [str(path.relative_to(output_root)) for path in outputs],
    }, indent=2, sort_keys=True) + "\n")
    return assembled
