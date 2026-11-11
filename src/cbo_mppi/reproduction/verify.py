"""Verify fresh results against thresholds frozen before recomputation."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cbo_mppi.figures.generate_all import EXPECTED

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / "data/reference/verification_targets.json"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(csv.DictReader(path.open()))


def _check(report: list[dict], name: str, observed: float, target: float, tolerance: float) -> None:
    difference = abs(observed - target)
    report.append({
        "name": name,
        "observed": observed,
        "target": target,
        "absolute_difference": difference,
        "absolute_tolerance": tolerance,
        "pass": difference <= tolerance or math.isclose(
            difference, tolerance, rel_tol=1e-12, abs_tol=1e-12
        ),
    })


def verify_nominal(output: Path, targets: dict, report: list[dict]) -> None:
    groups = defaultdict(list)
    for row in _rows(output / "assembled/figures/tables/nominal_per_seed.csv"):
        groups[(row["task"], row["method"], float(row["evaluation_tau"]))].append(row)
    for target in targets["nominal"]:
        key = (target["task"], target["method"], float(target["tau"]))
        rows = groups[key]
        if len(rows) != target["n"]:
            raise RuntimeError(f"nominal row-count mismatch for {key}: {len(rows)}")
        prefix = f"nominal/{key[0]}/{key[1]}/tau{key[2]:g}"
        _check(report, prefix + "/J_CL_median",
               float(np.median([float(row["J_CL"]) for row in rows])),
               target["J_CL_median"], target["J_CL_absolute_tolerance"])
        _check(report, prefix + "/F_CL_median",
               float(np.median([float(row["F_CL"]) for row in rows])),
               target["F_CL_median"], target["F_CL_absolute_tolerance"])
        _check(report, prefix + "/success_rate",
               float(np.mean([row["success"] == "true" for row in rows])),
               target["success_rate"], target["success_rate_absolute_tolerance"])
        _check(report, prefix + "/collision_rate",
               float(np.mean([row["collision"] == "true" for row in rows])),
               target["collision_rate"], target["collision_rate_absolute_tolerance"])


def verify_sensitivity(output: Path, targets: dict, report: list[dict]) -> None:
    rows = _rows(output / "assembled/figures/tables/sensitivity_per_seed.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"], float(row["tau"]), row["cell_id"])].append(row)
    for target in targets["sensitivity"]:
        task, tau = target["task"], float(target["tau"])
        candidates = []
        for (row_task, method, row_tau, _), values in groups.items():
            if row_task != task or method != "cbo_mppi" or row_tau != tau:
                continue
            first = values[0]
            if target["quantity"] == "alpha_ratio":
                match = (abs(float(first["alpha_ratio"]) - target["value"]) < 1e-9
                         and abs(float(first["kappa"]) - 0.5) < 1e-9)
            else:
                match = (abs(float(first["alpha_ratio"]) - 1.0) < 1e-9
                         and abs(float(first["kappa"]) - target["value"]) < 1e-9)
            if match:
                candidates.append(values)
        if len(candidates) != 1:
            raise RuntimeError(f"sensitivity cell match failure: {target}")
        cbo = candidates[0]
        mppi_groups = [values for (row_task, method, row_tau, _), values in groups.items()
                       if row_task == task and method == "mppi" and row_tau == tau]
        if len(mppi_groups) != 1:
            raise RuntimeError(f"sensitivity MPPI match failure: {task}/{tau}")
        mppi = mppi_groups[0]
        prefix = f"sensitivity/{task}/tau{tau:g}/{target['quantity']}={target['value']:g}"
        cbo_f = [float(row["F_CL"]) for row in cbo if row["F_CL"]]
        mppi_f = [float(row["F_CL"]) for row in mppi if row["F_CL"]]
        if not cbo_f or not mppi_f:
            raise RuntimeError(f"no finite sensitivity metrics: {prefix}")
        _check(report, prefix + "/cbo_F_median",
               float(np.median(cbo_f)),
               target["cbo_F_median"], target["cbo_F_absolute_tolerance"])
        _check(report, prefix + "/mppi_F_median",
               float(np.median(mppi_f)),
               target["mppi_F_median"], target["mppi_F_absolute_tolerance"])
        _check(report, prefix + "/cbo_success_rate",
               float(np.mean([row["success"] == "true" for row in cbo])),
               target["cbo_success_rate"], target["success_rate_absolute_tolerance"])


def verify_horizon(output: Path, targets: dict, report: list[dict]) -> None:
    rows = {
        (row["task"], int(row["horizon"]), row["method_id"]): row
        for row in _rows(output / "assembled/figures/horizon/figure_input.csv")
    }
    for target in targets["horizon"]:
        key = (target["task"], int(target["horizon"]), target["method"])
        row = rows[key]
        prefix = f"horizon/{key[0]}/H{key[1]}/{key[2]}"
        _check(report, prefix + "/J_CL_median", float(row["J_CL_median"]),
               target["J_CL_median"], target["J_CL_absolute_tolerance"])
        _check(report, prefix + "/F_CL_median", float(row["F_CL_median"]),
               target["F_CL_median"], target["F_CL_absolute_tolerance"])
        _check(report, prefix + "/success_rate", float(row["success_rate"]),
               target["success_rate"], target["success_rate_absolute_tolerance"])


def _finite_argmin(values: np.ndarray) -> int:
    return int(np.argmin(np.where(np.isfinite(values), values, np.inf)))


def verify_landscapes(output: Path, report: list[dict]) -> None:
    fresh_root = output / "assembled/figures"
    frozen_root = ROOT / "data/frozen/figures"
    with np.load(fresh_root / "navigation/free_energy_landscape.npz", allow_pickle=False) as fresh, \
         np.load(output / "landscapes/navigation/free_energy_landscape.npz", allow_pickle=False) as same:
        heights = np.asarray(fresh["heights"])
        nominal = np.asarray(fresh["nominal_costs"])
        free = np.asarray(fresh["free_energies"])
        # The duplicated load intentionally confirms assembly preserved the arrays.
        if not np.array_equal(nominal, np.asarray(same["nominal_costs"])):
            raise RuntimeError("assembled navigation landscape changed")
        _check(report, "landscape/navigation/height_count",
               float(len(heights)), 121.0, 0.0)
        _check(report, "landscape/navigation/free_energy_curve_count",
               float(free.shape[0]), 2.0, 0.0)
        for name, values in (("nominal", nominal), ("F15", free[0]), ("F300", free[1])):
            argmin = float(heights[_finite_argmin(values)])
            report.append({
                "name": f"landscape/navigation/{name}_finite_argmin",
                "observed": argmin,
                "target": "finite point inside frozen grid [-3, 3]",
                "absolute_difference": None,
                "absolute_tolerance": None,
                "pass": bool(np.isfinite(argmin) and -3.0 <= argmin <= 3.0),
            })

    with np.load(fresh_root / "pendulum/landscape_tau10.npz", allow_pickle=False) as fresh, \
         np.load(frozen_root / "pendulum/landscape_tau10.npz", allow_pickle=False) as frozen:
        for field in ("nom_cost", "F_hat"):
            fi = _finite_argmin(np.asarray(fresh[field]))
            bi = _finite_argmin(np.asarray(frozen[field]))
            fp = np.asarray(fresh["params"])[fi]
            bp = np.asarray(frozen["params"])[bi]
            _check(report, f"landscape/pendulum/{field}_argmin_a", float(fp[0]), float(bp[0]), 0.15)
            _check(report, f"landscape/pendulum/{field}_argmin_b", float(fp[1]), float(bp[1]), 0.15)

    with np.load(fresh_root / "planar_arm/landscape_arrays.npz", allow_pickle=False) as fresh, \
         np.load(frozen_root / "planar_arm/landscape_arrays.npz", allow_pickle=False) as frozen:
        h_fresh, h_frozen = np.asarray(fresh["h"]), np.asarray(frozen["h"])
        fields = (("nominal_cost", None), ("projected_free_energy", 0), ("projected_free_energy", 1))
        for field, column in fields:
            f = np.asarray(fresh[field]); b = np.asarray(frozen[field])
            if column is not None:
                f, b = f[:, column], b[:, column]
            _check(report, f"landscape/planar_arm/{field}/{column}_argmin_h",
                   float(h_fresh[_finite_argmin(f)]),
                   float(h_frozen[_finite_argmin(b)]), 0.03)


def run(output: Path) -> Path:
    targets = json.loads(TARGETS.read_text())
    checks: list[dict] = []
    verify_nominal(output, targets, checks)
    verify_sensitivity(output, targets, checks)
    verify_horizon(output, targets, checks)
    verify_landscapes(output, checks)
    missing_figures = [name for name in EXPECTED if not (output / "figures" / name).is_file()]
    passed = all(check["pass"] for check in checks) and not missing_figures
    report = {
        "schema_version": 1,
        "passed": passed,
        "timing_excluded": True,
        "target_file": str(TARGETS.relative_to(ROOT)),
        "check_count": len(checks),
        "failed_checks": [check for check in checks if not check["pass"]],
        "missing_figures": missing_figures,
        "checks": checks,
    }
    destination = output / "verification/report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"verification failed; see {destination}")
    return destination
