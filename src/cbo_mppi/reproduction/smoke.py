"""One-seed GPU smoke gate covering every task and controller family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import evaluate, nominal

ROOT = Path(__file__).resolve().parents[3]


def jobs():
    output = []
    for task in nominal.TASKS:
        seed = nominal.SEEDS[task][0]
        low_tau = nominal.TAUS[task][0]
        for method in nominal.METHODS:
            tau = None if method == "cbo_mpc" else low_tau
            output.append((nominal.select_cell(task, method, tau), seed))
    return tuple(output)


def run(output_root: Path, resume: bool = False) -> Path:
    rows = []
    for index, (cell, seed) in enumerate(jobs(), start=1):
        npz_path, json_path = evaluate.result_paths(cell, seed, output_root)
        if not (resume and npz_path.is_file() and json_path.is_file()):
            print(f"[smoke trajectory {index}/{len(jobs())}] {cell.task} {cell.cell_id}", flush=True)
            result = nominal.execute(cell, seed)
            nominal.save_result(result, cell, seed, output_root)
        for tau in evaluate.evaluation_taus(cell):
            metric = evaluate.metric_path(cell, seed, tau, output_root)
            if not (resume and metric.is_file()):
                print(f"[smoke evaluation] {cell.task} {cell.cell_id} tau={tau:g}", flush=True)
                evaluate.evaluate_one(cell, seed, tau, output_root)
            payload = json.loads(metric.read_text())
            rows.append({
                "task": cell.task,
                "method": cell.method,
                "cell_id": cell.cell_id,
                "seed": seed,
                "tau": tau,
                "success": payload["success"],
                "collision": payload["collision"],
                "J_CL": payload["J_CL"],
                "F_CL": payload["F_CL"],
                "feasible_fraction": payload["feasible_fraction"],
            })
    passed = (
        len(rows) == 20
        and all(row["J_CL"] is not None for row in rows)
        and all(row["F_CL"] is not None for row in rows)
        and all(0.0 <= row["feasible_fraction"] <= 1.0 for row in rows)
    )
    destination = output_root / "smoke/report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema_version": 1,
        "passed": passed,
        "trajectory_jobs": len(jobs()),
        "metric_rows": len(rows),
        "timing": "not measured",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"smoke gate failed: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.output_root.resolve(), resume=args.resume))


if __name__ == "__main__":
    main()
