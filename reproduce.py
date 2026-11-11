#!/usr/bin/env python3
"""Entry point for frozen-data replay and full manuscript recomputation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

STAGES = (
    "nominal", "sensitivity", "horizon", "evaluate",
    "landscapes", "aggregate", "figures", "verify",
)
IMPLEMENTED_SIMULATION_STAGES = {
    "nominal", "evaluate", "sensitivity", "horizon", "landscapes",
    "aggregate", "figures", "verify",
}
ROOT = Path(__file__).resolve().parent


def source_contract() -> dict:
    roots = [
        ROOT / "src/cbo_mppi", ROOT / "experiments", ROOT / "configs",
        ROOT / "data/frozen",
    ]
    files = [ROOT / "reproduce.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in roots:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    records = {}
    for path in sorted(set(files)):
        if path.suffix in {".pyc", ".so"} or "__pycache__" in path.parts:
            continue
        records[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    core = {
        "schema_version": 1,
        "timing": "excluded",
        "source_files": records,
        "verification_targets_sha256": hashlib.sha256(
            (ROOT / "data/reference/verification_targets.json").read_bytes()
        ).hexdigest(),
    }
    return {
        **core,
        "contract_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def ensure_source_contract(output: Path) -> Path:
    path = output / "campaign_contract.json"
    expected = source_contract()
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError(
                f"output directory was created by a different source contract: {path}"
            )
        return path
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return path


def valid_result(npz_path: Path, json_path: Path) -> bool:
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = metadata.get("npz_sha256")
    return bool(expected) and hashlib.sha256(npz_path.read_bytes()).hexdigest() == expected


def record_failure(output: Path, stage: str, identity: dict, error: BaseException) -> Path:
    root = output / "failures" / stage
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    path = root / f"{digest}.json"
    payload = {
        "schema_version": 1,
        "stage": stage,
        "identity": identity,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_nominal(output: Path, resume: bool) -> None:
    from cbo_mppi.reproduction import nominal

    total = len(nominal.cells()) * 100
    completed = 0
    for cell in nominal.cells():
        for seed in nominal.SEEDS[cell.task]:
            directory = output / "nominal" / cell.task / cell.cell_id
            npz_path = directory / f"seed{seed}.npz"
            json_path = directory / f"seed{seed}.json"
            if resume and valid_result(npz_path, json_path):
                completed += 1
                continue
            identity = {"task": cell.task, "cell": cell.manifest(), "seed": seed}
            print(f"[nominal {completed + 1}/{total}] {cell.task} {cell.cell_id} seed={seed}", flush=True)
            try:
                result = nominal.execute(cell, seed)
                nominal.save_result(result, cell, seed, output)
            except BaseException as error:
                failure = record_failure(output, "nominal", identity, error)
                raise RuntimeError(f"nominal run failed; record: {failure}") from error
            completed += 1


def run_evaluation(output: Path, resume: bool) -> None:
    from cbo_mppi.reproduction import evaluate, horizon, nominal, sensitivity

    nominal_jobs = [
        (cell, seed, tau)
        for cell in nominal.cells()
        for seed in nominal.SEEDS[cell.task]
        for tau in evaluate.evaluation_taus(cell)
    ]
    external_jobs = [
        (
            "sensitivity", cell.task, cell.cell_id, seed, cell.tau,
            output / "sensitivity" / cell.task / cell.cell_id / f"seed{seed}.npz",
            cell.manifest(),
        )
        for cell in sensitivity.cells()
        for seed in sensitivity.SEEDS
    ] + [
        (
            "horizon", cell.task, cell.cell_id, seed, cell.tau,
            output / "horizon" / cell.task / cell.cell_id / f"seed{seed}.npz",
            cell.manifest(),
        )
        for cell in horizon.cells()
        for seed in horizon.SEEDS
    ]
    total = len(nominal_jobs) + len(external_jobs)
    index = 0
    for cell, seed, tau in nominal_jobs:
        index += 1
        destination = evaluate.metric_path(cell, seed, tau, output)
        if resume and destination.is_file():
            continue
        identity = {
            "stage": "nominal", "task": cell.task,
            "cell_id": cell.cell_id, "seed": seed, "tau": tau,
        }
        print(
            f"[evaluate {index}/{total}] nominal {cell.task} {cell.cell_id} "
            f"seed={seed} tau={tau}", flush=True,
        )
        try:
            evaluate.evaluate_one(cell, seed, tau, output)
        except BaseException as error:
            failure = record_failure(output, "evaluate", identity, error)
            raise RuntimeError(f"evaluation failed; record: {failure}") from error
    for stage, task, cell_id, seed, tau, source_npz, metadata in external_jobs:
        index += 1
        destination = evaluate.external_metric_path(
            stage, task, cell_id, seed, tau, output
        )
        if resume and destination.is_file():
            continue
        identity = {
            "stage": stage, "task": task, "cell_id": cell_id,
            "seed": seed, "tau": tau,
        }
        print(
            f"[evaluate {index}/{total}] {stage} {task} {cell_id} "
            f"seed={seed} tau={tau}", flush=True,
        )
        try:
            evaluate.evaluate_external(
                stage, task, cell_id, seed, tau, source_npz, output,
                cell_metadata=metadata,
            )
        except BaseException as error:
            failure = record_failure(output, "evaluate", identity, error)
            raise RuntimeError(f"evaluation failed; record: {failure}") from error


def run_sensitivity(output: Path, resume: bool) -> None:
    from cbo_mppi.reproduction import sensitivity

    jobs = [
        (cell, seed)
        for cell in sensitivity.cells()
        for seed in sensitivity.SEEDS
    ]
    for index, (cell, seed) in enumerate(jobs, start=1):
        directory = output / "sensitivity" / cell.task / cell.cell_id
        npz_path = directory / f"seed{seed}.npz"
        json_path = directory / f"seed{seed}.json"
        if resume and valid_result(npz_path, json_path):
            continue
        identity = {"cell": cell.manifest(), "seed": seed}
        print(
            f"[sensitivity {index}/{len(jobs)}] {cell.task} "
            f"{cell.cell_id} seed={seed}", flush=True,
        )
        try:
            result = sensitivity.execute(cell, seed)
            sensitivity.save_result(result, cell, seed, output)
        except BaseException as error:
            failure = record_failure(output, "sensitivity", identity, error)
            raise RuntimeError(f"sensitivity run failed; record: {failure}") from error


def run_horizon(output: Path, resume: bool) -> None:
    from cbo_mppi.reproduction import horizon

    jobs = [
        (cell, seed)
        for cell in horizon.cells()
        for seed in horizon.SEEDS
    ]
    for index, (cell, seed) in enumerate(jobs, start=1):
        directory = output / "horizon" / cell.task / cell.cell_id
        npz_path = directory / f"seed{seed}.npz"
        json_path = directory / f"seed{seed}.json"
        if resume and valid_result(npz_path, json_path):
            continue
        identity = {"cell": cell.manifest(), "seed": seed}
        print(
            f"[horizon {index}/{len(jobs)}] {cell.task} "
            f"{cell.cell_id} seed={seed}", flush=True,
        )
        try:
            result = horizon.execute(cell, seed)
            horizon.save_result(result, cell, seed, output)
        except BaseException as error:
            failure = record_failure(output, "horizon", identity, error)
            raise RuntimeError(f"horizon run failed; record: {failure}") from error


def run_landscapes(output: Path, resume: bool) -> None:
    from cbo_mppi.reproduction import landscapes

    try:
        landscapes.run_all(output, resume=resume)
    except BaseException as error:
        failure = record_failure(output, "landscapes", {"campaign": "all"}, error)
        raise RuntimeError(f"landscape computation failed; record: {failure}") from error


def run_aggregate(output: Path, resume: bool) -> None:
    marker = output / "assembled/manifest.json"
    if resume and marker.is_file():
        return
    from cbo_mppi.reproduction import aggregate

    try:
        aggregate.run(output)
    except BaseException as error:
        failure = record_failure(output, "aggregate", {"campaign": "all"}, error)
        raise RuntimeError(f"aggregation failed; record: {failure}") from error


def run_figures(output: Path, resume: bool) -> None:
    from cbo_mppi.figures.generate_all import EXPECTED, generate

    destination = output / "figures"
    if resume and all((destination / name).is_file() for name in EXPECTED):
        return
    generate(destination, data_root=output / "assembled/figures")


def run_verify(output: Path, resume: bool) -> None:
    marker = output / "verification/report.json"
    if resume and marker.is_file() and json.loads(marker.read_text()).get("passed") is True:
        return
    from cbo_mppi.reproduction import verify

    verify.run(output)


def simulation_plan(selected: tuple[str, ...], output: Path, gpu_index: int) -> None:
    from cbo_mppi.reproduction import horizon, nominal, sensitivity

    print("Full numerical reproduction plan")
    print("  stages:", ", ".join(selected))
    print("  GPU index:", gpu_index)
    print("  output:", output)
    print("  timing benchmarks: excluded")
    print("  nominal trajectories:", sum(len(nominal.SEEDS[t]) * len(nominal.cells(t)) for t in nominal.TASKS))
    nominal_evaluations = sum(len(nominal.SEEDS[t]) * 8 for t in nominal.TASKS)
    sensitivity_count = len(sensitivity.cells()) * len(sensitivity.SEEDS)
    horizon_count = len(horizon.cells()) * len(horizon.SEEDS)
    print("  nominal free-energy jobs:", nominal_evaluations)
    print("  sensitivity trajectories:", sensitivity_count)
    print("  horizon trajectories:", horizon_count)
    print("  total free-energy jobs:", nominal_evaluations + sensitivity_count + horizon_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("frozen", "simulation"), default="frozen")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.source == "frozen":
        from cbo_mppi.figures.generate_all import generate
        generate(args.output / "figures", args.verify_against)
        return

    selected = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        parser.error("unknown stages: " + ", ".join(unknown))

    # This must be set before importing any JAX-backed reproduction module.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    output = args.output.resolve()
    simulation_plan(selected, output, args.gpu_index)
    if not args.execute:
        print("No computation started. Pass --execute to run implemented stages.")
        return
    incomplete = sorted(set(selected) - IMPLEMENTED_SIMULATION_STAGES)
    if incomplete:
        raise SystemExit("Unsupported execution stages: " + ", ".join(incomplete))
    ensure_source_contract(output)
    runners = {
        "nominal": run_nominal,
        "evaluate": run_evaluation,
        "sensitivity": run_sensitivity,
        "horizon": run_horizon,
        "landscapes": run_landscapes,
        "aggregate": run_aggregate,
        "figures": run_figures,
        "verify": run_verify,
    }
    for stage in selected:
        runners[stage](output, args.resume)


if __name__ == "__main__":
    main()
