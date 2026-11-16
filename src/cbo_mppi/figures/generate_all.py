"""Regenerate all nine manuscript figures from the frozen public inputs."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = (
    "arm_closed_loop_final_postures.pdf",
    "arm_landscape.pdf",
    "arm_posture_grid.pdf",
    "horizon_sweep_cost.pdf",
    "many_obstacles_equal_budget.pdf",
    "navigation_open_loop.pdf",
    "navigation_three_panel.pdf",
    "pendulum_closed_loop.pdf",
    "pendulum_landscape.pdf",
)


def run(command: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def rendered_hash(pdf: Path, work: Path, dpi: int = 150) -> str:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("pdftoppm is required for --verify-against")
    prefix = work / pdf.stem
    subprocess.run(
        [executable, "-singlefile", "-png", "-r", str(dpi), str(pdf), str(prefix)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return hashlib.sha256(prefix.with_suffix(".png").read_bytes()).hexdigest()


def verify(stage: Path, reference: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="cbo-mppi-figure-verify.") as tmp_name:
        tmp = Path(tmp_name)
        for name in EXPECTED:
            candidate = stage / name
            expected = reference / name
            if not expected.is_file():
                raise FileNotFoundError(f"missing reference figure: {expected}")
            candidate_dir = tmp / f"candidate-{candidate.stem}"
            reference_dir = tmp / f"reference-{expected.stem}"
            candidate_dir.mkdir()
            reference_dir.mkdir()
            if rendered_hash(candidate, candidate_dir) != rendered_hash(expected, reference_dir):
                raise RuntimeError(f"rendered figure differs from reference: {name}")
            print("MATCH", name)


def generate(
    output: Path,
    reference: Path | None = None,
    data_root: Path | None = None,
) -> None:
    if shutil.which("latex") is None:
        raise RuntimeError("a LaTeX installation is required to render manuscript fonts")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    python = sys.executable
    data = (ROOT / "data/frozen/figures" if data_root is None else data_root.resolve())

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as stage_name:
        stage = Path(stage_name)
        run([
            python, "-m", "cbo_mppi.figures.closed_loop",
            "--data-file", str(data / "closed_loop_figure_trajectories.npz"),
            "--output-dir", str(stage),
        ], env)
        navigation_command = [
            python, "-m", "cbo_mppi.figures.navigation",
            "--data-file", str(data / "closed_loop_figure_trajectories.npz"),
            "--output-dir", str(stage),
        ]
        route_data = data / "navigation/route_heights.npz"
        landscape_data = data / "navigation/free_energy_landscape.npz"
        if not route_data.is_file() or not landscape_data.is_file():
            raise FileNotFoundError(
                "Navigation figure inputs are missing: "
                f"{route_data}, {landscape_data}"
            )
        navigation_command += [
            "--route-data", str(route_data),
            "--landscape-data", str(landscape_data),
        ]
        run(navigation_command, env)
        run([
            python, "-m", "cbo_mppi.figures.planar_arm",
            "--data-root", str(data / "planar_arm"),
            "--output-dir", str(stage),
        ], env)
        run([
            python, "-m", "cbo_mppi.figures.horizon",
            "--input", str(data / "horizon/figure_input.csv"),
            "--output", str(stage / "horizon_sweep_cost.pdf"),
        ], env)
        run([
            python, "-m", "cbo_mppi.figures.pendulum",
            "--input", str(data / "pendulum/landscape_tau10.npz"),
            "--output-dir", str(stage),
        ], env)

        missing = [name for name in EXPECTED if not (stage / name).is_file()]
        if missing:
            raise RuntimeError("figure generation did not produce: " + ", ".join(missing))
        if reference is not None:
            verify(stage, reference.resolve())
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(stage, output)
    print(f"wrote {len(EXPECTED)} figures to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/figures")
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    generate(args.output, args.verify_against, args.data_root)


if __name__ == "__main__":
    main()
