"""Render the published Pendulum landscape from frozen Monte Carlo arrays."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cbo-mppi-mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data/frozen/figures/pendulum/landscape_tau10.npz"
DEFAULT_OUTPUT = ROOT / "artifacts/figures"

PAPER_RCPARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
}

J_COLOR = "#d62728"
F_COLOR = "#ff7f0e"


def _tex_tick(value, _position):
    return rf"\mbox{{{value:g}}}"


def generate(input_path: Path, output_dir: Path) -> Path:
    with np.load(input_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    a_vals = data["a_vals"]
    b_vals = data["b_vals"]
    params = data["params"]
    n = len(a_vals)
    j_index = int(np.argmin(data["nom_cost"]))
    f_index = int(np.argmin(data["F_hat"]))
    j_grid = data["nom_cost"].reshape(n, n)
    f_grid = data["F_hat"].reshape(n, n)

    matplotlib.rcParams.update(PAPER_RCPARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.20))
    for axis, grid in ((axes[0], j_grid), (axes[1], f_grid)):
        image = axis.pcolormesh(
            a_vals, b_vals, grid.T, cmap="viridis", shading="nearest", rasterized=True
        )
        fig.colorbar(image, ax=axis)
        axis.scatter(
            [params[j_index, 0]], [params[j_index, 1]], marker="x", s=35,
            linewidths=1.4, color=J_COLOR, label=r"$\arg\min J$", zorder=5,
        )
        axis.scatter(
            [params[f_index, 0]], [params[f_index, 1]], marker="+", s=45,
            linewidths=1.4, color=F_COLOR, label=r"$\arg\min \hat{F}$", zorder=5,
        )
        axis.set_xlabel(r"$a$")
        axis.set_ylabel(r"$b$")
        axis.legend(loc="upper right", framealpha=0.85, borderpad=0.25)
        axis.xaxis.set_major_formatter(FuncFormatter(_tex_tick))
        axis.yaxis.set_major_formatter(FuncFormatter(_tex_tick))
    fig.tight_layout(pad=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pendulum_landscape.pdf"
    fig.savefig(
        output, bbox_inches="tight", pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.input.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
