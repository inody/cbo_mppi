#!/usr/bin/env python3
"""Render Figure 8 from the frozen nominal-horizon summary."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
})

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data/frozen/figures/horizon/figure_input.csv"
OUTPUT = ROOT / "artifacts/figures/horizon_sweep_cost.pdf"

TASKS = ("navigation", "pendulum", "planar_arm")
TITLES = {"navigation": "Navigation", "pendulum": "Pendulum", "planar_arm": "Planar Arm"}
METHODS = ("mppi", "multistart_mppi", "cbo_mppi")
LABELS = {"mppi": "MPPI", "multistart_mppi": "multi-start MPPI", "cbo_mppi": "CBO-MPPI"}
COLORS = {"mppi": "#1f77b4", "multistart_mppi": "#ff7f0e", "cbo_mppi": "#2ca02c"}
MARKERS = {"mppi": "o", "multistart_mppi": "D", "cbo_mppi": "^"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric = (
        "horizon", "lookahead_seconds", "nominal_horizon", "nominal_lookahead_seconds",
        "J_CL_median", "J_CL_q25", "J_CL_q75", "F_CL_median", "F_CL_q25", "F_CL_q75",
    )
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.7), sharex="col")
    for col, task in enumerate(TASKS):
        task_rows = [row for row in rows if row["task"] == task]
        for row_index, prefix in ((0, "J_CL"), (1, "F_CL")):
            ax = axes[row_index, col]
            for method in METHODS:
                selected = sorted(
                    [row for row in task_rows if row["method_id"] == method],
                    key=lambda row: row["lookahead_seconds"],
                )
                x = np.asarray([row["lookahead_seconds"] for row in selected])
                med = np.asarray([row[f"{prefix}_median"] for row in selected])
                lo = np.asarray([row[f"{prefix}_q25"] for row in selected])
                hi = np.asarray([row[f"{prefix}_q75"] for row in selected])
                ax.plot(x, med, color=COLORS[method], marker=MARKERS[method], ms=3,
                        lw=1, label=LABELS[method])
                ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.14, lw=0)
            ax.axvline(task_rows[0]["nominal_lookahead_seconds"], color="0.35", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both", lw=0.4)
        axes[0, col].set_title(TITLES[task])
        axes[1, col].set_xlabel(r"lookahead $H\Delta t$ (s)")
        ticks = sorted({row["lookahead_seconds"] for row in task_rows})
        axes[1, col].set_xticks(ticks)
        axes[1, col].set_xticklabels([f"{value:g}" for value in ticks],
                                     rotation=45 if len(ticks) > 7 else 0)
        axes[1, col].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].set_ylabel(r"median $J_{\rm CL}$")
    axes[1, 0].set_ylabel(r"median $F_{\rm CL}$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
