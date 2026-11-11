#!/usr/bin/env python3
"""Regenerate the two published Planar Arm landscape PDFs from frozen arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/frozen/figures/planar_arm"
DEFAULT_OUTPUT = ROOT / "artifacts/figures"

LANDSCAPE_SIZE = (301.28 / 72.0, 200.48 / 72.0)
POSTURE_SIZE = (261.999 / 72.0, 187.6 / 72.0)


def configure_style() -> None:
    plt.rcParams.update({
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
    })


def normalize_finite(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    lower, upper = float(np.min(values[mask])), float(np.max(values[mask]))
    output[mask] = (values[mask] - lower) / (upper - lower) if upper > lower else 0.0
    return output


def forward_kinematics(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    angles = np.cumsum(q, dtype=np.float32)
    dx = np.cumsum(lengths * np.cos(angles), dtype=np.float32)
    dy = np.cumsum(lengths * np.sin(angles), dtype=np.float32)
    return np.stack([np.r_[np.float32(0), dx], np.r_[np.float32(0), dy]], axis=-1)


def draw_posture(ax, q, geometry, title, color) -> None:
    joints = forward_kinematics(q, geometry["link_lengths"])
    for center, radius in zip(geometry["obstacle_centers"], geometry["obstacle_radii"]):
        ax.add_patch(plt.Circle(center, radius, color="0.55", alpha=0.45, zorder=1))
        ax.add_patch(plt.Circle(center, radius + geometry["link_radius"], fill=False,
                                ec="0.45", lw=0.6, ls=":", zorder=2))
    ax.plot(joints[:, 0], joints[:, 1], "-", lw=2.2, color=color,
            solid_capstyle="round", zorder=3)
    ax.plot(joints[:, 0], joints[:, 1], ".", color=color, ms=3.8, zorder=4)
    ax.plot(*geometry["goal"], "k*", ms=6.5, zorder=5)
    ax.plot(0, 0, "ks", ms=3.2, zorder=5)
    ax.set(xlim=(-0.45, 2.55), ylim=(-0.75, 2.05), title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    design = json.loads((data_root / "frozen_design.json").read_text())
    geometry = design["authority"]["diagnostic_problem"]
    with np.load(data_root / "landscape_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    h = arrays["h"]
    nominal = arrays["nominal_cost"]
    free_energy = arrays["projected_free_energy"]
    taus = arrays["taus"]
    display = arrays["representative_display_indices"].astype(int)
    final_q = arrays["final_q"]
    j_index = int(np.nanargmin(nominal))
    f_indices = [int(np.nanargmin(free_energy[:, index])) for index in range(len(taus))]

    configure_style()
    fig, ax = plt.subplots(figsize=LANDSCAPE_SIZE)
    j_norm = normalize_finite(nominal)
    ax.plot(h, j_norm, color="#d62728", lw=1.3, label=r"$\bar J$ (nominal)")
    ax.scatter([h[j_index]], [j_norm[j_index]], color="#d62728", s=20, zorder=5)
    colors = ["#74c476", "#005a32"]
    for tau_index, tau in enumerate(taus):
        normalized = normalize_finite(free_energy[:, tau_index])
        ax.plot(h, normalized, color=colors[tau_index], lw=1.3,
                label=rf"$\widehat F$ ($\tau={tau:g}$)")
        index = f_indices[tau_index]
        ax.scatter([h[index]], [normalized[index]], color=colors[tau_index], s=20, zorder=5)
    ax.axvline(0.0, color="0.7", lw=0.9, ls=":")
    for index in display:
        ax.axvline(h[index], color="0.78", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel(r"posture parameter $h$")
    ax.set_ylabel("normalized objective")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper center", handlelength=1.4, borderpad=0.25)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.20, top=0.975)
    fig.savefig(output / "arm_landscape.pdf", dpi=300,
                metadata={"Creator": "nominal_planar_arm_landscape_v2"})
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=POSTURE_SIZE, sharex=True, sharey=True)
    sweep_colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(display)))
    for axis, index, color in zip(axes.ravel(), display, sweep_colors):
        draw_posture(axis, final_q[index], geometry, rf"$h={float(h[index]):g}$", color)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.03, hspace=0.13)
    fig.savefig(output / "arm_posture_grid.pdf", dpi=300,
                metadata={"Creator": "nominal_planar_arm_landscape_v2"})
    plt.close(fig)
    print(output / "arm_landscape.pdf")
    print(output / "arm_posture_grid.pdf")


if __name__ == "__main__":
    main()
