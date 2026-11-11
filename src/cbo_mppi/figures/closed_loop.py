#!/usr/bin/env python3
"""Generate the fresh-100 closed-loop figures from frozen trajectories."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cbo-mppi-mpl-cache")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/frozen/figures/closed_loop_figure_trajectories.npz"

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": .6,
    "lines.linewidth": 1.,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})

C = {"mppi": "#1f77b4", "cbo_mpc": "#d62728", "cbo_mppi": "#2ca02c"}
POS = {
    "mppi_low": (slice(0, 2), 0),
    "cbo_mppi_low": (slice(0, 2), 2),
    "cbo_mpc": (slice(1, 3), 1),
    "mppi_high": (slice(2, 4), 0),
    "cbo_mppi_high": (slice(2, 4), 2),
}

MANY_CENTERS = (
    (-7.70, -0.80), (-7.20, -6.40), (-6.25, 3.25), (-4.85, 1.15),
    (-3.55, 7.25), (-2.55, 5.30), (-1.85, 1.45), (-1.25, -2.05),
    (0.95, 1.45), (2.95, -6.20), (3.65, 4.80), (4.45, 2.35),
    (3.95, -0.75), (5.50, -3.75),
)
MANY_RADII = (
    .90, 1.05, .95, .90, .90, 1.05, .95,
    .95, 1.05, .95, .90, .90, .95, 1.05,
)


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"frozen figure data are absent: {path}")
    return np.load(path, allow_pickle=False)


def states(data, key: str, horizon: int):
    runs = np.asarray(data[key])
    if runs.ndim != 3 or runs.shape[1] != horizon + 1:
        raise ValueError(f"unexpected frozen trajectory shape for {key}: {runs.shape}")
    return runs


def axes(size=(7.16, 4.75), hspace=.28, wspace=.25):
    figure = plt.figure(figsize=size)
    grid = figure.add_gridspec(4, 3, wspace=wspace, hspace=hspace)
    return figure, {key: figure.add_subplot(grid[row, col])
                    for key, (row, col) in POS.items()}


def save(figure, path, pad=.01):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=pad,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def nav_panel(ax, centers, radii, start, goal, limits):
    for center, radius in zip(centers, radii):
        ax.add_patch(plt.Circle(center, radius, color=".58", alpha=.52, zorder=1))
    ax.plot(*start, "ks", ms=3.2)
    ax.plot(*goal, "k*", ms=6.5)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


def plot_nav(data, out, many=False):
    if many:
        prefix = "many_obstacles"
        horizon = 120
        centers = MANY_CENTERS
        radii = MANY_RADII
        start = (-8.5, -9.0)
        goal = (8.5, 9.0)
        limits = ((-10.5, 10.5), (-10.5, 10.5))
        taus = (15, 3000)
    else:
        prefix = "navigation"
        horizon = 80
        centers = ((0, .85), (0, -.85))
        radii = (.5, .5)
        start = (-4, 0)
        goal = (4, 0)
        limits = ((-5.2, 5.2), (-2.65, 2.65))
        taus = (15, 300)

    cells = {
        "mppi_low": (f"{prefix}_mppi_low", rf"MPPI ($\tau={taus[0]}$)", "mppi"),
        "mppi_high": (f"{prefix}_mppi_high", rf"MPPI ($\tau={taus[1]}$)", "mppi"),
        "cbo_mppi_low": (f"{prefix}_cbo_mppi_low", rf"CBO-MPPI ($\tau={taus[0]}$)", "cbo_mppi"),
        "cbo_mppi_high": (f"{prefix}_cbo_mppi_high", rf"CBO-MPPI ($\tau={taus[1]}$)", "cbo_mppi"),
        "cbo_mpc": (f"{prefix}_cbo_mpc", "CBO-MPC", "cbo_mpc"),
    }
    figure, panel_axes = axes(
        (7.16, 5.1) if many else (7.16, 3.65),
        hspace=.65 if many else .08,
        wspace=.25 if many else .18,
    )
    for panel, (key, title, method) in cells.items():
        ax = panel_axes[panel]
        nav_panel(ax, centers, radii, start, goal, limits)
        for trajectory in states(data, key, horizon):
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=C[method],
                    lw=.8, alpha=.42, zorder=2)
        if many:
            ax.set_title(title)
        else:
            ax.xaxis.labelpad = 1
            ax.yaxis.labelpad = 1
            ax.set_title(title, pad=2)
    save(figure, out, pad=.02 if many else .01)


def plot_pend(data, out):
    horizon = 170
    cells = {
        "mppi_low": ("pendulum_mppi_low", r"MPPI ($\tau=1$)", "mppi"),
        "mppi_high": ("pendulum_mppi_high", r"MPPI ($\tau=10$)", "mppi"),
        "cbo_mppi_low": ("pendulum_cbo_mppi_low", r"CBO-MPPI ($\tau=1$)", "cbo_mppi"),
        "cbo_mppi_high": ("pendulum_cbo_mppi_high", r"CBO-MPPI ($\tau=10$)", "cbo_mppi"),
        "cbo_mpc": ("pendulum_cbo_mpc", "CBO-MPC", "cbo_mpc"),
    }
    figure, panel_axes = axes((7.16, 4.0), hspace=.95, wspace=.18)
    for panel, (key, title, method) in cells.items():
        ax = panel_axes[panel]
        ax.plot(0, 0, "k*", ms=6.5)
        ax.set_xlim(-np.pi-.15, np.pi+.15)
        ax.set_ylim(-7, 7)
        ax.set_xticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", r"$0$", r"$\pi$"])
        ax.grid(alpha=.2)
        ax.set_xlabel(r"wrapped $\theta$", labelpad=1)
        ax.set_ylabel(r"$\omega$", labelpad=1)
        for trajectory in states(data, key, horizon):
            theta = np.arctan2(np.sin(trajectory[:, 0]), np.cos(trajectory[:, 0]))
            omega = trajectory[:, 1].copy()
            jumps = np.flatnonzero(np.abs(np.diff(theta)) > np.pi) + 1
            theta[jumps] = np.nan
            omega[jumps] = np.nan
            ax.plot(theta, omega, color=C[method], lw=.75, alpha=.38)
        ax.set_title(title, pad=2)
    save(figure, out)


def joints(q):
    angle = np.cumsum(q)
    return np.vstack((np.zeros((1, 2)), np.column_stack((
        np.cumsum(np.cos(angle)), np.cumsum(np.sin(angle)),
    ))))


def plot_arm(data, out):
    horizon = 35
    cells = {
        "mppi_low": ("planar_arm_mppi_low", r"MPPI ($\tau=3$)", "mppi"),
        "mppi_high": ("planar_arm_mppi_high", r"MPPI ($\tau=300$)", "mppi"),
        "cbo_mppi_low": ("planar_arm_cbo_mppi_low", r"CBO-MPPI ($\tau=3$)", "cbo_mppi"),
        "cbo_mppi_high": ("planar_arm_cbo_mppi_high", r"CBO-MPPI ($\tau=300$)", "cbo_mppi"),
        "cbo_mpc": ("planar_arm_cbo_mpc", "CBO-MPC", "cbo_mpc"),
    }
    figure, panel_axes = axes((7.16, 4.1), hspace=.95, wspace=.18)
    for panel, (key, title, method) in cells.items():
        ax = panel_axes[panel]
        for center, radius in [((1, .6), .25), ((1.5, .9), .2)]:
            ax.add_patch(plt.Circle(center, radius, color=".58", alpha=.52))
        ax.plot(1.8, 1.2, "k*", ms=6.5)
        ax.set_xlim(-.15, 2.35)
        ax.set_ylim(-.65, 1.55)
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        ax.set_xlabel(r"$x$", labelpad=1)
        ax.set_ylabel(r"$y$", labelpad=1)
        for trajectory in states(data, key, horizon):
            arm = joints(trajectory[-1, :3])
            ax.plot(arm[:, 0], arm[:, 1], "-.", color=C[method], lw=.8, alpha=.42)
        ax.set_title(title, pad=2)
    save(figure, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    args = parser.parse_args()
    with load_data(args.data_file) as data:
        plot_pend(data, args.output_dir / "pendulum_closed_loop.pdf")
        plot_arm(data, args.output_dir / "arm_closed_loop_final_postures.pdf")
        plot_nav(data, args.output_dir / "many_obstacles_equal_budget.pdf", True)


if __name__ == "__main__":
    main()
