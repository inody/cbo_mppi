"""Problem configuration and TOML loading for the navigation experiment."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp


class ProblemConfig(NamedTuple):
    dt: float
    goal: Any
    obstacle_centers: Any
    obstacle_radii: Any
    safety_margin: float
    q_goal_running: float
    q_goal_terminal: float
    q_velocity_running: float
    q_velocity_terminal: float
    q_control: float
    q_control_rate: float
    q_obstacle: float
    u_max: float
    hard_obstacles: bool = False


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def make_problem(raw: dict[str, Any]) -> ProblemConfig:
    problem = raw["problem"]
    return ProblemConfig(
        dt=float(problem["dt"]),
        goal=jnp.asarray(problem["goal"], dtype=jnp.float32),
        obstacle_centers=jnp.asarray(problem["obstacle_centers"], dtype=jnp.float32),
        obstacle_radii=jnp.asarray(problem["obstacle_radii"], dtype=jnp.float32),
        safety_margin=float(problem["safety_margin"]),
        q_goal_running=float(problem["q_goal_running"]),
        q_goal_terminal=float(problem["q_goal_terminal"]),
        q_velocity_running=float(problem["q_velocity_running"]),
        q_velocity_terminal=float(problem["q_velocity_terminal"]),
        q_control=float(problem["q_control"]),
        q_control_rate=float(problem["q_control_rate"]),
        q_obstacle=float(problem["q_obstacle"]),
        u_max=float(problem["u_max"]),
        hard_obstacles=bool(problem.get("hard_obstacles", False)),
    )
