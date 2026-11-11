"""Frozen task adapters for episodic Success and hard feasibility."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from experiments.navigation.config import ProblemConfig
from experiments.pendulum_swingup.config import PendulumConfig
from experiments.pendulum_swingup.costs import angle_error
from experiments.planar_arm.config import ArmConfig
from experiments.planar_arm.env import end_effector, link_obstacle_clearance

from .episodic import EpisodeMonitor


def navigation_goal_error(state: np.ndarray, problem: ProblemConfig) -> float:
    return float(np.linalg.norm(np.asarray(state[:2]) - np.asarray(problem.goal)))


def navigation_clearance(state: np.ndarray, problem: ProblemConfig) -> float:
    position = np.asarray(state[:2], dtype=float)
    centers = np.asarray(problem.obstacle_centers, dtype=float)
    radii = np.asarray(problem.obstacle_radii, dtype=float)
    return float(np.min(np.linalg.norm(position[None, :] - centers, axis=1) - radii))


def navigation_monitor(problem: ProblemConfig) -> EpisodeMonitor:
    return EpisodeMonitor(
        dt=problem.dt,
        success=lambda state: navigation_goal_error(state, problem) <= 0.6,
        clearance=lambda state: navigation_clearance(state, problem),
    )


def pendulum_residuals(
    state: np.ndarray, problem: PendulumConfig
) -> tuple[float, float]:
    state = np.asarray(state)
    theta_error = float(abs(angle_error(jnp.asarray(state[0] - problem.theta_goal))))
    omega_error = float(abs(state[1] - problem.omega_goal))
    return theta_error, omega_error


def pendulum_monitor(problem: PendulumConfig) -> EpisodeMonitor:
    def success(state: np.ndarray) -> bool:
        theta_error, omega_error = pendulum_residuals(state, problem)
        return theta_error < 0.25 and omega_error < 1.0

    return EpisodeMonitor(dt=problem.dt, success=success)


def arm_residuals(state: np.ndarray, problem: ArmConfig) -> tuple[float, float]:
    state = np.asarray(state)
    ee = np.asarray(end_effector(jnp.asarray(state[:3]), problem))
    ee_error = float(np.linalg.norm(ee - np.asarray(problem.goal)))
    velocity = float(np.linalg.norm(state[3:]))
    return ee_error, velocity


def arm_clearance(state: np.ndarray, problem: ArmConfig) -> float:
    return float(link_obstacle_clearance(jnp.asarray(state[:3]), problem))


def arm_monitor(problem: ArmConfig) -> EpisodeMonitor:
    def success(state: np.ndarray) -> bool:
        ee_error, velocity = arm_residuals(state, problem)
        return ee_error < 0.20 and velocity < 1.5

    return EpisodeMonitor(
        dt=problem.dt,
        success=success,
        clearance=lambda state: arm_clearance(state, problem),
    )


def monitor_for_task(task: str, problem) -> EpisodeMonitor:
    factories: dict[str, Callable] = {
        "two_obstacles": navigation_monitor,
        "many_obstacles": navigation_monitor,
        "pendulum": pendulum_monitor,
        "planar_arm": arm_monitor,
    }
    try:
        return factories[task](problem)
    except KeyError as exc:
        raise NotImplementedError(task) from exc
