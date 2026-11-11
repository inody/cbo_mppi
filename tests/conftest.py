from __future__ import annotations

import jax.numpy as jnp
import pytest

from experiments.navigation.config import ProblemConfig


@pytest.fixture
def small_problem() -> ProblemConfig:
    return ProblemConfig(
        dt=0.1,
        goal=jnp.asarray([1.0, 0.0]),
        obstacle_centers=jnp.asarray([[0.0, 2.0]]),
        obstacle_radii=jnp.asarray([0.25]),
        safety_margin=0.1,
        q_goal_running=0.1,
        q_goal_terminal=5.0,
        q_velocity_running=0.05,
        q_velocity_terminal=2.0,
        q_control=0.01,
        q_control_rate=0.01,
        q_obstacle=10.0,
        u_max=2.0,
        hard_obstacles=False,
    )
