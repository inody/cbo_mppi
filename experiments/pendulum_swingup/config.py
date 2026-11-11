from __future__ import annotations

import math
from typing import NamedTuple


class PendulumConfig(NamedTuple):
    dt: float = 0.05
    g: float = 9.81
    l: float = 1.0
    m: float = 1.0
    c: float = 0.05
    u_max: float = 3.0
    horizon: int = 160
    omega_max: float = 8.0

    # Cost weights for the *weighted* style (open-loop landscape). The
    # regulation style hardcodes the gym/honda coefficients (1.0, 0.1).
    q_angle: float = 0.05
    q_omega: float = 0.01
    q_u: float = 0.001
    q_du: float = 0.01
    Q_angle: float = 30.0
    Q_omega: float = 3.0

    theta0: float = math.pi
    omega0: float = 0.0
    theta_goal: float = 0.0
    omega_goal: float = 0.0

    theta_tol: float = 0.25
    omega_tol: float = 1.0


class LandscapeConfig(NamedTuple):
    tau: float
    sigma_local: float
    rho: float
    samples: int
    grid_n: int
    mode: str = "full_double"
    k1: float | None = None
    k2: float | None = None
    target_batch: int = 131072
    seed: int = 27200
