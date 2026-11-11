from __future__ import annotations

from typing import NamedTuple


class ArmConfig(NamedTuple):
    dt: float = 0.05
    link_lengths: tuple = (1.0, 1.0, 1.0)
    q_min: float = -2.5
    q_max: float = 2.5
    u_max: float = 5.0
    horizon: int = 40
    damping: float = 1.0
    link_radius: float = 0.05
    safety_margin: float = 0.0

    q_ee: float = 5.0
    q_v: float = 0.1
    q_u: float = 0.001
    q_du: float = 0.01
    q_obs: float = 50.0
    q_lim: float = 10.0
    Q_ee: float = 200.0
    Q_v: float = 1.0

    theta0: tuple = (-0.8, 1.0, -0.6)
    omega0: tuple = (0.0, 0.0, 0.0)
    goal: tuple = (1.8, 1.2)
    obstacle_centers: tuple = ((1.0, 0.6), (1.5, 0.9))
    obstacle_radii: tuple = (0.25, 0.20)

    ee_tol: float = 0.20
    vel_tol: float = 1.5
