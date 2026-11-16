from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/frozen/figures"


def test_closed_loop_bundle_shapes():
    with np.load(DATA / "closed_loop_figure_trajectories.npz", allow_pickle=False) as data:
        assert data["navigation_mppi_low"].shape == (100, 81, 4)
        assert data["many_obstacles_mppi_low"].shape == (100, 121, 4)
        assert data["pendulum_mppi_low"].shape == (100, 171, 2)
        assert data["planar_arm_mppi_low"].shape == (100, 36, 6)


def test_navigation_inputs_are_numerical_arrays():
    navigation = DATA / "navigation"
    assert not list(navigation.glob("*.pdf"))
    with np.load(navigation / "route_heights.npz", allow_pickle=False) as data:
        assert data["h_values"].shape == (9,)
        assert data["trajectories"].shape == (9, 61, 2)
        assert data["obstacle_centers"].shape == (2, 2)
        assert data["obstacle_radii"].shape == (2,)
    with np.load(navigation / "free_energy_landscape.npz", allow_pickle=False) as data:
        assert data["heights"].shape == (121,)
        assert data["nominal_costs"].shape == (121,)
        assert data["taus"].shape == (2,)
        assert data["free_energies"].shape == (2, 121)


def test_landscape_inputs_are_complete():
    with np.load(DATA / "pendulum/landscape_tau10.npz", allow_pickle=False) as data:
        assert data["F_hat"].shape == (6561,)
        assert data["nom_cost"].shape == (6561,)
    with np.load(DATA / "planar_arm/landscape_arrays.npz", allow_pickle=False) as data:
        assert data["h"].shape == (61,)
        assert data["projected_free_energy"].shape == (61, 2)
        assert data["representative_display_indices"].shape == (6,)
