import numpy as np

from cbo_mppi.reproduction.landscapes import navigation_routes


def test_navigation_route_family_is_generated_from_raw_horizon(tmp_path):
    path = navigation_routes(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["h_values"].shape == (9,)
        assert archive["trajectories"].shape == (9, 61, 2)
