import math

import jax
import numpy as np

from cbo_mppi.reproduction.evaluate import (
    _navigation_noise,
    _paired_noise,
    finite_free_energy,
)


def test_antithetic_noise_constructions():
    paired = np.asarray(_paired_noise(jax.random.PRNGKey(0), 8, 5, 1, 0.2, 0.9))
    assert paired.shape == (8, 5, 1)
    assert np.array_equal(paired[:4], -paired[4:])

    navigation = np.asarray(
        _navigation_noise(jax.random.PRNGKey(1), 8, 5, 0.2, 0.9)
    )
    assert navigation.shape == (8, 5, 2)
    assert np.array_equal(navigation[:2], -navigation[6:])


def test_free_energy_keeps_infeasible_samples_in_denominator():
    costs = np.asarray([1.0, 2.0, np.inf, np.inf])
    observed = finite_free_energy(costs, tau=1.0)
    expected = -math.log((math.exp(-1.0) + math.exp(-2.0)) / 4.0)
    assert math.isclose(observed, expected)
