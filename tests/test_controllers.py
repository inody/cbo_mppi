from __future__ import annotations

import jax
import jax.numpy as jnp

from cbo_mppi.config import CBOConfig, MPPIConfig, ProposedConfig
from cbo_mppi.controllers import (
    init_cbo_mpc,
    init_mppi,
    init_proposed,
    make_cbo_mpc_step,
    make_mppi_step,
    make_proposed_step,
)
from experiments.navigation.env import rollout_batch


def test_equal_rollout_accounting_and_output_shapes(small_problem):
    horizon = 5
    nominal = jnp.zeros((horizon, 2))
    current = jnp.zeros(4)
    previous = jnp.zeros(2)

    mppi_cfg = MPPIConfig(16, 10.0, 0.2, 0.8)
    mppi_step = make_mppi_step(rollout_batch)
    _, _, control, info = mppi_step(
        jax.random.PRNGKey(0), init_mppi(nominal), current, previous,
        small_problem, mppi_cfg,
    )
    assert control.shape == (2,)
    assert int(info["rollout_count"]) == 16

    cbo_cfg = CBOConfig(
        particles=4, cycles=4, alpha=0.1, lambda_=1.0, sigma=0.1,
        delta_s=0.2, init_sigma=0.2, init_noise_correlation=0.8,
        record_history=False,
    )
    cbo_step = make_cbo_mpc_step(rollout_batch)
    cbo_state = init_cbo_mpc(jax.random.PRNGKey(1), nominal, small_problem, cbo_cfg)
    _, _, control, info = cbo_step(
        jax.random.PRNGKey(2), cbo_state, current, previous,
        small_problem, cbo_cfg,
    )
    assert control.shape == (2,)
    assert int(info["rollout_count"]) == 16

    proposed_cfg = ProposedConfig(
        particles=2, cycles=2, local_samples=4, tau=10.0,
        local_noise_sigma=0.2, local_noise_correlation=0.8,
        alpha=0.1, lambda_=1.0, sigma=0.1, delta_s=0.2,
        init_sigma=0.2, init_noise_correlation=0.8,
        record_history=False,
    )
    proposed_step = make_proposed_step(rollout_batch)
    proposed_state = init_proposed(
        jax.random.PRNGKey(3), nominal, small_problem, proposed_cfg
    )
    _, _, control, info = proposed_step(
        jax.random.PRNGKey(4), proposed_state, current, previous,
        small_problem, proposed_cfg,
    )
    assert control.shape == (2,)
    assert int(info["rollout_count"]) == 16
    assert jnp.all(jnp.isfinite(control))
