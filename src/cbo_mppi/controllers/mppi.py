from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from cbo_mppi.config import MPPIConfig
from cbo_mppi.utils import (
    effective_sample_size,
    mppi_weighted_summaries,
    shift_control_sequence,
    temporally_correlated_normal,
    weighted_or_fallback,
)

RolloutFn = Callable[..., tuple[jnp.ndarray, jnp.ndarray]]

_REQUIRED = object()  # sentinel — rollout_fn has no default


class MPPIState(NamedTuple):
    nominal: jnp.ndarray
    route_direction: jnp.ndarray
    initial_noise_sigma: jnp.ndarray
    initial_route_sigma: jnp.ndarray
    first_step: jnp.ndarray


def init_mppi(
    base_nominal: jnp.ndarray,
    *,
    route_direction: jnp.ndarray | None = None,
    initial_noise_sigma: float | None = None,
    initial_route_sigma: float = 0.0,
) -> MPPIState:
    if route_direction is None:
        route_direction = jnp.zeros_like(base_nominal)
    noise_sigma = jnp.asarray(
        jnp.nan if initial_noise_sigma is None else initial_noise_sigma
    )
    return MPPIState(
        nominal=base_nominal,
        route_direction=route_direction,
        initial_noise_sigma=noise_sigma,
        initial_route_sigma=jnp.asarray(initial_route_sigma),
        first_step=jnp.asarray(True),
    )


def make_mppi_step(rollout_fn: RolloutFn):
    @partial(jax.jit, static_argnames=("cfg",))
    def mppi_step(
        key: jax.Array,
        state: MPPIState,
        current_state: jnp.ndarray,
        previous_control: jnp.ndarray,
        problem,
        cfg: MPPIConfig,
    ):
        next_key, sample_key, route_key = jax.random.split(key, 3)
        configured_initial_sigma = jnp.where(
            jnp.isfinite(state.initial_noise_sigma),
            state.initial_noise_sigma,
            cfg.noise_sigma,
        )
        noise_sigma = jnp.where(
            state.first_step, configured_initial_sigma, cfg.noise_sigma
        )
        route_sigma = jnp.where(state.first_step, state.initial_route_sigma, 0.0)
        noise = noise_sigma * temporally_correlated_normal(
            sample_key,
            (cfg.rollouts,) + state.nominal.shape,
            cfg.noise_correlation,
        )
        route_coefficients = route_sigma * jax.random.normal(
            route_key, (cfg.rollouts, 1, 1)
        )
        noise = noise + route_coefficients * state.route_direction
        samples = state.nominal[None, ...] + noise
        costs, _ = rollout_fn(
            current_state,
            samples,
            problem,
            previous_control,
            clip_inputs=cfg.clip_rollout_inputs,
        )
        weights, energy, valid = mppi_weighted_summaries(costs, cfg.tau)
        fallback = jnp.clip(state.nominal, -problem.u_max, problem.u_max)
        output_sequence = weighted_or_fallback(samples, weights, fallback)
        next_state = MPPIState(
            nominal=shift_control_sequence(output_sequence),
            route_direction=state.route_direction,
            initial_noise_sigma=state.initial_noise_sigma,
            initial_route_sigma=state.initial_route_sigma,
            first_step=jnp.asarray(False),
        )
        info = {
            "free_energy": energy,
            "minimum_candidate_cost": jnp.min(costs),
            "candidate_cost_std": jnp.nanstd(
                jnp.where(jnp.isfinite(costs), costs, jnp.nan)
            ),
            "ess": effective_sample_size(weights),
            "finite_cost_fraction": jnp.mean(jnp.isfinite(costs)),
            "valid": valid,
            "evaluated_nominal_sequence": state.nominal,
            "output_sequence": output_sequence,
            "prediction_sequences": samples,
            "prediction_scores": costs,
            "rollout_count": jnp.asarray(cfg.rollouts),
        }
        return next_key, next_state, output_sequence[0], info

    return mppi_step
