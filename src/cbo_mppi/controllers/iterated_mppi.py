from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from cbo_mppi.config import IteratedMPPIConfig
from cbo_mppi.controllers.mppi import MPPIState
from cbo_mppi.utils import (
    effective_sample_size,
    mppi_weighted_summaries,
    shift_control_sequence,
    temporally_correlated_normal,
    weighted_or_fallback,
)

RolloutFn = Callable[..., tuple[jnp.ndarray, jnp.ndarray]]


def init_iterated_mppi(
    base_nominal: jnp.ndarray,
    *,
    route_direction: jnp.ndarray | None = None,
    initial_noise_sigma: float | None = None,
    initial_route_sigma: float = 0.0,
) -> MPPIState:
    """Initialize multi-step MPPI with the same state convention as MPPI."""
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


def make_iterated_mppi_step(rollout_fn: RolloutFn):
    """Build sequential MPPI updates with a fixed total rollout identity.

    A round evaluates one batch around the current nominal and immediately
    replaces that nominal by the MPPI weighted mean. The final updated nominal
    supplies the applied control and the next-step warm start.
    """

    @partial(jax.jit, static_argnames=("cfg",))
    def iterated_mppi_step(
        key: jax.Array,
        state: MPPIState,
        current_state: jnp.ndarray,
        previous_control: jnp.ndarray,
        problem,
        cfg: IteratedMPPIConfig,
    ):
        if cfg.rounds <= 0:
            raise ValueError("iterated MPPI rounds must be positive")
        if cfg.rollouts_per_round <= 0:
            raise ValueError("iterated MPPI rollouts_per_round must be positive")

        next_key, first_sample_key, first_route_key = jax.random.split(key, 3)
        sample_shape = (cfg.rollouts_per_round,) + state.nominal.shape
        initial_samples = jnp.broadcast_to(state.nominal, sample_shape)
        initial_costs = jnp.full((cfg.rollouts_per_round,), jnp.inf)
        initial_weights = jnp.zeros((cfg.rollouts_per_round,))
        initial_energy = jnp.asarray(jnp.inf)
        initial_valid = jnp.asarray(False)
        initial_ess_history = jnp.zeros((cfg.rounds,))
        initial_energy_history = jnp.full((cfg.rounds,), jnp.inf)

        configured_initial_sigma = jnp.where(
            jnp.isfinite(state.initial_noise_sigma),
            state.initial_noise_sigma,
            cfg.noise_sigma,
        )

        def body(index, carry):
            (
                nominal,
                _,
                _,
                _,
                _,
                _,
                _,
                ess_history,
                energy_history,
            ) = carry
            sample_key = jax.lax.cond(
                index == 0,
                lambda _: first_sample_key,
                lambda _: jax.random.fold_in(first_sample_key, index),
                operand=None,
            )
            route_key = jax.lax.cond(
                index == 0,
                lambda _: first_route_key,
                lambda _: jax.random.fold_in(first_route_key, index),
                operand=None,
            )
            initial_round = state.first_step & (index == 0)
            noise_sigma = jnp.where(
                initial_round, configured_initial_sigma, cfg.noise_sigma
            )
            route_sigma = jnp.where(
                initial_round, state.initial_route_sigma, 0.0
            )
            noise = noise_sigma * temporally_correlated_normal(
                sample_key, sample_shape, cfg.noise_correlation
            )
            route_coefficients = route_sigma * jax.random.normal(
                route_key, (cfg.rollouts_per_round, 1, 1)
            )
            samples = nominal[None, ...] + noise + (
                route_coefficients * state.route_direction
            )
            costs, _ = rollout_fn(
                current_state,
                samples,
                problem,
                previous_control,
                clip_inputs=cfg.clip_rollout_inputs,
            )
            weights, energy, valid = mppi_weighted_summaries(costs, cfg.tau)
            fallback = jnp.clip(nominal, -problem.u_max, problem.u_max)
            updated_nominal = weighted_or_fallback(samples, weights, fallback)
            ess_history = ess_history.at[index].set(effective_sample_size(weights))
            energy_history = energy_history.at[index].set(energy)
            return (
                updated_nominal,
                nominal,
                samples,
                costs,
                weights,
                energy,
                valid,
                ess_history,
                energy_history,
            )

        (
            output_sequence,
            last_evaluated_nominal,
            last_samples,
            last_costs,
            last_weights,
            last_energy,
            last_valid,
            ess_history,
            energy_history,
        ) = jax.lax.fori_loop(
            0,
            cfg.rounds,
            body,
            (
                state.nominal,
                state.nominal,
                initial_samples,
                initial_costs,
                initial_weights,
                initial_energy,
                initial_valid,
                initial_ess_history,
                initial_energy_history,
            ),
        )
        next_state = MPPIState(
            nominal=shift_control_sequence(output_sequence),
            route_direction=state.route_direction,
            initial_noise_sigma=state.initial_noise_sigma,
            initial_route_sigma=state.initial_route_sigma,
            first_step=jnp.asarray(False),
        )
        info = {
            "free_energy": last_energy,
            "free_energy_by_round": energy_history,
            "minimum_candidate_cost": jnp.min(last_costs),
            "candidate_cost_std": jnp.nanstd(
                jnp.where(jnp.isfinite(last_costs), last_costs, jnp.nan)
            ),
            "ess": effective_sample_size(last_weights),
            "ess_by_round": ess_history,
            "finite_cost_fraction": jnp.mean(jnp.isfinite(last_costs)),
            "valid": last_valid,
            "evaluated_nominal_sequence": last_evaluated_nominal,
            "output_sequence": output_sequence,
            "prediction_sequences": last_samples,
            "prediction_scores": last_costs,
            "rollout_count": jnp.asarray(
                cfg.rounds * cfg.rollouts_per_round
            ),
        }
        return next_key, next_state, output_sequence[0], info

    return iterated_mppi_step
