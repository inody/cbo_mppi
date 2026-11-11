from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from cbo_mppi.config import CBOConfig
from cbo_mppi.utils import (
    effective_sample_size,
    normalized_exponential,
    rejuvenate_control_population,
    shift_control_sequence,
    temporally_correlated_normal,
    weighted_or_fallback,
)

RolloutFn = Callable[..., tuple[jnp.ndarray, jnp.ndarray]]


class CBOMPCState(NamedTuple):
    particles: jnp.ndarray


def init_cbo_mpc(
    key: jax.Array,
    base_nominal: jnp.ndarray,
    problem,
    cfg: CBOConfig,
    structured_direction: jnp.ndarray | None = None,
    structured_sigma: float = 0.0,
    noise_sigma: float | None = None,
    clip_initial: bool = True,
) -> CBOMPCState:
    noise_key, structured_key = jax.random.split(key)
    effective_noise_sigma = cfg.init_sigma if noise_sigma is None else noise_sigma
    perturbations = effective_noise_sigma * temporally_correlated_normal(
        noise_key,
        (cfg.particles,) + base_nominal.shape,
        cfg.init_noise_correlation,
    )
    if structured_direction is not None and structured_sigma > 0.0:
        coefficients = structured_sigma * jax.random.normal(
            structured_key, (cfg.particles, 1, 1)
        )
        perturbations = perturbations + coefficients * structured_direction
    particles = base_nominal[None, ...] + perturbations
    if clip_initial:
        particles = jnp.clip(particles, -problem.u_max, problem.u_max)
    return CBOMPCState(particles=particles)


def make_cbo_mpc_step(rollout_fn: RolloutFn):
    @partial(jax.jit, static_argnames=("cfg",))
    def cbo_mpc_step(
        key: jax.Array,
        state: CBOMPCState,
        current_state: jnp.ndarray,
        previous_control: jnp.ndarray,
        problem,
        cfg: CBOConfig,
    ):
        next_key, loop_key, warm_start_key = jax.random.split(key, 3)

        def population_statistics(points):
            flattened = points.reshape(cfg.particles, -1)
            centered = flattened - jnp.mean(flattened, axis=0, keepdims=True)
            denominator = max(cfg.particles - 1, 1)
            covariance_trace = jnp.sum(centered**2) / denominator
            return covariance_trace, jnp.sqrt(2.0 * covariance_trace)

        initial_costs = jnp.full((cfg.particles,), jnp.inf)
        _hk = cfg.cycles if cfg.record_history else 1
        _hn = cfg.particles if cfg.record_history else 1
        initial_cost_history = jnp.full((_hk, _hn), jnp.inf)
        initial_consensus_distance_history = jnp.zeros((_hk, _hn))
        initial_ess_history = jnp.zeros((cfg.cycles,))
        if cfg.record_history:
            initial_covariance_trace_history = jnp.zeros((cfg.cycles,))
            initial_rms_pairwise_distance_history = jnp.zeros((cfg.cycles,))
        else:
            initial_covariance_trace_history = jnp.zeros((1,))
            initial_rms_pairwise_distance_history = jnp.zeros((1,))
            _pre_cov_trace, _ = population_statistics(state.particles)

        def update_once(index, carry):
            particles, ess_history = carry
            costs, _ = rollout_fn(
                current_state,
                particles,
                problem,
                previous_control,
                clip_inputs=cfg.clip_rollout_inputs,
            )
            weights, _ = normalized_exponential(-cfg.alpha * costs)
            ess_history = ess_history.at[index].set(effective_sample_size(weights))
            fallback = jnp.mean(jnp.clip(particles, -problem.u_max, problem.u_max), axis=0)
            consensus = weighted_or_fallback(particles, weights, fallback)
            displacement = particles - consensus
            distances = jnp.linalg.norm(displacement.reshape(cfg.particles, -1), axis=1)
            noise = jax.random.normal(jax.random.fold_in(loop_key, index), particles.shape)
            updated = (
                particles
                - cfg.lambda_ * cfg.delta_s * displacement
                + cfg.sigma
                * distances[:, None, None]
                * noise
                * jnp.sqrt(cfg.delta_s)
            )
            return updated, ess_history

        def body(index, carry):
            (
                particles,
                _,
                cost_history,
                consensus_distance_history,
                covariance_trace_history,
                rms_pairwise_distance_history,
                ess_history,
            ) = carry
            costs, _ = rollout_fn(
                current_state,
                particles,
                problem,
                previous_control,
                clip_inputs=cfg.clip_rollout_inputs,
            )
            weights, _ = normalized_exponential(-cfg.alpha * costs)
            ess_history = ess_history.at[index].set(effective_sample_size(weights))
            fallback = jnp.mean(jnp.clip(particles, -problem.u_max, problem.u_max), axis=0)
            consensus = weighted_or_fallback(particles, weights, fallback)
            displacement = particles - consensus
            distances = jnp.linalg.norm(displacement.reshape(cfg.particles, -1), axis=1)
            if cfg.record_history:
                covariance_trace, rms_pairwise_distance = population_statistics(particles)
            noise = jax.random.normal(jax.random.fold_in(loop_key, index), particles.shape)
            proposed_update = (
                particles
                - cfg.lambda_ * cfg.delta_s * displacement
                + cfg.sigma
                * distances[:, None, None]
                * noise
                * jnp.sqrt(cfg.delta_s)
            )
            updated = jax.lax.cond(
                index < cfg.cycles - 1,
                lambda _: proposed_update,
                lambda _: particles,
                operand=None,
            )
            if cfg.record_history:
                cost_history = cost_history.at[index].set(costs)
                consensus_distance_history = consensus_distance_history.at[index].set(
                    distances
                )
                covariance_trace_history = covariance_trace_history.at[index].set(
                    covariance_trace
                )
                rms_pairwise_distance_history = rms_pairwise_distance_history.at[index].set(
                    rms_pairwise_distance
                )
            return (
                updated,
                costs,
                cost_history,
                consensus_distance_history,
                covariance_trace_history,
                rms_pairwise_distance_history,
                ess_history,
            )

        if cfg.record_history:
            (
                particles_k,
                last_costs,
                cost_history,
                consensus_distance_history,
                covariance_trace_history,
                rms_pairwise_distance_history,
                ess_history,
            ) = jax.lax.fori_loop(
                0,
                cfg.cycles,
                body,
                (
                    state.particles,
                    initial_costs,
                    initial_cost_history,
                    initial_consensus_distance_history,
                    initial_covariance_trace_history,
                    initial_rms_pairwise_distance_history,
                    initial_ess_history,
                ),
            )
        else:
            particles_k, ess_history = jax.lax.fori_loop(
                0,
                max(cfg.cycles - 1, 0),
                update_once,
                (state.particles, initial_ess_history),
            )
            last_costs, _ = rollout_fn(
                current_state,
                particles_k,
                problem,
                previous_control,
                clip_inputs=cfg.clip_rollout_inputs,
            )
            cost_history = initial_cost_history
            consensus_distance_history = initial_consensus_distance_history
            covariance_trace_history = initial_covariance_trace_history
            rms_pairwise_distance_history = initial_rms_pairwise_distance_history
        last_evaluated = particles_k
        best_index = jnp.argmin(last_costs)
        valid = jnp.any(jnp.isfinite(last_costs))
        fallback = jnp.mean(jnp.clip(last_evaluated, -problem.u_max, problem.u_max), axis=0)
        output_sequence = jnp.where(valid, last_evaluated[best_index], fallback)
        last_weights, _ = normalized_exponential(-cfg.alpha * last_costs)
        ess_history = ess_history.at[-1].set(effective_sample_size(last_weights))
        shifted_particles = shift_control_sequence(particles_k)
        shifted_output = shift_control_sequence(output_sequence)
        shifted_particles, rejuvenation_noise_rms, refreshed_fraction = (
            rejuvenate_control_population(
                warm_start_key,
                shifted_particles,
                shifted_output,
                mode=cfg.warm_start_mode,
                jitter_kind=cfg.warm_start_jitter_kind,
                sigma=cfg.warm_start_sigma,
                refresh_rate=cfg.warm_start_refresh_rate,
                min_cov_trace=cfg.warm_start_min_cov_trace,
                apply_to=cfg.warm_start_apply_to,
            )
        )
        warm_start_covariance_trace, warm_start_rms_pairwise_distance = (
            population_statistics(shifted_particles)
        )
        next_state = CBOMPCState(particles=shifted_particles)
        if cfg.record_history:
            cov_at_start = covariance_trace_history[0]
            cov_at_end = covariance_trace_history[-1]
        else:
            cov_at_start = _pre_cov_trace
            cov_at_end, _ = population_statistics(particles_k)
        info = {
            "minimum_candidate_cost": jnp.min(last_costs),
            "candidate_cost_std": jnp.nanstd(
                jnp.where(jnp.isfinite(last_costs), last_costs, jnp.nan)
            ),
            "ess": effective_sample_size(last_weights),
            "ess_by_cycle": ess_history,
            "finite_cost_fraction": jnp.mean(jnp.isfinite(last_costs)),
            "valid": valid,
            "selected_particle": best_index,
            "output_sequence": output_sequence,
            "prediction_sequences": last_evaluated,
            "prediction_scores": last_costs,
            "particle_covariance_trace_at_step_start": cov_at_start,
            "particle_covariance_trace_at_final_evaluation": cov_at_end,
            "warm_start_covariance_trace": warm_start_covariance_trace,
            "warm_start_rms_pairwise_distance": warm_start_rms_pairwise_distance,
            "rejuvenation_noise_rms": rejuvenation_noise_rms,
            "refreshed_particle_fraction": refreshed_fraction,
            "rollout_count": jnp.asarray(cfg.cycles * cfg.particles),
        }
        if cfg.record_history:
            info["particle_covariance_trace_by_cycle"] = covariance_trace_history
            info["particle_rms_pairwise_distance_by_cycle"] = rms_pairwise_distance_history
            info["consensus_distance_by_cycle"] = consensus_distance_history
            info["cost_by_cycle"] = cost_history
        return next_key, next_state, output_sequence[0], info

    return cbo_mpc_step
