from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from cbo_mppi.config import ProposedConfig
from cbo_mppi.utils import (
    rejuvenate_control_population,
    effective_sample_size,
    free_energy,
    mppi_weighted_summaries,
    normalized_exponential,
    shift_control_sequence,
    temporally_correlated_normal,
    weighted_or_fallback,
)

RolloutFn = Callable[..., tuple[jnp.ndarray, jnp.ndarray]]


class ProposedState(NamedTuple):
    centers: jnp.ndarray


def init_proposed(
    key: jax.Array,
    base_nominal: jnp.ndarray,
    problem,
    cfg: ProposedConfig,
    structured_direction: jnp.ndarray | None = None,
    structured_sigma: float = 0.0,
    noise_sigma: float | None = None,
    clip_initial: bool = True,
) -> ProposedState:
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
    centers = base_nominal[None, ...] + perturbations
    if clip_initial:
        centers = jnp.clip(centers, -problem.u_max, problem.u_max)
    return ProposedState(centers=centers)


def make_proposed_step(rollout_fn: RolloutFn):
    @partial(jax.jit, static_argnames=("cfg",))
    def proposed_step(
        key: jax.Array,
        state: ProposedState,
        current_state: jnp.ndarray,
        previous_control: jnp.ndarray,
        problem,
        cfg: ProposedConfig,
    ):
        next_key, loop_key, warm_start_key = jax.random.split(key, 3)
        sample_shape = (cfg.particles, cfg.local_samples) + state.centers.shape[1:]

        def population_statistics(points):
            flattened = points.reshape(cfg.particles, -1)
            centered = flattened - jnp.mean(flattened, axis=0, keepdims=True)
            denominator = max(cfg.particles - 1, 1)
            covariance_trace = jnp.sum(centered**2) / denominator
            rms_pairwise_distance = jnp.sqrt(2.0 * covariance_trace)
            return covariance_trace, rms_pairwise_distance

        initial_samples = jnp.broadcast_to(state.centers[:, None, ...], sample_shape)
        initial_costs = jnp.full((cfg.particles, cfg.local_samples), jnp.inf)
        initial_free_energy = jnp.full((cfg.particles,), jnp.inf)
        _hk = cfg.cycles if cfg.record_history else 1
        _hn = cfg.particles if cfg.record_history else 1
        initial_free_energy_history = jnp.full((_hk, _hn), jnp.inf)
        initial_local_ess_history = jnp.zeros((_hk, _hn))
        initial_feasible_count_history = jnp.zeros((_hk, _hn), dtype=jnp.int32)
        initial_consensus_distance_history = jnp.zeros((_hk, _hn))
        initial_mean_free_energy_history = jnp.full((cfg.cycles,), jnp.inf)
        initial_min_free_energy_history = jnp.full((cfg.cycles,), jnp.inf)
        initial_outer_ess_history = jnp.zeros((cfg.cycles,))
        initial_mean_local_mppi_displacement_history = jnp.zeros((cfg.cycles,))
        initial_mean_consensus_distance_history = jnp.zeros((cfg.cycles,))
        initial_mean_gradient_step_norm_history = jnp.zeros((cfg.cycles,))
        initial_max_gradient_step_norm_history = jnp.zeros((cfg.cycles,))
        initial_gradient_clip_fraction_history = jnp.zeros((cfg.cycles,))
        if cfg.record_history:
            initial_covariance_trace_history = jnp.zeros((cfg.cycles,))
            initial_rms_pairwise_distance_history = jnp.zeros((cfg.cycles,))
            initial_split_agreement_history = jnp.zeros((cfg.cycles,), dtype=jnp.bool_)
            initial_j_f_disagreement_history = jnp.zeros((cfg.cycles,), dtype=jnp.bool_)
        else:
            initial_covariance_trace_history = jnp.zeros((1,))
            initial_rms_pairwise_distance_history = jnp.zeros((1,))
            initial_split_agreement_history = jnp.zeros((1,), dtype=jnp.bool_)
            initial_j_f_disagreement_history = jnp.zeros((1,), dtype=jnp.bool_)
            _pre_cov_trace, _ = population_statistics(state.centers)

        def body(index, carry):
            (
                centers,
                _,
                _,
                _,
                free_energy_history,
                local_ess_history,
                feasible_count_history,
                consensus_distance_history,
                covariance_trace_history,
                rms_pairwise_distance_history,
                split_agreement_history,
                j_f_disagreement_history,
                mean_free_energy_history,
                min_free_energy_history,
                outer_ess_history,
                mean_local_mppi_displacement_history,
                mean_consensus_distance_history,
                mean_gradient_step_norm_history,
                max_gradient_step_norm_history,
                gradient_clip_fraction_history,
            ) = carry
            sample_key = jax.random.fold_in(loop_key, 2 * index)
            cbo_key = jax.random.fold_in(loop_key, 2 * index + 1)
            if cfg.shared_local_noise:
                shared_noise = temporally_correlated_normal(
                    sample_key,
                    sample_shape[1:],
                    cfg.local_noise_correlation,
                )
                local_noise = cfg.local_noise_sigma * jnp.broadcast_to(
                    shared_noise[None, ...], sample_shape
                )
            else:
                local_noise = cfg.local_noise_sigma * temporally_correlated_normal(
                    sample_key, sample_shape, cfg.local_noise_correlation
                )
            samples = centers[:, None, ...] + local_noise
            flat_samples = samples.reshape(
                cfg.particles * cfg.local_samples, *state.centers.shape[1:]
            )
            flat_costs, _ = rollout_fn(
                current_state,
                flat_samples,
                problem,
                previous_control,
                clip_inputs=cfg.clip_rollout_inputs,
            )
            costs = flat_costs.reshape(cfg.particles, cfg.local_samples)
            all_local_weights, estimated_free_energy, _ = mppi_weighted_summaries(
                costs, cfg.tau, axis=1
            )
            outer_weights, _ = normalized_exponential(-cfg.alpha * estimated_free_energy)
            outer_ess_history = outer_ess_history.at[index].set(
                effective_sample_size(outer_weights)
            )
            fallback = jnp.mean(jnp.clip(centers, -problem.u_max, problem.u_max), axis=0)
            consensus = weighted_or_fallback(centers, outer_weights, fallback)
            displacement = centers - consensus
            distances = jnp.linalg.norm(displacement.reshape(cfg.particles, -1), axis=1)
            center_fallbacks = jnp.clip(centers, -problem.u_max, problem.u_max)
            local_sequences = jax.vmap(weighted_or_fallback)(
                samples, all_local_weights, center_fallbacks
            )
            local_mppi_displacement = local_sequences - centers
            local_mppi_displacement_norm = jnp.linalg.norm(
                local_mppi_displacement.reshape(cfg.particles, -1), axis=1
            )
            raw_gradient_step = cfg.eta_grad * local_mppi_displacement
            raw_gradient_step_norm = cfg.eta_grad * local_mppi_displacement_norm
            if cfg.grad_clip > 0.0:
                gradient_scale = jnp.minimum(
                    1.0, cfg.grad_clip / jnp.maximum(raw_gradient_step_norm, 1e-12)
                )
                gradient_step = raw_gradient_step * gradient_scale.reshape(
                    (cfg.particles,) + (1,) * (centers.ndim - 1)
                )
                gradient_clip_fraction = jnp.mean(
                    raw_gradient_step_norm > cfg.grad_clip
                )
            else:
                gradient_step = raw_gradient_step
                gradient_clip_fraction = jnp.asarray(0.0)
            gradient_step_norm = jnp.linalg.norm(
                gradient_step.reshape(cfg.particles, -1), axis=1
            )
            if cfg.record_history:
                local_ess = jax.vmap(effective_sample_size)(all_local_weights)
                feasible_count = jnp.sum(jnp.isfinite(costs), axis=1)
                covariance_trace, rms_pairwise_distance = population_statistics(centers)
                if cfg.local_samples >= 2:
                    even_free_energy = free_energy(costs[:, ::2], cfg.tau, axis=1)
                    odd_free_energy = free_energy(costs[:, 1::2], cfg.tau, axis=1)
                    split_agreement = jnp.argmin(even_free_energy) == jnp.argmin(
                        odd_free_energy
                    )
                else:
                    split_agreement = jnp.asarray(False)
                free_energy_best = jnp.argmin(estimated_free_energy)
                best_local_cost_center = jnp.argmin(jnp.min(costs, axis=1))
                j_f_disagreement = free_energy_best != best_local_cost_center
            cbo_noise = jax.random.normal(cbo_key, centers.shape)
            base_update = (
                centers
                - cfg.lambda_ * cfg.delta_s * displacement
                + cfg.sigma
                * distances[:, None, None]
                * cbo_noise
                * jnp.sqrt(cfg.delta_s)
            )
            # Keep the default eta_grad=0 CBO branch arithmetically identical to
            # the frozen implementation. The independent_mppi branch is an
            # experiment-only no-consensus ablation: every center follows only its
            # own local MPPI weighted mean.
            if cfg.center_update_rule == "cbo":
                if cfg.eta_grad == 0.0:
                    proposed_update = base_update
                else:
                    proposed_update = base_update + gradient_step
            elif cfg.center_update_rule == "independent_mppi":
                proposed_update = local_sequences
            else:
                raise ValueError(
                    f"unknown proposed center update rule: {cfg.center_update_rule}"
                )
            # K counts evaluations. The final evaluated population is used for
            # output and warm start, so only the first K-1 rounds update particles.
            updated = jax.lax.cond(
                index < cfg.cycles - 1,
                lambda _: proposed_update,
                lambda _: centers,
                operand=None,
            )
            if cfg.record_history:
                free_energy_history = free_energy_history.at[index].set(
                    estimated_free_energy
                )
                local_ess_history = local_ess_history.at[index].set(local_ess)
                feasible_count_history = feasible_count_history.at[index].set(
                    feasible_count
                )
                consensus_distance_history = consensus_distance_history.at[index].set(
                    distances
                )
                covariance_trace_history = covariance_trace_history.at[index].set(
                    covariance_trace
                )
                rms_pairwise_distance_history = rms_pairwise_distance_history.at[index].set(
                    rms_pairwise_distance
                )
                split_agreement_history = split_agreement_history.at[index].set(
                    split_agreement
                )
                j_f_disagreement_history = j_f_disagreement_history.at[index].set(
                    j_f_disagreement
                )
            mean_free_energy_history = mean_free_energy_history.at[index].set(
                jnp.mean(estimated_free_energy)
            )
            min_free_energy_history = min_free_energy_history.at[index].set(
                jnp.min(estimated_free_energy)
            )
            mean_local_mppi_displacement_history = (
                mean_local_mppi_displacement_history.at[index].set(
                    jnp.mean(local_mppi_displacement_norm)
                )
            )
            mean_consensus_distance_history = (
                mean_consensus_distance_history.at[index].set(jnp.mean(distances))
            )
            mean_gradient_step_norm_history = (
                mean_gradient_step_norm_history.at[index].set(
                    jnp.mean(gradient_step_norm)
                )
            )
            max_gradient_step_norm_history = max_gradient_step_norm_history.at[
                index
            ].set(jnp.max(gradient_step_norm))
            gradient_clip_fraction_history = gradient_clip_fraction_history.at[
                index
            ].set(gradient_clip_fraction)
            return (
                updated,
                samples,
                costs,
                estimated_free_energy,
                free_energy_history,
                local_ess_history,
                feasible_count_history,
                consensus_distance_history,
                covariance_trace_history,
                rms_pairwise_distance_history,
                split_agreement_history,
                j_f_disagreement_history,
                mean_free_energy_history,
                min_free_energy_history,
                outer_ess_history,
                mean_local_mppi_displacement_history,
                mean_consensus_distance_history,
                mean_gradient_step_norm_history,
                max_gradient_step_norm_history,
                gradient_clip_fraction_history,
            )

        (
            centers_k,
            last_samples,
            last_costs,
            last_free_energy,
            free_energy_history,
            local_ess_history,
            feasible_count_history,
            consensus_distance_history,
            covariance_trace_history,
            rms_pairwise_distance_history,
            split_agreement_history,
            j_f_disagreement_history,
            mean_free_energy_history,
            min_free_energy_history,
            outer_ess_history,
            mean_local_mppi_displacement_history,
            mean_consensus_distance_history,
            mean_gradient_step_norm_history,
            max_gradient_step_norm_history,
            gradient_clip_fraction_history,
        ) = jax.lax.fori_loop(
            0,
            cfg.cycles,
            body,
            (
                state.centers,
                initial_samples,
                initial_costs,
                initial_free_energy,
                initial_free_energy_history,
                initial_local_ess_history,
                initial_feasible_count_history,
                initial_consensus_distance_history,
                initial_covariance_trace_history,
                initial_rms_pairwise_distance_history,
                initial_split_agreement_history,
                initial_j_f_disagreement_history,
                initial_mean_free_energy_history,
                initial_min_free_energy_history,
                initial_outer_ess_history,
                initial_mean_local_mppi_displacement_history,
                initial_mean_consensus_distance_history,
                initial_mean_gradient_step_norm_history,
                initial_max_gradient_step_norm_history,
                initial_gradient_clip_fraction_history,
            ),
        )
        last_evaluated_centers = centers_k
        selected_center = jnp.argmin(last_free_energy)
        selected_costs = last_costs[selected_center]
        local_weights, _, valid = mppi_weighted_summaries(selected_costs, cfg.tau)
        fallback = jnp.clip(
            last_evaluated_centers[selected_center], -problem.u_max, problem.u_max
        )
        outer_weights, _ = normalized_exponential(-cfg.alpha * last_free_energy)
        if cfg.output_rule == "weighted":
            output_sequence = weighted_or_fallback(
                last_samples[selected_center], local_weights, fallback
            )
        elif cfg.output_rule == "outer_weighted":
            all_local_weights, _, _ = mppi_weighted_summaries(
                last_costs, cfg.tau, axis=1
            )
            center_fallbacks = jnp.clip(
                last_evaluated_centers, -problem.u_max, problem.u_max
            )
            local_sequences = jax.vmap(weighted_or_fallback)(
                last_samples, all_local_weights, center_fallbacks
            )
            output_sequence = weighted_or_fallback(
                local_sequences, outer_weights, fallback
            )
        elif cfg.output_rule == "best_sample":
            best_sample = last_samples[selected_center, jnp.argmin(selected_costs)]
            output_sequence = jnp.where(valid, best_sample, fallback)
        elif cfg.output_rule == "center":
            output_sequence = fallback
        else:
            raise ValueError(f"unknown proposed output rule: {cfg.output_rule}")
        shifted_centers = shift_control_sequence(centers_k)
        shifted_output = shift_control_sequence(output_sequence)
        shifted_centers, rejuvenation_noise_rms, refreshed_fraction = (
            rejuvenate_control_population(
                warm_start_key,
                shifted_centers,
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
            population_statistics(shifted_centers)
        )
        next_state = ProposedState(centers=shifted_centers)
        sorted_free_energy = jnp.sort(last_free_energy)
        free_energy_gap = jnp.where(
            cfg.particles >= 2,
            sorted_free_energy[1] - sorted_free_energy[0],
            jnp.inf,
        )
        best_local_cost_center = jnp.argmin(jnp.min(last_costs, axis=1))
        if cfg.record_history:
            cov_at_start = covariance_trace_history[0]
            cov_at_end = covariance_trace_history[-1]
        else:
            cov_at_start = _pre_cov_trace
            cov_at_end, _ = population_statistics(centers_k)
        selected_local_ess = effective_sample_size(local_weights)
        info = {
            "eta_grad": jnp.asarray(cfg.eta_grad),
            "grad_clip": jnp.asarray(cfg.grad_clip),
            "free_energy": last_free_energy[selected_center],
            "minimum_candidate_cost": jnp.min(selected_costs),
            "candidate_cost_std": jnp.nanstd(
                jnp.where(jnp.isfinite(selected_costs), selected_costs, jnp.nan)
            ),
            "local_ess": selected_local_ess,
            "selected_center_local_ess": selected_local_ess,
            "outer_ess": effective_sample_size(outer_weights),
            "outer_ess_by_cycle": outer_ess_history,
            "finite_cost_fraction": jnp.mean(jnp.isfinite(last_costs)),
            "selected_finite_cost_fraction": jnp.mean(jnp.isfinite(selected_costs)),
            "valid": valid,
            "selected_particle": selected_center,
            "selected_center_index": selected_center,
            "selected_center_sequence": last_evaluated_centers[selected_center],
            "output_sequence": output_sequence,
            "prediction_sequences": last_evaluated_centers,
            "prediction_scores": last_free_energy,
            "prediction_local_sequences": last_samples[selected_center],
            "prediction_local_scores": selected_costs,
            "best_local_cost_center": best_local_cost_center,
            "j_f_selected_disagreement": selected_center != best_local_cost_center,
            "free_energy_gap": free_energy_gap,
            "center_covariance_trace_at_step_start": cov_at_start,
            "center_covariance_trace_at_final_evaluation": cov_at_end,
            "warm_start_covariance_trace": warm_start_covariance_trace,
            "warm_start_rms_pairwise_distance": warm_start_rms_pairwise_distance,
            "rejuvenation_noise_rms": rejuvenation_noise_rms,
            "refreshed_particle_fraction": refreshed_fraction,
            "rollout_count": jnp.asarray(
                cfg.cycles * cfg.particles * cfg.local_samples
            ),
            "mean_free_energy_by_cycle": mean_free_energy_history,
            "min_free_energy_by_cycle": min_free_energy_history,
            "mean_local_mppi_displacement_by_cycle": (
                mean_local_mppi_displacement_history
            ),
            "mean_consensus_distance_by_cycle": mean_consensus_distance_history,
            "mean_gradient_step_norm_by_cycle": mean_gradient_step_norm_history,
            "max_gradient_step_norm_by_cycle": max_gradient_step_norm_history,
            "gradient_clip_fraction_by_cycle": gradient_clip_fraction_history,
        }
        if cfg.record_history:
            info["selected_local_ess_fraction"] = (
                local_ess_history[-1, selected_center] / cfg.local_samples
            )
            info["center_covariance_trace_by_cycle"] = covariance_trace_history
            info["center_rms_pairwise_distance_by_cycle"] = rms_pairwise_distance_history
            info["split_half_ranking_agreement_by_cycle"] = split_agreement_history
            info["j_f_disagreement_by_cycle"] = j_f_disagreement_history
            info["consensus_distance_by_cycle"] = consensus_distance_history
            info["local_ess_by_cycle"] = local_ess_history
            info["feasible_local_sample_count_by_cycle"] = feasible_count_history
            info["free_energy_by_cycle"] = free_energy_history
        return next_key, next_state, output_sequence[0], info

    return proposed_step
