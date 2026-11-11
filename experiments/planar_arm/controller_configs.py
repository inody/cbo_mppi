"""Controller config presets for the planar arm."""

from __future__ import annotations

from cbo_mppi.config import CBOConfig, MPPIConfig, ProposedConfig


def mppi_preset(*, budget, tau=10.0, noise_sigma=1.0, noise_correlation=0.9):
    return MPPIConfig(
        rollouts=budget, tau=tau, noise_sigma=noise_sigma,
        noise_correlation=noise_correlation, clip_rollout_inputs=True,
    )


def cbo_mpc_preset(*, budget, cycles=8, alpha=1.0, lambda_=1.0,
                   sigma=0.1, delta_s=0.1, init_sigma=1.0,
                   init_noise_correlation=0.9, warm_start_sigma=1.0):
    return CBOConfig(
        particles=budget // cycles, cycles=cycles, alpha=alpha,
        lambda_=lambda_, sigma=sigma, delta_s=delta_s,
        init_sigma=init_sigma, init_noise_correlation=init_noise_correlation,
        warm_start_mode="shift_jitter", warm_start_jitter_kind="smooth",
        warm_start_sigma=warm_start_sigma,
        record_history=True, clip_rollout_inputs=True,
    )


def proposed_preset(*, budget, cycles=4, local_samples=16, tau=10.0,
                    local_noise_sigma=0.5, local_noise_correlation=0.9,
                    alpha=1.0, lambda_=1.0, sigma=0.1, delta_s=0.1,
                    init_sigma=1.0, init_noise_correlation=0.9,
                    eta_grad=1.0, warm_start_sigma=1.0):
    return ProposedConfig(
        particles=budget // (cycles * local_samples),
        cycles=cycles, local_samples=local_samples, tau=tau,
        local_noise_sigma=local_noise_sigma,
        local_noise_correlation=local_noise_correlation,
        alpha=alpha, lambda_=lambda_, sigma=sigma, delta_s=delta_s,
        init_sigma=init_sigma, init_noise_correlation=init_noise_correlation,
        output_rule="weighted", shared_local_noise=False,
        warm_start_mode="shift_jitter", warm_start_jitter_kind="smooth",
        warm_start_sigma=warm_start_sigma,
        record_history=True, eta_grad=eta_grad, grad_clip=0.0,
        clip_rollout_inputs=True,
    )
