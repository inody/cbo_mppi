"""Controller config presets for the pendulum swing-up closed-loop comparison.

All three controllers share the scalar-control pendulum rollout and operate on
control sequences of shape (horizon, 1). Budgets are matched so that
rollouts-per-step are approximately equal across methods.
"""

from __future__ import annotations

from cbo_mppi.config import CBOConfig, MPPIConfig, ProposedConfig


def mppi_preset(
    *,
    budget: int,
    tau: float = 10.0,
    noise_sigma: float = 1.0,
    noise_correlation: float = 0.9,
) -> MPPIConfig:
    return MPPIConfig(
        rollouts=budget,
        tau=tau,
        noise_sigma=noise_sigma,
        noise_correlation=noise_correlation,
        clip_rollout_inputs=True,
    )


def cbo_mpc_preset(
    *,
    budget: int,
    cycles: int = 8,
    alpha: float = 1.0,
    lambda_: float = 1.0,
    sigma: float = 0.1,
    delta_s: float = 0.1,
    init_sigma: float = 1.0,
    init_noise_correlation: float = 0.9,
    warm_start_sigma: float = 1.0,
) -> CBOConfig:
    if budget % cycles:
        raise ValueError("budget must be divisible by cycles")
    return CBOConfig(
        particles=budget // cycles,
        cycles=cycles,
        alpha=alpha,
        lambda_=lambda_,
        sigma=sigma,
        delta_s=delta_s,
        init_sigma=init_sigma,
        init_noise_correlation=init_noise_correlation,
        warm_start_mode="shift_jitter",
        warm_start_jitter_kind="smooth",
        warm_start_sigma=warm_start_sigma,
        warm_start_refresh_rate=0.0,
        warm_start_min_cov_trace=0.0,
        warm_start_apply_to="all_particles",
        record_history=True,
        clip_rollout_inputs=True,
    )


def proposed_preset(
    *,
    budget: int,
    cycles: int = 4,
    local_samples: int = 16,
    tau: float = 10.0,
    local_noise_sigma: float = 0.25,
    local_noise_correlation: float = 0.9,
    alpha: float = 1.0,
    lambda_: float = 1.0,
    sigma: float = 0.1,
    delta_s: float = 0.1,
    init_sigma: float = 1.0,
    init_noise_correlation: float = 0.9,
    eta_grad: float = 1.0,
    grad_clip: float = 0.0,
    output_rule: str = "weighted",
    warm_start_sigma: float = 1.0,
) -> ProposedConfig:
    divisor = cycles * local_samples
    if budget % divisor:
        raise ValueError("budget must be divisible by cycles*local_samples")
    return ProposedConfig(
        particles=budget // divisor,
        cycles=cycles,
        local_samples=local_samples,
        tau=tau,
        local_noise_sigma=local_noise_sigma,
        local_noise_correlation=local_noise_correlation,
        alpha=alpha,
        lambda_=lambda_,
        sigma=sigma,
        delta_s=delta_s,
        init_sigma=init_sigma,
        init_noise_correlation=init_noise_correlation,
        output_rule=output_rule,
        shared_local_noise=False,
        warm_start_mode="shift_jitter",
        warm_start_jitter_kind="smooth",
        warm_start_sigma=warm_start_sigma,
        warm_start_refresh_rate=0.0,
        warm_start_min_cov_trace=0.0,
        warm_start_apply_to="all_particles",
        record_history=True,
        eta_grad=eta_grad,
        grad_clip=grad_clip,
        clip_rollout_inputs=True,
    )
