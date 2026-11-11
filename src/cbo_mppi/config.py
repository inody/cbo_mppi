from __future__ import annotations

from typing import NamedTuple


class MPPIConfig(NamedTuple):
    rollouts: int
    tau: float
    noise_sigma: float
    noise_correlation: float
    clip_rollout_inputs: bool = False


class IteratedMPPIConfig(NamedTuple):
    rollouts_per_round: int
    rounds: int
    tau: float
    noise_sigma: float
    noise_correlation: float
    clip_rollout_inputs: bool = False


class CBOConfig(NamedTuple):
    particles: int
    cycles: int
    alpha: float
    lambda_: float
    sigma: float
    delta_s: float
    init_sigma: float
    init_noise_correlation: float
    warm_start_mode: str = "shift"
    warm_start_jitter_kind: str = "smooth"
    warm_start_sigma: float = 0.0
    warm_start_refresh_rate: float = 0.0
    warm_start_min_cov_trace: float = 0.0
    warm_start_apply_to: str = "all_particles"
    record_history: bool = True
    clip_rollout_inputs: bool = False


class ProposedConfig(NamedTuple):
    particles: int
    cycles: int
    local_samples: int
    tau: float
    local_noise_sigma: float
    local_noise_correlation: float
    alpha: float
    lambda_: float
    sigma: float
    delta_s: float
    init_sigma: float
    init_noise_correlation: float
    output_rule: str = "weighted"
    shared_local_noise: bool = False
    warm_start_mode: str = "shift"
    warm_start_jitter_kind: str = "smooth"
    warm_start_sigma: float = 0.0
    warm_start_refresh_rate: float = 0.0
    warm_start_min_cov_trace: float = 0.0
    warm_start_apply_to: str = "all_particles"
    record_history: bool = True
    eta_grad: float = 0.0
    grad_clip: float = 0.0
    clip_rollout_inputs: bool = False
    center_update_rule: str = "cbo"


def validate_isotropic_cbo(dimension: int, lambda_: float, sigma: float) -> None:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if lambda_ <= 0 or sigma < 0:
        raise ValueError("lambda must be positive and sigma nonnegative")
    if 2.0 * lambda_ <= dimension * sigma**2:
        bound = (2.0 * lambda_ / dimension) ** 0.5
        raise ValueError(
            "isotropic CBO condition 2*lambda > D*sigma^2 is violated; "
            f"require sigma < {bound:.6g} for D={dimension}"
        )
