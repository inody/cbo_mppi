"""Shared first-hitting semantics for episodic closed-loop experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

TerminationReason = Literal["success", "timeout", "collision"]
StatePredicate = Callable[[np.ndarray], bool]
ClearanceFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class EpisodeTermination:
    reason: TerminationReason
    step: int
    time: float
    success: bool
    collision: bool
    minimum_clearance: float | None


class EpisodeMonitor:
    """Observe states after updates and stop at collision or first Success.

    Collision is checked before Success at each state. Contact is feasible:
    only strictly negative clearance is a collision.
    """

    def __init__(
        self,
        *,
        dt: float,
        success: StatePredicate,
        clearance: ClearanceFunction | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.success_predicate = success
        self.clearance_function = clearance
        self.minimum_clearance = np.inf if clearance is not None else None
        self.termination: EpisodeTermination | None = None
        self.last_step = -1

    def observe(self, state: np.ndarray, *, step: int) -> EpisodeTermination | None:
        if self.termination is not None:
            raise RuntimeError("cannot observe a terminated episode")
        if step != self.last_step + 1:
            raise ValueError(f"expected state step {self.last_step + 1}, got {step}")
        self.last_step = step
        state = np.asarray(state)
        if not np.all(np.isfinite(state)):
            raise ValueError(f"nonfinite state at step {step}")
        clearance = None
        if self.clearance_function is not None:
            clearance = float(self.clearance_function(state))
            if not np.isfinite(clearance):
                raise ValueError(f"nonfinite clearance at step {step}")
            self.minimum_clearance = min(float(self.minimum_clearance), clearance)
            if clearance < 0.0:
                self.termination = self._make("collision", step)
                return self.termination
        if bool(self.success_predicate(state)):
            self.termination = self._make("success", step)
        return self.termination

    def timeout(self, *, step: int) -> EpisodeTermination:
        if self.termination is not None:
            return self.termination
        if step != self.last_step:
            raise ValueError(f"timeout step {step} differs from last observed {self.last_step}")
        self.termination = self._make("timeout", step)
        return self.termination

    def _make(self, reason: TerminationReason, step: int) -> EpisodeTermination:
        return EpisodeTermination(
            reason=reason,
            step=int(step),
            time=float(step * self.dt),
            success=reason == "success",
            collision=reason == "collision",
            minimum_clearance=(
                None if self.minimum_clearance is None else float(self.minimum_clearance)
            ),
        )


def termination_summary(
    termination: EpisodeTermination,
    *,
    rollout_budget_per_step: int,
) -> dict[str, object]:
    if rollout_budget_per_step <= 0:
        raise ValueError("rollout budget must be positive")
    return {
        "episodic": True,
        "termination_reason": termination.reason,
        "success": termination.success,
        "collision": termination.collision,
        "success_step": termination.step if termination.success else None,
        "success_time": termination.time if termination.success else None,
        "episode_steps": termination.step,
        "episode_time": termination.time,
        "minimum_clearance_episode": termination.minimum_clearance,
        "rollouts_per_step": int(rollout_budget_per_step),
        "total_rollouts": int(rollout_budget_per_step * termination.step),
    }


def first_hitting_prefix(
    states: np.ndarray,
    controls: np.ndarray,
    *,
    dt: float,
    success: StatePredicate,
    clearance: ClearanceFunction | None = None,
) -> tuple[np.ndarray, np.ndarray, EpisodeTermination]:
    """Apply the shared semantics to a full artifact and return its prefix."""
    states = np.asarray(states)
    controls = np.asarray(controls)
    if states.ndim != 2 or controls.ndim != 2:
        raise ValueError("states and controls must be rank-two")
    if states.shape[0] != controls.shape[0] + 1:
        raise ValueError("states must contain exactly one more sample than controls")
    monitor = EpisodeMonitor(dt=dt, success=success, clearance=clearance)
    termination = monitor.observe(states[0], step=0)
    if termination is not None:
        return states[:1].copy(), controls[:0].copy(), termination
    for step in range(1, states.shape[0]):
        termination = monitor.observe(states[step], step=step)
        if termination is not None:
            return states[: step + 1].copy(), controls[:step].copy(), termination
    termination = monitor.timeout(step=controls.shape[0])
    return states.copy(), controls.copy(), termination
