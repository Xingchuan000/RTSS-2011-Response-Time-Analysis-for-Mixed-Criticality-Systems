"""Mutable quality state owned by the q-AMC runtime engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import QAmcProfileBundle, QAmcTaskProfile


@dataclass(frozen=True, slots=True)
class QAmcQualitySnapshot:
    level_by_task: dict[str, int]


@dataclass(frozen=True, slots=True)
class QAmcOverrunDecision:
    task_name: str
    old_level: int
    new_level: int | None
    action: Literal["DEGRADE_NEXT_RELEASE", "ENTER_TERMINAL_HI"]


@dataclass(slots=True)
class QAmcRuntimeController:
    _profiles: dict[str, QAmcTaskProfile]
    _current_level_by_task: dict[str, int]

    @classmethod
    def from_profile_bundle(cls, bundle: QAmcProfileBundle) -> "QAmcRuntimeController":
        return cls(
            _profiles=dict(bundle.profiles),
            _current_level_by_task={
                name: profile.initial_runtime_level for name, profile in bundle.profiles.items()
            },
        )

    def current_level(self, task_name: str) -> int:
        try:
            return self._current_level_by_task[task_name]
        except KeyError as exc:
            raise KeyError(f"QAMC_UNKNOWN_TASK:{task_name}") from exc

    def snapshot(self) -> QAmcQualitySnapshot:
        return QAmcQualitySnapshot(level_by_task=dict(self._current_level_by_task))

    def profile(self, task_name: str) -> QAmcTaskProfile:
        try:
            return self._profiles[task_name]
        except KeyError as exc:
            raise KeyError(f"QAMC_UNKNOWN_TASK:{task_name}") from exc

    def plan_lo_overrun(self, task_name: str) -> QAmcOverrunDecision:
        profile = self.profile(task_name)
        old = self.current_level(task_name)
        if profile.can_degrade(old):
            return QAmcOverrunDecision(task_name, old, profile.next_lower_level(old), "DEGRADE_NEXT_RELEASE")
        return QAmcOverrunDecision(task_name, old, None, "ENTER_TERMINAL_HI")

    def commit(self, decision: QAmcOverrunDecision) -> None:
        if decision.action == "DEGRADE_NEXT_RELEASE":
            assert decision.new_level is not None
            if self.current_level(decision.task_name) != decision.old_level:
                raise RuntimeError("QAMC_QUALITY_STATE_CHANGED_BEFORE_COMMIT")
            self._current_level_by_task[decision.task_name] = decision.new_level

    def on_lo_overrun(self, task_name: str) -> QAmcOverrunDecision:
        decision = self.plan_lo_overrun(task_name)
        self.commit(decision)
        return decision


__all__ = ["QAmcOverrunDecision", "QAmcQualitySnapshot", "QAmcRuntimeController"]
