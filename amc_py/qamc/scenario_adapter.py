"""Adapters for inspecting q-AMC demand without changing a source scenario."""

from __future__ import annotations

from dataclasses import dataclass

from amc_py.models import Task
from amc_py.runtime_scenarios import ExecutionScenario

from .demand import QAmcDemandSnapshot, map_full_cost_to_quality_cost
from .models import QAmcTaskProfile


@dataclass(frozen=True, slots=True)
class QAmcScenarioAdapter:
    """Read-only view that maps full-quality samples at release time."""

    base_scenario: ExecutionScenario
    profile: QAmcTaskProfile
    runtime_level: int

    def actual_cost_for(self, task: Task, release_index: int) -> int:
        if task.name != self.profile.task_name:
            raise ValueError("QAMC_SCENARIO_TASK_PROFILE_MISMATCH")
        return map_full_cost_to_quality_cost(
            full_quality_actual_cost=self.base_scenario.actual_cost_for(task, release_index),
            profile=self.profile,
            runtime_level=self.runtime_level,
        ).quality_specific_actual_cost

    def demand_snapshot_for(self, task: Task, release_index: int) -> QAmcDemandSnapshot:
        if task.name != self.profile.task_name:
            raise ValueError("QAMC_SCENARIO_TASK_PROFILE_MISMATCH")
        return map_full_cost_to_quality_cost(
            full_quality_actual_cost=self.base_scenario.actual_cost_for(task, release_index),
            profile=self.profile,
            runtime_level=self.runtime_level,
        )


__all__ = ["QAmcScenarioAdapter"]
