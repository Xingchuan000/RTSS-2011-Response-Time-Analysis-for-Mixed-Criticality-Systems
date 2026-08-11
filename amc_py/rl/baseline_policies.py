"""Mask-aware non-learning baseline selectors for formal10 C-AMC-sem."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import random

from amc_py.models import Criticality, Task
from amc_py.rl.actions import BudgetAction
from amc_py.rl.types import AgentObservation


@dataclass(slots=True)
class RandomValidSelector:
    """Select uniformly from the actions admitted by the current mask."""

    seed: int
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def select_action_id(
        self,
        *,
        observation: AgentObservation,
        valid_action_mask: Sequence[bool],
        actions: Sequence[BudgetAction],
    ) -> int | None:
        del observation
        valid_ids = [
            action.action_id
            for action, valid in zip(actions, valid_action_mask, strict=True)
            if bool(valid)
        ]
        if not valid_ids:
            return None
        return int(self._rng.choice(valid_ids))


@dataclass(frozen=True, slots=True)
class PressureThresholdValidSelector:
    """Select legal single-task LO budget actions from budget pressure."""

    ordered_tasks: tuple[Task, ...]
    u_low: float
    u_high: float

    def __post_init__(self) -> None:
        if not self.u_low < self.u_high:
            raise ValueError("u_low must be < u_high")

    def _budget_util(self, observation: AgentObservation) -> float:
        return sum(
            float(observation.raw_budgets[task.name]) / float(task.period)
            for task in self.ordered_tasks
        )

    def _pressure(self, observation: AgentObservation, task: Task) -> float:
        budget = max(1.0, float(observation.raw_budgets[task.name]))
        recent = float(observation.raw_recent_costs.get(task.name, 0))
        return recent / budget

    def select_action_id(
        self,
        *,
        observation: AgentObservation,
        valid_action_mask: Sequence[bool],
        actions: Sequence[BudgetAction],
    ) -> int | None:
        util = self._budget_util(observation)
        lo_tasks = [
            task for task in self.ordered_tasks if task.criticality is Criticality.LO
        ]
        pressure = {
            task.name: self._pressure(observation, task) for task in lo_tasks
        }

        if util <= self.u_low:
            ordered_names = [
                task.name
                for task in sorted(
                    lo_tasks,
                    key=lambda task: (
                        -pressure[task.name],
                        self.ordered_tasks.index(task),
                    ),
                )
            ]
            for task_name in ordered_names:
                for action, valid in zip(actions, valid_action_mask, strict=True):
                    if (
                        bool(valid)
                        and action.increase_task == task_name
                        and not action.decrease_tasks
                        and not action.is_noop
                    ):
                        return int(action.action_id)
            return None

        if util >= self.u_high:
            ordered_names = [
                task.name
                for task in sorted(
                    lo_tasks,
                    key=lambda task: (
                        pressure[task.name],
                        self.ordered_tasks.index(task),
                    ),
                )
            ]
            for task_name in ordered_names:
                for action, valid in zip(actions, valid_action_mask, strict=True):
                    if (
                        bool(valid)
                        and action.increase_task is None
                        and tuple(action.decrease_tasks) == (task_name,)
                        and not action.is_noop
                    ):
                        return int(action.action_id)
            return None

        return None
