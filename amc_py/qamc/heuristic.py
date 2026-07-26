"""Quality-blind budget-pressure heuristic for q-AMC comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from amc_py.rl.actions import BudgetAction
from amc_py.rl.types import AgentObservation


@dataclass(frozen=True, slots=True)
class QAmcBudgetPressureHeuristic:
    increase_threshold: float = 0.90
    decrease_threshold: float = 0.60

    def select_action_id(
        self,
        observation: AgentObservation,
        actions: Sequence[BudgetAction],
        valid_mask: Sequence[bool],
    ) -> int | None:
        pressure = {
            name: float(observation.raw_recent_costs.get(name, 0))
            / max(1.0, float(budget))
            for name, budget in observation.raw_budgets.items()
        }
        if not pressure:
            return None
        highest = max(pressure, key=pressure.get)
        lowest = min(pressure, key=pressure.get)
        max_pressure = pressure[highest]
        min_pressure = pressure[lowest]
        candidates: list[tuple[float, int]] = []
        explicit_noop: int | None = None
        for action, valid in zip(actions, valid_mask, strict=True):
            if not valid:
                continue
            if action.is_noop:
                explicit_noop = action.action_id
                continue
            score = float("-inf")
            if action.is_residual_ranked:
                action_type = str(action.residual_action_type or "")
                rank_bonus = -0.001 * float(action.residual_rank or 0)
                if max_pressure >= self.increase_threshold and "increase" in action_type:
                    score = 2.0 + max_pressure + rank_bonus
                elif min_pressure <= self.decrease_threshold and "decrease" in action_type:
                    score = 1.0 - min_pressure + rank_bonus
            else:
                if (
                    max_pressure >= self.increase_threshold
                    and action.increase_task == highest
                ):
                    score = 2.0 + max_pressure
                    score -= sum(pressure.get(name, 1.0) for name in action.decrease_tasks)
                elif (
                    action.increase_task is None
                    and action.decrease_tasks
                    and min_pressure <= self.decrease_threshold
                    and lowest in action.decrease_tasks
                ):
                    score = 1.0 - min_pressure
            if score != float("-inf"):
                candidates.append((score, action.action_id))
        if candidates:
            return max(candidates, key=lambda item: (item[0], -item[1]))[1]
        return explicit_noop


__all__ = ["QAmcBudgetPressureHeuristic"]
