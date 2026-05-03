"""离散预算动作空间与候选预算更新构建。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from collections.abc import Sequence

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


@dataclass(frozen=True, slots=True)
class BudgetAction:
    """单个离散预算动作。"""

    action_id: int
    increase_task: str
    decrease_tasks: tuple[str, str]
    increase_ratio: float = 1.10
    decrease_ratio: float = 0.95


def build_budget_action_space(ordered_tasks: Sequence[Task]) -> tuple[BudgetAction, ...]:
    """构建确定性动作空间，动作数为 n(n-1)(n-2)/2。"""

    names = [task.name for task in ordered_tasks]
    actions: list[BudgetAction] = []
    action_id = 0

    for increase_name in names:
        candidates = [name for name in names if name != increase_name]
        for dec_a, dec_b in combinations(candidates, 2):
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_name,
                    decrease_tasks=(dec_a, dec_b),
                )
            )
            action_id += 1

    return tuple(actions)


def apply_budget_action_candidate(
    *,
    action: BudgetAction,
    budget_state: BudgetState,
    ordered_tasks: Sequence[Task],
) -> dict[str, int]:
    """将动作转换为候选更新，不直接改写原 BudgetState。"""

    task_map = {task.name: task for task in ordered_tasks}
    candidate: dict[str, int] = {}

    old_inc = budget_state.budgets[action.increase_task]
    inc_task = task_map[action.increase_task]
    inc_value = math.ceil(old_inc * action.increase_ratio)

    if inc_task.criticality is Criticality.HI:
        upper_bound = inc_task.c_hi if inc_task.c_hi > 0 else inc_task.deadline
        inc_value = min(inc_value, upper_bound)
    else:
        inc_value = min(inc_value, inc_task.deadline)

    candidate[action.increase_task] = max(1, inc_value)

    for dec_name in action.decrease_tasks:
        old_dec = budget_state.budgets[dec_name]
        dec_value = math.floor(old_dec * action.decrease_ratio)
        candidate[dec_name] = max(1, dec_value)

    return candidate
