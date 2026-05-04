"""离散预算动作空间与候选预算更新构建。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from collections.abc import Sequence

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


@dataclass(frozen=True, slots=True)
class BudgetAction:
    """单个离散预算动作。"""

    action_id: int
    increase_task: str | None
    decrease_tasks: tuple[str, ...]
    increase_ratio: float = 0.10
    decrease_ratio: float = 0.05
    action_space_type: str = "triple"
    is_noop: bool = False


def build_budget_action_space(
    ordered_tasks: Sequence[Task],
    *,
    action_space: str = "triple",
    budget_increase_ratio: float = 0.10,
    budget_decrease_ratio: float = 0.05,
    include_explicit_noop: bool = False,
) -> tuple[BudgetAction, ...]:
    """构建确定性动作空间。

    - `triple`: 增加 1 个任务预算，降低 2 个任务预算；
    - `pair`: 增加 1 个任务预算，降低 1 个任务预算；
    - `single`: 只增加或只降低 1 个任务预算；
    - 可选 `include_explicit_noop` 追加显式 NoOp 动作。
    """

    if action_space not in {"triple", "pair", "single"}:
        raise ValueError(f"不支持的 action_space: {action_space}")
    if budget_increase_ratio <= 0.0:
        raise ValueError("budget_increase_ratio 必须为正数")
    if budget_decrease_ratio <= 0.0 or budget_decrease_ratio >= 1.0:
        raise ValueError("budget_decrease_ratio 必须在 (0, 1) 区间内")

    names = [task.name for task in ordered_tasks]
    actions: list[BudgetAction] = []
    action_id = 0

    if action_space == "triple":
        for increase_name in names:
            candidates = [name for name in names if name != increase_name]
            for dec_a, dec_b in combinations(candidates, 2):
                actions.append(
                    BudgetAction(
                        action_id=action_id,
                        increase_task=increase_name,
                        decrease_tasks=(dec_a, dec_b),
                        increase_ratio=budget_increase_ratio,
                        decrease_ratio=budget_decrease_ratio,
                        action_space_type=action_space,
                    )
                )
                action_id += 1
    elif action_space == "pair":
        for increase_name, decrease_name in permutations(names, 2):
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_name,
                    decrease_tasks=(decrease_name,),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1
    else:
        for increase_name in names:
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_name,
                    decrease_tasks=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1
        for decrease_name in names:
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=None,
                    decrease_tasks=(decrease_name,),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1

    if include_explicit_noop:
        actions.append(
            BudgetAction(
                action_id=action_id,
                increase_task=None,
                decrease_tasks=(),
                increase_ratio=budget_increase_ratio,
                decrease_ratio=budget_decrease_ratio,
                action_space_type=action_space,
                is_noop=True,
            )
        )

    return tuple(actions)


def apply_budget_action_candidate(
    *,
    action: BudgetAction,
    budget_state: BudgetState,
    ordered_tasks: Sequence[Task],
) -> dict[str, int]:
    """将动作转换为候选更新，不直接改写原 BudgetState。"""

    if action.is_noop:
        return {}

    task_map = {task.name: task for task in ordered_tasks}
    candidate: dict[str, int] = {}

    if action.increase_task is not None:
        old_inc = budget_state.budgets[action.increase_task]
        inc_task = task_map[action.increase_task]
        inc_value = math.ceil(old_inc * (1.0 + action.increase_ratio))

        if inc_task.criticality is Criticality.HI:
            upper_bound = inc_task.c_hi if inc_task.c_hi > 0 else inc_task.deadline
            inc_value = min(inc_value, upper_bound)
        else:
            inc_value = min(inc_value, inc_task.deadline)

        candidate[action.increase_task] = max(1, inc_value)

    for dec_name in action.decrease_tasks:
        old_dec = budget_state.budgets[dec_name]
        dec_value = math.floor(old_dec * (1.0 - action.decrease_ratio))
        candidate[dec_name] = max(1, dec_value)

    return candidate
