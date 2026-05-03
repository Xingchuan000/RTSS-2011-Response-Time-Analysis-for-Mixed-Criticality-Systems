"""运行时预算候选安全过滤器。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


@dataclass(frozen=True, slots=True)
class RuntimeSafetyReport:
    """安全检查报告。"""

    accepted: bool
    reason: str
    checked_tasks: tuple[str, ...]


@dataclass(slots=True)
class RuntimeBudgetSafetyChecker:
    """AMC-rtb 运行时预算的保守可行性检查器。"""

    ordered_tasks: Sequence[Task]
    design_r_lo: Mapping[str, int]
    check_lo_tasks: bool = True

    def __post_init__(self) -> None:
        """校验 `design_r_lo` 是否完整，禁止静默缺失。"""

        for task in self.ordered_tasks:
            if task.criticality is Criticality.HI and task.name not in self.design_r_lo:
                raise ValueError(f"design_r_lo 缺少 HI 任务 {task.name} 的设计时 R_LO")

    def validate_candidate(self, candidate_budgets: Mapping[str, int]) -> RuntimeSafetyReport:
        """校验完整 budget 向量是否满足保守约束。"""

        task_names = tuple(task.name for task in self.ordered_tasks)
        for name in task_names:
            if name not in candidate_budgets:
                return RuntimeSafetyReport(False, f"missing_budget:{name}", task_names)
            if candidate_budgets[name] <= 0:
                return RuntimeSafetyReport(False, f"non_positive_budget:{name}", task_names)

        for idx, task_i in enumerate(self.ordered_tasks):
            hp = self.ordered_tasks[:idx]
            if task_i.criticality is Criticality.HI:
                r_lo_i = self.design_r_lo[task_i.name]

                lhs_lo = candidate_budgets[task_i.name]
                for task_j in hp:
                    lhs_lo += math.ceil(r_lo_i / task_j.period) * candidate_budgets[task_j.name]
                if lhs_lo > r_lo_i:
                    return RuntimeSafetyReport(False, f"hi_lo_mode_violation:{task_i.name}", task_names)

                lhs_switch = task_i.c_hi
                for task_j in hp:
                    if task_j.criticality is Criticality.LO:
                        lhs_switch += math.ceil(r_lo_i / task_j.period) * candidate_budgets[task_j.name]
                    else:
                        lhs_switch += math.ceil(task_i.deadline / task_j.period) * task_j.c_hi
                if lhs_switch > task_i.deadline:
                    return RuntimeSafetyReport(False, f"hi_mode_switch_violation:{task_i.name}", task_names)

            if self.check_lo_tasks and task_i.criticality is Criticality.LO:
                lhs_lo_task = candidate_budgets[task_i.name]
                for task_j in hp:
                    lhs_lo_task += math.ceil(task_i.deadline / task_j.period) * candidate_budgets[task_j.name]
                if lhs_lo_task > task_i.deadline:
                    return RuntimeSafetyReport(False, f"lo_mode_violation:{task_i.name}", task_names)

        return RuntimeSafetyReport(True, "accepted", task_names)


def merge_budget_candidate(current: BudgetState, updates: Mapping[str, int]) -> dict[str, int]:
    """将局部 updates 与当前完整预算向量合并。"""

    merged = dict(current.budgets)
    merged.update(updates)
    return merged


def has_effective_budget_updates(current: BudgetState, updates: Mapping[str, int]) -> bool:
    """判断候选更新是否真的改变了所有被选中的任务预算。"""

    for task_name, updated_budget in updates.items():
        if task_name not in current.budgets:
            return False
        # 若预算被上下界裁剪后没有发生变化，则该动作在当前状态下视为无效。
        if current.budgets[task_name] == updated_budget:
            return False
    return True
