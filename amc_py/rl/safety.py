"""运行时预算候选安全过滤器。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


@dataclass(frozen=True, slots=True)
class RuntimeSafetyReport:
    """安全检查报告。"""

    accepted: bool
    reason: str
    checked_tasks: tuple[str, ...]
    diagnostics: tuple[dict[str, str | int | float], ...] = ()


@dataclass(slots=True)
class RuntimeBudgetSafetyChecker:
    """AMC-rtb 运行时预算的保守可行性检查器。"""

    ordered_tasks: Sequence[Task]
    design_r_lo: Mapping[str, int]
    check_lo_tasks: bool = True
    _task_names: tuple[str, ...] = field(init=False, repr=False)
    _task_index: dict[str, int] = field(init=False, repr=False)
    _constraint_rows: list[tuple[str, str, float, np.ndarray]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """校验 `design_r_lo` 是否完整，禁止静默缺失。"""

        for task in self.ordered_tasks:
            if task.criticality is Criticality.HI and task.name not in self.design_r_lo:
                raise ValueError(f"design_r_lo 缺少 HI 任务 {task.name} 的设计时 R_LO")
        # 预构建线性约束系数，供增量 safety check 与批量 mask 复用。
        self._task_names = tuple(task.name for task in self.ordered_tasks)
        self._task_index = {name: idx for idx, name in enumerate(self._task_names)}
        self._constraint_rows = self._build_constraint_rows()

    def _build_constraint_rows(self) -> list[tuple[str, str, float, np.ndarray]]:
        """构建线性约束，统一成 `A @ B <= bound` 的形式。"""

        rows: list[tuple[str, str, float, np.ndarray]] = []
        n_tasks = len(self.ordered_tasks)
        for idx, task_i in enumerate(self.ordered_tasks):
            hp = self.ordered_tasks[:idx]
            if task_i.criticality is Criticality.HI:
                r_lo_i = self.design_r_lo[task_i.name]
                coeff_lo = np.zeros(n_tasks, dtype=np.float64)
                coeff_lo[idx] = 1.0
                for task_j in hp:
                    jdx = self._task_index[task_j.name]
                    coeff_lo[jdx] += float(math.ceil(r_lo_i / task_j.period))
                rows.append((task_i.name, "hi_lo_mode", float(r_lo_i), coeff_lo))

                coeff_switch = np.zeros(n_tasks, dtype=np.float64)
                rhs_switch = float(task_i.deadline - task_i.c_hi)
                for task_j in hp:
                    if task_j.criticality is Criticality.LO:
                        jdx = self._task_index[task_j.name]
                        coeff_switch[jdx] += float(math.ceil(r_lo_i / task_j.period))
                    else:
                        rhs_switch -= float(math.ceil(task_i.deadline / task_j.period) * task_j.c_hi)
                rows.append((task_i.name, "hi_mode_switch", rhs_switch, coeff_switch))

            if self.check_lo_tasks and task_i.criticality is Criticality.LO:
                coeff_lo_task = np.zeros(n_tasks, dtype=np.float64)
                coeff_lo_task[idx] = 1.0
                for task_j in hp:
                    jdx = self._task_index[task_j.name]
                    coeff_lo_task[jdx] += float(math.ceil(task_i.deadline / task_j.period))
                rows.append((task_i.name, "lo_deadline_bound", float(task_i.deadline), coeff_lo_task))
        return rows

    def build_linear_constraints(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 `A, bound`，满足 `A @ budgets <= bound`。"""

        if not self._constraint_rows:
            return np.zeros((0, len(self.ordered_tasks)), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        matrix_a = np.stack([row[3] for row in self._constraint_rows], axis=0)
        bounds = np.array([row[2] for row in self._constraint_rows], dtype=np.float64)
        return matrix_a, bounds

    def validate_candidate(self, candidate_budgets: Mapping[str, int]) -> RuntimeSafetyReport:
        """校验完整 budget 向量是否满足保守约束。"""

        task_names = self._task_names
        diagnostics: list[dict[str, str | int | float]] = []
        for name in task_names:
            if name not in candidate_budgets:
                return RuntimeSafetyReport(False, f"missing_budget:{name}", task_names, tuple(diagnostics))
            if candidate_budgets[name] <= 0:
                return RuntimeSafetyReport(False, f"non_positive_budget:{name}", task_names, tuple(diagnostics))

        budget_array = np.array([float(candidate_budgets[name]) for name in task_names], dtype=np.float64)
        for task_name, constraint_name, rhs, coeff in self._constraint_rows:
            lhs = float(np.dot(coeff, budget_array))
            diagnostics.append(
                {
                    "task": task_name,
                    "constraint": constraint_name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "slack": rhs - lhs,
                }
            )
            if lhs > rhs:
                if constraint_name == "hi_lo_mode":
                    reason = f"hi_lo_mode_violation:{task_name}"
                elif constraint_name == "hi_mode_switch":
                    reason = f"hi_mode_switch_violation:{task_name}"
                else:
                    reason = f"lo_mode_violation:{task_name}"
                return RuntimeSafetyReport(False, reason, task_names, tuple(diagnostics))

        return RuntimeSafetyReport(True, "accepted", task_names, tuple(diagnostics))


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
