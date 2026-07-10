"""运行时预算候选安全过滤器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


def build_effective_safety_budget_vector(
    ordered_tasks: Sequence[Task],
    candidate_budgets: Mapping[str, int],
    active_release_budget_max: Mapping[str, int] | None = None,
    lo_overrun_guard_units: int = 0,
) -> dict[str, int]:
    """统一构造 carry-over envelope 与 LO guard 后的安全检查向量。"""
    active_release_budget_max = active_release_budget_max or {}
    return {
        task.name: max(int(candidate_budgets[task.name]), int(active_release_budget_max.get(task.name, 0)))
        + (lo_overrun_guard_units if task.criticality is Criticality.LO else 0)
        for task in ordered_tasks
    }


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
    _constraint_rows: list[tuple[str, str, int, np.ndarray]] = field(init=False, repr=False)

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

        rows: list[tuple[str, str, int, np.ndarray]] = []
        n_tasks = len(self.ordered_tasks)
        for idx, task_i in enumerate(self.ordered_tasks):
            hp = self.ordered_tasks[:idx]
            if task_i.criticality is Criticality.HI:
                r_lo_i = self.design_r_lo[task_i.name]
                coeff_lo = np.zeros(n_tasks, dtype=np.int64)
                coeff_lo[idx] = 1.0
                for task_j in hp:
                    jdx = self._task_index[task_j.name]
                    coeff_lo[jdx] += (r_lo_i + task_j.period - 1) // task_j.period
                rows.append((task_i.name, "hi_lo_mode", int(r_lo_i), coeff_lo))

                coeff_switch = np.zeros(n_tasks, dtype=np.int64)
                rhs_switch = int(task_i.deadline - task_i.c_hi)
                for task_j in hp:
                    if task_j.criticality is Criticality.LO:
                        jdx = self._task_index[task_j.name]
                        coeff_switch[jdx] += (r_lo_i + task_j.period - 1) // task_j.period
                    else:
                        rhs_switch -= ((task_i.deadline + task_j.period - 1) // task_j.period) * task_j.c_hi
                rows.append((task_i.name, "hi_mode_switch", rhs_switch, coeff_switch))

            if self.check_lo_tasks and task_i.criticality is Criticality.LO:
                coeff_lo_task = np.zeros(n_tasks, dtype=np.int64)
                coeff_lo_task[idx] = 1.0
                for task_j in hp:
                    jdx = self._task_index[task_j.name]
                    coeff_lo_task[jdx] += (task_i.deadline + task_j.period - 1) // task_j.period
                rows.append((task_i.name, "lo_deadline_bound", int(task_i.deadline), coeff_lo_task))
        return rows

    def build_linear_constraints(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 `A, bound`，满足 `A @ budgets <= bound`。"""

        if not self._constraint_rows:
            return np.zeros((0, len(self.ordered_tasks)), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        matrix_a = np.stack([row[3] for row in self._constraint_rows], axis=0)
        bounds = np.array([row[2] for row in self._constraint_rows], dtype=np.int64)
        return matrix_a, bounds

    def validate_candidate(
        self,
        candidate_budgets: Mapping[str, int],
        active_release_budget_max: Mapping[str, int] | None = None,
        lo_overrun_guard_units: int = 0,
    ) -> RuntimeSafetyReport:
        """校验完整 budget 向量是否满足保守约束。"""

        task_names = self._task_names
        if lo_overrun_guard_units < 0:
            raise ValueError("lo_overrun_guard_units 不能为负数")
        diagnostics: list[dict[str, str | int | float]] = []
        for name in task_names:
            if name not in candidate_budgets:
                return RuntimeSafetyReport(False, f"missing_budget:{name}", task_names, tuple(diagnostics))
            if candidate_budgets[name] <= 0:
                return RuntimeSafetyReport(False, f"non_positive_budget:{name}", task_names, tuple(diagnostics))

        active_release_budget_max = active_release_budget_max or {}
        checked = build_effective_safety_budget_vector(self.ordered_tasks, candidate_budgets, active_release_budget_max, lo_overrun_guard_units)
        effective: dict[str, int] = {name: max(int(candidate_budgets[name]), int(active_release_budget_max.get(name, 0))) for name in task_names}
        budget_array = np.array([int(checked[name]) for name in task_names], dtype=object)
        for task_name, constraint_name, rhs, coeff in self._constraint_rows:
            lhs = sum(int(coeff[index]) * int(budget_array[index]) for index in range(len(task_names)))
            diagnostics.append(
                {
                    "task": task_name,
                    "constraint": constraint_name,
                    "lhs": int(lhs),
                    "rhs": rhs,
                    "slack": int(rhs - lhs),
                    "candidate_budget": int(candidate_budgets[task_name]),
                    "active_release_budget_max": int(active_release_budget_max.get(task_name, 0)),
                    "envelope_budget": int(effective[task_name]),
                    "checked_effective_budget": int(checked[task_name]),
                    "lo_overrun_guard_units": int(lo_overrun_guard_units),
                }
            )
            if lhs > int(rhs):
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
