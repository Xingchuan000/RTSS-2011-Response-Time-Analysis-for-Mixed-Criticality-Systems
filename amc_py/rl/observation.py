"""构建 RL 观测向量。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.types import AgentObservation


@dataclass(frozen=True, slots=True)
class TaskNormalizationBound:
    """单个任务的归一化区间定义。"""

    min_cost: float
    max_cost: float


NormalizationBounds = dict[str, TaskNormalizationBound]


def build_default_normalization_bounds(tasks: Sequence[Task]) -> NormalizationBounds:
    """按默认策略构建归一化边界。"""

    bounds: NormalizationBounds = {}
    for task in tasks:
        if task.criticality is Criticality.HI:
            bounds[task.name] = TaskNormalizationBound(min_cost=0.0, max_cost=float(task.c_hi))
        else:
            bounds[task.name] = TaskNormalizationBound(
                min_cost=0.0,
                max_cost=float(max(task.c_lo, task.deadline)),
            )
    return bounds


def _normalize(value: int, lo: float, hi: float) -> float:
    """将整数值归一化并裁剪到 [0, 1]。"""

    if hi <= lo:
        raise ValueError("normalization bound 非法：max_cost 必须大于 min_cost")
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def build_observation(
    *,
    time: int,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    monitor: RuntimeMonitor,
    bounds: NormalizationBounds | None = None,
) -> AgentObservation:
    """按任务顺序构建观测，状态格式为 [(B_i, c_i), ...]。"""

    active_bounds = bounds or build_default_normalization_bounds(ordered_tasks)
    state_values: list[float] = []
    raw_budgets: dict[str, int] = {}
    raw_recent_costs: dict[str, int] = {}

    for task in ordered_tasks:
        task_name = task.name
        budget = budget_state.budgets[task_name]
        recent_cost = monitor.recent_execution.get(task_name, 0)

        if task_name not in active_bounds:
            raise ValueError(f"normalization bounds 缺少任务 {task_name}")
        task_bound = active_bounds[task_name]
        lo = float(task_bound.min_cost)
        hi = float(task_bound.max_cost)

        state_values.append(_normalize(budget, lo, hi))
        state_values.append(_normalize(recent_cost, lo, hi))

        raw_budgets[task_name] = budget
        raw_recent_costs[task_name] = recent_cost

    return AgentObservation(
        time=time,
        state_vector=tuple(state_values),
        raw_budgets=raw_budgets,
        raw_recent_costs=raw_recent_costs,
    )
