"""运行时动态预算容器（阶段 2）。

本模块用于分离“设计时任务参数”和“运行时预算向量”：
- `Task.c_lo` 保持为设计时参数，不在仿真过程中被修改；
- `BudgetState` 维护可独立更新/复制的运行时预算字典。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Sequence

from .models import Task


@dataclass(slots=True)
class BudgetState:
    """运行时预算状态。

    字段说明：
    - `budgets`: 当前生效的运行时预算（任务名 -> 预算值）；
    - `initial_budgets`: 初始化时预算快照，便于对照或复位。
    """

    budgets: dict[str, int]
    initial_budgets: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_tasks(cls, tasks: Sequence[Task]) -> "BudgetState":
        """用任务集的 `c_lo` 初始化运行时预算。"""

        budgets = {task.name: task.c_lo for task in tasks}
        return cls(budgets=dict(budgets), initial_budgets=dict(budgets))

    def budget_of(self, task: Task) -> int:
        """查询指定任务当前运行时预算。"""

        try:
            return self.budgets[task.name]
        except KeyError as exc:
            raise KeyError(f"missing runtime budget for task {task.name!r}") from exc

    def set_budget(self, task_name: str, value: int) -> None:
        """更新指定任务预算，预算值必须为正整数。"""

        if value <= 0:
            raise ValueError("runtime budget must be > 0")
        if task_name not in self.budgets:
            raise KeyError(f"missing runtime budget for task {task_name!r}")
        self.budgets[task_name] = value

    def apply_updates(self, updates: Mapping[str, int]) -> None:
        """批量应用预算更新。"""

        for task_name, value in updates.items():
            self.set_budget(task_name, value)

    def copy(self) -> "BudgetState":
        """返回当前预算状态的深拷贝。"""

        return BudgetState(
            budgets=dict(self.budgets),
            initial_budgets=dict(self.initial_budgets),
        )


@dataclass(frozen=True, slots=True)
class BudgetUpdate:
    """描述一个时刻的预算更新指令。"""

    time: int
    updates: dict[str, int]


__all__ = ["BudgetState", "BudgetUpdate"]
