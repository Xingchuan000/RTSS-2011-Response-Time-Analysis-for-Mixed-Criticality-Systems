"""运行时预算状态测试（阶段 2）。"""

from __future__ import annotations

import pytest

from amc_py.budget_runtime import BudgetState, BudgetUpdate
from amc_py.models import Criticality, Task


def _tasks() -> list[Task]:
    """构造一组用于预算测试的任务。"""

    return [
        Task("h", period=10, deadline=10, c_lo=3, c_hi=5, criticality=Criticality.HI),
        Task("l", period=12, deadline=12, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def test_from_tasks_uses_c_lo_as_initial_budget() -> None:
    """from_tasks 应按每个任务的 c_lo 初始化预算。"""

    tasks = _tasks()
    state = BudgetState.from_tasks(tasks)
    assert state.budgets == {"h": 3, "l": 2}
    assert state.initial_budgets == {"h": 3, "l": 2}


def test_budget_of_and_set_budget_work() -> None:
    """应支持查询和更新单个任务预算。"""

    tasks = _tasks()
    state = BudgetState.from_tasks(tasks)
    assert state.budget_of(tasks[0]) == 3

    state.set_budget("h", 4)
    assert state.budget_of(tasks[0]) == 4


@pytest.mark.parametrize("invalid", [0, -1, -9])
def test_set_budget_rejects_non_positive_values(invalid: int) -> None:
    """预算必须为正数，0 和负数都应抛错。"""

    state = BudgetState.from_tasks(_tasks())
    with pytest.raises(ValueError, match="runtime budget must be > 0"):
        state.set_budget("h", invalid)


def test_budget_of_raises_key_error_when_task_missing() -> None:
    """若预算表缺少指定任务，应抛 KeyError。"""

    tasks = _tasks()
    state = BudgetState.from_tasks(tasks)
    del state.budgets["h"]

    with pytest.raises(KeyError, match="missing runtime budget"):
        state.budget_of(tasks[0])


def test_copy_is_deep_copy() -> None:
    """copy 后新旧对象互不影响。"""

    state = BudgetState.from_tasks(_tasks())
    cloned = state.copy()
    cloned.set_budget("h", 9)

    assert state.budgets["h"] == 3
    assert cloned.budgets["h"] == 9


def test_set_budget_raises_key_error_for_unknown_task() -> None:
    """更新不存在的任务预算应抛 KeyError。"""

    state = BudgetState.from_tasks(_tasks())
    with pytest.raises(KeyError, match="missing runtime budget"):
        state.set_budget("unknown", 3)


def test_apply_updates_updates_multiple_tasks() -> None:
    """批量更新应逐项生效。"""

    state = BudgetState.from_tasks(_tasks())
    state.apply_updates({"h": 4, "l": 3})
    assert state.budgets["h"] == 4
    assert state.budgets["l"] == 3


def test_budget_update_dataclass_holds_time_and_updates() -> None:
    """BudgetUpdate 应正确保存更新时间和更新字典。"""

    update = BudgetUpdate(time=5, updates={"h": 4})
    assert update.time == 5
    assert update.updates == {"h": 4}
