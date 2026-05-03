"""运行时预算按时间更新测试（阶段 8）。"""

from __future__ import annotations

import pytest

from amc_py.budget_runtime import BudgetUpdate
from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


def _hi(name: str, period: int, c_lo: int, c_hi: int) -> Task:
    """构造 HI 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


def _lo(name: str, period: int, c_lo: int) -> Task:
    """构造 LO 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_budget_defaults_to_c_lo_when_no_updates() -> None:
    """无更新时预算应等于 c_lo。"""

    hi = _hi("h", period=10, c_lo=3, c_hi=5)
    scenario = make_table_scenario(actual_costs={("h", 0): 4})
    result = simulate_ordered_taskset([hi], scenario, RuntimeConfig(end_time=6))
    assert result.mode_change_count() == 1
    assert result.mode_switch is not None
    assert result.mode_switch.budget_at_switch == 3


def test_time_zero_update_changes_hi_overrun_decision() -> None:
    """time=0 将 HI budget 提升后，原本会切换的作业可不切换。"""

    hi = _hi("h", period=10, c_lo=3, c_hi=5)
    scenario = make_table_scenario(actual_costs={("h", 0): 4})
    result = simulate_ordered_taskset(
        [hi],
        scenario,
        RuntimeConfig(end_time=6),
        budget_updates=[BudgetUpdate(time=0, updates={"h": 4})],
    )
    assert result.mode_change_count() == 0
    assert len(result.budget_update_events) == 1


def test_time_zero_update_changes_lo_cancellation_decision() -> None:
    """time=0 将 LO budget 提升后，原本会取消的 LO job 可继续执行。"""

    lo = _lo("l", period=10, c_lo=3)
    scenario = make_table_scenario(actual_costs={("l", 0): 4})
    result = simulate_ordered_taskset(
        [lo],
        scenario,
        RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC_PLUS),
        budget_updates=[BudgetUpdate(time=0, updates={"l": 4})],
    )
    assert result.lo_job_cancellation_count() == 0


def test_late_budget_update_only_affects_future_jobs() -> None:
    """在 time=5 更新预算，只应影响 time>=5 的后续作业。"""

    hi = _hi("h", period=5, c_lo=3, c_hi=5)
    scenario = make_table_scenario(
        actual_costs={
            ("h", 0): 4,
            ("h", 1): 4,
        }
    )
    result = simulate_ordered_taskset(
        [hi],
        scenario,
        RuntimeConfig(end_time=12),
        budget_updates=[BudgetUpdate(time=5, updates={"h": 4})],
    )
    assert result.mode_change_count() == 1
    assert result.mode_switches[0].triggering_release_index == 0


def test_invalid_budget_update_value_raises_value_error() -> None:
    """预算更新值为 0 或负数时应抛 ValueError。"""

    hi = _hi("h", period=10, c_lo=3, c_hi=5)
    scenario = make_table_scenario(actual_costs={("h", 0): 4})
    with pytest.raises(ValueError, match="runtime budget must be > 0"):
        simulate_ordered_taskset(
            [hi],
            scenario,
            RuntimeConfig(end_time=6),
            budget_updates=[BudgetUpdate(time=0, updates={"h": 0})],
        )


def test_unknown_task_budget_update_raises_key_error() -> None:
    """更新不存在任务预算时应抛 KeyError。"""

    hi = _hi("h", period=10, c_lo=3, c_hi=5)
    scenario = make_table_scenario(actual_costs={("h", 0): 4})
    with pytest.raises(KeyError, match="missing runtime budget"):
        simulate_ordered_taskset(
            [hi],
            scenario,
            RuntimeConfig(end_time=6),
            budget_updates=[BudgetUpdate(time=0, updates={"unknown": 4})],
        )

