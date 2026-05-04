"""事件驱动 runtime 的动态预算更新测试（阶段六）。"""

from __future__ import annotations

import pytest

from amc_py.budget_runtime import BudgetUpdate
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
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


def test_budget_update_prevents_lo_cancellation() -> None:
    """提升 LO 预算后，应避免原本的 AMC+ 局部取消。"""

    lo = _lo("l", period=10, c_lo=3)
    scenario = make_table_scenario({("l", 0): 5})
    no_update = simulate_ordered_taskset_event_driven(
        [lo],
        scenario,
        config=RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    with_update = simulate_ordered_taskset_event_driven(
        [lo],
        scenario,
        config=RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
        budget_updates=[BudgetUpdate(time=0, updates={"l": 5})],
    )
    assert no_update.lo_job_cancellation_count() == 1
    assert with_update.lo_job_cancellation_count() == 0


def test_budget_update_prevents_hi_mode_switch() -> None:
    """提升 HI 预算后，应避免原本的 HI 模式切换。"""

    hi = _hi("h", period=10, c_lo=3, c_hi=5)
    scenario = make_table_scenario({("h", 0): 5})
    no_update = simulate_ordered_taskset_event_driven(
        [hi],
        scenario,
        config=RuntimeConfig(end_time=8),
    )
    with_update = simulate_ordered_taskset_event_driven(
        [hi],
        scenario,
        config=RuntimeConfig(end_time=8),
        budget_updates=[BudgetUpdate(time=0, updates={"h": 5})],
    )
    assert no_update.mode_change_count() == 1
    assert with_update.mode_change_count() == 0


def test_budget_update_does_not_retroactively_trigger_overrun_for_released_job() -> None:
    """下调预算不应追溯影响已释放 job 的 overrun 判定预算。"""

    lo = _lo("l", period=20, c_lo=5)
    scenario = make_table_scenario({("l", 0): 8})
    result = simulate_ordered_taskset_event_driven(
        [lo],
        scenario,
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
        budget_updates=[BudgetUpdate(time=3, updates={"l": 2})],
    )
    assert result.lo_job_cancellation_count() == 1
    assert result.job_cancellations[0].cancel_time == 6
    assert result.job_cancellations[0].budget_at_cancel == 5


def test_budget_update_event_is_recorded() -> None:
    """预算更新事件应记录到结果结构。"""

    lo = _lo("l", period=10, c_lo=3)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("l", 0): 3}),
        config=RuntimeConfig(end_time=8),
        budget_updates=[
            BudgetUpdate(time=0, updates={"l": 3}),
            BudgetUpdate(time=5, updates={"l": 2}),
        ],
    )
    assert len(result.budget_update_events) == 2
    assert result.budget_update_events[0].time == 0
    assert result.budget_update_events[1].time == 5


def test_invalid_budget_update_raises() -> None:
    """非法预算更新应直接抛错。"""

    lo = _lo("l", period=10, c_lo=3)
    scenario = make_table_scenario({("l", 0): 3})
    with pytest.raises(ValueError, match="非负整数"):
        simulate_ordered_taskset_event_driven(
            [lo],
            scenario,
            config=RuntimeConfig(end_time=8),
            budget_updates=[BudgetUpdate(time=-1, updates={"l": 3})],
        )
