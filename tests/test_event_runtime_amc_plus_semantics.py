"""事件驱动 runtime 的 AMC / AMC+ 语义测试（阶段四）。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import make_single_lo_overrun_scenario, make_table_scenario


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


def test_amc_plus_lo_overrun_cancels_only_that_job() -> None:
    """AMC+ 下 LO overrun 只取消该 job，不切 HI 模式。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4),
        config=RuntimeConfig(end_time=15, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.mode_change_count() == 0
    assert result.lo_job_cancellation_count() == 1
    assert result.final_mode is SystemMode.LO


def test_standard_amc_lo_overrun_switches_to_hi() -> None:
    """标准 AMC 下 LO overrun 应触发 LO->HI 切换。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4),
        config=RuntimeConfig(end_time=15, semantics=RuntimeSemantics.AMC),
    )
    assert result.mode_change_count() == 1
    assert result.lo_job_cancellation_count() == 0
    assert result.mode_switches[0].reason == "lo_budget_overrun_standard_amc"


def test_amc_plus_hi_overrun_switches_to_hi() -> None:
    """AMC+ 下 HI overrun 仍应触发切换。"""

    tasks = [_hi("h", period=10, c_lo=1, c_hi=3), _lo("l", period=10, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("h", 0): 3}),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.mode_change_count() == 1
    assert result.lo_job_cancellation_count() == 0


def test_amc_plus_lo_overrun_does_not_suppress_future_lo_releases() -> None:
    """AMC+ 的 LO 局部取消不应影响后续 LO 周期释放。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=1), _lo("l", period=2, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("l", 0): 2}),
        config=RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    # period=2, end_time=8 时理论 release_index 为 0,1,2,3；都应出现。
    assert [job.release_index for job in result.jobs_of("l")] == [0, 1, 2, 3]


def test_standard_amc_lo_overrun_suppresses_future_lo_releases_until_recovery() -> None:
    """标准 AMC 的 LO overrun 触发 HI 后，应在恢复前抑制 LO 释放。"""

    lo = _lo("l", period=1, c_lo=1)
    hi = _hi("h", period=20, c_lo=1, c_hi=3)
    result = simulate_ordered_taskset_event_driven(
        [lo, hi],
        make_table_scenario({("l", 0): 3, ("h", 0): 3}),
        config=RuntimeConfig(end_time=7, semantics=RuntimeSemantics.AMC),
    )
    lo_release_indexes = [job.release_index for job in result.jobs_of("l")]
    assert result.mode_change_count() == 1
    assert result.mode_recovery_count() == 1
    # 在 HI 运行区间（t=2~5）内，未来 LO release 被抑制。
    assert 2 not in lo_release_indexes
    assert 3 not in lo_release_indexes
    assert 4 not in lo_release_indexes
    # 恢复后未来 release 正常继续。
    assert 5 in lo_release_indexes
    assert 6 in lo_release_indexes
