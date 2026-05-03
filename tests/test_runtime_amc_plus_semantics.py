"""AMC 与 AMC+ 语义分支测试（阶段 5）。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import (
    make_single_hi_overrun_scenario,
    make_single_lo_overrun_scenario,
)


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


def test_amc_plus_lo_overrun_cancels_without_mode_switch() -> None:
    """AMC+ 下 LO 超预算应局部取消，不触发模式切换。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    result = simulate_ordered_taskset(
        tasks,
        make_single_lo_overrun_scenario("l", release_index=0, actual_cost=3),
        RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.lo_job_cancellation_count() == 1
    assert result.mode_change_count() == 0
    assert result.final_mode is SystemMode.LO


def test_amc_lo_overrun_switches_mode() -> None:
    """标准 AMC 下 LO 超预算应触发模式切换。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    result = simulate_ordered_taskset(
        tasks,
        make_single_lo_overrun_scenario("l", release_index=0, actual_cost=3),
        RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC),
    )
    assert result.mode_change_count() == 1
    assert result.mode_switch is not None
    assert result.mode_switch.reason == "lo_budget_overrun_standard_amc"
    assert result.mode_recovery_count() >= 1
    assert result.final_mode is SystemMode.LO


def test_hi_overrun_switches_mode_in_both_semantics() -> None:
    """无论 AMC 还是 AMC+，HI 超预算都应触发模式切换。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4)]
    scenario = make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi")

    plus_result = simulate_ordered_taskset(
        tasks,
        scenario,
        RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC_PLUS),
    )
    amc_result = simulate_ordered_taskset(
        tasks,
        scenario,
        RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC),
    )

    assert plus_result.mode_change_count() == 1
    assert amc_result.mode_change_count() == 1


def test_runtime_config_default_semantics_is_amc_plus() -> None:
    """默认配置语义应为 AMC_PLUS。"""

    assert RuntimeConfig().semantics is RuntimeSemantics.AMC_PLUS
