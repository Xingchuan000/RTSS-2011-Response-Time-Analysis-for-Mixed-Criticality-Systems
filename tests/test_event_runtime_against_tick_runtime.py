"""事件驱动 runtime 与 tick runtime 的对照回归测试（阶段十）。"""

from __future__ import annotations

from collections.abc import Sequence

from amc_py.budget_runtime import BudgetUpdate
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import (
    ExecutionScenario,
    make_nominal_scenario,
    make_single_lo_overrun_scenario,
    make_table_scenario,
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


def _assert_core_stats_equal(
    tasks: Sequence[Task],
    scenario: ExecutionScenario,
    config: RuntimeConfig,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> None:
    """断言 event runtime 与 tick runtime 的核心统计一致。"""

    tick_result = simulate_ordered_taskset(
        ordered_tasks=tasks,
        scenario=scenario,
        config=config,
        budget_updates=budget_updates,
    )
    event_result = simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=config,
        budget_updates=budget_updates,
    )
    assert event_result.mode_change_count() == tick_result.mode_change_count()
    assert (
        event_result.lo_job_cancellation_count()
        == tick_result.lo_job_cancellation_count()
    )
    assert event_result.mode_recovery_count() == tick_result.mode_recovery_count()
    assert len(event_result.deadline_misses) == len(tick_result.deadline_misses)


def test_against_tick_nominal_no_overrun() -> None:
    """无 overrun 场景下，两种 runtime 核心统计应一致。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
    )


def test_against_tick_amc_plus_lo_overrun() -> None:
    """AMC+ 的 LO overrun 场景下，两种 runtime 核心统计应一致。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4),
        config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
    )


def test_against_tick_standard_amc_lo_overrun() -> None:
    """标准 AMC 的 LO overrun 场景下，两种 runtime 核心统计应一致。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4),
        config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC),
    )


def test_against_tick_amc_plus_hi_overrun() -> None:
    """AMC+ 的 HI overrun 场景下，两种 runtime 核心统计应一致。"""

    tasks = [_hi("h", period=10, c_lo=1, c_hi=3), _lo("l", period=10, c_lo=1)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_table_scenario({("h", 0): 3}),
        config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
    )


def test_against_tick_dynamic_budget_avoids_overrun() -> None:
    """动态预算提高后避免 overrun 的场景下，两种 runtime 核心统计应一致。"""

    tasks = [_lo("l", period=10, c_lo=3)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_table_scenario({("l", 0): 5}),
        config=RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC_PLUS),
        budget_updates=[BudgetUpdate(time=0, updates={"l": 5})],
    )


def test_against_tick_preemption_case() -> None:
    """高优先级抢占低优先级场景下，两种 runtime 核心统计应一致。"""

    tasks = [_lo("high", period=3, c_lo=1), _lo("low", period=20, c_lo=6)]
    _assert_core_stats_equal(
        tasks=tasks,
        scenario=make_table_scenario({("low", 0): 6, ("high", 1): 1}),
        config=RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC_PLUS),
    )
