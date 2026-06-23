"""AMC-RA / AMC-RH 的 tick runtime 与 event runtime 对照测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_runtime_degradation_metrics
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


def _assert_ra_rh_stats_equal(config: RuntimeConfig) -> None:
    """断言两种 runtime 在 RA/RH 关键统计上保持一致。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    scenario = make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo")
    tick_result = simulate_ordered_taskset(tasks, scenario, config=config)
    event_result = simulate_ordered_taskset_event_driven(tasks, scenario, config=config)

    assert event_result.mode_change_count() == tick_result.mode_change_count()
    assert event_result.mode_recovery_count() == tick_result.mode_recovery_count()
    assert len(event_result.deadline_misses) == len(tick_result.deadline_misses)
    assert len([job for job in event_result.jobs if job.dropped]) == len(
        [job for job in tick_result.jobs if job.dropped]
    )
    assert sum(
        1
        for event in event_result.job_cancellations
        if event.reason == "lo_release_dropped_in_degraded_mode"
    ) == sum(
        1
        for event in tick_result.job_cancellations
        if event.reason == "lo_release_dropped_in_degraded_mode"
    )


def test_amc_ra_against_event_runtime() -> None:
    """AMC_RA 下 tick / event runtime 的核心统计应一致。"""

    _assert_ra_rh_stats_equal(
        RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        )
    )


def test_amc_rh_against_event_runtime() -> None:
    """AMC_RH 下 tick / event runtime 的核心统计应一致。"""

    _assert_ra_rh_stats_equal(
        RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RH,
            record_dropped_lo_releases=True,
        )
    )


def test_tick_and_event_both_record_dropped_lo_release_cancellation() -> None:
    """同一 RA 场景下，tick/event runtime 都应记录 degraded-mode dropped LO release 的 cancellation。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    scenario = make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo")
    config = RuntimeConfig(
        end_time=9,
        semantics=RuntimeSemantics.AMC_RA,
        record_dropped_lo_releases=True,
    )
    tick_result = simulate_ordered_taskset(tasks, scenario, config=config)
    event_result = simulate_ordered_taskset_event_driven(tasks, scenario, config=config)

    tick_reasons = [event.reason for event in tick_result.job_cancellations]
    event_reasons = [event.reason for event in event_result.job_cancellations]
    assert "lo_release_dropped_in_degraded_mode" in tick_reasons
    assert "lo_release_dropped_in_degraded_mode" in event_reasons
    assert compute_runtime_degradation_metrics(tick_result).jne > 0
