"""tick runtime 在 degraded mode 下记录 dropped LO release cancellation 的测试。"""

from __future__ import annotations

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


def test_tick_runtime_records_dropped_lo_release_cancellation_in_degraded_mode() -> None:
    """tick runtime 在 degraded mode 下 dropped LO release 时，应同时记录 dropped job 与 cancellation。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    result = simulate_ordered_taskset(
        tasks,
        make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        ),
    )

    dropped_lo_jobs = [
        job for job in result.jobs
        if job.task.criticality is Criticality.LO and job.dropped
    ]
    dropped_release_cancellations = [
        event for event in result.job_cancellations
        if event.reason == "lo_release_dropped_in_degraded_mode"
    ]

    assert dropped_lo_jobs
    assert dropped_release_cancellations
    assert all(event.executed_at_cancel == 0 for event in dropped_release_cancellations)
    assert compute_runtime_degradation_metrics(result).jne >= len(dropped_lo_jobs)


def test_tick_runtime_does_not_record_dropped_release_cancellation_by_default() -> None:
    """默认配置下，不应额外记录 dropped LO release job 或对应 cancellation。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    result = simulate_ordered_taskset(
        tasks,
        make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=False,
        ),
    )

    # 默认配置下，degraded mode 中“未来 LO release 被直接 dropped”的记录不应出现；
    # 但切换瞬间已有的 active LO job 仍可能因 mode switch 被正常丢弃，因此这里只排除
    # drop_time == release_time 这类“未执行就被 suppress 的新 release”。
    assert not any(
        job.task.criticality is Criticality.LO
        and job.dropped
        and job.drop_time == job.release_time
        for job in result.jobs
    )
    assert not any(
        event.reason == "lo_release_dropped_in_degraded_mode"
        for event in result.job_cancellations
    )
