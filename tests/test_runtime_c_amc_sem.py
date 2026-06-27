"""tick runtime 的 C-AMC-sem without DVFS 兼容性测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import make_table_scenario


def _hi(name: str, period: int, c_lo: int, c_hi: int) -> Task:
    """构造 HI 关键级任务，便于复用测试场景。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


def _lo(name: str, period: int, c_lo: int) -> Task:
    """构造 LO 关键级任务，保持样例简洁。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_c_amc_sem_tick_switches_at_hi_abnormal_arrival() -> None:
    """tick runtime 中，HI abnormal job 应在 release time 立即触发切换。"""

    hi = _hi("H", period=10, c_lo=2, c_hi=5)
    lo = _lo("L", period=10, c_lo=2)
    result = simulate_ordered_taskset(
        [hi, lo],
        make_table_scenario({("H", 0): 5}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.C_AMC_SEM),
    )

    assert result.mode_change_count() == 1
    switch = result.mode_switches[0]
    assert switch.switch_time == 0
    assert switch.reason == "semi_clairvoyant_hi_abnormal_arrival"
    assert switch.executed_at_switch == 0
    assert switch.budget_at_switch == 2


def test_c_amc_sem_tick_mode_switch_does_not_drop_active_lo_job() -> None:
    """C-AMC-sem 切换到 HI mode 时不应丢弃已经 active 的 LO job。"""

    lo = _lo("L", period=20, c_lo=6)
    hi = _hi("H", period=5, c_lo=1, c_hi=3)
    result = simulate_ordered_taskset(
        [lo, hi],
        make_table_scenario({("H", 0): 1, ("H", 1): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=12,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l0 = next(job for job in result.jobs if job.task.name == "L" and job.release_index == 0)
    assert not l0.dropped


def test_c_amc_sem_tick_hi_mode_does_not_suppress_lo_release() -> None:
    """C-AMC-sem 在 HI mode 中仍应创建 LO release，而不是直接 suppress。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=4)
    result = simulate_ordered_taskset(
        [hi, lo],
        make_table_scenario({("H", 0): 12}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    assert any(job.task.name == "L" and job.release_index == 1 for job in result.jobs)
    assert not any(
        event.reason == "lo_release_dropped_in_degraded_mode"
        for event in result.job_cancellations
    )


def test_c_amc_sem_tick_hi_mode_lo_release_uses_degraded_budget_and_capped_cost() -> None:
    """HI mode 中释放的 LO job 应使用 degraded budget，并截断 actual_cost。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=6)
    result = simulate_ordered_taskset(
        [hi, lo],
        make_table_scenario({("H", 0): 12, ("L", 1): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l1 = next(job for job in result.jobs if job.task.name == "L" and job.release_index == 1)
    assert l1.runtime_budget_at_release == 3
    assert l1.actual_cost == 3
    assert l1.actual_cost <= l1.runtime_budget_at_release


def test_c_amc_sem_tick_matches_event_runtime_core_semantics() -> None:
    """代表性场景下，tick runtime 与 event runtime 的核心 C-AMC-sem 语义应一致。"""

    tasks = [
        _hi("H", period=20, c_lo=8, c_hi=12),
        _lo("L", period=5, c_lo=6),
    ]
    scenario = make_table_scenario(
        {("H", 0): 12, ("L", 1): 6},
        default_hi="c_lo",
        default_lo="c_lo",
    )
    config = RuntimeConfig(
        end_time=15,
        semantics=RuntimeSemantics.C_AMC_SEM,
        record_dropped_lo_releases=True,
        c_amc_sem_lo_degradation_ratio=0.5,
    )

    tick_result = simulate_ordered_taskset(tasks, scenario, config=config)
    event_result = simulate_ordered_taskset_event_driven(tasks, scenario, config=config)

    assert tick_result.mode_change_count() == event_result.mode_change_count()
    assert tick_result.mode_recovery_count() == event_result.mode_recovery_count()
    assert tick_result.final_mode == event_result.final_mode
    tick_l1 = next(job for job in tick_result.jobs if job.task.name == "L" and job.release_index == 1)
    event_l1 = next(job for job in event_result.jobs if job.task.name == "L" and job.release_index == 1)
    assert tick_l1.runtime_budget_at_release == event_l1.runtime_budget_at_release
    assert tick_l1.actual_cost == event_l1.actual_cost
    assert tick_l1.released_in_mode == event_l1.released_in_mode
    assert tick_l1.is_degraded == event_l1.is_degraded
    assert tick_l1.service_quality_if_completed == event_l1.service_quality_if_completed


def test_c_amc_sem_tick_degraded_lo_job_records_quality_metadata() -> None:
    """tick runtime 的 degraded LO job 也应记录质量加权所需 metadata。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=6)
    result = simulate_ordered_taskset(
        [hi, lo],
        make_table_scenario({("H", 0): 12, ("L", 1): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l1 = next(job for job in result.jobs if job.task.name == "L" and job.release_index == 1)
    assert l1.released_in_mode is SystemMode.HI
    assert l1.is_degraded is True
    assert l1.service_quality_if_completed == 0.5
    assert l1.original_actual_cost == 6
    assert l1.original_runtime_budget_at_release == 6
    assert l1.runtime_budget_at_release == 3
    assert l1.actual_cost == 3


def test_c_amc_sem_tick_primary_on_switch_time_keeps_same_batch_lo_primary() -> None:
    """tick runtime 开启 strict 边界后，同一 batch 的 LO release 不应降级。"""

    hi = _hi("H", period=10, c_lo=2, c_hi=5)
    lo = _lo("L", period=10, c_lo=6)
    result = simulate_ordered_taskset(
        [hi, lo],
        make_table_scenario({("H", 0): 5, ("L", 0): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=10,
            semantics=RuntimeSemantics.C_AMC_SEM,
            c_amc_sem_lo_degradation_ratio=0.5,
            c_amc_sem_primary_on_switch_time=True,
        ),
    )

    l0 = next(job for job in result.jobs if job.task.name == "L" and job.release_index == 0)
    assert l0.released_in_mode is SystemMode.LO
    assert l0.is_degraded is False
    assert l0.service_quality_if_completed == 1.0
    assert l0.runtime_budget_at_release == 6
    assert l0.actual_cost == 6
