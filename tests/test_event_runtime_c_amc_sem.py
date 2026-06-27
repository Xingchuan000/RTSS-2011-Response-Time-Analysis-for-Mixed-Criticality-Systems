"""C-AMC-sem without DVFS 的事件运行时测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import (
    compute_lo_job_loss_breakdown_metrics,
    compute_runtime_degradation_metrics,
)
from amc_py.models import Criticality, Task
from amc_py.runtime_models import (
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
    RuntimeConfig,
    RuntimeSemantics,
    SystemMode,
)
from amc_py.runtime_scenarios import make_table_scenario


def _hi(name: str, period: int, c_lo: int, c_hi: int) -> Task:
    """构造 HI 关键级任务，减少测试样板代码。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


def _lo(name: str, period: int, c_lo: int) -> Task:
    """构造 LO 关键级任务，保持测试意图清晰。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_c_amc_sem_switches_at_hi_abnormal_arrival() -> None:
    """HI abnormal job 应在释放时刻直接触发 C-AMC-sem 模式切换。"""

    hi = _hi("H", period=10, c_lo=2, c_hi=5)
    lo = _lo("L", period=10, c_lo=2)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 5}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.C_AMC_SEM),
    )

    assert result.mode_change_count() == 1
    switch = result.mode_switches[0]
    assert switch.switch_time == 0
    assert switch.triggering_task == "H"
    assert switch.triggering_release_index == 0
    assert switch.executed_at_switch == 0
    assert switch.budget_at_switch == 2
    assert switch.reason == "semi_clairvoyant_hi_abnormal_arrival"


def test_c_amc_sem_does_not_drop_active_lo_on_mode_switch() -> None:
    """C-AMC-sem 切换时不应把已经 active 的 LO job 直接丢弃。"""

    lo = _lo("L", period=20, c_lo=6)
    hi = _hi("H", period=5, c_lo=1, c_hi=3)
    result = simulate_ordered_taskset_event_driven(
        [lo, hi],
        make_table_scenario({("H", 0): 1, ("H", 1): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=12,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    reasons = [event.reason for event in result.lo_job_losses]
    assert LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH not in reasons


def test_c_amc_sem_does_not_suppress_lo_release_in_hi_mode() -> None:
    """C-AMC-sem 在 HI mode 中仍应保留 LO release，而不是直接 suppress。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=4)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 12}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    lo_jobs = result.jobs_of("L")
    assert any(job.release_index == 1 for job in lo_jobs)
    reasons = [event.reason for event in result.lo_job_losses]
    assert LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE not in reasons


def test_c_amc_sem_hi_mode_lo_release_uses_degraded_budget_and_capped_cost() -> None:
    """HI mode 新释放的 LO job 应使用 degraded budget，并截断 actual_cost。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 12, ("L", 1): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l1 = next(job for job in result.jobs_of("L") if job.release_index == 1)
    assert l1.runtime_budget_at_release == 3
    assert l1.actual_cost == 3
    assert not l1.dropped


def test_c_amc_sem_metrics_use_existing_jne_breakdown() -> None:
    """C-AMC-sem 仍应复用现有 metrics.py 的 JNE/LDM 统计口径。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=4)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 12}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    degradation = compute_runtime_degradation_metrics(result)
    breakdown = compute_lo_job_loss_breakdown_metrics(result, degradation)
    assert degradation.jne >= 0
    assert breakdown.lo_release_dropped_in_degraded_mode == 0
    assert breakdown.lo_active_dropped_on_mode_switch == 0


def test_c_amc_sem_degraded_lo_job_records_quality_metadata() -> None:
    """HI mode 中的 degraded LO job 应完整记录质量统计所需 metadata。"""

    hi = _hi("H", period=20, c_lo=8, c_hi=12)
    lo = _lo("L", period=5, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 12, ("L", 1): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=15,
            semantics=RuntimeSemantics.C_AMC_SEM,
            record_dropped_lo_releases=True,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l1 = next(job for job in result.jobs_of("L") if job.release_index == 1)
    assert l1.released_in_mode is SystemMode.HI
    assert l1.is_degraded is True
    assert l1.service_quality_if_completed == 0.5
    assert l1.original_actual_cost == 6
    assert l1.original_runtime_budget_at_release == 6
    assert l1.runtime_budget_at_release == 3
    assert l1.actual_cost == 3


def test_c_amc_sem_primary_on_switch_time_keeps_same_batch_lo_primary() -> None:
    """开启 strict 边界后，同一 arrival batch 的 LO release 仍保持 primary。"""

    hi = _hi("H", period=10, c_lo=2, c_hi=5)
    lo = _lo("L", period=10, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 5, ("L", 0): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=10,
            semantics=RuntimeSemantics.C_AMC_SEM,
            c_amc_sem_lo_degradation_ratio=0.5,
            c_amc_sem_primary_on_switch_time=True,
        ),
    )

    l0 = next(job for job in result.jobs_of("L") if job.release_index == 0)
    assert l0.released_in_mode is SystemMode.LO
    assert l0.is_degraded is False
    assert l0.service_quality_if_completed == 1.0
    assert l0.runtime_budget_at_release == 6
    assert l0.actual_cost == 6


def test_c_amc_sem_default_same_time_behavior_is_unchanged() -> None:
    """默认配置必须保持旧的“同一时刻已切 HI 即 degraded”行为。"""

    hi = _hi("H", period=10, c_lo=2, c_hi=5)
    lo = _lo("L", period=10, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 5, ("L", 0): 6}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=10,
            semantics=RuntimeSemantics.C_AMC_SEM,
            c_amc_sem_lo_degradation_ratio=0.5,
        ),
    )

    l0 = next(job for job in result.jobs_of("L") if job.release_index == 0)
    assert l0.released_in_mode is SystemMode.HI
    assert l0.is_degraded is True
    assert l0.runtime_budget_at_release == 3
    assert l0.actual_cost == 3
