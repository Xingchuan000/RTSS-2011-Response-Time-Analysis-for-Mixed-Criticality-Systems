"""reason-level JNE / LO job loss 事件测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import (
    compute_lo_job_loss_breakdown_metrics,
    compute_runtime_degradation_metrics,
)
from amc_py.models import Criticality, Task
from amc_py.runtime_models import (
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_BUDGET_CANCELLATION,
    LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
    RuntimeConfig,
    RuntimeSemantics,
)
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


def test_lo_budget_overrun_records_budget_cancellation_loss() -> None:
    """LO budget overrun 需要同时写入旧 cancellation 与新 loss 事件。"""

    lo = _lo("L", period=10, c_lo=1)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("L", 0): 3}, default_lo="c_lo"),
        config=RuntimeConfig(end_time=5, semantics=RuntimeSemantics.AMC_PLUS),
    )

    assert result.lo_job_cancellation_count() == 1
    assert any(loss.reason == LO_LOSS_BUDGET_CANCELLATION for loss in result.lo_job_losses)


def test_degraded_mode_release_drop_records_release_loss() -> None:
    """degraded mode 中被直接抑制的 LO release 应记录 release-drop reason。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=8,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        ),
    )

    assert any(
        loss.reason == LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE
        for loss in result.lo_job_losses
    )


def test_mode_switch_active_lo_drop_records_active_drop_loss() -> None:
    """mode switch 时 active LO drop 应进入 JNE，但不追加 cancellation。"""

    hi = _hi("H", period=10, c_lo=1, c_hi=3)
    lo = _lo("L", period=10, c_lo=2)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_RA),
    )

    assert any(
        loss.reason == LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH
        for loss in result.lo_job_losses
    )
    assert result.lo_job_cancellation_count() == 0
    assert compute_runtime_degradation_metrics(result).jne >= 1


def test_loss_breakdown_matches_runtime_degradation_metrics() -> None:
    """breakdown 总数应与 reason 分项之和一致。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("H1", 0): 3, ("L1", 0): 2}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=8,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        ),
    )

    breakdown = compute_lo_job_loss_breakdown_metrics(result)

    assert breakdown.lo_job_losses_total == (
        breakdown.lo_budget_cancellations
        + breakdown.lo_release_dropped_in_degraded_mode
        + breakdown.lo_active_dropped_on_mode_switch
    )
