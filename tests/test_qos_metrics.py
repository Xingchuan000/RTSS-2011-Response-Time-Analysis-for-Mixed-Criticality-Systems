"""QoS 指标 helper 测试。"""

from __future__ import annotations

from amc_py.metrics import (
    compute_lo_job_loss_breakdown_metrics,
    compute_lo_quality_weighted_metrics,
    compute_service_quality_metrics,
)
from amc_py.models import Criticality, Task
from amc_py.runtime_models import (
    Job,
    JobCancellationEvent,
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_BUDGET_CANCELLATION,
    LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
    LoJobLossEvent,
    SimulationResult,
    SystemMode,
)


def _build_lo_job(task: Task, release_index: int) -> Job:
    """构造测试用 LO job。"""

    release_time = release_index * task.period
    return Job(
        task=task,
        release_index=release_index,
        release_time=release_time,
        absolute_deadline=release_time + task.deadline,
        actual_cost=task.c_lo,
        runtime_budget_at_release=task.c_lo,
    )


def test_compute_service_quality_metrics_basic_case() -> None:
    """2 个 LO job、1 个取消、1 个完成时应得到计划文档要求的指标。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO)
    hi_task = Task(name="HI1", period=20, deadline=20, c_lo=3, c_hi=5, criticality=Criticality.HI)
    job_a = _build_lo_job(lo_task, 0)
    job_b = _build_lo_job(lo_task, 1)
    job_b.completion_time = 15
    hi_job = Job(
        task=hi_task,
        release_index=0,
        release_time=0,
        absolute_deadline=20,
        actual_cost=3,
        runtime_budget_at_release=3,
        completion_time=3,
    )
    result = SimulationResult(
        jobs=[job_a, job_b, hi_job],
        job_cancellations=[
            JobCancellationEvent(
                cancel_time=5,
                task="LO1",
                release_index=0,
                executed_at_cancel=1,
                budget_at_cancel=1,
            )
        ],
    )

    metrics = compute_service_quality_metrics(result)

    assert metrics.released_lo_jobs == 2
    assert metrics.cancelled_lo_jobs == 1
    assert metrics.completed_lo_jobs == 1
    assert metrics.lc_service_loss == 0.5
    assert metrics.lc_qos == 0.5
    assert metrics.min_lc_service == 0.5


def test_compute_service_quality_metrics_without_lo_jobs() -> None:
    """没有 LO job 时，LC service loss 应回落到 0 且 QoS 为 1。"""

    hi_task = Task(name="HI1", period=20, deadline=20, c_lo=3, c_hi=5, criticality=Criticality.HI)
    hi_job = Job(
        task=hi_task,
        release_index=0,
        release_time=0,
        absolute_deadline=20,
        actual_cost=3,
        runtime_budget_at_release=3,
        completion_time=3,
    )
    result = SimulationResult(jobs=[hi_job])

    metrics = compute_service_quality_metrics(result)

    assert metrics.released_lo_jobs == 0
    assert metrics.lc_service_loss == 0.0
    assert metrics.lc_qos == 1.0
    assert metrics.min_lc_service is None


def test_compute_lo_job_loss_breakdown_metrics_uses_reason_level_losses() -> None:
    """当新结果已包含 reason-level loss 时，应直接按 reason 精确计数。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO)
    result = SimulationResult(
        jobs=[_build_lo_job(lo_task, 0), _build_lo_job(lo_task, 1), _build_lo_job(lo_task, 2)],
        lo_job_losses=[
            LoJobLossEvent(
                loss_time=2,
                task="LO1",
                release_index=0,
                release_time=0,
                executed_at_loss=2,
                budget_at_loss=1,
                reason=LO_LOSS_BUDGET_CANCELLATION,
            ),
            LoJobLossEvent(
                loss_time=4,
                task="LO1",
                release_index=1,
                release_time=10,
                executed_at_loss=0,
                budget_at_loss=2,
                reason=LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
            ),
            LoJobLossEvent(
                loss_time=5,
                task="LO1",
                release_index=2,
                release_time=20,
                executed_at_loss=1,
                budget_at_loss=2,
                reason=LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
            ),
        ],
    )

    metrics = compute_lo_job_loss_breakdown_metrics(result)

    assert metrics.lo_job_losses_total == 3
    assert metrics.lo_budget_cancellations == 1
    assert metrics.lo_release_dropped_in_degraded_mode == 1
    assert metrics.lo_active_dropped_on_mode_switch == 1
    assert metrics.jne_residual_not_in_cancellations == 0


def test_compute_lo_job_loss_breakdown_metrics_falls_back_to_old_cancellations() -> None:
    """旧结果只有 cancellation 时，fallback 也应能给出稳定统计。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO)
    job_a = _build_lo_job(lo_task, 0)
    job_a.dropped = True
    job_b = _build_lo_job(lo_task, 1)
    job_b.dropped = True
    result = SimulationResult(
        jobs=[job_a, job_b],
        job_cancellations=[
            JobCancellationEvent(
                cancel_time=5,
                task="LO1",
                release_index=0,
                executed_at_cancel=2,
                budget_at_cancel=1,
                reason="lo_budget_overrun",
            )
        ],
    )

    metrics = compute_lo_job_loss_breakdown_metrics(result)

    assert metrics.lo_budget_cancellations == 1
    assert metrics.lo_release_dropped_in_degraded_mode == 0
    assert metrics.lo_active_dropped_on_mode_switch == 1
    assert metrics.lo_job_losses_total == 2


def test_compute_lo_job_loss_breakdown_metrics_returns_none_share_when_jne_zero() -> None:
    """当 JNE 为 0 时，active drop share 没有定义，应返回 None。"""

    result = SimulationResult()

    metrics = compute_lo_job_loss_breakdown_metrics(result)

    assert metrics.lo_job_losses_total == 0
    assert metrics.active_drop_share_of_jne is None


def test_compute_lo_quality_weighted_metrics_binary_methods() -> None:
    """传统 binary 方法在新指标下应退化为 0/1 服务。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO)
    job0 = _build_lo_job(lo_task, 0)
    job0.completion_time = 2
    job1 = _build_lo_job(lo_task, 1)
    job1.dropped = True
    result = SimulationResult(jobs=[job0, job1])

    metrics = compute_lo_quality_weighted_metrics(result)

    assert metrics.lo_full_quality_completed == 1
    assert metrics.lo_degraded_released == 0
    assert metrics.lo_zero_service_jobs == 1
    assert metrics.lo_quality_qos == 0.5
    assert metrics.lo_equiv_jne == 1.0
    assert metrics.lo_equiv_jne_rate == 0.5


def test_compute_lo_quality_weighted_metrics_counts_degraded_completion_as_partial_service() -> None:
    """degraded LO completion 应按 service_quality_if_completed 计入部分服务。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=6, c_hi=6, criticality=Criticality.LO)
    job0 = _build_lo_job(lo_task, 0)
    job0.completion_time = 3
    job0.is_degraded = True
    job0.released_in_mode = SystemMode.HI
    job0.service_quality_if_completed = 0.5
    job0.runtime_budget_at_release = 3
    job0.original_runtime_budget_at_release = 6
    job0.actual_cost = 3
    job0.original_actual_cost = 6
    result = SimulationResult(jobs=[job0])

    metrics = compute_lo_quality_weighted_metrics(result)

    assert metrics.lo_degraded_released == 1
    assert metrics.lo_degraded_completed == 1
    assert metrics.lo_quality_qos == 0.5
    assert metrics.lo_quality_loss == 0.5
    assert metrics.lo_equiv_jne == 0.5
    assert metrics.lo_full_quality_ratio == 0.0
    assert metrics.lo_degraded_release_ratio == 1.0
    assert metrics.lo_degraded_completion_ratio == 1.0
    assert metrics.lo_zero_service_ratio == 0.0
    assert metrics.lo_degraded_budget_ratio_mean == 0.5


def test_compute_lo_quality_weighted_metrics_degraded_completion_ratio_uses_released_lo_jobs() -> None:
    """degraded completion ratio 的分母应为全部 released LO jobs。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=6, c_hi=6, criticality=Criticality.LO)
    full_quality_job = _build_lo_job(lo_task, 0)
    full_quality_job.completion_time = 2

    degraded_job = _build_lo_job(lo_task, 1)
    degraded_job.completion_time = 13
    degraded_job.is_degraded = True
    degraded_job.released_in_mode = SystemMode.HI
    degraded_job.service_quality_if_completed = 0.5
    degraded_job.runtime_budget_at_release = 3
    degraded_job.original_runtime_budget_at_release = 6
    degraded_job.actual_cost = 3
    degraded_job.original_actual_cost = 6

    result = SimulationResult(jobs=[full_quality_job, degraded_job])

    metrics = compute_lo_quality_weighted_metrics(result)

    assert metrics.lo_degraded_released == 1
    assert metrics.lo_degraded_completed == 1
    assert metrics.lo_degraded_completion_ratio == 0.5
    assert metrics.lo_degraded_release_ratio == 0.5
    assert metrics.lo_degraded_among_completed_ratio == 0.5
    assert metrics.lo_quality_qos == 0.75
    assert metrics.lo_equiv_jne == 0.5
    assert metrics.lo_equiv_jne_rate == 0.25


def test_compute_lo_quality_weighted_metrics_degraded_not_completed_is_zero_service() -> None:
    """degraded LO 若未按时完成，服务贡献必须强制记为 0。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=6, c_hi=6, criticality=Criticality.LO)
    job0 = _build_lo_job(lo_task, 0)
    job0.is_degraded = True
    job0.service_quality_if_completed = 0.5
    job0.dropped = True
    result = SimulationResult(
        jobs=[job0],
        job_cancellations=[
            JobCancellationEvent(
                cancel_time=4,
                task="LO1",
                release_index=0,
                executed_at_cancel=4,
                budget_at_cancel=3,
                reason=LO_LOSS_BUDGET_CANCELLATION,
            )
        ],
    )

    metrics = compute_lo_quality_weighted_metrics(result)

    assert metrics.lo_degraded_released == 1
    assert metrics.lo_degraded_completed == 0
    assert metrics.lo_degraded_cancelled == 1
    assert metrics.lo_degraded_not_completed == 1
    assert metrics.lo_zero_service_jobs == 1
    assert metrics.lo_quality_qos == 0.0
    assert metrics.lo_equiv_jne == 1.0


def test_compute_lo_quality_weighted_metrics_degraded_cancelled_accepts_legacy_tick_reason() -> None:
    """legacy tick runtime 的 cancellation reason 也应计入 degraded_cancelled。"""

    lo_task = Task(name="LO1", period=10, deadline=10, c_lo=6, c_hi=6, criticality=Criticality.LO)
    job0 = _build_lo_job(lo_task, 0)
    job0.is_degraded = True
    job0.released_in_mode = SystemMode.HI
    job0.service_quality_if_completed = 0.5
    job0.dropped = True
    result = SimulationResult(
        jobs=[job0],
        job_cancellations=[
            JobCancellationEvent(
                cancel_time=4,
                task="LO1",
                release_index=0,
                executed_at_cancel=4,
                budget_at_cancel=3,
                reason="lo_budget_overrun",
            )
        ],
    )

    metrics = compute_lo_quality_weighted_metrics(result)

    assert metrics.lo_degraded_released == 1
    assert metrics.lo_degraded_completed == 0
    assert metrics.lo_degraded_not_completed == 1
    assert metrics.lo_degraded_cancelled == 1
    assert metrics.lo_zero_service_ratio == 1.0
    assert metrics.lo_quality_qos == 0.0
    assert metrics.lo_equiv_jne == 1.0
