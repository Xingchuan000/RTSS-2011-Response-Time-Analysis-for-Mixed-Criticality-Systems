"""q-AMC-specific LO job loss accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from amc_py.models import Criticality
from amc_py.runtime_models import SimulationResult


@dataclass(frozen=True, slots=True)
class QAmcLossMetrics:
    released_lo_jobs: int
    completed_positive_quality_jobs: int
    overrun_stopped_zero_quality_jobs: int
    deadline_lost_zero_quality_jobs: int
    min_threshold_fallback_zero_quality_jobs: int
    hi_mode_discard_zero_quality_jobs: int
    other_zero_quality_jobs: int


def compute_qamc_loss_metrics(result: SimulationResult) -> QAmcLossMetrics:
    jobs = [
        job
        for job in result.jobs
        if job.task.criticality is Criticality.LO
        and job.qamc_target_runtime_level_at_release is not None
    ]
    completed_positive = 0
    overrun_stopped = 0
    deadline_lost = 0
    min_threshold_fallback = 0
    hi_mode_discard = 0
    other = 0
    min_exhaustions = {
        (event.task, event.trigger_release_index)
        for event in result.qamc_min_quality_exhaustions
    }
    for job in jobs:
        provided = float(job.qamc_provided_normalized_quality or 0.0)
        if job.completion_time is not None and provided > 0.0:
            completed_positive += 1
        elif job.qamc_stopped_by_overrun:
            if (job.task.name, job.release_index) in min_exhaustions:
                min_threshold_fallback += 1
            else:
                overrun_stopped += 1
        elif job.qamc_result_lost_due_to_deadline:
            deadline_lost += 1
        elif job.dropped:
            hi_mode_discard += 1
        else:
            other += 1
    metrics = QAmcLossMetrics(
        released_lo_jobs=len(jobs),
        completed_positive_quality_jobs=completed_positive,
        overrun_stopped_zero_quality_jobs=overrun_stopped,
        deadline_lost_zero_quality_jobs=deadline_lost,
        min_threshold_fallback_zero_quality_jobs=min_threshold_fallback,
        hi_mode_discard_zero_quality_jobs=hi_mode_discard,
        other_zero_quality_jobs=other,
    )
    accounted = sum(
        value for key, value in asdict(metrics).items() if key != "released_lo_jobs"
    )
    if accounted != metrics.released_lo_jobs:
        raise AssertionError(
            f"QAMC_LOSS_CONSERVATION_FAILED:{accounted}:{metrics.released_lo_jobs}"
        )
    return metrics


def qamc_loss_metrics_to_row(
    metrics: QAmcLossMetrics,
    prefix: str = "qamc_loss_",
) -> dict[str, int]:
    return {f"{prefix}{name}": int(value) for name, value in asdict(metrics).items()}


__all__ = [
    "QAmcLossMetrics",
    "compute_qamc_loss_metrics",
    "qamc_loss_metrics_to_row",
]
