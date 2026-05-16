"""统一的运行时服务质量指标计算 helper。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from amc_py.models import Criticality
from amc_py.runtime_models import SimulationResult


@dataclass(frozen=True)
class ServiceQualityMetrics:
    """封装单次运行时仿真的低关键任务服务质量指标。"""

    released_lo_jobs: int
    cancelled_lo_jobs: int
    completed_lo_jobs: int
    lo_deadline_misses: int
    hi_deadline_misses: int
    lc_service_loss: float
    lc_qos: float
    min_lc_service: float | None
    budget_adjust_count: int
    mean_abs_budget_change: float | None


def safe_relative_reduction(baseline: float, method: float) -> float | None:
    """安全计算相对损失下降比例。

    当 baseline 本身不发生 LC service loss 时，分母为 0，没有定义可比较的相对改善，
    因此第一版按计划文档要求返回 None，而不是人为制造比例值。
    """

    if baseline <= 0.0:
        return None
    return (baseline - method) / baseline


def compute_service_quality_metrics(
    result: SimulationResult,
    *,
    include_lo_deadline_misses: bool = False,
) -> ServiceQualityMetrics:
    """从统一的 `SimulationResult` 中提取 QoS 相关指标。

    第一版严格按计划文档实现：
    - `lc_service_loss` 默认仅以 LO cancellation 计入损失；
    - `lo_deadline_misses` 先单独统计，但默认不计入 `lc_service_loss`；
    - 若遇到无法从 `result.jobs` 反查 criticality 的 miss，保守计入 HI miss。
    """

    lo_jobs = [job for job in result.jobs if job.task.criticality is Criticality.LO]
    released_lo_jobs = len(lo_jobs)
    cancelled_lo_jobs = len(result.job_cancellations)
    completed_lo_jobs = sum(
        1
        for job in lo_jobs
        if job.completion_time is not None and not job.dropped
    )

    task_criticality = {job.task.name: job.task.criticality for job in result.jobs}
    lo_deadline_misses = 0
    hi_deadline_misses = 0
    for miss in result.deadline_misses:
        criticality = task_criticality.get(miss.task)
        if criticality is Criticality.LO:
            lo_deadline_misses += 1
        else:
            # unknown task 也保守按 HI miss 处理，避免把潜在安全问题低估为 LO 指标。
            hi_deadline_misses += 1

    loss_numerator = cancelled_lo_jobs
    if include_lo_deadline_misses:
        loss_numerator += lo_deadline_misses
    if released_lo_jobs > 0:
        lc_service_loss = float(loss_numerator) / float(released_lo_jobs)
    else:
        lc_service_loss = 0.0
    lc_qos = 1.0 - lc_service_loss

    per_task_released: dict[str, int] = {}
    per_task_cancelled: dict[str, int] = {}
    for job in lo_jobs:
        task_name = job.task.name
        per_task_released[task_name] = per_task_released.get(task_name, 0) + 1
    for cancellation in result.job_cancellations:
        task_name = cancellation.task
        per_task_cancelled[task_name] = per_task_cancelled.get(task_name, 0) + 1
    service_values: list[float] = []
    for task_name, released_count in per_task_released.items():
        if released_count <= 0:
            continue
        cancelled_count = per_task_cancelled.get(task_name, 0)
        service_values.append(1.0 - (float(cancelled_count) / float(released_count)))
    min_lc_service = min(service_values) if service_values else None

    budget_adjust_count = len(result.budget_update_events)
    budget_change_values: list[float] = []
    previous_budget_by_task: dict[str, int] = {}
    for job in result.jobs:
        previous_budget_by_task.setdefault(job.task.name, job.task.c_lo)
    for event in result.budget_update_events:
        for task_name, new_budget in event.updates.items():
            old_budget = previous_budget_by_task.get(task_name)
            if old_budget is None:
                continue
            budget_change_values.append(abs(float(new_budget) - float(old_budget)))
            previous_budget_by_task[task_name] = int(new_budget)
    mean_abs_budget_change = (
        sum(budget_change_values) / float(len(budget_change_values))
        if budget_change_values
        else None
    )

    return ServiceQualityMetrics(
        released_lo_jobs=released_lo_jobs,
        cancelled_lo_jobs=cancelled_lo_jobs,
        completed_lo_jobs=completed_lo_jobs,
        lo_deadline_misses=lo_deadline_misses,
        hi_deadline_misses=hi_deadline_misses,
        lc_service_loss=lc_service_loss,
        lc_qos=lc_qos,
        min_lc_service=min_lc_service,
        budget_adjust_count=budget_adjust_count,
        mean_abs_budget_change=mean_abs_budget_change,
    )


def service_metrics_to_row(
    metrics: ServiceQualityMetrics,
    prefix: str = "",
) -> dict[str, int | float | None]:
    """把服务质量指标转成扁平 row 字典。

    `prefix` 用于统一生成 `baseline_...` / `dqn_...` 等字段，避免训练期和评估期
    各自手写字段名导致口径漂移。
    """

    return {
        f"{prefix}released_lo_jobs": metrics.released_lo_jobs,
        f"{prefix}cancelled_lo_jobs": metrics.cancelled_lo_jobs,
        f"{prefix}completed_lo_jobs": metrics.completed_lo_jobs,
        f"{prefix}lo_deadline_misses": metrics.lo_deadline_misses,
        f"{prefix}hi_deadline_misses": metrics.hi_deadline_misses,
        f"{prefix}lc_service_loss": metrics.lc_service_loss,
        f"{prefix}lc_qos": metrics.lc_qos,
        f"{prefix}min_lc_service": metrics.min_lc_service,
        f"{prefix}budget_adjust_count": metrics.budget_adjust_count,
        f"{prefix}mean_abs_budget_change": metrics.mean_abs_budget_change,
    }


def mean_optional(rows: list[Mapping[str, object]], key: str) -> float | None:
    """对可空数字字段求均值。"""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / float(len(values))
