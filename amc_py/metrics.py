"""统一的运行时服务质量指标计算 helper。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from amc_py.models import Criticality
from amc_py.runtime_models import (
    Job,
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_BUDGET_CANCELLATION,
    LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
    SimulationResult,
)


LO_BUDGET_CANCELLATION_REASONS = {
    LO_LOSS_BUDGET_CANCELLATION,
    "lo_budget_overrun",
    "lo_budget_overrun_standard_amc",
}


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


@dataclass(frozen=True)
class RuntimeDegradationMetrics:
    """论文风格的 degraded mode 统计指标。"""

    hdm: int
    jne: int
    ldm: int
    nid: int
    tid: int
    total_time: int
    tid_ratio: float
    nid_per_1e6_time: float
    mean_degraded_interval: float | None
    safety_feasible: int


@dataclass(frozen=True)
class LoJobLossBreakdownMetrics:
    """LO job loss 的原因级拆分指标。"""

    lo_job_losses_total: int
    lo_budget_cancellations: int
    lo_release_dropped_in_degraded_mode: int
    lo_active_dropped_on_mode_switch: int
    jne_residual_not_in_cancellations: int
    active_drop_share_of_jne: float | None


@dataclass(frozen=True)
class LoQualityWeightedMetrics:
    """LO 任务按服务质量加权后的指标集合。"""

    lo_equiv_jne: float
    lo_equiv_jne_rate: float
    lo_quality_qos: float
    lo_quality_loss: float
    lo_full_quality_completed: int
    lo_full_quality_ratio: float
    lo_degraded_released: int
    lo_degraded_completed: int
    lo_degraded_cancelled: int
    lo_degraded_deadline_missed: int
    lo_degraded_not_completed: int
    lo_degraded_release_ratio: float
    lo_degraded_completion_ratio: float
    lo_degraded_among_completed_ratio: float | None
    lo_degraded_quality_sum: float
    lo_degraded_budget_sum: float
    lo_degraded_original_budget_sum: float
    lo_degraded_budget_ratio_mean: float | None
    lo_degraded_exec_time_sum: float
    lo_degraded_exec_time_ratio: float | None
    lo_zero_service_jobs: int
    lo_zero_service_ratio: float
    lo_full_quality_service_sum: float
    lo_total_service_sum: float


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


def compute_degraded_intervals(result: SimulationResult) -> list[tuple[int, int]]:
    """按 mode switch / recovery 序列还原 degraded mode 的时间区间。"""

    intervals: list[tuple[int, int]] = []
    recovery_index = 0
    recoveries = result.mode_recoveries

    for switch in result.mode_switches:
        while (
            recovery_index < len(recoveries)
            and recoveries[recovery_index].recovery_time < switch.switch_time
        ):
            recovery_index += 1
        if recovery_index < len(recoveries):
            end = recoveries[recovery_index].recovery_time
            recovery_index += 1
        else:
            end = result.end_time
        intervals.append((switch.switch_time, end))
    return intervals


def compute_runtime_degradation_metrics(result: SimulationResult) -> RuntimeDegradationMetrics:
    """计算 AMC-RA / AMC-RH baseline 对照所需的 degraded mode 指标。"""

    task_criticality = {job.task.name: job.task.criticality for job in result.jobs}
    hdm = sum(
        1
        for miss in result.deadline_misses
        if task_criticality.get(miss.task) is Criticality.HI
    )
    ldm = sum(
        1
        for miss in result.deadline_misses
        if task_criticality.get(miss.task) is Criticality.LO
    )
    jne = sum(
        1
        for job in result.jobs
        if job.task.criticality is Criticality.LO
        and job.dropped
        and job.completion_time is None
    )
    intervals = compute_degraded_intervals(result)
    tid = sum(end - start for start, end in intervals)
    total_time = result.end_time
    tid_ratio = float(tid) / float(total_time) if total_time > 0 else 0.0
    nid_per_1e6_time = (
        float(len(result.mode_switches)) / float(total_time) * 1_000_000.0
        if total_time > 0
        else 0.0
    )
    mean_degraded_interval = (
        None
        if len(result.mode_switches) == 0
        else float(tid) / float(len(result.mode_switches))
    )
    safety_feasible = 1 if hdm == 0 else 0
    return RuntimeDegradationMetrics(
        hdm=hdm,
        jne=jne,
        ldm=ldm,
        nid=len(result.mode_switches),
        tid=tid,
        total_time=total_time,
        tid_ratio=tid_ratio,
        nid_per_1e6_time=nid_per_1e6_time,
        mean_degraded_interval=mean_degraded_interval,
        safety_feasible=safety_feasible,
    )


def _lo_deadline_miss_keys(result: SimulationResult) -> set[tuple[str, int]]:
    """提取所有 LO deadline miss 的 job 键。"""

    task_criticality = {job.task.name: job.task.criticality for job in result.jobs}
    return {
        (miss.task, miss.release_index)
        for miss in result.deadline_misses
        if task_criticality.get(miss.task) is Criticality.LO
    }


def lo_job_service_quality(
    job: Job,
    *,
    lo_deadline_miss_keys: set[tuple[str, int]],
    terminal: bool,
) -> float | None:
    """返回已终结 LO job 的服务质量；未终结时返回 ``None``。

    该判定是质量加权最终指标与 interval reward tracker 的共享口径。
    """

    key = (job.task.name, job.release_index)
    if job.completion_time is not None:
        if (
            job.completion_time <= job.absolute_deadline
            and not job.dropped
            and key not in lo_deadline_miss_keys
        ):
            return max(0.0, min(1.0, float(job.service_quality_if_completed)))
        return 0.0

    if job.dropped or key in lo_deadline_miss_keys:
        return 0.0
    if terminal:
        return 0.0
    return None


def _loss_reason_by_job_key(result: SimulationResult) -> dict[tuple[str, int], str]:
    """按 job 键汇总 LO loss / cancellation 的原因。"""

    reasons: dict[tuple[str, int], str] = {}
    for loss in result.lo_job_losses:
        reasons[(loss.task, loss.release_index)] = loss.reason
    for cancellation in result.job_cancellations:
        reasons.setdefault((cancellation.task, cancellation.release_index), cancellation.reason)
    return reasons


def compute_lo_quality_weighted_metrics(result: SimulationResult) -> LoQualityWeightedMetrics:
    """计算 LO 任务的质量加权服务指标。

    旧口径 `lc_service_loss / lc_qos` 仍保持“取消即损失”的 binary 解释；
    这里新增的是与其并行存在的 quality-weighted 口径，用于单独观察
    degraded LO completion 给低关键任务侧带来的部分服务贡献。
    """

    lo_jobs = [job for job in result.jobs if job.task.criticality is Criticality.LO]
    released_lo_jobs = len(lo_jobs)
    lo_miss_keys = _lo_deadline_miss_keys(result)
    loss_reason_by_key = _loss_reason_by_job_key(result)

    lo_equiv_jne = 0.0
    lo_full_quality_completed = 0
    lo_degraded_released = 0
    lo_degraded_completed = 0
    lo_degraded_cancelled = 0
    lo_degraded_deadline_missed = 0
    lo_degraded_not_completed = 0
    lo_degraded_quality_sum = 0.0
    lo_degraded_budget_sum = 0.0
    lo_degraded_original_budget_sum = 0.0
    lo_degraded_exec_time_sum = 0.0
    lo_zero_service_jobs = 0
    lo_full_quality_service_sum = 0.0
    lo_total_service_sum = 0.0
    total_lo_exec_time_sum = 0.0
    degraded_budget_ratios: list[float] = []
    completed_lo_jobs = 0

    for job in lo_jobs:
        key = (job.task.name, job.release_index)
        service = lo_job_service_quality(
            job,
            lo_deadline_miss_keys=lo_miss_keys,
            terminal=True,
        )
        is_completed_on_time = (
            service is not None
            and job.completion_time is not None
            and job.completion_time <= job.absolute_deadline
            and not job.dropped
            and key not in lo_miss_keys
        )
        if is_completed_on_time:
            completed_lo_jobs += 1
        # terminal=True guarantees a concrete value for every released LO job.
        assert service is not None

        lo_equiv_jne += 1.0 - service
        lo_total_service_sum += service
        total_lo_exec_time_sum += float(job.executed_time)

        if service == 0.0:
            lo_zero_service_jobs += 1

        if is_completed_on_time and not job.is_degraded:
            lo_full_quality_completed += 1
            lo_full_quality_service_sum += service

        if job.is_degraded:
            lo_degraded_released += 1
            degraded_budget = float(job.runtime_budget_at_release or 0)
            original_budget = float(job.original_runtime_budget_at_release or job.task.c_lo)
            lo_degraded_budget_sum += degraded_budget
            lo_degraded_original_budget_sum += original_budget
            lo_degraded_exec_time_sum += float(job.executed_time)
            if original_budget > 0.0:
                degraded_budget_ratios.append(degraded_budget / original_budget)
            if is_completed_on_time:
                lo_degraded_completed += 1
                lo_degraded_quality_sum += service
            else:
                lo_degraded_not_completed += 1
                # tick runtime 旧路径仍可能输出 legacy cancellation reason，
                # 这里在 metrics 层兼容这些既有字符串，避免 degraded_cancelled
                # 统计只在 event runtime 下正确、在 tick runtime 下被低估。
                if loss_reason_by_key.get(key) in LO_BUDGET_CANCELLATION_REASONS:
                    lo_degraded_cancelled += 1
                if key in lo_miss_keys:
                    lo_degraded_deadline_missed += 1

    lo_quality_qos = (
        lo_total_service_sum / float(released_lo_jobs)
        if released_lo_jobs > 0
        else 1.0
    )
    lo_quality_loss = 1.0 - lo_quality_qos
    lo_equiv_jne_rate = (
        lo_equiv_jne / float(released_lo_jobs)
        if released_lo_jobs > 0
        else 0.0
    )

    return LoQualityWeightedMetrics(
        lo_equiv_jne=lo_equiv_jne,
        lo_equiv_jne_rate=lo_equiv_jne_rate,
        lo_quality_qos=lo_quality_qos,
        lo_quality_loss=lo_quality_loss,
        lo_full_quality_completed=lo_full_quality_completed,
        lo_full_quality_ratio=(
            float(lo_full_quality_completed) / float(released_lo_jobs)
            if released_lo_jobs > 0
            else 0.0
        ),
        lo_degraded_released=lo_degraded_released,
        lo_degraded_completed=lo_degraded_completed,
        lo_degraded_cancelled=lo_degraded_cancelled,
        lo_degraded_deadline_missed=lo_degraded_deadline_missed,
        lo_degraded_not_completed=lo_degraded_not_completed,
        lo_degraded_release_ratio=(
            float(lo_degraded_released) / float(released_lo_jobs)
            if released_lo_jobs > 0
            else 0.0
        ),
        lo_degraded_completion_ratio=(
            float(lo_degraded_completed) / float(released_lo_jobs)
            if released_lo_jobs > 0
            else 0.0
        ),
        lo_degraded_among_completed_ratio=(
            float(lo_degraded_completed) / float(completed_lo_jobs)
            if completed_lo_jobs > 0
            else None
        ),
        lo_degraded_quality_sum=lo_degraded_quality_sum,
        lo_degraded_budget_sum=lo_degraded_budget_sum,
        lo_degraded_original_budget_sum=lo_degraded_original_budget_sum,
        lo_degraded_budget_ratio_mean=(
            sum(degraded_budget_ratios) / float(len(degraded_budget_ratios))
            if degraded_budget_ratios
            else None
        ),
        lo_degraded_exec_time_sum=lo_degraded_exec_time_sum,
        lo_degraded_exec_time_ratio=(
            lo_degraded_exec_time_sum / float(total_lo_exec_time_sum)
            if total_lo_exec_time_sum > 0.0
            else None
        ),
        lo_zero_service_jobs=lo_zero_service_jobs,
        lo_zero_service_ratio=(
            float(lo_zero_service_jobs) / float(released_lo_jobs)
            if released_lo_jobs > 0
            else 0.0
        ),
        lo_full_quality_service_sum=lo_full_quality_service_sum,
        lo_total_service_sum=lo_total_service_sum,
    )


def compute_lo_job_loss_breakdown_metrics(
    result: SimulationResult,
    degradation: RuntimeDegradationMetrics | None = None,
) -> LoJobLossBreakdownMetrics:
    """计算 LO job loss 的 reason-level breakdown。

    新结果优先读取 `result.lo_job_losses`。
    当读取到旧结果对象时，允许从 `job_cancellations` 与 `JNE` 做最小兼容回推，
    但该 fallback 仅用于旧数据兼容，不作为新口径主来源。
    """

    if degradation is None:
        degradation = compute_runtime_degradation_metrics(result)

    losses = result.lo_job_losses
    if losses:
        lo_budget_cancellations = sum(
            1 for loss in losses if loss.reason == LO_LOSS_BUDGET_CANCELLATION
        )
        lo_release_dropped_in_degraded_mode = sum(
            1
            for loss in losses
            if loss.reason == LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE
        )
        lo_active_dropped_on_mode_switch = sum(
            1
            for loss in losses
            if loss.reason == LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH
        )
        lo_job_losses_total = len(losses)
    else:
        lo_budget_cancellations = sum(
            1
            for cancellation in result.job_cancellations
            if cancellation.reason
            in {
                "lo_budget_overrun",
                "lo_budget_overrun_standard_amc",
                LO_LOSS_BUDGET_CANCELLATION,
            }
        )
        lo_release_dropped_in_degraded_mode = sum(
            1
            for cancellation in result.job_cancellations
            if cancellation.reason == LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE
        )
        lo_active_dropped_on_mode_switch = max(0, degradation.jne - len(result.job_cancellations))
        lo_job_losses_total = (
            lo_budget_cancellations
            + lo_release_dropped_in_degraded_mode
            + lo_active_dropped_on_mode_switch
        )

    jne_residual_not_in_cancellations = max(0, degradation.jne - len(result.job_cancellations))
    active_drop_share_of_jne = (
        None
        if degradation.jne == 0
        else lo_active_dropped_on_mode_switch / degradation.jne
    )
    return LoJobLossBreakdownMetrics(
        lo_job_losses_total=lo_job_losses_total,
        lo_budget_cancellations=lo_budget_cancellations,
        lo_release_dropped_in_degraded_mode=lo_release_dropped_in_degraded_mode,
        lo_active_dropped_on_mode_switch=lo_active_dropped_on_mode_switch,
        jne_residual_not_in_cancellations=jne_residual_not_in_cancellations,
        active_drop_share_of_jne=active_drop_share_of_jne,
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


def lo_job_loss_breakdown_to_row(
    metrics: LoJobLossBreakdownMetrics,
    prefix: str = "",
) -> dict[str, int | float | None]:
    """把 reason-level LO loss 指标展平成 row。"""

    return {
        f"{prefix}lo_job_losses_total": metrics.lo_job_losses_total,
        f"{prefix}lo_budget_cancellations": metrics.lo_budget_cancellations,
        f"{prefix}lo_release_dropped_in_degraded_mode": metrics.lo_release_dropped_in_degraded_mode,
        f"{prefix}lo_active_dropped_on_mode_switch": metrics.lo_active_dropped_on_mode_switch,
        f"{prefix}jne_residual_not_in_cancellations": metrics.jne_residual_not_in_cancellations,
        f"{prefix}active_drop_share_of_jne": metrics.active_drop_share_of_jne,
    }


def lo_quality_weighted_metrics_to_row(
    metrics: LoQualityWeightedMetrics,
    prefix: str = "",
) -> dict[str, int | float | None]:
    """把质量加权 LO 指标展平成 CSV row。"""

    return {
        f"{prefix}lo_equiv_jne": metrics.lo_equiv_jne,
        f"{prefix}lo_equiv_jne_rate": metrics.lo_equiv_jne_rate,
        f"{prefix}lo_quality_qos": metrics.lo_quality_qos,
        f"{prefix}lo_quality_loss": metrics.lo_quality_loss,
        f"{prefix}lo_full_quality_completed": metrics.lo_full_quality_completed,
        f"{prefix}lo_full_quality_ratio": metrics.lo_full_quality_ratio,
        f"{prefix}lo_degraded_released": metrics.lo_degraded_released,
        f"{prefix}lo_degraded_completed": metrics.lo_degraded_completed,
        f"{prefix}lo_degraded_cancelled": metrics.lo_degraded_cancelled,
        f"{prefix}lo_degraded_deadline_missed": metrics.lo_degraded_deadline_missed,
        f"{prefix}lo_degraded_not_completed": metrics.lo_degraded_not_completed,
        f"{prefix}lo_degraded_release_ratio": metrics.lo_degraded_release_ratio,
        f"{prefix}lo_degraded_completion_ratio": metrics.lo_degraded_completion_ratio,
        f"{prefix}lo_degraded_among_completed_ratio": metrics.lo_degraded_among_completed_ratio,
        f"{prefix}lo_degraded_quality_sum": metrics.lo_degraded_quality_sum,
        f"{prefix}lo_degraded_budget_sum": metrics.lo_degraded_budget_sum,
        f"{prefix}lo_degraded_original_budget_sum": metrics.lo_degraded_original_budget_sum,
        f"{prefix}lo_degraded_budget_ratio_mean": metrics.lo_degraded_budget_ratio_mean,
        f"{prefix}lo_degraded_exec_time_sum": metrics.lo_degraded_exec_time_sum,
        f"{prefix}lo_degraded_exec_time_ratio": metrics.lo_degraded_exec_time_ratio,
        f"{prefix}lo_zero_service_jobs": metrics.lo_zero_service_jobs,
        f"{prefix}lo_zero_service_ratio": metrics.lo_zero_service_ratio,
        f"{prefix}lo_full_quality_service_sum": metrics.lo_full_quality_service_sum,
        f"{prefix}lo_total_service_sum": metrics.lo_total_service_sum,
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
