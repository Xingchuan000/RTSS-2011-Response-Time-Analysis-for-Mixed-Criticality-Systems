"""QoS pressure 任务集筛选相关公共函数。"""

from __future__ import annotations


EASY_MAX_LOSS = 0.10
MEDIUM_MAX_LOSS = 0.30
HARD_MAX_LOSS = 0.50


def classify_qos_pressure_bucket(lc_service_loss: float | None) -> str:
    """根据 baseline LC service loss 分类任务集压力桶。

    分类规则严格按实现计划：
    - easy: lc_service_loss < 0.10
    - medium: 0.10 <= lc_service_loss <= 0.30
    - hard: 0.30 < lc_service_loss <= 0.50
    - overloaded: lc_service_loss > 0.50
    - unknown: 无法解析或缺失
    """

    if lc_service_loss is None:
        return "unknown"

    value = float(lc_service_loss)
    if value < EASY_MAX_LOSS:
        return "easy"
    if value <= MEDIUM_MAX_LOSS:
        return "medium"
    if value <= HARD_MAX_LOSS:
        return "hard"
    return "overloaded"


def recommend_for_qos_dqn(
    *,
    amcrtb_schedulable: bool,
    baseline_hi_deadline_misses_sum: int | float | None,
    baseline_lc_service_loss_mean: float | None,
    baseline_released_lo_jobs_mean: float | None,
    baseline_cancelled_lo_jobs_mean: float | None,
    baseline_mode_changes_mean: float | None,
    static_sweep_relative_lc_loss_reduction: float | None = None,
    min_lc_service_loss: float = 0.10,
    max_lc_service_loss: float = 0.30,
    min_released_lo_jobs: float = 100.0,
    min_cancelled_lo_jobs: float = 10.0,
    min_mode_changes: float = 1.0,
    min_static_sweep_reduction: float | None = None,
) -> tuple[bool, str]:
    """基于固定门槛判断任务集是否推荐用于 QoS DQN 训练。

    返回 `(is_recommended, reason)`：
    - 当通过全部条件时返回 `(True, "ok")`
    - 否则返回 `(False, <稳定拒绝原因字符串>)`

    注意：
    - 判定顺序固定，便于 CSV 统计与可复现；
    - 不做任何兜底推断，缺失值视为不满足条件。
    """

    if not amcrtb_schedulable:
        return False, "not_schedulable"

    if baseline_hi_deadline_misses_sum is None or float(baseline_hi_deadline_misses_sum) > 0.0:
        return False, "hi_deadline_miss"

    if baseline_lc_service_loss_mean is None:
        return False, "loss_too_low"
    loss = float(baseline_lc_service_loss_mean)
    if loss < float(min_lc_service_loss):
        return False, "loss_too_low"
    if loss > float(max_lc_service_loss):
        return False, "loss_too_high"

    if baseline_released_lo_jobs_mean is None or float(baseline_released_lo_jobs_mean) < float(min_released_lo_jobs):
        return False, "insufficient_released_lo_jobs"

    if baseline_cancelled_lo_jobs_mean is None or float(baseline_cancelled_lo_jobs_mean) < float(min_cancelled_lo_jobs):
        return False, "insufficient_cancelled_lo_jobs"

    if baseline_mode_changes_mean is None or float(baseline_mode_changes_mean) < float(min_mode_changes):
        return False, "insufficient_mode_changes"

    if min_static_sweep_reduction is not None:
        if static_sweep_relative_lc_loss_reduction is None:
            return False, "insufficient_static_sweep_reduction"
        if float(static_sweep_relative_lc_loss_reduction) < float(min_static_sweep_reduction):
            return False, "insufficient_static_sweep_reduction"

    return True, "ok"
