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
    min_stable_static_sweep_reduction: float = 0.0,
    stable_static_delta: float = 0.05,
    allow_relaxed_stable: bool = True,
    exclude_tradeoff_only: bool = False,
    stable005_static_relative_lc_loss_reduction: float | None = None,
    stable010_static_relative_lc_loss_reduction: float | None = None,
    tradeoff_only_flag_005: bool | None = None,
    tradeoff_only_flag_010: bool | None = None,
    single_stable005_static_relative_lc_loss_reduction: float | None = None,
    single_stable010_static_relative_lc_loss_reduction: float | None = None,
    min_single_stable_static_sweep_reduction: float = 0.0,
    require_single_action_improvement: bool = False,
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

    # 稳定改进约束：当用户显式要求稳定静态改进阈值时，优先按 5% 限制判定；
    # 若允许放宽，则在 5% 不满足时再尝试 10% 限制。
    if float(min_stable_static_sweep_reduction) > 0.0:
        stable005 = None if stable005_static_relative_lc_loss_reduction is None else float(stable005_static_relative_lc_loss_reduction)
        stable010 = None if stable010_static_relative_lc_loss_reduction is None else float(stable010_static_relative_lc_loss_reduction)
        primary_is_005 = abs(float(stable_static_delta) - 0.05) < 1e-12
        primary_value = stable005 if primary_is_005 else stable010
        if primary_value is not None and primary_value >= float(min_stable_static_sweep_reduction):
            pass
        elif (
            allow_relaxed_stable
            and primary_is_005
            and stable010 is not None
            and stable010 >= float(min_stable_static_sweep_reduction)
        ):
            pass
        else:
            return False, "insufficient_stable_static_sweep_reduction"

    # trade-off-only 排除：显式要求时，一旦命中任一对应标记就拒绝。
    if exclude_tradeoff_only:
        if bool(tradeoff_only_flag_005) or bool(tradeoff_only_flag_010):
            return False, "tradeoff_only"

    if require_single_action_improvement:
        single005 = None if single_stable005_static_relative_lc_loss_reduction is None else float(single_stable005_static_relative_lc_loss_reduction)
        single010 = None if single_stable010_static_relative_lc_loss_reduction is None else float(single_stable010_static_relative_lc_loss_reduction)
        if single005 is not None and single005 >= float(min_single_stable_static_sweep_reduction):
            pass
        elif allow_relaxed_stable and single010 is not None and single010 >= float(min_single_stable_static_sweep_reduction):
            pass
        else:
            return False, "insufficient_single_action_stable_improvement"

    return True, "ok"


def classify_improvement_type(
    *,
    static_qos_best_found_valid: bool | None,
    static_qos_best_relative_lc_loss_reduction: float | None,
    stable005_static_relative_lc_loss_reduction: float | None,
    stable010_static_relative_lc_loss_reduction: float | None,
) -> str:
    """把任务集的静态改进形态分类为稳定改进或 trade-off-only 等类别。"""

    if not bool(static_qos_best_found_valid):
        return "no_static_improvement"

    static_best = (
        0.0 if static_qos_best_relative_lc_loss_reduction is None else float(static_qos_best_relative_lc_loss_reduction)
    )
    stable005 = 0.0 if stable005_static_relative_lc_loss_reduction is None else float(stable005_static_relative_lc_loss_reduction)
    stable010 = 0.0 if stable010_static_relative_lc_loss_reduction is None else float(stable010_static_relative_lc_loss_reduction)

    if stable005 >= 0.02:
        return "stable005_improvable"
    if stable010 >= 0.02:
        return "stable010_improvable"
    if static_best >= 0.05:
        return "tradeoff_only"
    return "weak_or_no_improvement"


def classify_single_improvement_type(
    *,
    single_static_qos_best_found_valid: bool | None,
    single_static_qos_best_relative_lc_loss_reduction: float | None,
    single_stable005_static_relative_lc_loss_reduction: float | None,
    single_stable010_static_relative_lc_loss_reduction: float | None,
) -> str:
    """把 single-action 代理改进形态分类。"""

    if not bool(single_static_qos_best_found_valid):
        return "single_no_improvement"
    single_best = 0.0 if single_static_qos_best_relative_lc_loss_reduction is None else float(single_static_qos_best_relative_lc_loss_reduction)
    stable005 = 0.0 if single_stable005_static_relative_lc_loss_reduction is None else float(single_stable005_static_relative_lc_loss_reduction)
    stable010 = 0.0 if single_stable010_static_relative_lc_loss_reduction is None else float(single_stable010_static_relative_lc_loss_reduction)
    if stable005 >= 0.01:
        return "single_stable005_improvable"
    if stable010 >= 0.01:
        return "single_stable010_improvable"
    if single_best >= 0.05:
        return "single_tradeoff_only"
    return "single_weak_or_no_improvement"
