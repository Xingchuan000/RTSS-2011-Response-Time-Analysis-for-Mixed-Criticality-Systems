"""统一的 QoS 选模规则。"""

from __future__ import annotations

import json
from typing import Mapping


def _float_or_default(value: object, default: float) -> float:
    """把可空值安全转成 float。"""

    if value in (None, ""):
        return default
    return float(value)


def is_hi_safe(row: Mapping[str, object]) -> bool:
    """判断候选是否满足 HI 安全约束。"""

    hi_miss_value = row.get("hi_deadline_misses_sum")
    if hi_miss_value in (None, ""):
        hi_miss_value = row.get("deadline_misses_sum", 0)
    return int(float(hi_miss_value)) == 0


def is_conservative_qos_valid(row: Mapping[str, object]) -> bool:
    """保守 QoS：HI safe 且 mode_changes 不高于 baseline。"""

    return is_hi_safe(row) and float(row["mode_changes_mean"]) <= float(row["baseline_mode_changes_mean"])


def is_qos_stable_valid(row: Mapping[str, object], delta: float = 0.05) -> bool:
    """QoS-Stable：HI safe 且 mode_changes 仅允许轻微高于 baseline。"""

    return is_hi_safe(row) and float(row["mode_changes_mean"]) <= float(row["baseline_mode_changes_mean"]) * (
        1.0 + delta
    )


def is_qos_best_valid(row: Mapping[str, object]) -> bool:
    """QoS-Best：仅要求 HI safe。"""

    return is_hi_safe(row)


def is_lo_quality_qos_best_valid(row: Mapping[str, object]) -> bool:
    """LO quality QoS 选模：只要求满足 HI safety 硬约束。"""

    return is_hi_safe(row)


def lo_quality_qos_best_sort_key(
    row: Mapping[str, object],
) -> tuple[float, float, float, int]:
    """按跨 AMC-family 的 LO quality QoS 主指标选择 checkpoint。"""

    return (
        -float(row["lo_quality_qos_mean"]),
        float(row["mode_changes_mean"]),
        _float_or_default(row.get("mean_abs_budget_change_mean"), 0.0),
        int(float(row["episode"])),
    )


def is_zero_service_qos_valid(
    row: Mapping[str, object],
    *,
    mode_delta: float = 0.05,
) -> bool:
    """Zero-Service-QoS：HI safe 且 mode_changes 不超过 baseline 允许范围。"""

    if not is_hi_safe(row):
        return False
    baseline_mode = _float_or_default(row.get("baseline_mode_changes_mean"), float("inf"))
    dqn_mode = _float_or_default(row.get("mode_changes_mean"), float("inf"))
    return dqn_mode <= baseline_mode * (1.0 + mode_delta)


def _sum_json_counter(value: object) -> float:
    """把 JSON 计数字段安全聚合成总和。

    这里专门给 `policy_action_*_json` 一类字段使用，保持和计划文档一致：
    - 空值直接按 0 处理；
    - 字符串先尝试 JSON 解析；
    - 解析后只接受 Mapping；
    - 每个 value 尽量转成 float 累加。
    """

    if value in (None, ""):
        return 0.0
    if isinstance(value, Mapping):
        data = value
    else:
        try:
            data = json.loads(str(value))
        except Exception:
            return 0.0
    if not isinstance(data, Mapping):
        return 0.0
    total = 0.0
    for item in data.values():
        try:
            total += float(item)
        except (TypeError, ValueError):
            continue
    return total


def recovery_action_stats(row: Mapping[str, object]) -> dict[str, float]:
    """汇总 recovery 相关动作统计。

    这组统计只读验证结果里的 policy action 计数字段，不改变任何训练或环境语义。
    """

    selected_action_count = _sum_json_counter(row.get("policy_action_hist_json"))
    increase_count = _sum_json_counter(row.get("policy_action_is_increase_sum_json"))
    decrease_count = _sum_json_counter(row.get("policy_action_is_decrease_sum_json"))
    recovery_decrease_count = _sum_json_counter(row.get("policy_action_safe_recovery_decrease_sum_json"))
    over_increase_count = _sum_json_counter(row.get("policy_action_over_increase_sum_json"))

    denom = max(1.0, selected_action_count)
    return {
        "selected_action_count": selected_action_count,
        "increase_count": increase_count,
        "decrease_count": decrease_count,
        "recovery_decrease_count": recovery_decrease_count,
        "over_increase_count": over_increase_count,
        "increase_rate": increase_count / denom,
        "decrease_rate": decrease_count / denom,
        "recovery_decrease_rate": recovery_decrease_count / denom,
        "over_increase_rate": over_increase_count / denom,
    }


def is_qos_recovery_stable_valid(
    row: Mapping[str, object],
    *,
    mode_delta: float = 0.05,
    max_increase_rate: float = 0.90,
    min_recovery_decrease_rate: float = 0.03,
    max_over_increase_rate: float = 0.90,
    require_positive_qos: bool = True,
) -> bool:
    """QoS-Recovery-Stable：在 QoS stable 的基础上再约束恢复动作分布。"""

    if not is_qos_stable_valid(row, delta=mode_delta):
        return False

    if require_positive_qos:
        relative_lc_loss_reduction = _float_or_default(row.get("relative_lc_loss_reduction"), 0.0)
        if relative_lc_loss_reduction <= 0.0:
            return False

    stats = recovery_action_stats(row)
    if stats["selected_action_count"] <= 0.0:
        return False
    if stats["increase_rate"] > max_increase_rate:
        return False
    if stats["recovery_decrease_rate"] < min_recovery_decrease_rate:
        return False
    if stats["over_increase_rate"] > max_over_increase_rate:
        return False
    return True


def qos_recovery_stable_sort_key(row: Mapping[str, object]) -> tuple[float, float, float, float, float, int]:
    """QoS-Recovery-Stable 的排序 key。

    排序优先级按计划文档执行：
    1. `lc_service_loss_mean` 越低越好；
    2. `over_increase_rate` 越低越好；
    3. `increase_rate` 越低越好；
    4. `recovery_decrease_rate` 越高越好，因此取负；
    5. `mode_changes_mean` 越低越好；
    6. `episode` 越早越好。
    """

    stats = recovery_action_stats(row)
    return (
        float(row["lc_service_loss_mean"]),
        stats["over_increase_rate"],
        stats["increase_rate"],
        -stats["recovery_decrease_rate"],
        float(row["mode_changes_mean"]),
        int(float(row["episode"])),
    )


def qos_sort_key(row: Mapping[str, object]) -> tuple[float, float, float, float, int]:
    """QoS 选模排序 key。

    排序方向严格按计划文档：
    1. `lc_service_loss_mean` 越低越好；
    2. `min_lc_service_mean` 越高越好，因此取负；
    3. `mode_changes_mean` 越低越好；
    4. `budget_adjust_count_mean` 越低越好；
    5. `episode` 越早越好。
    """

    return (
        float(row["lc_service_loss_mean"]),
        -_float_or_default(row.get("min_lc_service_mean"), -1.0),
        float(row["mode_changes_mean"]),
        _float_or_default(row.get("budget_adjust_count_mean"), 0.0),
        int(float(row["episode"])),
    )


def zero_service_qos_sort_key(row: Mapping[str, object]) -> tuple[float, float, float, float, float, int]:
    """Zero-service 选模排序。

    排序优先级严格按计划文档：
    1. `lo_zero_service_ratio_mean` 越低越好；
    2. `lo_active_drop_rate_mean` 越低越好；
    3. `lo_budget_cancellation_rate_mean` 越低越好；
    4. `mode_changes_mean` 越低越好；
    5. `mean_abs_budget_change_mean` 越低越好；
    6. `episode` 越早越好。
    """

    return (
        _float_or_default(row.get("lo_zero_service_ratio_mean"), float("inf")),
        _float_or_default(row.get("lo_active_drop_rate_mean"), float("inf")),
        _float_or_default(row.get("lo_budget_cancellation_rate_mean"), float("inf")),
        _float_or_default(row.get("mode_changes_mean"), float("inf")),
        _float_or_default(row.get("mean_abs_budget_change_mean"), float("inf")),
        int(float(row.get("episode", 10**12))),
    )


def filter_by_best_type(
    rows: list[Mapping[str, object]],
    *,
    best_type: str,
    delta: float = 0.05,
) -> list[Mapping[str, object]]:
    """按 best_type 过滤候选 validation rows。"""

    if best_type == "legacy_lo_cancel":
        return [row for row in rows if int(float(row.get("deadline_misses_sum", 0))) == 0]
    if best_type == "conservative_qos":
        return [row for row in rows if is_conservative_qos_valid(row)]
    if best_type == "qos_stable":
        return [row for row in rows if is_qos_stable_valid(row, delta)]
    if best_type == "qos_best":
        return [row for row in rows if is_qos_best_valid(row)]
    if best_type == "lo_quality_qos_best":
        return [row for row in rows if is_lo_quality_qos_best_valid(row)]
    if best_type == "qos_recovery_stable":
        return [row for row in rows if is_qos_recovery_stable_valid(row, mode_delta=delta)]
    if best_type == "zero_service_qos":
        return [row for row in rows if is_zero_service_qos_valid(row, mode_delta=delta)]
    raise ValueError(f"Unsupported best_type: {best_type}")


def best_row_for_type(
    rows: list[Mapping[str, object]],
    *,
    best_type: str,
    delta: float = 0.05,
) -> Mapping[str, object] | None:
    """选择指定 best_type 的最佳候选。"""

    candidates = filter_by_best_type(rows, best_type=best_type, delta=delta)
    if not candidates:
        return None
    if best_type == "legacy_lo_cancel":
        return min(candidates, key=lambda row: float(row["lo_cancellations_mean"]))
    if best_type == "qos_recovery_stable":
        return min(candidates, key=qos_recovery_stable_sort_key)
    if best_type == "lo_quality_qos_best":
        return min(candidates, key=lo_quality_qos_best_sort_key)
    if best_type == "zero_service_qos":
        return min(candidates, key=zero_service_qos_sort_key)
    return min(candidates, key=qos_sort_key)
