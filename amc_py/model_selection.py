"""统一的 QoS 选模规则。"""

from __future__ import annotations

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
    return min(candidates, key=qos_sort_key)
