"""候选树选择：安全性是硬 gate，fallback 频率只在兼容模式下作排序偏好。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


# ---- 安全解析函数：字段缺失、None、NaN、inf、非法字符串时不会抛未处理异常 ----

def _parse_finite_float(value: object, *, require_finite: bool = True) -> float | None:
    """安全解析为有限浮点数；None/NaN/inf/非法字符串返回 None。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    if require_finite and not math.isfinite(f):
        return None
    return f


def _parse_int_zero(value: object) -> int | None:
    """安全解析为整数；None/NaN/非法字符串返回 None。"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_rate(value: object) -> float | None:
    """安全解析为 [0, 1] 范围内的概率值；None/NaN/inf/非法/越界返回 None。"""
    f = _parse_finite_float(value, require_finite=True)
    if f is None:
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    selection_mode: str = "strict_top1_v1"
    max_invalid_rate: float | None = 0.20
    max_fallback_rate: float | None = 0.50
    max_no_valid_action_rate: float | None = None
    min_qos_retention: float | None = None
    complexity_qos_ratio: float = 0.98
    qos_field: str = "validation_lo_quality_qos_mean"
    invalid_field: str = "validation_tree_raw_top1_invalid_rate_mean"
    fallback_field: str = "validation_tree_ranked_fallback_rate_mean"
    no_valid_field: str = "validation_tree_no_valid_action_rate_mean"
    deadline_field: str = "validation_deadline_misses_sum"
    hi_deadline_field: str = "validation_hi_deadline_misses_sum"
    lo_deadline_field: str = "validation_lo_deadline_misses_sum"
    selected_invalid_field: str = "validation_selected_invalid_mask_actions_sum"
    mismatch_field: str = "validation_formal_v1_mask_step_mismatch_count_sum"

    @classmethod
    def strict_top1_v1(cls) -> "SelectionConfig":
        return cls(selection_mode="strict_top1_v1", max_invalid_rate=0.20, max_fallback_rate=0.50)

    @classmethod
    def performance_compatible(cls, **kwargs: object) -> "SelectionConfig":
        values = {"selection_mode": "performance_compatible", "max_invalid_rate": None, "max_fallback_rate": None, "max_no_valid_action_rate": None}
        values.update(kwargs)
        return cls(**values)


def evaluate_candidate_gates(row: dict[str, object], config: SelectionConfig) -> tuple[bool, list[str]]:
    """唯一的 gate 判断入口；selection 和报告必须共用它。

    使用安全解析函数：字段缺失、None、NaN、inf、非法字符串时
    不会抛未处理异常，而是添加明确的 rejection reason。
    QoS 错误不得阻止继续检查 deadline、selected-invalid 和 mismatch。
    """
    reasons: list[str] = []
    qos = _parse_finite_float(row.get(config.qos_field))
    if qos is None:
        reasons.append("missing_or_nonfinite_qos")

    if config.selection_mode == "performance_compatible":
        # performance_compatible：deadline/selected-invalid/mismatch 必须是确定有限整数
        deadline = _parse_int_zero(row.get(config.deadline_field))
        hi_dl = _parse_int_zero(row.get(config.hi_deadline_field))
        lo_dl = _parse_int_zero(row.get(config.lo_deadline_field))
        sel_invalid = _parse_int_zero(row.get(config.selected_invalid_field))
        mismatch = _parse_int_zero(row.get(config.mismatch_field))
        if deadline is None:
            reasons.append("missing_or_invalid:deadline_misses")
        if hi_dl is None:
            reasons.append("missing_or_invalid:hi_deadline_misses")
        if lo_dl is None:
            reasons.append("missing_or_invalid:lo_deadline_misses")
        if sel_invalid is None:
            reasons.append("missing_or_invalid:selected_invalid_mask_actions")
        if mismatch is None:
            reasons.append("missing_or_invalid:formal_v1_mask_step_mismatch")
        # 只有所有值都可解析时才做数值判断
        if all(v is not None for v in (deadline, hi_dl, lo_dl, sel_invalid, mismatch)):
            if deadline != 0:
                reasons.append("deadline_misses>0")
            if hi_dl != 0:
                reasons.append("hi_deadline_misses>0")
            if lo_dl != 0:
                reasons.append("lo_deadline_misses>0")
            if sel_invalid != 0:
                reasons.append("selected_invalid_mask_actions>0")
            if mismatch != 0:
                reasons.append("formal_v1_mask_step_mismatch>0")
    else:
        # strict_top1_v1：deadline 使用安全默认 0
        for field, label in ((config.deadline_field, "deadline_misses"), (config.hi_deadline_field, "hi_deadline_misses"), (config.lo_deadline_field, "lo_deadline_misses")):
            value = _parse_int_zero(row.get(field))
            if value is None:
                reasons.append(f"missing_or_invalid:{field}")
            elif value != 0:
                reasons.append(f"{label}>0")
        if config.max_invalid_rate is not None:
            invalid_rate = _parse_rate(row.get(config.invalid_field))
            if invalid_rate is None or invalid_rate > config.max_invalid_rate:
                reasons.append("raw_top1_invalid_rate_exceeded")
        fallback_value = _parse_rate(row.get(config.fallback_field, row.get("validation_tree_fallback_rate_mean")))
        if config.max_fallback_rate is not None and (fallback_value is None or fallback_value > config.max_fallback_rate):
            reasons.append("fallback_rate_exceeded")

    # optional rate 门限：仅在 performance_compatible 模式且显式指定 limit 时才判断
    optional = ((config.invalid_field, config.max_invalid_rate, "raw_top1_invalid_rate"), (config.fallback_field, config.max_fallback_rate, "ranked_fallback_rate"), (config.no_valid_field, config.max_no_valid_action_rate, "no_valid_action_rate"))
    if config.selection_mode == "performance_compatible":
        for field, limit, label in optional:
            if limit is None:
                continue
            rate_value = _parse_rate(row.get(field))
            if rate_value is None or rate_value > limit:
                reasons.append(f"{label}_exceeded")
    return not reasons, reasons


def select_best_tree(candidates: list[dict[str, object]], *, config: SelectionConfig) -> dict[str, object]:
    if not candidates:
        raise ValueError("候选列表不能为空")
    gated = []
    for row in candidates:
        passed, reasons = evaluate_candidate_gates(row, config)
        if passed:
            gated.append(row)
    if not gated:
        details = [evaluate_candidate_gates(row, config)[1] for row in candidates]
        raise ValueError(f"没有候选通过 selection gate: {details}")
    best_qos = max(float(row[config.qos_field]) for row in gated)
    band = [row for row in gated if float(row[config.qos_field]) >= best_qos * config.complexity_qos_ratio]
    band.sort(key=lambda row: (
        int(row.get("tree_depth", 10**9)), int(row.get("tree_node_count", 10**9)), int(row.get("tree_leaf_count", 10**9)),
        float(row.get(config.no_valid_field, 0.0)), float(row.get(config.fallback_field, row.get("validation_tree_fallback_rate_mean", 0.0))), float(row.get(config.invalid_field, 0.0)),
        -float(row.get("validation_lo_quality_qos_mean", 0.0)),
    ))
    return band[0]
