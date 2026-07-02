"""候选树选择逻辑。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """候选树筛选阈值。"""

    max_invalid_rate: float = 0.20
    max_fallback_rate: float = 0.50
    min_qos_retention: float = 0.0
    complexity_qos_ratio: float = 0.98
    metric_prefix: str = "validation_"
    qos_field: str = "validation_lo_quality_qos_mean"
    invalid_field: str = "validation_tree_raw_top1_invalid_rate_mean"
    fallback_field: str = "validation_tree_fallback_rate_mean"
    deadline_field: str = "validation_deadline_misses_sum"
    hi_deadline_field: str = "validation_hi_deadline_misses_sum"
    lo_deadline_field: str = "validation_lo_deadline_misses_sum"
    parent_qos_field: str = "parent_lo_quality_qos"
    retention_field: str = "qos_retention"


def select_best_tree(candidates: list[dict[str, object]], *, config: SelectionConfig) -> dict[str, object]:
    """按计划里的 gate 顺序挑选最优候选。"""

    if not candidates:
        raise ValueError("候选列表不能为空")
    gated = [
        row for row in candidates
        if int(row.get(config.deadline_field, 0)) == 0
        and int(row.get(config.hi_deadline_field, 0)) == 0
        and int(row.get(config.lo_deadline_field, 0)) == 0
        and float(row.get(config.invalid_field, 0.0)) <= config.max_invalid_rate
        and float(row.get(config.fallback_field, 0.0)) <= config.max_fallback_rate
        and float(row.get(config.qos_field, 0.0)) > float(row.get(config.parent_qos_field, float("-inf")))
        and (
            row.get(config.retention_field) is None
            or float(row.get(config.retention_field, 0.0)) >= config.min_qos_retention
        )
    ]
    if not gated:
        raise ValueError("没有候选通过 selection gate")
    best_qos = max(float(row.get(config.qos_field, 0.0)) for row in gated)
    complexity_band = [
        row for row in gated
        if float(row.get(config.qos_field, 0.0)) >= best_qos * config.complexity_qos_ratio
    ]
    complexity_band.sort(
        key=lambda row: (
            int(row.get("tree_depth", 10**9)),
            int(row.get("tree_node_count", 10**9)),
            int(row.get("tree_leaf_count", 10**9)),
            -float(row.get(config.qos_field, 0.0)),
        )
    )
    return complexity_band[0]
