"""VIPER 候选树选择测试。"""

from __future__ import annotations

import pytest

from amc_py.viper.selection import (
    SelectionConfig,
    evaluate_candidate_gates,
    select_best_tree,
    _parse_finite_float,
    _parse_int_zero,
    _parse_rate,
)


def test_select_best_tree_rejects_deadline_miss_candidates() -> None:
    with pytest.raises(ValueError):
        select_best_tree(
            [
                {
                    "validation_deadline_misses_sum": 1,
                    "validation_hi_deadline_misses_sum": 0,
                    "validation_lo_deadline_misses_sum": 0,
                    "validation_tree_raw_top1_invalid_rate_mean": 0.0,
                    "validation_tree_fallback_rate_mean": 0.0,
                    "validation_lo_quality_qos_mean": 0.9,
                    "parent_lo_quality_qos": 0.1,
                    "tree_depth": 2,
                    "tree_node_count": 3,
                    "tree_leaf_count": 2,
                }
            ],
            config=SelectionConfig(),
        )


def test_select_best_tree_prefers_smaller_tree_within_98pct_qos_band() -> None:
    chosen = select_best_tree(
        [
            {
                "validation_deadline_misses_sum": 0,
                "validation_hi_deadline_misses_sum": 0,
                "validation_lo_deadline_misses_sum": 0,
                "validation_tree_raw_top1_invalid_rate_mean": 0.0,
                "validation_tree_fallback_rate_mean": 0.0,
                "validation_lo_quality_qos_mean": 1.0,
                "parent_lo_quality_qos": 0.1,
                "tree_depth": 5,
                "tree_node_count": 20,
                "tree_leaf_count": 10,
            },
            {
                "validation_deadline_misses_sum": 0,
                "validation_hi_deadline_misses_sum": 0,
                "validation_lo_deadline_misses_sum": 0,
                "validation_tree_raw_top1_invalid_rate_mean": 0.0,
                "validation_tree_fallback_rate_mean": 0.0,
                "validation_lo_quality_qos_mean": 0.99,
                "parent_lo_quality_qos": 0.1,
                "tree_depth": 2,
                "tree_node_count": 5,
                "tree_leaf_count": 3,
            },
        ],
        config=SelectionConfig(),
    )
    assert chosen["tree_depth"] == 2


# ========== Patch 4 新增测试 ==========


def test_none_safety_value_fails_closed_without_crash() -> None:
    """None 安全字段不应使 evaluate_candidate_gates 抛未处理异常。"""
    row = {
        "validation_lo_quality_qos_mean": 0.9,
        "validation_deadline_misses_sum": None,
        "validation_hi_deadline_misses_sum": 0,
        "validation_lo_deadline_misses_sum": 0,
        "validation_selected_invalid_mask_actions_sum": 0,
        "validation_formal_v1_mask_step_mismatch_count_sum": 0,
    }
    passed, reasons = evaluate_candidate_gates(row, SelectionConfig.performance_compatible())
    assert not passed
    assert any("deadline" in r for r in reasons)


def test_nan_safety_value_fails_closed_without_crash() -> None:
    """NaN 安全字段不应使 evaluate_candidate_gates 抛未处理异常。"""
    row = {
        "validation_lo_quality_qos_mean": float("nan"),
        "validation_deadline_misses_sum": 0,
        "validation_hi_deadline_misses_sum": 0,
        "validation_lo_deadline_misses_sum": 0,
        "validation_selected_invalid_mask_actions_sum": 0,
        "validation_formal_v1_mask_step_mismatch_count_sum": 0,
    }
    passed, reasons = evaluate_candidate_gates(row, SelectionConfig.performance_compatible())
    assert not passed
    assert "missing_or_nonfinite_qos" in reasons


def test_all_rejection_reasons_are_collected() -> None:
    """多个 safety 字段同时非法时，应收集全部拒绝原因而不是只报第一个。"""
    row = {
        "validation_lo_quality_qos_mean": 0.9,
        "validation_deadline_misses_sum": 1,
        "validation_hi_deadline_misses_sum": 1,
        "validation_lo_deadline_misses_sum": 1,
        "validation_selected_invalid_mask_actions_sum": 1,
        "validation_formal_v1_mask_step_mismatch_count_sum": 1,
    }
    passed, reasons = evaluate_candidate_gates(row, SelectionConfig.performance_compatible())
    assert not passed
    assert len(reasons) >= 5  # deadline_misses, hi, lo, selected_invalid, mismatch


def test_optional_rate_cli_rejects_out_of_range_value() -> None:
    """optional rate 值超出 [0,1] 时应产生 rejection reason。"""
    row = {
        "validation_lo_quality_qos_mean": 0.9,
        "validation_deadline_misses_sum": 0,
        "validation_hi_deadline_misses_sum": 0,
        "validation_lo_deadline_misses_sum": 0,
        "validation_selected_invalid_mask_actions_sum": 0,
        "validation_formal_v1_mask_step_mismatch_count_sum": 0,
        "validation_tree_raw_top1_invalid_rate_mean": 1.5,  # 超出 [0,1]
    }
    config = SelectionConfig.performance_compatible(max_invalid_rate=0.2)
    passed, reasons = evaluate_candidate_gates(row, config)
    assert not passed
    assert "raw_top1_invalid_rate" in str(reasons)


def test_parse_safe_functions() -> None:
    """安全解析函数应在非法输入时返回 None 而不抛异常。"""
    assert _parse_finite_float(None) is None
    assert _parse_finite_float("abc") is None
    assert _parse_finite_float(float("nan")) is None
    assert _parse_finite_float(1.5) == 1.5
    assert _parse_int_zero(None) is None
    assert _parse_int_zero("abc") is None
    assert _parse_int_zero(42) == 42
    assert _parse_rate(1.5) is None
    assert _parse_rate(-0.1) is None
    assert _parse_rate(0.5) == 0.5


def _assert_performance_compatible_accepts(field: str, value: float) -> None:
    """高 raw-invalid/fallback 本身不是安全失败：selected action 合法即可通过。"""
    row = {
        "validation_lo_quality_qos_mean": 0.9,
        "validation_deadline_misses_sum": 0,
        "validation_hi_deadline_misses_sum": 0,
        "validation_lo_deadline_misses_sum": 0,
        "validation_selected_invalid_mask_actions_sum": 0,
        "validation_formal_v1_mask_step_mismatch_count_sum": 0,
        "validation_tree_raw_top1_invalid_rate_mean": 0.0,
        "validation_tree_ranked_fallback_rate_mean": 0.0,
    }
    row[field] = value
    passed, reasons = evaluate_candidate_gates(row, SelectionConfig.performance_compatible())
    assert passed
    assert reasons == []


def test_performance_compatible_accepts_raw_invalid_80_percent() -> None:
    _assert_performance_compatible_accepts(
        "validation_tree_raw_top1_invalid_rate_mean", 0.8
    )


def test_performance_compatible_accepts_ranked_fallback_80_percent() -> None:
    _assert_performance_compatible_accepts(
        "validation_tree_ranked_fallback_rate_mean", 0.8
    )
