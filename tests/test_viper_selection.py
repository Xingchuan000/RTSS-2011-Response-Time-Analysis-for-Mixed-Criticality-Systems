"""VIPER 候选树选择测试。"""

from __future__ import annotations

import pytest

from amc_py.viper.selection import SelectionConfig, select_best_tree


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
