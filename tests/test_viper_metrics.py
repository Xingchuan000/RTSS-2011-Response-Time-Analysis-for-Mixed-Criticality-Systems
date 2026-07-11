"""VIPER metrics 与 summary 测试。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from amc_py.viper.metrics import compute_offline_tree_metrics, retention_higher_is_better, retention_lower_is_better
from amc_py.viper.dataset import ViperSample
from amc_py.viper.tree_policy import TreeBudgetPolicy
from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.integer_tree import compile_sklearn_tree_to_integer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_retention_directions_are_correct() -> None:
    assert retention_higher_is_better(0.2, 0.6, 0.4) == pytest.approx(0.5)
    assert retention_lower_is_better(10.0, 4.0, 7.0) == pytest.approx(0.5)
    assert retention_higher_is_better(0.5, 0.5, 0.5) is None
    assert retention_lower_is_better(3.0, 3.5, 3.2) is None


def test_summarize_viper_results_writes_all_required_outputs(tmp_path: Path) -> None:
    eval_csv = tmp_path / "eval.csv"
    rows = [
        {
            "seed": "0",
            "method": "c_amc_sem_baseline",
            "lo_quality_qos": "0.4",
            "lo_zero_service_ratio": "0.4",
            "lo_equiv_jne": "10.0",
            "tree_depth": "",
            "tree_node_count": "",
            "tree_leaf_count": "",
            "tree_raw_top1_invalid_rate": "",
            "tree_fallback_rate": "",
            "deadline_misses": "0",
            "hi_deadline_misses": "0",
            "lo_deadline_misses": "0",
        },
        {
            "seed": "0",
            "method": "dqn_agent",
            "lo_quality_qos": "0.8",
            "lo_zero_service_ratio": "0.1",
            "lo_equiv_jne": "4.0",
            "tree_depth": "",
            "tree_node_count": "",
            "tree_leaf_count": "",
            "tree_raw_top1_invalid_rate": "",
            "tree_fallback_rate": "",
            "deadline_misses": "0",
            "hi_deadline_misses": "0",
            "lo_deadline_misses": "0",
        },
        {
            "seed": "0",
            "method": "viper_tree_agent",
            "lo_quality_qos": "0.7",
            "lo_zero_service_ratio": "0.2",
            "lo_equiv_jne": "6.0",
            "tree_depth": "2",
            "tree_node_count": "5",
            "tree_leaf_count": "3",
            "tree_raw_top1_invalid_rate": "0.1",
            "tree_fallback_rate": "0.2",
            "deadline_misses": "0",
            "hi_deadline_misses": "0",
            "lo_deadline_misses": "0",
        },
    ]
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_dir = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_viper_results.py",
            "--eval-csv",
            str(eval_csv),
            "--parent-method",
            "c_amc_sem_baseline",
            "--teacher-method",
            "dqn_agent",
            "--tree-method",
            "viper_tree_agent",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    assert (output_dir / "per_seed_tree_vs_parent_vs_dqn.csv").exists()
    assert (output_dir / "retention_summary.csv").exists()
    assert (output_dir / "tree_complexity_summary.csv").exists()
    assert (output_dir / "tree_safety_summary.csv").exists()


def test_summarize_viper_results_fails_when_methods_are_missing_by_default(tmp_path: Path) -> None:
    eval_csv = tmp_path / "eval_missing.csv"
    rows = [{"seed": "0", "method": "dqn_agent", "lo_quality_qos": "0.8", "lo_zero_service_ratio": "0.1", "lo_equiv_jne": "4.0"}]
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_dir = tmp_path / "summary_missing"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_viper_results.py",
            "--eval-csv",
            str(eval_csv),
            "--parent-method",
            "c_amc_sem_baseline",
            "--teacher-method",
            "dqn_agent",
            "--tree-method",
            "viper_tree_agent",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "available_methods" in (result.stderr + result.stdout)


# ========== Patch 3 新增测试 ==========


def _make_ranked_policy_for_metric_test() -> TreeBudgetPolicy:
    """构造一个简单的 ranked 兼容策略用于离线指标测试。"""
    sklearn = pytest.importorskip("sklearn.tree")
    DecisionTreeClassifier = sklearn.DecisionTreeClassifier
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    return TreeBudgetPolicy(
        classifier=clf,
        metadata={"state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0", "fallback_mode": "ranked_valid_or_none"},
        feature_names=("f0",),
        action_definitions=[{"action_id": 0}, {"action_id": 1}, {"action_id": 2}],
    )


def test_offline_all_invalid_counts_noop_and_no_valid() -> None:
    """离线 all-invalid 应同时计入 no-valid 和 noop，不计入 ranked fallback。"""
    policy = _make_ranked_policy_for_metric_test()
    # 构造 all-invalid 样本：所有 action 均非法
    sample = ViperSample(
        teacher_id="t0", taskset_seed=None, scenario_seed=0, scenario_split="train",
        horizon=100, decision_index=0, time=0,
        state_vector=(1.0,), valid_action_mask=(False, False, False),
        teacher_action_id=0, teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0), q_best=1.0, q_second_best=0.0,
        q_worst=0.0, q_margin_second=1.0, viper_weight=1.0,
        behavior_policy="oracle", behavior_action_id=0, tree_iteration=None,
        raw_budgets_json="{}", raw_recent_costs_json="{}", mask_reject_reasons_json="{}",
    )
    metrics = compute_offline_tree_metrics(policy, [sample])
    assert metrics["no_valid_action_rate_on_dataset"] > 0
    assert metrics["noop_fallback_rate_on_dataset"] == metrics["no_valid_action_rate_on_dataset"]
    assert metrics["ranked_fallback_rate_on_dataset"] == 0.0


def test_offline_all_invalid_not_counted_as_ranked_fallback() -> None:
    """离线 all-invalid 不应计入 ranked fallback。"""
    policy = _make_ranked_policy_for_metric_test()
    sample = ViperSample(
        teacher_id="t0", taskset_seed=None, scenario_seed=0, scenario_split="train",
        horizon=100, decision_index=0, time=0,
        state_vector=(1.0,), valid_action_mask=(False, False, False),
        teacher_action_id=0, teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0), q_best=1.0, q_second_best=0.0,
        q_worst=0.0, q_margin_second=1.0, viper_weight=1.0,
        behavior_policy="oracle", behavior_action_id=0, tree_iteration=None,
        raw_budgets_json="{}", raw_recent_costs_json="{}", mask_reject_reasons_json="{}",
    )
    metrics = compute_offline_tree_metrics(policy, [sample])
    assert metrics["noop_fallback_rate_on_dataset"] > 0
    assert metrics["ranked_fallback_rate_on_dataset"] == 0.0
