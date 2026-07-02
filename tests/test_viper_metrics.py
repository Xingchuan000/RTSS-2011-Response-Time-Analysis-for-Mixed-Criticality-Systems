"""VIPER metrics 与 summary 测试。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from amc_py.viper.metrics import retention_higher_is_better, retention_lower_is_better

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
