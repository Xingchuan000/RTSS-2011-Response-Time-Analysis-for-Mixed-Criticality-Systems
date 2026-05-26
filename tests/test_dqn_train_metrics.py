"""DQN 训练指标输出测试。"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_metrics_csv_is_generated_with_required_columns(tmp_path: Path) -> None:
    """最小 RTSS11 训练应产出 train_metrics.csv 且包含关键字段。"""

    output_dir = tmp_path / "dqn_train_metrics"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "rtss11",
            "--total-util",
            "0.55",
            "--num-tasks",
            "20",
            "--cf",
            "2.0",
            "--cp",
            "0.5",
            "--require-schedulable",
            "--episodes",
            "2",
            "--end-time",
            "2000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    model_path = output_dir / "model_final.pt"
    metrics_path = output_dir / "train_metrics.csv"
    assert model_path.exists()
    assert metrics_path.exists()

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 2
    required_columns = {
        "episode",
        "total_reward",
        "epsilon",
        "accepted_actions",
        "noop_actions",
        "deadline_misses",
        "valid_action_count_mean",
        "masked_action_count_mean",
    }
    assert required_columns.issubset(set(rows[0].keys()))


def test_train_metrics_csv_includes_increase_coverage_columns(tmp_path: Path) -> None:
    """coverage 探索模式下，train_metrics.csv 应包含 coverage 统计字段。"""

    output_dir = tmp_path / "dqn_train_metrics_coverage"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "rtss11",
            "--total-util",
            "0.55",
            "--num-tasks",
            "20",
            "--cf",
            "2.0",
            "--cp",
            "0.5",
            "--require-schedulable",
            "--episodes",
            "2",
            "--end-time",
            "2000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--exploration-mode",
            "epsilon_increase_coverage",
            "--safe-increase-explore-prob",
            "1.0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    metrics_path = output_dir / "train_metrics.csv"
    assert metrics_path.exists()

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 2
    required_columns = {
        "exploration_mode",
        "safe_increase_explore_prob",
        "exploration_increase_coverage_action_count",
        "exploration_increase_coverage_tie_count",
        "exploration_increase_coverage_action_rate",
        "exploration_increase_coverage_tie_rate",
        "increase_coverage_min_count",
        "increase_coverage_max_count",
        "increase_coverage_mean_count",
        "increase_coverage_std_count",
    }
    assert required_columns.issubset(set(rows[0].keys()))
    assert rows[0]["exploration_mode"] == "epsilon_increase_coverage"
