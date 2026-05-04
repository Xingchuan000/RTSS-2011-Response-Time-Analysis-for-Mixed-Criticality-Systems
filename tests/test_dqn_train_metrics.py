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
