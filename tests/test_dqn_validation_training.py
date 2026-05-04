"""DQN 训练内 validation 流程测试。"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_writes_validation_metrics_and_best_checkpoint(tmp_path: Path) -> None:
    """开启 validate-every 后应产出 validation CSV 与 best/final 模型。"""

    output_dir = tmp_path / "dqn_validation"
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
            "3",
            "--end-time",
            "2000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--validation-seeds",
            "100,101",
            "--validate-every",
            "1",
            "--validation-end-time",
            "2000",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    validation_metrics_path = output_dir / "validation_metrics.csv"
    validation_unified_summary_path = output_dir / "validation_unified_summary.csv"
    best_model_path = output_dir / "model_best.pt"
    final_model_path = output_dir / "model_final.pt"
    assert validation_metrics_path.exists()
    assert validation_unified_summary_path.exists()
    assert best_model_path.exists()
    assert final_model_path.exists()

    with validation_metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 3
    assert "deadline_misses_sum" in rows[0]
    with validation_unified_summary_path.open("r", encoding="utf-8", newline="") as f:
        unified_rows = list(csv.DictReader(f))
    assert unified_rows
    assert "noop_action_rate_mean" in unified_rows[0]
