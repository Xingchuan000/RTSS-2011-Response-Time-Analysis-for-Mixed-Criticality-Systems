"""DQN 训练诊断绘图脚本测试。"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plot_dqn_training_generates_png_files(tmp_path: Path) -> None:
    """使用最小 fake train_log.csv 也应能生成至少一个 PNG 文件。"""

    train_log = tmp_path / "train_log.csv"
    output_dir = tmp_path / "plots"
    rows = [
        {
            "episode": 0,
            "step": 0,
            "sim_time": 10,
            "reward": 1.0,
            "episode_reward": 1.0,
            "loss": "",
            "epsilon": 1.0,
            "action_id": 0,
            "accepted": True,
            "rejected": False,
            "reject_reason": "",
            "valid_action_count": 2,
            "masked_action_count": 1,
            "noop_due_to_no_valid_action": False,
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        },
        {
            "episode": 0,
            "step": 1,
            "sim_time": 20,
            "reward": -0.5,
            "episode_reward": 0.5,
            "loss": 0.25,
            "epsilon": 0.9,
            "action_id": "",
            "accepted": False,
            "rejected": False,
            "reject_reason": "no_valid_action",
            "valid_action_count": 0,
            "masked_action_count": 3,
            "noop_due_to_no_valid_action": True,
            "mode_changes": 1,
            "lo_cancellations": 1,
            "deadline_misses": 0,
        },
    ]
    with train_log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "amc-repro",
            "python",
            "scripts/plot_dqn_training.py",
            "--train-log",
            str(train_log),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    png_files = list(output_dir.glob("*.png"))
    assert png_files
