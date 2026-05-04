"""DQN 多 seed 训练测试。"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_per_episode_seed_mode_changes_episode_seed_and_keeps_zero_deadline_miss(tmp_path: Path) -> None:
    """per-episode 模式下 episode seed 应变化，且 deadline miss 维持 0。"""

    output_dir = tmp_path / "dqn_multiseed"
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
            "--train-seed-mode",
            "per-episode",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    metrics_path = output_dir / "train_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    episode_seeds = {int(row["episode_seed"]) for row in rows}
    assert len(episode_seeds) == 3
    assert all(int(row["deadline_misses"]) == 0 for row in rows)
