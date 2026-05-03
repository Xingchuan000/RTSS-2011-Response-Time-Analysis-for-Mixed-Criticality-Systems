"""RTSS2011 workload 训练脚本 smoke 测试。"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_dqn_rtss11_smoke_runs_and_writes_outputs(tmp_path: Path) -> None:
    """`--workload rtss11` 应可快速训练并生成产物。"""

    output_dir = tmp_path / "dqn_rtss11"
    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "amc-repro",
            "python",
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
            "--batch-size",
            "6",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    log_path = output_dir / "train_log.csv"
    model_path = output_dir / "model_final.pt"
    assert log_path.exists()
    assert model_path.exists()

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert rows[0]["workload"] == "rtss11"
    assert all(int(row["deadline_misses"]) == 0 for row in rows)
