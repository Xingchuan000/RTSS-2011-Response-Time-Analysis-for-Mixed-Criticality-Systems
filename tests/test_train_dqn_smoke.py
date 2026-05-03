"""DQN smoke 训练脚本测试。"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_dqn_smoke_runs_and_writes_outputs(tmp_path: Path) -> None:
    """脚本应可快速跑通并生成日志与模型。"""

    output_dir = tmp_path / "dqn_smoke"
    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "amc-repro",
            "python",
            "scripts/train_dqn_smoke.py",
            "--episodes",
            "2",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    log_path = output_dir / "train_log.csv"
    model_path = output_dir / "model.pt"
    assert log_path.exists()
    assert model_path.exists()

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    loss_values = [row["loss"] for row in rows if row["loss"] != ""]
    assert any(value.lower() != "nan" for value in loss_values)
