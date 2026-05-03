"""正式 DQN 评估 CLI 测试。"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_dqn_amc_cli_runs_after_training(tmp_path: Path) -> None:
    """训练后的正式模型应可被评估 CLI 加载并输出汇总。"""

    output_dir = tmp_path / "dqn_amc"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_summary.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "amc-repro",
            "python",
            "scripts/train_dqn_amc.py",
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
        env=env,
    )
    subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "amc-repro",
            "python",
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0,1",
            "--end-time",
            "50",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert eval_path.exists()
    with eval_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    methods = {row["method"] for row in rows}
    assert "dqn_agent" in methods
    assert "amc_plus_baseline" in methods
    assert "noop_agent" in methods
