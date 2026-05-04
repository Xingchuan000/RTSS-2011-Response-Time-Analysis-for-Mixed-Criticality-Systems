"""RTSS2011 DQN 端到端 smoke 测试。"""

from __future__ import annotations

import csv
import os
import sys
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_rtss11_dqn_end_to_end_smoke(tmp_path: Path) -> None:
    """覆盖 taskset 生成、训练、评估、汇总的最小闭环。"""

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    root = tmp_path / "rtss11_e2e"
    train_dir = root / "train_u055_seed0"
    model_path = train_dir / "model_final.pt"
    eval_path = root / "u055_eval.csv"
    summary_path = root / "u055_summary.csv"
    improvement_path = root / "u055_summary_improvement.csv"

    # 1) 短训练
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "rtss11",
            "--total-util",
            "0.55",
            "--num-tasks",
            "8",
            "--cf",
            "2.0",
            "--cp",
            "0.5",
            "--require-schedulable",
            "--episodes",
            "2",
            "--end-time",
            "1000",
            "--agent-period",
            "1000",
            "--seed",
            "0",
            "--batch-size",
            "6",
            "--output-dir",
            str(train_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    # 2) 短评估（同一批 seed 对比 baseline 与 dqn）
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--workload",
            "rtss11",
            "--total-util",
            "0.55",
            "--num-tasks",
            "8",
            "--cf",
            "2.0",
            "--cp",
            "0.5",
            "--require-schedulable",
            "--model",
            str(model_path),
            "--seeds",
            "100,101",
            "--end-time",
            "1000",
            "--agent-period",
            "1000",
            "--baselines",
            "amc_plus_baseline,noop_agent,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    # 3) 汇总
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_dqn_rtss11_results.py",
            "--input",
            str(eval_path),
            "--output",
            str(summary_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    # 4) 核心验收
    assert model_path.exists()
    assert eval_path.exists()
    assert summary_path.exists()
    assert improvement_path.exists()

    with eval_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(int(row["deadline_misses"]) == 0 for row in rows)
