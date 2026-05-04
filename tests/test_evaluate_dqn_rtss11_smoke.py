"""RTSS2011 workload 评估脚本 smoke 测试。"""

from __future__ import annotations

import csv
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_dqn_rtss11_smoke_outputs_expected_columns_and_methods(tmp_path: Path) -> None:
    """rtss11 评估应输出关键字段，并包含 baseline/noop/dqn 三类方法。"""

    output_dir = tmp_path / "dqn_rtss11"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_rtss11.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

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
            "--batch-size",
            "6",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
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
            "--model",
            str(model_path),
            "--seeds",
            "100,101",
            "--end-time",
            "2000",
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

    with eval_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    methods = {row["method"] for row in rows}
    assert "amc_plus_baseline" in methods
    assert "noop_agent" in methods
    assert "dqn_agent" in methods

    required_columns = {
        "workload",
        "total_util",
        "num_tasks",
        "cf",
        "cp",
        "seed",
        "taskset_seed",
        "scenario_seed",
        "method",
        "amc_rtb_schedulable",
        "attempts",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "budget_overruns",
        "accepted_actions",
        "rejected_actions",
        "noop_actions",
        "total_reward",
        "check_safety",
        "safety_checked_actions",
        "safety_accepted_actions",
        "safety_rejected_actions",
        "valid_action_count_mean",
        "masked_action_count_mean",
        "masked_action_count_max",
        "mask_rejection_rate_mean",
        "selected_invalid_mask_actions",
        "end_time",
        "agent_period",
    }
    assert required_columns.issubset(set(rows[0].keys()))
    assert all(int(row["deadline_misses"]) == 0 for row in rows)
