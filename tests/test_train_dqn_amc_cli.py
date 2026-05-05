"""正式 DQN 训练 CLI 测试。"""

from __future__ import annotations

import csv
import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_dqn_amc_cli_runs_and_writes_expected_outputs(tmp_path: Path) -> None:
    """正式训练 CLI 应输出训练日志、模型和配置文件。"""

    output_dir = tmp_path / "dqn_amc"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "2",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--checkpoint",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert (output_dir / "train_log.csv").exists()
    assert (output_dir / "model_final.pt").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "checkpoints" / "model_episode_0001.pt").exists()

    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        config_payload = json.load(f)
    assert "dqn_config" in config_payload
    assert "normalization_bounds" in config_payload
    assert "budget_floor_ratio" in config_payload

    with (output_dir / "train_log.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "valid_action_count" in rows[0]


def test_train_dqn_amc_cli_is_reasonably_reproducible_for_fixed_seed(tmp_path: Path) -> None:
    """固定 seed 时两次训练的日志应保持一致。"""

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    output_a = tmp_path / "run_a"
    output_b = tmp_path / "run_b"
    base_cmd = [
        sys.executable,
        "scripts/train_dqn_amc.py",
        "--episodes",
        "2",
        "--end-time",
        "50",
        "--seed",
        "5",
    ]
    subprocess.run(base_cmd + ["--output-dir", str(output_a)], check=True, cwd=PROJECT_ROOT, env=env)
    subprocess.run(base_cmd + ["--output-dir", str(output_b)], check=True, cwd=PROJECT_ROOT, env=env)

    text_a = (output_a / "train_log.csv").read_text(encoding="utf-8")
    text_b = (output_b / "train_log.csv").read_text(encoding="utf-8")
    assert text_a == text_b


def test_train_cli_rejects_legacy_reward_mode(tmp_path: Path) -> None:
    """训练 CLI 不应再接受旧 reward mode。"""

    output_dir = tmp_path / "legacy_reward_mode"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--output-dir",
            str(output_dir),
            "--reward-mode",
            "event_delta_no_job_start",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode != 0


def test_train_cli_rejects_invalid_budget_floor_ratio(tmp_path: Path) -> None:
    """训练 CLI 应拒绝超出 [0,1] 的 budget floor 参数。"""

    output_dir = tmp_path / "invalid_budget_floor"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--output-dir",
            str(output_dir),
            "--budget-floor-ratio",
            "1.1",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode != 0
