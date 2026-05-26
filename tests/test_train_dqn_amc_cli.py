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
    assert config_payload["validation_workers"] == 1
    assert config_payload["log_step_every"] == 1

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


def test_train_cli_rejects_action_aware_mask_mode_outside_action_aware_single(tmp_path: Path) -> None:
    """action-aware-mask-mode 仅允许 action_aware+single。"""

    output_dir = tmp_path / "invalid_action_aware_mask_mode"
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
            "--q-network-type",
            "mlp",
            "--action-space",
            "single",
            "--action-aware-mask-mode",
            "increase_noop",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode != 0


def test_train_cli_supports_parallel_validation_and_disabling_step_log(tmp_path: Path) -> None:
    """并行 validation 与关闭 step 日志时，训练仍应正常完成并写出空表头 CSV。"""

    output_dir = tmp_path / "parallel_validation"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "2",
            "--end-time",
            "100",
            "--validate-every",
            "1",
            "--validation-seeds",
            "100:101",
            "--validation-workers",
            "2",
            "--log-step-every",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    train_log_path = output_dir / "train_log.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    assert train_log_path.exists()
    assert validation_metrics_path.exists()

    with train_log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames is not None
    assert "valid_action_count" in fieldnames
    assert rows == []

    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        config_payload = json.load(f)
    assert config_payload["validation_workers"] == 2
    assert config_payload["log_step_every"] == 0


def test_train_cli_supports_plateau_balanced_exploration_fields(tmp_path: Path) -> None:
    """plateau-balanced 探索模式应写出对应的配置与统计字段。"""

    output_dir = tmp_path / "plateau_balanced"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "2",
            "--end-time",
            "100",
            "--validate-every",
            "1",
            "--validation-seeds",
            "100:101",
            "--exploration-mode",
            "epsilon_plateau_soft_target_balanced",
            "--plateau-balanced-mix-prob",
            "0.25",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        config_payload = json.load(f)
    assert config_payload["exploration_mode"] == "epsilon_plateau_soft_target_balanced"
    assert config_payload["plateau_balanced_mix_prob"] == 0.25
    assert "plateau_balanced_total_bursts" in config_payload

    with (output_dir / "train_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        train_rows = list(csv.DictReader(f))
    assert train_rows
    assert "plateau_balanced_active" in train_rows[0]
    assert "plateau_balanced_action_rate" in train_rows[0]

    with (output_dir / "validation_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        validation_rows = list(csv.DictReader(f))
    assert validation_rows
    assert "plateau_current_reduction" in validation_rows[0]
    assert "plateau_balanced_triggered" in validation_rows[0]


def test_train_cli_rejects_invalid_validation_workers(tmp_path: Path) -> None:
    """validation worker 数小于 1 时，训练 CLI 应显式报错。"""

    output_dir = tmp_path / "invalid_validation_workers"
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
            "--validation-workers",
            "0",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    assert result.returncode != 0


def test_train_cli_constraint_guided_pair_smoke(tmp_path: Path) -> None:
    """训练 CLI 应支持 constraint_guided_pair 并写入对应配置。"""

    output_dir = tmp_path / "constraint_guided_pair_train"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "1",
            "--end-time",
            "60",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "constraint_guided_pair",
            "--constraint-guided-pair-top-k-risk",
            "3",
            "--constraint-guided-pair-top-k-decrease",
            "5",
            "--include-explicit-noop",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    with (output_dir / "config.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["requested_action_space"] == "constraint_guided_pair"
    assert payload["action_space"] == "constraint_guided_transfer"
    assert payload["constraint_guided_pair_top_k_risk"] == 3
    assert payload["constraint_guided_pair_top_k_decrease"] == 5
    # bundled transfer 口径：top_k_risk=3 且开启 explicit noop 时动作维度应为 4。
    assert payload["action_space_size"] == 4
    assert (output_dir / "train_metrics.csv").exists()
