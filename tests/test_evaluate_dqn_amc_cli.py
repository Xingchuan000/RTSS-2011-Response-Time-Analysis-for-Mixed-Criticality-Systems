"""正式 DQN 评估 CLI 测试。"""

from __future__ import annotations

import csv
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 全部行，供串并行结果对比复用。"""

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_evaluate_dqn_amc_cli_runs_after_training(tmp_path: Path) -> None:
    """训练后的正式模型应可被评估 CLI 加载并输出汇总。"""

    output_dir = tmp_path / "dqn_amc"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_summary.csv"
    unified_summary_path = output_dir / "eval_summary_unified_summary.csv"
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
    assert unified_summary_path.exists()
    with eval_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with unified_summary_path.open("r", encoding="utf-8", newline="") as f:
        unified_rows = list(csv.DictReader(f))

    assert rows
    assert unified_rows
    methods = {row["method"] for row in rows}
    assert "dqn_agent" in methods
    assert "amc_plus_baseline" in methods
    assert "noop_agent" in methods
    expected_summary_fields = {
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "dqn_mode_changes_mean",
        "dqn_lo_cancellations_mean",
        "mode_change_ratio",
        "lo_cancellation_ratio",
        "accepted_action_count_mean",
        "rejected_action_count_mean",
        "noop_action_count_mean",
        "noop_action_rate_mean",
        "noop_q_rank_mean",
        "noop_q_sample_count",
        "masked_action_count_mean",
        "valid_action_count_mean",
    }
    assert expected_summary_fields.issubset(set(unified_rows[0].keys()))
    assert "noop_q_rank_mean" in rows[0]
    baseline_row = next(row for row in rows if row["method"] == "amc_plus_baseline")
    assert baseline_row["noop_q_rank_mean"] == ""
    assert "budget_floor_ratio" in rows[0]
    assert "masked_budget_floor_violation_count" in rows[0]
    assert "masked_budget_floor_violation_rate" in rows[0]


def test_evaluate_cli_rejects_legacy_reward_mode(tmp_path: Path) -> None:
    """评估 CLI 不应再接受旧 reward mode。"""

    output_dir = tmp_path / "dqn_amc_legacy_reward_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--reward-mode",
            "event_delta_no_job_start",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_rejects_invalid_budget_floor_ratio(tmp_path: Path) -> None:
    """评估 CLI 应拒绝超出 [0,1] 的 budget floor 参数。"""

    output_dir = tmp_path / "dqn_amc_invalid_floor_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--budget-floor-ratio",
            "-0.1",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_supports_parallel_seed_workers(tmp_path: Path) -> None:
    """评估 CLI 开启多进程按 seed 并行后应正常产出结果。"""

    output_dir = tmp_path / "dqn_amc_parallel_eval"
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
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    eval_path = output_dir / "eval_parallel.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0,1",
            "--end-time",
            "50",
            "--evaluation-workers",
            "2",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    assert rows
    assert {row["method"] for row in rows} >= {"dqn_agent", "amc_plus_baseline"}


def test_evaluate_cli_parallel_and_serial_outputs_match(tmp_path: Path) -> None:
    """固定模型与 seeds 时，串行/并行评估输出应保持一致。"""

    output_dir = tmp_path / "dqn_amc_eval_consistency"
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
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    serial_eval_path = output_dir / "eval_serial.csv"
    parallel_eval_path = output_dir / "eval_parallel.csv"
    base_cmd = [
        sys.executable,
        "scripts/evaluate_dqn_amc.py",
        "--model",
        str(output_dir / "model_final.pt"),
        "--seeds",
        "0,1",
        "--end-time",
        "50",
    ]
    subprocess.run(
        base_cmd + ["--evaluation-workers", "1", "--output", str(serial_eval_path)],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    subprocess.run(
        base_cmd + ["--evaluation-workers", "2", "--output", str(parallel_eval_path)],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert _read_csv_rows(serial_eval_path) == _read_csv_rows(parallel_eval_path)
    assert _read_csv_rows(
        serial_eval_path.with_name(f"{serial_eval_path.stem}_unified_summary.csv")
    ) == _read_csv_rows(
        parallel_eval_path.with_name(f"{parallel_eval_path.stem}_unified_summary.csv")
    )


def test_evaluate_cli_rejects_invalid_evaluation_workers(tmp_path: Path) -> None:
    """评估 worker 数小于 1 时，评估 CLI 应显式报错。"""

    output_dir = tmp_path / "dqn_amc_invalid_eval_workers"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--evaluation-workers",
            "0",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0
