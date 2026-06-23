"""DQN 训练 run 对比脚本测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.compare_dqn_training_runs import compare_runs


def _write_run_dir(base: Path, name: str, lo_values: tuple[float, float]) -> Path:
    """构造最小训练目录结构。"""

    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_util": 0.65,
                "reward_mode": "event_delta_no_job_start",
                "action_space": "pair",
                "train_seed_mode": "per-episode",
                "train_seeds": [0, 1, 2],
                "runtime_config": {"end_time": 50000},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (run_dir / "validation_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "validation_seed_count",
                "deadline_misses_sum",
                "hi_deadline_misses_sum",
                "mode_changes_mean",
                "lo_cancellations_mean",
                "baseline_mode_changes_mean",
                "baseline_lo_cancellations_mean",
                "lc_service_loss_mean",
                "baseline_lc_service_loss_mean",
                "relative_lc_loss_reduction",
                "lc_qos_mean",
                "min_lc_service_mean",
                "mode_change_delta_ratio",
                "budget_adjust_count_mean",
                "dqn_mode_changes_delta_mean",
                "dqn_lo_cancellations_delta_mean",
                "accepted_actions_mean",
                "noop_actions_mean",
                "valid_action_count_mean",
                "masked_action_count_mean",
                "no_safe_action_steps_mean",
                "reward_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode": 1,
                "validation_seed_count": 2,
                "deadline_misses_sum": 0,
                "hi_deadline_misses_sum": 0,
                "mode_changes_mean": 10,
                "lo_cancellations_mean": lo_values[0],
                "baseline_mode_changes_mean": 12,
                "baseline_lo_cancellations_mean": 20,
                "lc_service_loss_mean": 0.45,
                "baseline_lc_service_loss_mean": 1.0,
                "relative_lc_loss_reduction": 0.55,
                "lc_qos_mean": 0.55,
                "min_lc_service_mean": 0.50,
                "mode_change_delta_ratio": -0.1666666667,
                "budget_adjust_count_mean": 2,
                "dqn_mode_changes_delta_mean": -2,
                "dqn_lo_cancellations_delta_mean": lo_values[0] - 20,
                "accepted_actions_mean": 3,
                "noop_actions_mean": 1,
                "valid_action_count_mean": 20,
                "masked_action_count_mean": 360,
                "no_safe_action_steps_mean": 0,
                "reward_mean": -10,
            }
        )
        writer.writerow(
            {
                "episode": 2,
                "validation_seed_count": 2,
                "deadline_misses_sum": 0,
                "hi_deadline_misses_sum": 0,
                "mode_changes_mean": 9,
                "lo_cancellations_mean": lo_values[1],
                "baseline_mode_changes_mean": 12,
                "baseline_lo_cancellations_mean": 20,
                "lc_service_loss_mean": 0.40,
                "baseline_lc_service_loss_mean": 1.0,
                "relative_lc_loss_reduction": 0.60,
                "lc_qos_mean": 0.60,
                "min_lc_service_mean": 0.55,
                "mode_change_delta_ratio": -0.25,
                "budget_adjust_count_mean": 3,
                "dqn_mode_changes_delta_mean": -3,
                "dqn_lo_cancellations_delta_mean": lo_values[1] - 20,
                "accepted_actions_mean": 4,
                "noop_actions_mean": 1,
                "valid_action_count_mean": 22,
                "masked_action_count_mean": 358,
                "no_safe_action_steps_mean": 0,
                "reward_mean": -9,
            }
        )
    return run_dir


def test_compare_runs_writes_required_columns(tmp_path: Path) -> None:
    """对比脚本应按约定字段输出 CSV。"""

    run_a = _write_run_dir(tmp_path, "run_a", (9.0, 8.0))
    run_b = _write_run_dir(tmp_path, "run_b", (11.0, 10.0))
    output = tmp_path / "comparison.csv"
    compare_runs(
        [run_a, run_b],
        output,
        best_type="legacy_lo_cancel",
        qos_stable_mode_delta=0.05,
    )

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {
        "run",
        "total_util",
        "reward_mode",
        "action_space",
        "episodes",
        "end_time",
        "train_seed_mode",
        "best_episode",
        "best_deadline_misses_sum",
        "best_mode_changes_mean",
        "best_lo_cancellations_mean",
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "dqn_mode_delta",
        "dqn_lo_cancel_delta",
        "valid_action_count_mean",
        "masked_action_count_mean",
        "no_safe_action_steps",
    }.issubset(set(rows[0].keys()))
