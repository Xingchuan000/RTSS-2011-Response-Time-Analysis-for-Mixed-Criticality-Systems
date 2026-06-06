"""evaluate summary 的 task-level JSON 日志回归测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.evaluate_dqn_amc import TASK_LEVEL_INFO_KEYS, _eval_summary_fieldnames, _write_unified_summary_csv


def test_unified_summary_csv_includes_task_level_json_columns(tmp_path: Path) -> None:
    """unified summary CSV 头部必须包含 task-level JSON 列。"""

    eval_fieldnames = _eval_summary_fieldnames()
    for key in TASK_LEVEL_INFO_KEYS:
        assert key in eval_fieldnames

    rows = [
        {
            "workload": "small",
            "total_util": 0.65,
            "num_tasks": 3,
            "cf": 2.0,
            "cp": 0.5,
            "method": "amc_plus_baseline",
            "mode_changes": 10,
            "lo_cancellations": 2,
            "deadline_misses": 0,
            "released_lo_jobs": 20,
            "lc_service_loss": 1.0,
            "lc_qos": 0.9,
            "min_lc_service": 0.7,
            "accepted_actions": 0,
            "rejected_actions": 0,
            "noop_actions": 0,
            "explicit_noop_actions": 0,
            "noop_action_rate": 0.0,
            "explicit_noop_action_rate": 0.0,
            "accepted_action_rate": 0.0,
            "rejection_rate": 0.0,
            "masked_action_count_mean": 0.0,
            "valid_action_count_mean": 0.0,
        },
        {
            "workload": "small",
            "total_util": 0.65,
            "num_tasks": 3,
            "cf": 2.0,
            "cp": 0.5,
            "method": "dqn_agent",
            "mode_changes": 8,
            "lo_cancellations": 1,
            "deadline_misses": 0,
            "released_lo_jobs": 21,
            "lc_service_loss": 0.8,
            "lc_qos": 0.92,
            "min_lc_service": 0.75,
            "accepted_actions": 5,
            "rejected_actions": 1,
            "noop_actions": 1,
            "explicit_noop_actions": 0,
            "noop_action_rate": 0.1,
            "explicit_noop_action_rate": 0.0,
            "accepted_action_rate": 0.5,
            "rejection_rate": 0.1,
            "masked_action_count_mean": 3.0,
            "valid_action_count_mean": 2.0,
            "final_budget_ratio_by_task_json": "{\"a\": 1.0}",
            "max_budget_ratio_by_task_json": "{\"a\": 1.2}",
            "min_budget_ratio_by_task_json": "{\"a\": 0.9}",
            "increase_count_by_task_json": "{\"a\": 2}",
            "decrease_count_by_task_json": "{\"a\": 1}",
            "recovery_decrease_count_by_task_json": "{\"a\": 1}",
            "over_increase_count_by_task_json": "{\"a\": 1}",
            "consecutive_increase_max_by_task_json": "{\"a\": 3}",
            "over_budget_dwell_steps_by_task_json": "{\"a\": 4}",
        },
    ]

    output = tmp_path / "eval.csv"
    _write_unified_summary_csv(output, rows)
    summary_path = output.with_name(f"{output.stem}_unified_summary.csv")
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))

    for key in TASK_LEVEL_INFO_KEYS:
        assert key in header
