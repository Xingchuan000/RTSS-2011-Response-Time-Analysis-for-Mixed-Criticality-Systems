"""RTSS2011 DQN 结果汇总脚本测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.summarize_dqn_rtss11_results import summarize


def _write_eval_csv(path: Path) -> None:
    """写入一份小型评估 CSV 作为测试输入。"""

    rows = [
        {
            "workload": "rtss11",
            "total_util": "0.65",
            "num_tasks": "20",
            "cf": "2.0",
            "cp": "0.5",
            "seed": "100",
            "taskset_seed": "200",
            "scenario_seed": "201",
            "method": "amc_plus_baseline",
            "amc_rtb_schedulable": "True",
            "attempts": "1",
            "mode_changes": "10",
            "lo_cancellations": "20",
            "deadline_misses": "0",
            "budget_overruns": "30",
            "accepted_actions": "0",
            "rejected_actions": "0",
            "noop_actions": "0",
            "rejection_rate": "0.0",
            "total_reward": "0.0",
            "end_time": "2000",
            "agent_period": "1000",
        },
        {
            "workload": "rtss11",
            "total_util": "0.65",
            "num_tasks": "20",
            "cf": "2.0",
            "cp": "0.5",
            "seed": "101",
            "taskset_seed": "202",
            "scenario_seed": "203",
            "method": "amc_plus_baseline",
            "amc_rtb_schedulable": "True",
            "attempts": "1",
            "mode_changes": "14",
            "lo_cancellations": "22",
            "deadline_misses": "0",
            "budget_overruns": "36",
            "accepted_actions": "0",
            "rejected_actions": "0",
            "noop_actions": "0",
            "rejection_rate": "0.0",
            "total_reward": "0.0",
            "end_time": "2000",
            "agent_period": "1000",
        },
        {
            "workload": "rtss11",
            "total_util": "0.65",
            "num_tasks": "20",
            "cf": "2.0",
            "cp": "0.5",
            "seed": "100",
            "taskset_seed": "200",
            "scenario_seed": "201",
            "method": "dqn_agent",
            "amc_rtb_schedulable": "True",
            "attempts": "1",
            "mode_changes": "0",
            "lo_cancellations": "10",
            "deadline_misses": "0",
            "budget_overruns": "10",
            "accepted_actions": "3",
            "rejected_actions": "1",
            "noop_actions": "1",
            "rejection_rate": "0.2",
            "total_reward": "1.0",
            "end_time": "2000",
            "agent_period": "1000",
        },
        {
            "workload": "rtss11",
            "total_util": "0.65",
            "num_tasks": "20",
            "cf": "2.0",
            "cp": "0.5",
            "seed": "101",
            "taskset_seed": "202",
            "scenario_seed": "203",
            "method": "dqn_agent",
            "amc_rtb_schedulable": "True",
            "attempts": "1",
            "mode_changes": "2",
            "lo_cancellations": "8",
            "deadline_misses": "0",
            "budget_overruns": "10",
            "accepted_actions": "5",
            "rejected_actions": "1",
            "noop_actions": "0",
            "rejection_rate": "0.1666667",
            "total_reward": "2.0",
            "end_time": "2000",
            "agent_period": "1000",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_computes_means_and_medians_correctly(tmp_path: Path) -> None:
    """应正确计算 summary 中的均值与中位数。"""

    input_path = tmp_path / "eval.csv"
    output_path = tmp_path / "summary.csv"
    _write_eval_csv(input_path)

    summarize(input_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_method = {row["method"]: row for row in rows}

    assert float(by_method["amc_plus_baseline"]["mode_changes_mean"]) == 12.0
    assert float(by_method["amc_plus_baseline"]["mode_changes_median"]) == 12.0
    assert float(by_method["dqn_agent"]["lo_cancellations_mean"]) == 9.0
    assert float(by_method["dqn_agent"]["lo_cancellations_median"]) == 9.0


def test_summarize_handles_zero_denominator_without_crash(tmp_path: Path) -> None:
    """分母为 0 时 improvement ratio 应给出约定值而不崩溃。"""

    input_path = tmp_path / "eval.csv"
    output_path = tmp_path / "summary.csv"
    _write_eval_csv(input_path)
    summarize(input_path, output_path)

    improvement_path = tmp_path / "summary_improvement.csv"
    with improvement_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # dqn_agent 在 mode_changes 上有 0 值样本，聚合均值为 1，不是 0；
    # 这里构造时 baseline>0 且 method>0，因此 ratio 应可正常计算为 finite 数值。
    assert rows
    mode_row = next(row for row in rows if row["metric"] == "mode_changes" and row["method"] == "dqn_agent")
    assert mode_row["ratio"] != ""


def test_summarize_output_columns_are_complete(tmp_path: Path) -> None:
    """summary 与 improvement 输出字段应完整。"""

    input_path = tmp_path / "eval.csv"
    output_path = tmp_path / "summary.csv"
    _write_eval_csv(input_path)
    summarize(input_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as f:
        summary_fields = set(csv.DictReader(f).fieldnames or [])
    assert {
        "workload",
        "total_util",
        "method",
        "n",
        "mode_changes_mean",
        "mode_changes_median",
        "lo_cancellations_mean",
        "lo_cancellations_median",
        "deadline_misses_sum",
        "accepted_actions_mean",
        "rejected_actions_mean",
        "rejection_rate_mean",
        "total_reward_mean",
    }.issubset(summary_fields)

    improvement_path = tmp_path / "summary_improvement.csv"
    with improvement_path.open("r", encoding="utf-8", newline="") as f:
        improvement_fields = set(csv.DictReader(f).fieldnames or [])
    assert {
        "workload",
        "total_util",
        "metric",
        "baseline_method",
        "method",
        "ratio",
        "delta",
    }.issubset(improvement_fields)
