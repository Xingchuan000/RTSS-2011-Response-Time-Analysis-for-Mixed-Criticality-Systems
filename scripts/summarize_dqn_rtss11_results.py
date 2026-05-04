"""汇总 RTSS2011 DQN 评估结果。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


def _to_float(row: dict[str, str], key: str) -> float:
    """把 CSV 字段解析为 float。"""

    value = row.get(key, "")
    if value is None or value == "":
        return 0.0
    return float(value)


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    """按 workload/total_util/method 进行分组。"""

    return (
        row.get("workload", ""),
        row.get("total_util", ""),
        row.get("method", ""),
    )


def _safe_ratio(baseline_value: float, method_value: float) -> tuple[str, float]:
    """安全计算改善比例，避免分母为 0 崩溃。"""

    if method_value == 0.0:
        if baseline_value > 0.0:
            return "inf", float("inf")
        return "nan", float("nan")
    return "finite", baseline_value / method_value


def summarize(input_path: Path, output_path: Path, baseline_method: str = "amc_plus_baseline") -> None:
    """从评估 CSV 生成 summary 与 improvement 文件。"""

    if not input_path.exists():
        raise SystemExit(
            f"Input CSV not found: {input_path}. "
            "Please run scripts/evaluate_dqn_amc.py successfully before summarizing."
        )

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = _group_key(row)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str | float | int]] = []
    for (workload, total_util, method), group_rows in sorted(grouped.items()):
        mode_changes = [_to_float(row, "mode_changes") for row in group_rows]
        lo_cancellations = [_to_float(row, "lo_cancellations") for row in group_rows]
        deadline_misses = [_to_float(row, "deadline_misses") for row in group_rows]
        accepted_actions = [_to_float(row, "accepted_actions") for row in group_rows]
        rejected_actions = [_to_float(row, "rejected_actions") for row in group_rows]
        rejection_rate = [_to_float(row, "rejection_rate") for row in group_rows]
        total_reward = [_to_float(row, "total_reward") for row in group_rows]

        summary_rows.append(
            {
                "workload": workload,
                "total_util": total_util,
                "method": method,
                "n": len(group_rows),
                "mode_changes_mean": mean(mode_changes),
                "mode_changes_median": median(mode_changes),
                "lo_cancellations_mean": mean(lo_cancellations),
                "lo_cancellations_median": median(lo_cancellations),
                "deadline_misses_sum": sum(deadline_misses),
                "accepted_actions_mean": mean(accepted_actions),
                "rejected_actions_mean": mean(rejected_actions),
                "rejection_rate_mean": mean(rejection_rate),
                "total_reward_mean": mean(total_reward),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 生成 improvement 文件：同 workload + total_util 组内，相对 baseline 比较。
    by_workload_util: dict[tuple[str, str], list[dict[str, str | float | int]]] = {}
    for row in summary_rows:
        by_workload_util.setdefault((str(row["workload"]), str(row["total_util"])), []).append(row)

    improvement_rows: list[dict[str, str | float]] = []
    for (workload, total_util), rows_of_group in sorted(by_workload_util.items()):
        baseline_row = next((row for row in rows_of_group if row["method"] == baseline_method), None)
        if baseline_row is None:
            continue

        baseline_mode = float(baseline_row["mode_changes_mean"])
        baseline_lo_cancel = float(baseline_row["lo_cancellations_mean"])

        for row in rows_of_group:
            method = str(row["method"])
            if method == baseline_method:
                continue

            method_mode = float(row["mode_changes_mean"])
            method_lo_cancel = float(row["lo_cancellations_mean"])

            mode_ratio_type, mode_ratio = _safe_ratio(baseline_mode, method_mode)
            lo_ratio_type, lo_ratio = _safe_ratio(baseline_lo_cancel, method_lo_cancel)

            mode_ratio_value = "inf" if mode_ratio_type == "inf" else ("nan" if math.isnan(mode_ratio) else mode_ratio)
            lo_ratio_value = "inf" if lo_ratio_type == "inf" else ("nan" if math.isnan(lo_ratio) else lo_ratio)

            improvement_rows.extend(
                [
                    {
                        "workload": workload,
                        "total_util": total_util,
                        "metric": "mode_changes",
                        "baseline_method": baseline_method,
                        "method": method,
                        "ratio": mode_ratio_value,
                        "delta": method_mode - baseline_mode,
                    },
                    {
                        "workload": workload,
                        "total_util": total_util,
                        "metric": "lo_cancellations",
                        "baseline_method": baseline_method,
                        "method": method,
                        "ratio": lo_ratio_value,
                        "delta": method_lo_cancel - baseline_lo_cancel,
                    },
                ]
            )

    improvement_path = output_path.with_name(f"{output_path.stem}_improvement.csv")
    improvement_fields = [
        "workload",
        "total_util",
        "metric",
        "baseline_method",
        "method",
        "ratio",
        "delta",
    ]
    with improvement_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=improvement_fields)
        writer.writeheader()
        writer.writerows(improvement_rows)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-method", type=str, default="amc_plus_baseline")
    return parser


def main() -> None:
    """执行汇总。"""

    args = build_parser().parse_args()
    summarize(args.input, args.output, baseline_method=args.baseline_method)


if __name__ == "__main__":
    main()
