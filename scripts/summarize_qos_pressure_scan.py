"""汇总 QoS pressure 扫描分布。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def get_float(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    return float(raw)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Path(args.scan_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("scan CSV 为空")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("qos_pressure_bucket", "unknown"))].append(row)

    summary_rows: list[dict[str, object]] = []
    for bucket, bucket_rows in sorted(groups.items()):
        loss_values = [v for v in (get_float(r, "baseline_lc_service_loss_mean") for r in bucket_rows) if v is not None]
        qos_values = [v for v in (get_float(r, "baseline_lc_qos_mean") for r in bucket_rows) if v is not None]
        mode_values = [v for v in (get_float(r, "baseline_mode_changes_mean") for r in bucket_rows) if v is not None]
        cancelled_values = [v for v in (get_float(r, "baseline_cancelled_lo_jobs_mean") for r in bucket_rows) if v is not None]
        reduction_values = [v for v in (get_float(r, "static_sweep_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        # 新增 stable-improvement 维度统计：用于区分“真实稳定可提升”与“仅 trade-off”样本。
        static_qos_best_reduction_values = [v for v in (get_float(r, "static_qos_best_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        stable005_reduction_values = [v for v in (get_float(r, "stable005_static_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        stable010_reduction_values = [v for v in (get_float(r, "stable010_static_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        single_stable005_reduction_values = [v for v in (get_float(r, "single_stable005_static_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        single_stable010_reduction_values = [v for v in (get_float(r, "single_stable010_static_relative_lc_loss_reduction") for r in bucket_rows) if v is not None]
        tradeoff_gap_005_values = [v for v in (get_float(r, "tradeoff_gap_005") for r in bucket_rows) if v is not None]
        tradeoff_gap_010_values = [v for v in (get_float(r, "tradeoff_gap_010") for r in bucket_rows) if v is not None]
        stable005_found_valid_count = sum(1 for r in bucket_rows if str(r.get("stable005_static_found_valid", "")).strip().lower() == "true")
        stable010_found_valid_count = sum(1 for r in bucket_rows if str(r.get("stable010_static_found_valid", "")).strip().lower() == "true")
        single_stable005_found_count = sum(1 for r in bucket_rows if str(r.get("single_stable005_static_found_valid", "")).strip().lower() == "true")
        single_stable010_found_count = sum(1 for r in bucket_rows if str(r.get("single_stable010_static_found_valid", "")).strip().lower() == "true")
        tradeoff_only_flag_005_count = sum(1 for r in bucket_rows if str(r.get("tradeoff_only_flag_005", "")).strip().lower() == "true")
        tradeoff_only_flag_010_count = sum(1 for r in bucket_rows if str(r.get("tradeoff_only_flag_010", "")).strip().lower() == "true")
        improvement_type_counter = Counter(str(r.get("improvement_type", "")).strip() for r in bucket_rows if str(r.get("improvement_type", "")).strip())
        single_improvement_type_counter = Counter(str(r.get("single_improvement_type", "")).strip() for r in bucket_rows if str(r.get("single_improvement_type", "")).strip())
        recommended_count = sum(1 for r in bucket_rows if str(r.get("recommended_for_qos_dqn", "")).strip().lower() == "true")

        summary_rows.append(
            {
                "bucket": bucket,
                "count": len(bucket_rows),
                "baseline_lc_service_loss_mean": mean(loss_values),
                "baseline_lc_service_loss_median": (float(median(loss_values)) if loss_values else ""),
                "baseline_lc_qos_mean": mean(qos_values),
                "baseline_mode_changes_mean": mean(mode_values),
                "baseline_cancelled_lo_jobs_mean": mean(cancelled_values),
                "static_sweep_relative_lc_loss_reduction_mean": mean(reduction_values),
                "static_qos_best_relative_lc_loss_reduction_mean": mean(static_qos_best_reduction_values),
                "stable005_static_found_valid_count": stable005_found_valid_count,
                "stable005_static_relative_lc_loss_reduction_mean": mean(stable005_reduction_values),
                "stable005_static_relative_lc_loss_reduction_median": (float(median(stable005_reduction_values)) if stable005_reduction_values else ""),
                "stable005_static_relative_lc_loss_reduction_max": (max(stable005_reduction_values) if stable005_reduction_values else ""),
                "stable010_static_found_valid_count": stable010_found_valid_count,
                "stable010_static_relative_lc_loss_reduction_mean": mean(stable010_reduction_values),
                "stable010_static_relative_lc_loss_reduction_median": (float(median(stable010_reduction_values)) if stable010_reduction_values else ""),
                "stable010_static_relative_lc_loss_reduction_max": (max(stable010_reduction_values) if stable010_reduction_values else ""),
                "single_stable005_found_count": single_stable005_found_count,
                "single_stable005_static_relative_lc_loss_reduction_mean": mean(single_stable005_reduction_values),
                "single_stable005_static_relative_lc_loss_reduction_max": (max(single_stable005_reduction_values) if single_stable005_reduction_values else ""),
                "single_stable010_found_count": single_stable010_found_count,
                "single_stable010_static_relative_lc_loss_reduction_mean": mean(single_stable010_reduction_values),
                "single_stable010_static_relative_lc_loss_reduction_max": (max(single_stable010_reduction_values) if single_stable010_reduction_values else ""),
                "tradeoff_gap_005_mean": mean(tradeoff_gap_005_values),
                "tradeoff_gap_010_mean": mean(tradeoff_gap_010_values),
                "tradeoff_only_flag_005_count": tradeoff_only_flag_005_count,
                "tradeoff_only_flag_010_count": tradeoff_only_flag_010_count,
                "improvement_type_counts": ";".join(f"{k}:{v}" for k, v in sorted(improvement_type_counter.items())),
                "single_improvement_type_counts": ";".join(f"{k}:{v}" for k, v in sorted(single_improvement_type_counter.items())),
                "recommended_count": recommended_count,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "count",
                "baseline_lc_service_loss_mean",
                "baseline_lc_service_loss_median",
                "baseline_lc_qos_mean",
                "baseline_mode_changes_mean",
                "baseline_cancelled_lo_jobs_mean",
                "static_sweep_relative_lc_loss_reduction_mean",
                "static_qos_best_relative_lc_loss_reduction_mean",
                "stable005_static_found_valid_count",
                "stable005_static_relative_lc_loss_reduction_mean",
                "stable005_static_relative_lc_loss_reduction_median",
                "stable005_static_relative_lc_loss_reduction_max",
                "stable010_static_found_valid_count",
                "stable010_static_relative_lc_loss_reduction_mean",
                "stable010_static_relative_lc_loss_reduction_median",
                "stable010_static_relative_lc_loss_reduction_max",
                "single_stable005_found_count",
                "single_stable005_static_relative_lc_loss_reduction_mean",
                "single_stable005_static_relative_lc_loss_reduction_max",
                "single_stable010_found_count",
                "single_stable010_static_relative_lc_loss_reduction_mean",
                "single_stable010_static_relative_lc_loss_reduction_max",
                "tradeoff_gap_005_mean",
                "tradeoff_gap_010_mean",
                "tradeoff_only_flag_005_count",
                "tradeoff_only_flag_010_count",
                "improvement_type_counts",
                "single_improvement_type_counts",
                "recommended_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    bucket_counter = Counter(str(row.get("qos_pressure_bucket", "unknown")) for row in rows)
    schedulable_rows = sum(1 for row in rows if str(row.get("amcrtb_schedulable", "")).strip().lower() == "true")
    recommended_rows = sum(1 for row in rows if str(row.get("recommended_for_qos_dqn", "")).strip().lower() == "true")

    print("[QoS Pressure Scan Summary]", flush=True)
    print(f"total_rows={len(rows)}", flush=True)
    print(f"schedulable_rows={schedulable_rows}", flush=True)
    print(f"recommended_rows={recommended_rows}", flush=True)
    print(f"easy_count={bucket_counter.get('easy', 0)}", flush=True)
    print(f"medium_count={bucket_counter.get('medium', 0)}", flush=True)
    print(f"hard_count={bucket_counter.get('hard', 0)}", flush=True)
    print(f"overloaded_count={bucket_counter.get('overloaded', 0)}", flush=True)
    print(f"unknown_count={bucket_counter.get('unknown', 0)}", flush=True)


if __name__ == "__main__":
    main()
