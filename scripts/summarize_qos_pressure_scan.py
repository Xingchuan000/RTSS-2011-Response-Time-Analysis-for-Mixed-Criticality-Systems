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
