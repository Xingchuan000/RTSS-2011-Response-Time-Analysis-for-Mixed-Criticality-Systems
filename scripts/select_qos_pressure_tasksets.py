"""从 QoS pressure 扫描结果中筛选训练 manifest。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bool_text(value: object) -> bool:
    """解析 CSV 中的布尔文本。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_float(row: dict[str, str], key: str) -> float | None:
    """安全读取浮点字段，空值返回 None。"""

    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    return float(raw)


def reject_reason(row: dict[str, str], args: argparse.Namespace, has_static_col: bool) -> str | None:
    """按固定顺序返回筛选拒绝原因；通过时返回 None。"""

    if str(row.get("qos_pressure_bucket", "")).strip() != args.bucket:
        return "not_bucket"

    if not parse_bool_text(row.get("amcrtb_schedulable", "")):
        return "not_schedulable"

    hi_miss = get_float(row, "baseline_hi_deadline_misses_sum")
    if hi_miss is None or hi_miss > 0.0:
        return "hi_deadline_miss"

    loss = get_float(row, "baseline_lc_service_loss_mean")
    if loss is None or loss < args.min_baseline_lc_service_loss:
        return "loss_below_min"
    if loss > args.max_baseline_lc_service_loss:
        return "loss_above_max"

    released = get_float(row, "baseline_released_lo_jobs_mean")
    if released is None or released < args.min_released_lo_jobs:
        return "insufficient_released_lo_jobs"

    cancelled = get_float(row, "baseline_cancelled_lo_jobs_mean")
    if cancelled is None or cancelled < args.min_cancelled_lo_jobs:
        return "insufficient_cancelled_lo_jobs"

    mode_changes = get_float(row, "baseline_mode_changes_mean")
    if mode_changes is None or mode_changes < args.min_mode_changes:
        return "insufficient_mode_changes"

    if args.min_static_sweep_reduction is not None and has_static_col:
        reduction = get_float(row, "static_sweep_relative_lc_loss_reduction")
        if reduction is None or reduction < args.min_static_sweep_reduction:
            return "insufficient_static_sweep_reduction"

    if args.require_recommended and not parse_bool_text(row.get("recommended_for_qos_dqn", "")):
        return "not_recommended"

    return None


def build_parser() -> argparse.ArgumentParser:
    """构建 QoS pressure 选择脚本参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-csv", type=str, required=True)
    parser.add_argument("--bucket", choices=["easy", "medium", "hard", "overloaded", "unknown"], required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-baseline-lc-service-loss", type=float, default=0.10)
    parser.add_argument("--max-baseline-lc-service-loss", type=float, default=0.30)
    parser.add_argument("--min-released-lo-jobs", type=float, default=100.0)
    parser.add_argument("--min-cancelled-lo-jobs", type=float, default=10.0)
    parser.add_argument("--min-mode-changes", type=float, default=1.0)
    parser.add_argument("--min-static-sweep-reduction", type=float, default=None)
    parser.add_argument("--require-recommended", action="store_true")
    parser.add_argument("--target-loss-center", type=float, default=0.20)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--rejections-output", type=str, required=True)
    return parser


def main() -> None:
    """执行筛选并写出 selected/rejections 两份 CSV。"""

    args = build_parser().parse_args()

    scan_path = Path(args.scan_csv)
    with scan_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("scan CSV 为空")

    has_static_col = "static_sweep_relative_lc_loss_reduction" in rows[0]

    selected: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for row in rows:
        reason = reject_reason(row, args, has_static_col)
        if reason is None:
            selected.append(dict(row))
        else:
            rejected_row = dict(row)
            rejected_row["select_reject_reason"] = reason
            rejected.append(rejected_row)

    def sort_key(row: dict[str, str]):
        loss = float(row["baseline_lc_service_loss_mean"])
        reduction_raw = row.get("static_sweep_relative_lc_loss_reduction", "")
        reduction = float(reduction_raw) if str(reduction_raw).strip() else float("-inf")
        mode_changes = float(row["baseline_mode_changes_mean"])
        candidate_seed = int(float(row["candidate_seed"]))
        return (
            abs(loss - args.target_loss_center),
            -reduction,
            mode_changes,
            candidate_seed,
        )

    selected_sorted = sorted(selected, key=sort_key)
    selected_topk = selected_sorted[: args.top_k]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_topk)

    rejections_path = Path(args.rejections_output)
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_fields = list(rows[0].keys()) + ["select_reject_reason"]
    with rejections_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rejection_fields)
        writer.writeheader()
        writer.writerows(rejected)

    selected_loss_values = [float(row["baseline_lc_service_loss_mean"]) for row in selected_topk]
    selected_qos_values = [float(row["baseline_lc_qos_mean"]) for row in selected_topk if str(row.get("baseline_lc_qos_mean", "")).strip()]
    selected_reduction_values = [
        float(row["static_sweep_relative_lc_loss_reduction"])
        for row in selected_topk
        if str(row.get("static_sweep_relative_lc_loss_reduction", "")).strip()
    ]

    loss_mean = sum(selected_loss_values) / len(selected_loss_values) if selected_loss_values else ""
    qos_mean = sum(selected_qos_values) / len(selected_qos_values) if selected_qos_values else ""
    reduction_mean = sum(selected_reduction_values) / len(selected_reduction_values) if selected_reduction_values else ""

    print("[QoS Select Summary]", flush=True)
    print(f"scan_rows={len(rows)}", flush=True)
    print(f"selected={len(selected_topk)}", flush=True)
    print(f"rejected={len(rejected)}", flush=True)
    print(f"bucket={args.bucket}", flush=True)
    print(f"baseline_lc_service_loss_mean_selected={loss_mean}", flush=True)
    print(f"baseline_lc_qos_mean_selected={qos_mean}", flush=True)
    print(f"static_sweep_relative_lc_loss_reduction_mean_selected={reduction_mean}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
