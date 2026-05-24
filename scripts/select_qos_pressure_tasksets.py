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

    # 稳定 static 改进筛选：按用户指定 delta 选主字段，必要时允许 0.05 放宽到 0.10。
    if args.min_stable_static_sweep_reduction > 0.0:
        # 主字段由 --stable-static-delta 决定，确保筛选标准与训练目标一致。
        primary_field = "stable005_static_relative_lc_loss_reduction" if args.stable_static_delta == "0.05" else "stable010_static_relative_lc_loss_reduction"
        relaxed_field = "stable010_static_relative_lc_loss_reduction"
        if primary_field not in row:
            return "missing_stable_static_field"
        primary = get_float(row, primary_field)
        if primary is not None and primary >= args.min_stable_static_sweep_reduction:
            pass
        elif args.allow_relaxed_stable_static and args.stable_static_delta == "0.05":
            # 仅在 0.05 主约束下允许放宽到 0.10，避免语义混乱。
            relaxed = get_float(row, relaxed_field)
            if relaxed is None or relaxed < args.min_stable_static_sweep_reduction:
                return "insufficient_stable_static_sweep_reduction"
        else:
            return "insufficient_stable_static_sweep_reduction"

    if args.exclude_tradeoff_only:
        # 按主 stable delta 选择对应 tradeoff-only 标志位，保持判定口径一致。
        tradeoff_flag_field = "tradeoff_only_flag_005" if args.stable_static_delta == "0.05" else "tradeoff_only_flag_010"
        if tradeoff_flag_field not in row:
            return "missing_tradeoff_flag_field"
        if parse_bool_text(row.get(tradeoff_flag_field, "")):
            return "tradeoff_only"

    if args.require_single_action_improvement:
        primary_field = "single_stable005_static_relative_lc_loss_reduction" if args.single_stable_delta == "0.05" else "single_stable010_static_relative_lc_loss_reduction"
        relaxed_field = "single_stable010_static_relative_lc_loss_reduction"
        if primary_field not in row:
            return "insufficient_single_action_stable_improvement"
        primary = get_float(row, primary_field)
        if primary is not None and primary >= args.min_single_stable_sweep_reduction:
            pass
        elif args.allow_relaxed_single_stable and args.single_stable_delta == "0.05":
            relaxed = get_float(row, relaxed_field)
            if relaxed is None or relaxed < args.min_single_stable_sweep_reduction:
                return "insufficient_single_action_stable_improvement"
        else:
            return "insufficient_single_action_stable_improvement"

    if args.exclude_single_tradeoff_only:
        if str(row.get("single_improvement_type", "")).strip() == "single_tradeoff_only":
            return "single_tradeoff_only"

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
    parser.add_argument("--min-stable-static-sweep-reduction", type=float, default=0.0)
    parser.add_argument("--stable-static-delta", choices=["0.05", "0.10"], default="0.05")
    parser.add_argument("--allow-relaxed-stable-static", action="store_true")
    parser.add_argument("--exclude-tradeoff-only", action="store_true")
    parser.add_argument("--prefer-stable-static", action="store_true")
    parser.add_argument("--min-single-stable-sweep-reduction", type=float, default=0.0)
    parser.add_argument("--single-stable-delta", choices=["0.05", "0.10"], default="0.05")
    parser.add_argument("--allow-relaxed-single-stable", action="store_true")
    parser.add_argument("--require-single-action-improvement", action="store_true")
    parser.add_argument("--prefer-single-action-stable", action="store_true")
    parser.add_argument("--exclude-single-tradeoff-only", action="store_true")
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
        stable005_raw = row.get("stable005_static_relative_lc_loss_reduction", "")
        stable010_raw = row.get("stable010_static_relative_lc_loss_reduction", "")
        stable005 = float(stable005_raw) if str(stable005_raw).strip() else float("-inf")
        stable010 = float(stable010_raw) if str(stable010_raw).strip() else float("-inf")
        valid_increase_raw = row.get("valid_increase_count_mean", "")
        valid_decrease_raw = row.get("valid_decrease_count_mean", "")
        valid_increase = float(valid_increase_raw) if str(valid_increase_raw).strip() else float("-inf")
        valid_decrease = float(valid_decrease_raw) if str(valid_decrease_raw).strip() else float("-inf")
        total_events_raw = row.get("baseline_total_events_mean", "")
        total_events = float(total_events_raw) if str(total_events_raw).strip() else float("inf")
        if args.prefer_stable_static:
            # 稳定改进优先排序：先看 stable005，再看 stable010，再看动作空间，再看 loss 中心距离。
            return (
                -stable005,
                -stable010,
                -valid_increase,
                -valid_decrease,
                abs(loss - args.target_loss_center),
                total_events,
                candidate_seed,
            )
        if args.prefer_single_action_stable:
            single005_raw = row.get("single_stable005_static_relative_lc_loss_reduction", "")
            single010_raw = row.get("single_stable010_static_relative_lc_loss_reduction", "")
            single005 = float(single005_raw) if str(single005_raw).strip() else float("-inf")
            single010 = float(single010_raw) if str(single010_raw).strip() else float("-inf")
            return (
                -single005,
                -single010,
                -stable005,
                -stable010,
                abs(loss - args.target_loss_center),
                -valid_increase,
                -valid_decrease,
                total_events,
                candidate_seed,
            )
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
    stable_primary_field = "stable005_static_relative_lc_loss_reduction" if args.stable_static_delta == "0.05" else "stable010_static_relative_lc_loss_reduction"
    stable_primary_values = [float(row[stable_primary_field]) for row in selected_topk if str(row.get(stable_primary_field, "")).strip()]
    stable_primary_mean = sum(stable_primary_values) / len(stable_primary_values) if stable_primary_values else ""
    print(f"static_sweep_relative_lc_loss_reduction_mean_selected={reduction_mean}", flush=True)
    print(f"{stable_primary_field}_mean_selected={stable_primary_mean}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
