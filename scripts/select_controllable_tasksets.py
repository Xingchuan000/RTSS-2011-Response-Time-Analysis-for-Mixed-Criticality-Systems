"""基于 task-level controllability 指标筛选可训练任务集。

设计目标：
1. 读取 headroom summary、原始 manifest、controllability summary；
2. 按 candidate_seed 合并并执行硬过滤；
3. 计算 controllable_score_v2，并按多指标排序选出 topK；
4. 输出选中 summary、可直接训练的 manifest、以及 rejection 明细。

注意：
- 本脚本严格按计划文档实现，不引入额外策略性兜底逻辑；
- 字段兼容仅用于解决历史文件的命名差异，不改变筛选语义。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLS = {
    "seed": ["candidate_seed", "taskset_seed", "fixed_taskset_seed"],
    "valid_inc": ["valid_increase_count_mean", "fast_valid_increase_count_mean"],
    "valid_dec": ["valid_decrease_count_mean", "fast_valid_decrease_count_mean"],
    "events_per_1m": ["baseline_total_events_per_1m", "fast_baseline_total_events_mean"],
    "mode_per_1m": ["baseline_mode_changes_per_1m", "fast_baseline_mode_changes_mean"],
    "lo_cancel_per_1m": ["baseline_lo_cancellations_per_1m", "fast_baseline_lo_cancellations_mean"],
    "lo_cancel_ratio": ["baseline_lo_cancellation_ratio_total", "fast_baseline_lo_cancellation_ratio_total"],
    "deadline_misses": ["baseline_deadline_misses_sum", "deadline_misses_sum"],
    "top1_share": ["task_level_top1_cancel_share_mean", "top1_cancel_share_mean"],
    "top2_share": ["task_level_top2_cancel_share_mean", "top2_cancel_share_mean"],
    "top3_share": ["task_level_top3_cancel_share_mean", "top3_cancel_share_mean"],
    "hhi": ["cancel_concentration_hhi_mean", "task_level_cancel_concentration_hhi_mean"],
    "inc_cov": ["valid_increase_cancel_coverage_mean"],
    "inc_top2_hit": ["valid_increase_top2_cancel_hit_count_mean"],
    "inc_top3_hit": ["valid_increase_top3_cancel_hit_count_mean"],
}


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--controllability-summary-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-column", type=str, default="candidate_seed")

    parser.add_argument("--min-valid-increase", type=float, default=3.0)
    parser.add_argument("--min-valid-decrease", type=float, default=3.0)
    parser.add_argument("--min-baseline-lo-cancellations-per-1m", type=float, default=22.0)
    parser.add_argument("--max-baseline-mode-changes-per-1m", type=float, default=30.0)
    parser.add_argument("--min-baseline-lo-cancellation-ratio", type=float, default=0.45)
    parser.add_argument("--max-baseline-lo-cancellation-ratio", type=float, default=0.75)
    parser.add_argument("--min-valid-increase-cancel-coverage", type=float, default=0.35)
    parser.add_argument("--min-valid-increase-top2-hit-count", type=float, default=1.0)
    parser.add_argument("--min-valid-increase-top3-hit-count", type=float, default=1.5)
    parser.add_argument("--max-top1-cancel-share", type=float, default=0.75)
    parser.add_argument("--max-deadline-misses", type=float, default=0.0)

    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--output-rejections", type=Path, default=None)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-top-n", type=int, default=30)
    return parser


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """在 DataFrame 中按候选名顺序查找字段。"""

    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"缺少必需字段，候选列表: {candidates}")
    return None


def find_col_merged(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """在 merge 后数据中查找字段。

    说明：
    - merge 时右表重名字段会被加上 `_ctrl` 后缀；
    - 因此这里按 `原名 -> 原名_ctrl` 顺序查找。
    """

    expanded: list[str] = []
    for col in candidates:
        expanded.append(col)
        expanded.append(f"{col}_ctrl")
    return find_col(df, expanded, required=required)


def append_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    """为未通过过滤的样本追加拒绝原因（多原因使用分号拼接）。"""

    failed = ~mask
    empty = series == ""
    series.loc[failed & empty] = reason
    series.loc[failed & ~empty] = series.loc[failed & ~empty] + ";" + reason
    return series


def main() -> None:
    """执行 selector 主流程。"""

    args = build_parser().parse_args()

    summary = pd.read_csv(args.summary_csv)
    manifest = pd.read_csv(args.manifest_csv)
    ctrl = pd.read_csv(args.controllability_summary_csv)

    seed_col_summary = find_col(summary, [args.seed_column] + COLS["seed"])
    seed_col_manifest = find_col(manifest, [args.seed_column] + COLS["seed"])
    seed_col_ctrl = find_col(ctrl, [args.seed_column] + COLS["seed"])

    summary[seed_col_summary] = summary[seed_col_summary].astype(int)
    manifest[seed_col_manifest] = manifest[seed_col_manifest].astype(int)
    ctrl[seed_col_ctrl] = ctrl[seed_col_ctrl].astype(int)

    summary = summary.rename(columns={seed_col_summary: args.seed_column})
    manifest = manifest.rename(columns={seed_col_manifest: args.seed_column})
    ctrl = ctrl.rename(columns={seed_col_ctrl: args.seed_column})

    merged = summary.merge(ctrl, on=args.seed_column, how="inner", suffixes=("", "_ctrl"))

    valid_inc_col = find_col_merged(merged, COLS["valid_inc"])
    valid_dec_col = find_col_merged(merged, COLS["valid_dec"])
    lo_cancel_per_1m_col = find_col_merged(merged, COLS["lo_cancel_per_1m"])
    mode_per_1m_col = find_col_merged(merged, COLS["mode_per_1m"])
    lo_cancel_ratio_col = find_col_merged(merged, COLS["lo_cancel_ratio"])
    deadline_col = find_col_merged(merged, COLS["deadline_misses"])
    inc_cov_col = find_col_merged(merged, COLS["inc_cov"])
    inc_top2_hit_col = find_col_merged(merged, COLS["inc_top2_hit"])
    inc_top3_hit_col = find_col_merged(merged, COLS["inc_top3_hit"])
    top1_share_col = find_col_merged(merged, COLS["top1_share"])
    top2_share_col = find_col_merged(merged, COLS["top2_share"], required=False)
    top3_share_col = find_col_merged(merged, COLS["top3_share"], required=False)
    hhi_col = find_col_merged(merged, COLS["hhi"], required=False)

    # 先准备 rejection reason，再逐条施加硬过滤。
    rejection_reasons = pd.Series("", index=merged.index, dtype=str)

    cond_deadline = merged[deadline_col].fillna(0) <= args.max_deadline_misses
    rejection_reasons = append_reason(rejection_reasons, cond_deadline, "baseline_deadline_miss")

    cond_valid_inc = merged[valid_inc_col] >= args.min_valid_increase
    rejection_reasons = append_reason(rejection_reasons, cond_valid_inc, "valid_increase_too_low")

    cond_valid_dec = merged[valid_dec_col] >= args.min_valid_decrease
    rejection_reasons = append_reason(rejection_reasons, cond_valid_dec, "valid_decrease_too_low")

    cond_lo_cancel = merged[lo_cancel_per_1m_col] >= args.min_baseline_lo_cancellations_per_1m
    rejection_reasons = append_reason(rejection_reasons, cond_lo_cancel, "lo_cancellations_too_low")

    cond_mode = merged[mode_per_1m_col] <= args.max_baseline_mode_changes_per_1m
    rejection_reasons = append_reason(rejection_reasons, cond_mode, "mode_changes_too_high")

    cond_ratio_low = merged[lo_cancel_ratio_col] >= args.min_baseline_lo_cancellation_ratio
    rejection_reasons = append_reason(rejection_reasons, cond_ratio_low, "lo_cancel_ratio_too_low")

    cond_ratio_high = merged[lo_cancel_ratio_col] <= args.max_baseline_lo_cancellation_ratio
    rejection_reasons = append_reason(rejection_reasons, cond_ratio_high, "lo_cancel_ratio_too_high")

    cond_inc_cov = merged[inc_cov_col] >= args.min_valid_increase_cancel_coverage
    rejection_reasons = append_reason(rejection_reasons, cond_inc_cov, "valid_increase_coverage_too_low")

    cond_inc_top2 = merged[inc_top2_hit_col] >= args.min_valid_increase_top2_hit_count
    rejection_reasons = append_reason(rejection_reasons, cond_inc_top2, "top2_valid_increase_hit_too_low")

    cond_inc_top3 = merged[inc_top3_hit_col] >= args.min_valid_increase_top3_hit_count
    rejection_reasons = append_reason(rejection_reasons, cond_inc_top3, "top3_valid_increase_hit_too_low")

    cond_top1 = merged[top1_share_col] <= args.max_top1_cancel_share
    rejection_reasons = append_reason(rejection_reasons, cond_top1, "top1_cancel_share_too_high")

    pass_mask = (
        cond_deadline
        & cond_valid_inc
        & cond_valid_dec
        & cond_lo_cancel
        & cond_mode
        & cond_ratio_low
        & cond_ratio_high
        & cond_inc_cov
        & cond_inc_top2
        & cond_inc_top3
        & cond_top1
    )

    merged["controllable_score_v2"] = (
        merged[inc_cov_col]
        * merged[inc_top3_hit_col]
        * merged[lo_cancel_per_1m_col]
        / merged[mode_per_1m_col].clip(lower=1.0)
    )

    # 与计划一致：保留惩罚版 score 作为参考列，不作为唯一排序依据。
    merged["controllable_score_v2_penalized"] = (
        merged["controllable_score_v2"] * (1.0 - merged[top1_share_col].clip(lower=0.0, upper=1.0))
    )

    passed = merged[pass_mask].copy()
    sort_cols = [inc_cov_col, inc_top3_hit_col, "controllable_score_v2", lo_cancel_per_1m_col, mode_per_1m_col, top1_share_col]
    ascending = [False, False, False, False, True, True]
    passed = passed.sort_values(sort_cols, ascending=ascending)
    selected = passed.head(args.top_k).copy()

    selected_seed_order = selected[args.seed_column].astype(int).tolist()
    seed_to_order = {seed: idx for idx, seed in enumerate(selected_seed_order)}
    out_manifest = manifest[manifest[args.seed_column].isin(selected_seed_order)].copy()
    out_manifest["_order"] = out_manifest[args.seed_column].map(seed_to_order)
    out_manifest = out_manifest.sort_values("_order").drop(columns=["_order"])

    rejected = merged[~pass_mask].copy()
    rejected["rejection_reasons"] = rejection_reasons[~pass_mask].values

    summary_out_cols = [
        args.seed_column,
        "controllable_score_v2",
        inc_cov_col,
        inc_top2_hit_col,
        inc_top3_hit_col,
        top1_share_col,
        lo_cancel_per_1m_col,
        mode_per_1m_col,
        lo_cancel_ratio_col,
        valid_inc_col,
        valid_dec_col,
        deadline_col,
    ]
    if top2_share_col is not None:
        summary_out_cols.append(top2_share_col)
    if top3_share_col is not None:
        summary_out_cols.append(top3_share_col)
    if hhi_col is not None:
        summary_out_cols.append(hhi_col)

    # dry-run 模式只打印，不写输出文件。
    if args.dry_run:
        print("=== selector dry-run ===")
        print(f"summary_rows={len(summary)}")
        print(f"controllability_rows={len(ctrl)}")
        print(f"merged_rows={len(merged)}")
        print(f"passed_filters={len(passed)}")
        print(f"selected_top_k={len(selected)}")
        print(f"selected_seeds={selected_seed_order}")

        print("\n=== passed top rows ===")
        print(passed[summary_out_cols].head(args.print_top_n).to_string(index=False))

        print("\n=== rejection reason counts ===")
        reason_counts: dict[str, int] = {}
        for value in rejected["rejection_reasons"].astype(str):
            for reason in value.split(";"):
                reason = reason.strip()
                if not reason:
                    continue
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            for reason, count in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True):
                print(f"{reason}: {count}")
        else:
            print("(none)")

        print("\n=== key metric describe (passed) ===")
        describe_cols = [inc_cov_col, inc_top2_hit_col, inc_top3_hit_col, lo_cancel_per_1m_col, mode_per_1m_col, top1_share_col]
        print(passed[describe_cols].describe().to_string())
        return

    if args.output_summary is None or args.output_manifest is None or args.output_rejections is None:
        raise ValueError("非 dry-run 模式下必须提供 --output-summary/--output-manifest/--output-rejections")

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejections.parent.mkdir(parents=True, exist_ok=True)

    selected[summary_out_cols].to_csv(args.output_summary, index=False)
    out_manifest.to_csv(args.output_manifest, index=False)

    rejection_out_cols = [
        args.seed_column,
        "rejection_reasons",
        "controllable_score_v2",
        inc_cov_col,
        inc_top2_hit_col,
        inc_top3_hit_col,
        top1_share_col,
        lo_cancel_per_1m_col,
        mode_per_1m_col,
        lo_cancel_ratio_col,
    ]
    rejected[rejection_out_cols].to_csv(args.output_rejections, index=False)

    print("=== selector completed ===")
    print(f"summary_rows={len(summary)}")
    print(f"controllability_rows={len(ctrl)}")
    print(f"merged_rows={len(merged)}")
    print(f"passed_filters={len(passed)}")
    print(f"selected_top_k={len(selected)}")
    print(f"selected_seeds={selected_seed_order}")

    print("\n=== rejection reason counts ===")
    reason_counts: dict[str, int] = {}
    for value in rejected["rejection_reasons"].astype(str):
        for reason in value.split(";"):
            reason = reason.strip()
            if not reason:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{reason}: {count}")
    else:
        print("(none)")


if __name__ == "__main__":
    main()
