#!/usr/bin/env python3
"""Learnability-first selector for single-action MC-FairGen tasksets.

This selector is intended to preserve the original singlev3 philosophy:
first keep tasksets with healthy single-action headroom, then prefer
moderate QoS pressure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-rejections", type=Path, required=True)

    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument("--min-valid-increase", type=float, default=3.0)
    parser.add_argument("--min-valid-decrease", type=float, default=3.0)
    parser.add_argument("--min-balance", type=float, default=0.20)

    parser.add_argument("--min-events", type=float, default=40.0)
    parser.add_argument("--max-events", type=float, default=250.0)
    parser.add_argument("--min-events-per-1m", type=float, default=40.0)
    parser.add_argument("--max-events-per-1m", type=float, default=250.0)

    parser.add_argument("--min-lo-cancellations", type=float, default=10.0)
    parser.add_argument("--min-lo-cancellation-ratio", type=float, default=0.30)
    parser.add_argument("--max-lo-cancellation-ratio", type=float, default=0.75)

    parser.add_argument("--max-mode-changes", type=float, default=80.0)
    parser.add_argument("--max-mode-changes-per-1m", type=float, default=80.0)
    parser.add_argument("--min-lo-cancellations-per-1m", type=float, default=10.0)
    parser.add_argument("--use-per-1m-metrics", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--require-recommended-for-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--seed-column",
        type=str,
        default="candidate_seed",
    )

    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def add_reject_reason(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    reasons = []

    for _, row in df.iterrows():
        r = []

        if args.require_recommended_for_dqn and "recommended_for_dqn" in df.columns:
            if str(row.get("recommended_for_dqn", "")).lower() not in ["true", "1", "yes", "y"]:
                r.append("not_recommended_for_dqn")

        if "baseline_deadline_misses_sum" in df.columns:
            if float(row.get("baseline_deadline_misses_sum", 0.0)) > 0.0:
                r.append("deadline_misses")

        if float(row.get("valid_increase_count_mean", 0.0)) < args.min_valid_increase:
            r.append("low_valid_increase")

        if float(row.get("valid_decrease_count_mean", 0.0)) < args.min_valid_decrease:
            r.append("low_valid_decrease")

        balance = float(row.get("increase_decrease_balance", 0.0))
        if balance < args.min_balance:
            r.append("low_increase_decrease_balance")

        use_per_1m = bool(args.use_per_1m_metrics and "baseline_total_events_per_1m" in df.columns)
        if use_per_1m:
            events = float(row.get("baseline_total_events_per_1m", 0.0))
            max_mode = args.max_mode_changes_per_1m
            min_events = args.min_events_per_1m
            max_events = args.max_events_per_1m
            min_lo_cancels = args.min_lo_cancellations_per_1m
            mode_value = float(row.get("baseline_mode_changes_per_1m", row.get("baseline_mode_changes_mean", 0.0)))
            cancels = float(row.get("baseline_lo_cancellations_per_1m", row.get("baseline_lo_cancellations_mean", 0.0)))
        else:
            events = float(row.get("baseline_total_events_mean", 0.0))
            max_mode = args.max_mode_changes
            min_events = args.min_events
            max_events = args.max_events
            min_lo_cancels = args.min_lo_cancellations
            mode_value = float(row.get("baseline_mode_changes_mean", 0.0))
            cancels = float(row.get("baseline_lo_cancellations_mean", 0.0))
        if events < min_events:
            r.append("too_few_events")
        if events > max_events:
            r.append("too_many_events")
        if cancels < min_lo_cancels:
            r.append("too_few_lo_cancellations")

        ratio = float(row.get("baseline_lo_cancellation_ratio_total", 0.0))
        if ratio < args.min_lo_cancellation_ratio:
            r.append("low_lo_cancellation_ratio")
        if ratio > args.max_lo_cancellation_ratio:
            r.append("too_high_lo_cancellation_ratio")

        if mode_value > max_mode:
                r.append("too_many_mode_changes")

        reasons.append(";".join(r))

    out = df.copy()
    out["select_reject_reason"] = reasons
    return out


def main() -> None:
    args = parse_args()
    if args.use_per_1m_metrics:
        print("WARNING: --min-events/--max-events/--max-mode-changes/--min-lo-cancellations 为旧参数，已优先使用 per-1M 参数。")

    summary = pd.read_csv(args.summary_csv)
    manifest = pd.read_csv(args.manifest_csv)

    if args.seed_column not in summary.columns:
        raise SystemExit(f"Missing seed column in summary: {args.seed_column}")
    if args.seed_column not in manifest.columns:
        raise SystemExit(f"Missing seed column in manifest: {args.seed_column}")

    required = [
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "baseline_total_events_mean",
        "baseline_lo_cancellations_mean",
        "baseline_lo_cancellation_ratio_total",
    ]
    missing = [c for c in required if c not in summary.columns]
    if missing:
        raise SystemExit(f"Missing required summary columns: {missing}")

    df = add_reject_reason(summary, args)

    selected = df[df["select_reject_reason"].eq("")].copy()
    rejected = df[~df["select_reject_reason"].eq("")].copy()

    if selected.empty:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_rejections.parent.mkdir(parents=True, exist_ok=True)

        selected.to_csv(args.output_summary, index=False)
        rejected.to_csv(args.output_rejections, index=False)
        manifest.iloc[0:0].to_csv(args.output_manifest, index=False)

        print("selected=0")
        print("No tasksets passed the learnability-first filters.")
        print("Top rejection reasons:")
        print(rejected["select_reject_reason"].value_counts().head(20).to_string())
        return

    selected["valid_min_mean"] = selected[
        ["valid_increase_count_mean", "valid_decrease_count_mean"]
    ].min(axis=1)

    selected["valid_total_mean"] = (
        selected["valid_increase_count_mean"] + selected["valid_decrease_count_mean"]
    )

    selected["balance_distance_to_one"] = (
        selected["increase_decrease_balance"] - 1.0
    ).abs()

    sort_cols = [
        "valid_min_mean",
        "valid_total_mean",
        "balance_distance_to_one",
        "baseline_lo_cancellation_ratio_total",
        "baseline_lo_cancellations_mean",
        "baseline_total_events_mean",
    ]
    ascending = [
        False,
        False,
        True,
        False,
        False,
        True,
    ]

    if "baseline_mode_changes_mean" in selected.columns:
        sort_cols.append("baseline_mode_changes_mean")
        ascending.append(True)

    sort_cols.append(args.seed_column)
    ascending.append(True)

    selected = selected.sort_values(sort_cols, ascending=ascending)
    top = selected.head(args.top_k).copy()

    seeds = top[args.seed_column].astype(int).tolist()

    manifest = manifest.copy()
    manifest[args.seed_column] = manifest[args.seed_column].astype(int)

    top_manifest = (
        manifest[manifest[args.seed_column].isin(seeds)]
        .set_index(args.seed_column)
        .loc[seeds]
        .reset_index()
    )

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejections.parent.mkdir(parents=True, exist_ok=True)

    top.to_csv(args.output_summary, index=False)
    top_manifest.to_csv(args.output_manifest, index=False)
    rejected.to_csv(args.output_rejections, index=False)

    print("[Learnability-first Select Summary]")
    print(f"summary_rows={len(summary)}")
    print(f"passed_filters={len(selected)}")
    print(f"selected_top_k={len(top)}")
    print(f"rejected={len(rejected)}")
    print("selected_seeds=" + ",".join(map(str, seeds)))

    show_cols = [
        args.seed_column,
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "valid_min_mean",
        "valid_total_mean",
        "increase_decrease_balance",
        "baseline_lo_cancellation_ratio_total",
        "baseline_lo_cancellations_mean",
        "baseline_total_events_mean",
        "baseline_mode_changes_mean",
    ]
    show_cols = [c for c in show_cols if c in top.columns]

    print("\nTop selected:")
    print(top[show_cols].to_string(index=False))

    print("\nTop rejection reasons:")
    print(rejected["select_reject_reason"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
