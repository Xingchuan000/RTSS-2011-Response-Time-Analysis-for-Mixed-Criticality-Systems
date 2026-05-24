#!/usr/bin/env python3
"""
High-opportunity selector for controlled_medium_scale500 tasksets.

This script selects tasksets before DQN training using only pre-training
headroom/baseline summary and controllability probe summary.

Default inputs match the current AMC project layout:
  outputs/taskset_slack_scan/controlled_medium_scale500/headroom_quick_top300_summary_0_3000_e3m_s200_206.csv
  outputs/tasksets/controlled_medium_scale500/prescreen_top300_0_3000.csv
  outputs/taskset_controllability/controlled_medium_scale500/prescreen_top300_controllability_summary_e1m_s200_210.csv

Default outputs:
  outputs/tasksets/controlled_medium_scale500/high_opportunity_v1_top{10,12,15}_e1m_s200_210.csv
  outputs/taskset_controllability/controlled_medium_scale500/high_opportunity_v1_ranked_summary_e1m_s200_210.csv
  outputs/taskset_controllability/controlled_medium_scale500/high_opportunity_v1_rejections_e1m_s200_210.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_topk(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--top-k must contain at least one integer")
    try:
        topks = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--top-k must be comma-separated integers") from exc
    if any(k <= 0 for k in topks):
        raise argparse.ArgumentTypeError("all --top-k values must be positive")
    return topks


def norm_seed_col(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "candidate_seed" in df.columns:
        return df
    for c in ["seed", "taskset_seed", "fixed_taskset_seed"]:
        if c in df.columns:
            return df.rename(columns={c: "candidate_seed"})
    raise ValueError(f"No candidate_seed-like column found in {source_name}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Required column(s) missing: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select high-opportunity controllable tasksets before DQN training."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/x1ngchuan/Documents/AMC"),
        help="AMC project root directory.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Headroom/baseline summary CSV. Defaults to controlled_medium_scale500 top300 summary.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Input manifest CSV containing candidate_seed. Defaults to prescreen_top300_0_3000.csv.",
    )
    parser.add_argument(
        "--controllability-csv",
        type=Path,
        default=None,
        help="Controllability probe summary CSV. Defaults to prescreen_top300 controllability summary.",
    )
    parser.add_argument(
        "--out-taskset-dir",
        type=Path,
        default=None,
        help="Directory for selected manifest files.",
    )
    parser.add_argument(
        "--out-summary-dir",
        type=Path,
        default=None,
        help="Directory for ranked/rejected selector summaries.",
    )
    parser.add_argument(
        "--tag",
        default="high_opportunity_v1",
        help="Output filename tag.",
    )
    parser.add_argument(
        "--suffix",
        default="e1m_s200_210",
        help="Output filename suffix.",
    )
    parser.add_argument(
        "--top-k",
        type=parse_topk,
        default=[10, 12, 15],
        help="Comma-separated top-k manifest sizes to export, e.g. 10,12,15.",
    )

    # Selector thresholds: default high_opportunity_v1.
    parser.add_argument("--min-valid-increase-count", type=float, default=3.0)
    parser.add_argument("--min-valid-decrease-count", type=float, default=3.0)
    parser.add_argument("--min-baseline-lo-cancellations-per-1m", type=float, default=22.0)
    parser.add_argument("--max-baseline-mode-changes-per-1m", type=float, default=30.0)
    parser.add_argument("--min-baseline-lo-cancellation-ratio", type=float, default=0.48)
    parser.add_argument("--max-baseline-lo-cancellation-ratio", type=float, default=0.75)
    parser.add_argument("--min-valid-increase-cancel-coverage", type=float, default=0.35)
    parser.add_argument("--min-valid-increase-top2-hit", type=float, default=1.0)
    parser.add_argument("--min-valid-increase-top3-hit", type=float, default=2.0)
    parser.add_argument("--max-top1-cancel-share", type=float, default=0.75)

    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.root
    base = "controlled_medium_scale500"

    if args.summary_csv is None:
        args.summary_csv = root / "outputs" / "taskset_slack_scan" / base / "headroom_quick_top300_summary_0_3000_e3m_s200_206.csv"
    if args.manifest_csv is None:
        args.manifest_csv = root / "outputs" / "tasksets" / base / "prescreen_top300_0_3000.csv"
    if args.controllability_csv is None:
        args.controllability_csv = root / "outputs" / "taskset_controllability" / base / "prescreen_top300_controllability_summary_e1m_s200_210.csv"
    if args.out_taskset_dir is None:
        args.out_taskset_dir = root / "outputs" / "tasksets" / base
    if args.out_summary_dir is None:
        args.out_summary_dir = root / "outputs" / "taskset_controllability" / base

    return args


def main() -> int:
    parser = build_parser()
    args = resolve_default_paths(parser.parse_args())

    for p, label in [
        (args.summary_csv, "summary CSV"),
        (args.manifest_csv, "manifest CSV"),
        (args.controllability_csv, "controllability CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    args.out_taskset_dir.mkdir(parents=True, exist_ok=True)
    args.out_summary_dir.mkdir(parents=True, exist_ok=True)

    summary = norm_seed_col(pd.read_csv(args.summary_csv), "summary_csv")
    manifest = norm_seed_col(pd.read_csv(args.manifest_csv), "manifest_csv")
    ctrl = norm_seed_col(pd.read_csv(args.controllability_csv), "controllability_csv")

    summary_cols = ["candidate_seed"] + [c for c in summary.columns if c != "candidate_seed"]
    ctrl_cols = ["candidate_seed"] + [c for c in ctrl.columns if c != "candidate_seed"]

    merged = summary[summary_cols].merge(
        ctrl[ctrl_cols],
        on="candidate_seed",
        how="inner",
        suffixes=("", "_ctrl"),
    )

    required = [
        "candidate_seed",
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_lo_cancellation_ratio",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "top1_cancel_share",
    ]
    require_columns(merged, required)

    cond = (
        (merged["valid_increase_count_mean"] >= args.min_valid_increase_count)
        & (merged["valid_decrease_count_mean"] >= args.min_valid_decrease_count)
        & (merged["baseline_lo_cancellations_per_1m"] >= args.min_baseline_lo_cancellations_per_1m)
        & (merged["baseline_mode_changes_per_1m"] <= args.max_baseline_mode_changes_per_1m)
        & (merged["baseline_lo_cancellation_ratio"] >= args.min_baseline_lo_cancellation_ratio)
        & (merged["baseline_lo_cancellation_ratio"] <= args.max_baseline_lo_cancellation_ratio)
        & (merged["valid_increase_cancel_coverage_mean"] >= args.min_valid_increase_cancel_coverage)
        & (merged["valid_increase_top2_cancel_hit_count_mean"] >= args.min_valid_increase_top2_hit)
        & (merged["valid_increase_top3_cancel_hit_count_mean"] >= args.min_valid_increase_top3_hit)
        & (merged["top1_cancel_share"] <= args.max_top1_cancel_share)
    )

    passed = merged[cond].copy()
    rejected = merged[~cond].copy()

    passed["top_hit_factor"] = (
        1.0
        + 0.20 * passed["valid_increase_top2_cancel_hit_count_mean"]
        + 0.10 * passed["valid_increase_top3_cancel_hit_count_mean"]
    )
    passed["source_diversity_factor"] = 1.0 - 0.25 * passed["top1_cancel_share"]
    passed["mode_safety_factor"] = 1.0 - passed["baseline_mode_changes_per_1m"] / 60.0
    passed["mode_safety_factor"] = passed["mode_safety_factor"].clip(lower=0.1, upper=1.0)

    passed["expected_reachable_gain_score"] = (
        passed["baseline_lo_cancellations_per_1m"]
        * passed["valid_increase_cancel_coverage_mean"]
        * passed["top_hit_factor"]
        * passed["source_diversity_factor"]
        * passed["mode_safety_factor"]
    )

    sort_cols = [
        "expected_reachable_gain_score",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "baseline_lo_cancellations_per_1m",
    ]
    passed = passed.sort_values(sort_cols, ascending=[False, False, False, False])

    ranked_path = args.out_summary_dir / f"{args.tag}_ranked_summary_{args.suffix}.csv"
    rejected_path = args.out_summary_dir / f"{args.tag}_rejections_{args.suffix}.csv"

    passed.to_csv(ranked_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    print("=== high-opportunity selector ===")
    print(f"summary_csv          = {args.summary_csv}")
    print(f"manifest_csv         = {args.manifest_csv}")
    print(f"controllability_csv  = {args.controllability_csv}")
    print(f"summary_rows         = {len(summary)}")
    print(f"manifest_rows        = {len(manifest)}")
    print(f"controllability_rows = {len(ctrl)}")
    print(f"merged_rows          = {len(merged)}")
    print(f"passed_filters       = {len(passed)}")
    print(f"rejected             = {len(rejected)}")
    print(f"ranked_summary       = {ranked_path}")
    print(f"rejections           = {rejected_path}")

    if passed.empty:
        print("[WARN] no taskset passed the high-opportunity filters; no top-k manifests written.")
        return 0

    for k in args.top_k:
        selected = passed.head(k).copy()
        seeds = selected["candidate_seed"].astype(int).tolist()

        selected_manifest = manifest[
            manifest["candidate_seed"].astype(int).isin(seeds)
        ].copy()

        order = {seed: i for i, seed in enumerate(seeds)}
        selected_manifest["_rank"] = selected_manifest["candidate_seed"].astype(int).map(order)
        selected_manifest = selected_manifest.sort_values("_rank").drop(columns=["_rank"])

        out_manifest = args.out_taskset_dir / f"{args.tag}_top{k}_{args.suffix}.csv"
        out_summary = args.out_summary_dir / f"{args.tag}_top{k}_summary_{args.suffix}.csv"

        selected_manifest.to_csv(out_manifest, index=False)
        selected.to_csv(out_summary, index=False)

        print()
        print(f"top{k}_seeds    = {seeds}")
        print(f"top{k}_manifest = {out_manifest}")
        print(f"top{k}_summary  = {out_summary}")

    print()
    show_cols = [
        "candidate_seed",
        "expected_reachable_gain_score",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_lo_cancellation_ratio",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "top1_cancel_share",
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
    ]
    show_cols = [c for c in show_cols if c in passed.columns]
    print("=== top ranked preview ===")
    print(passed[show_cols].head(max(args.top_k)).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
