#!/usr/bin/env python3
"""
Select hybrid-controllable MC tasksets using two lanes:

1) direct lane: high valid-increase cancellation coverage;
2) indirect lane: medium valid-increase coverage but high cancellation pressure and low mode pressure.

This script is intended to complement select_controllable_tasksets.py. It produces a top-k
manifest/summary from an existing headroom summary, taskset manifest, and task-level
controllability summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    if "candidate_seed" not in df.columns:
        raise ValueError(f"{name} must contain candidate_seed column: {path}")
    df = df.copy()
    df["candidate_seed"] = df["candidate_seed"].astype(int)
    return df


def _merge_inputs(summary: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    merged = summary.merge(
        cont,
        on="candidate_seed",
        suffixes=("", "_cont"),
        how="inner",
    )

    # Prefer base columns, but fill missing values from *_cont duplicates.
    for col in list(merged.columns):
        if not col.endswith("_cont"):
            continue
        base = col[:-5]
        if base not in merged.columns:
            merged[base] = merged[col]
        else:
            merged[base] = merged[base].fillna(merged[col])

    # Defragment after filling duplicate columns.
    return merged.copy()


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mode = out["baseline_mode_changes_per_1m"].clip(lower=1.0)

    out["direct_score"] = (
        out["valid_increase_cancel_coverage_mean"]
        * out["valid_increase_top3_cancel_hit_count_mean"]
        * out["baseline_lo_cancellations_per_1m"]
        / mode
    )

    out["indirect_score"] = (
        out["baseline_lo_cancellations_per_1m"]
        * out["baseline_lo_cancellation_ratio_total"]
        / mode
        * out["valid_increase_top3_cancel_hit_count_mean"]
        * (1.0 - 0.5 * out["task_level_top1_cancel_share_mean"])
    )

    # Useful unified score for inspecting selected rows.
    out["hybrid_score"] = np.where(
        out["valid_increase_cancel_coverage_mean"] >= 0.35,
        out["direct_score"],
        out["indirect_score"],
    )
    return out.copy()


def _build_conditions(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, pd.Series, pd.Series]:
    base = pd.Series(True, index=df.index)

    if "baseline_deadline_misses_sum" in df.columns and args.require_zero_deadline_misses:
        base &= df["baseline_deadline_misses_sum"].fillna(0).eq(0)

    base &= df["valid_increase_count_mean"] >= args.min_valid_increase
    base &= df["valid_decrease_count_mean"] >= args.min_valid_decrease
    base &= df["valid_increase_top2_cancel_hit_count_mean"] >= args.min_valid_increase_top2_hit_count
    base &= df["valid_increase_top3_cancel_hit_count_mean"] >= args.min_valid_increase_top3_hit_count
    base &= df["task_level_top1_cancel_share_mean"] <= args.max_top1_cancel_share
    base &= df["baseline_lo_cancellation_ratio_total"].between(
        args.min_baseline_lo_cancellation_ratio,
        args.max_baseline_lo_cancellation_ratio,
    )

    direct = base.copy()
    direct &= df["valid_increase_cancel_coverage_mean"] >= args.direct_min_valid_increase_cancel_coverage
    direct &= df["baseline_lo_cancellations_per_1m"] >= args.direct_min_baseline_lo_cancellations_per_1m
    direct &= df["baseline_mode_changes_per_1m"] <= args.direct_max_baseline_mode_changes_per_1m

    indirect = base.copy()
    indirect &= df["valid_increase_cancel_coverage_mean"] >= args.indirect_min_valid_increase_cancel_coverage
    indirect &= df["baseline_lo_cancellations_per_1m"] >= args.indirect_min_baseline_lo_cancellations_per_1m
    indirect &= df["baseline_mode_changes_per_1m"] <= args.indirect_max_baseline_mode_changes_per_1m
    indirect &= df["baseline_lo_cancellation_ratio_total"] >= args.indirect_min_baseline_lo_cancellation_ratio

    return base, direct, indirect


def _rank_direct(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(
        [
            "valid_increase_cancel_coverage_mean",
            "valid_increase_top3_cancel_hit_count_mean",
            "direct_score",
            "baseline_lo_cancellations_per_1m",
            "baseline_mode_changes_per_1m",
            "task_level_top1_cancel_share_mean",
            "candidate_seed",
        ],
        ascending=[False, False, False, False, True, True, True],
    ).copy()


def _rank_indirect(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(
        [
            "indirect_score",
            "baseline_lo_cancellations_per_1m",
            "baseline_lo_cancellation_ratio_total",
            "baseline_mode_changes_per_1m",
            "valid_increase_cancel_coverage_mean",
            "task_level_top1_cancel_share_mean",
            "candidate_seed",
        ],
        ascending=[False, False, False, True, False, True, True],
    ).copy()


def _select_rows(
    direct: pd.DataFrame,
    indirect: pd.DataFrame,
    relaxed: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    selected = []
    selected_seeds: set[int] = set()

    anchor_seeds = []
    if args.anchor_seeds:
        anchor_seeds = [int(x.strip()) for x in args.anchor_seeds.split(",") if x.strip()]

    pool_for_anchors = pd.concat([direct, indirect, relaxed], ignore_index=True)
    pool_for_anchors = pool_for_anchors.drop_duplicates(subset=["candidate_seed"], keep="first")

    for seed in anchor_seeds:
        hit = pool_for_anchors[pool_for_anchors["candidate_seed"].astype(int).eq(seed)]
        if hit.empty:
            if args.allow_missing_anchors:
                print(f"WARNING: anchor seed {seed} not found in candidate pools; skipping")
                continue
            raise ValueError(f"Anchor seed {seed} does not pass any lane/backfill pool")
        row = hit.iloc[0].copy()
        row["hybrid_lane"] = f"anchor_{row.get('hybrid_lane', 'candidate')}"
        selected.append(row)
        selected_seeds.add(seed)

    direct_count = 0
    for _, row in direct.iterrows():
        seed = int(row["candidate_seed"])
        if seed in selected_seeds:
            continue
        selected.append(row)
        selected_seeds.add(seed)
        direct_count += 1
        if direct_count >= args.direct_quota:
            break

    indirect_count = 0
    for _, row in indirect.iterrows():
        seed = int(row["candidate_seed"])
        if seed in selected_seeds:
            continue
        selected.append(row)
        selected_seeds.add(seed)
        indirect_count += 1
        if indirect_count >= args.indirect_quota:
            break

    if len(selected) < args.top_k:
        for _, row in direct.iterrows():
            seed = int(row["candidate_seed"])
            if seed in selected_seeds:
                continue
            selected.append(row)
            selected_seeds.add(seed)
            if len(selected) >= args.top_k:
                break

    if len(selected) < args.top_k:
        for _, row in relaxed.iterrows():
            seed = int(row["candidate_seed"])
            if seed in selected_seeds:
                continue
            selected.append(row)
            selected_seeds.add(seed)
            if len(selected) >= args.top_k:
                break

    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected).head(args.top_k).reset_index(drop=True)
    out["hybrid_rank"] = np.arange(1, len(out) + 1)
    return out.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select hybrid direct/indirect controllable tasksets."
    )

    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--controllability-summary-csv", type=Path, required=True)

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--direct-quota", type=int, default=12)
    parser.add_argument("--indirect-quota", type=int, default=8)
    parser.add_argument("--anchor-seeds", type=str, default="")
    parser.add_argument("--allow-missing-anchors", action="store_true")

    parser.add_argument("--min-valid-increase", type=float, default=3.0)
    parser.add_argument("--min-valid-decrease", type=float, default=3.0)
    parser.add_argument("--min-valid-increase-top2-hit-count", type=float, default=1.0)
    parser.add_argument("--min-valid-increase-top3-hit-count", type=float, default=2.0)
    parser.add_argument("--max-top1-cancel-share", type=float, default=0.75)
    parser.add_argument("--min-baseline-lo-cancellation-ratio", type=float, default=0.45)
    parser.add_argument("--max-baseline-lo-cancellation-ratio", type=float, default=0.75)
    parser.add_argument("--require-zero-deadline-misses", action="store_true", default=True)
    parser.add_argument("--allow-deadline-misses", dest="require_zero_deadline_misses", action="store_false")

    parser.add_argument("--direct-min-valid-increase-cancel-coverage", type=float, default=0.35)
    parser.add_argument("--direct-min-baseline-lo-cancellations-per-1m", type=float, default=20.0)
    parser.add_argument("--direct-max-baseline-mode-changes-per-1m", type=float, default=30.0)

    parser.add_argument("--indirect-min-valid-increase-cancel-coverage", type=float, default=0.30)
    parser.add_argument("--indirect-min-baseline-lo-cancellations-per-1m", type=float, default=30.0)
    parser.add_argument("--indirect-max-baseline-mode-changes-per-1m", type=float, default=18.0)
    parser.add_argument("--indirect-min-baseline-lo-cancellation-ratio", type=float, default=0.60)

    parser.add_argument("--relaxed-backfill-min-valid-increase-cancel-coverage", type=float, default=0.30)
    parser.add_argument("--relaxed-backfill-min-baseline-lo-cancellations-per-1m", type=float, default=20.0)
    parser.add_argument("--relaxed-backfill-max-baseline-mode-changes-per-1m", type=float, default=30.0)

    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-rejections", type=Path)
    parser.add_argument("--output-candidates", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-top-n", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = _read_csv(args.summary_csv, "summary csv")
    manifest = _read_csv(args.manifest_csv, "manifest csv")
    cont = _read_csv(args.controllability_summary_csv, "controllability summary csv")

    required = [
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_lo_cancellation_ratio_total",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "task_level_top1_cancel_share_mean",
    ]

    merged = _merge_inputs(summary, cont)
    _require_columns(merged, required)
    merged = _add_scores(merged)

    base_cond, direct_cond, indirect_cond = _build_conditions(merged, args)

    direct = merged[direct_cond].copy()
    direct["hybrid_lane"] = "direct"
    direct = _rank_direct(direct)

    indirect = merged[indirect_cond].copy()
    indirect["hybrid_lane"] = "indirect"
    indirect = _rank_indirect(indirect)

    relaxed_cond = base_cond.copy()
    relaxed_cond &= merged["valid_increase_cancel_coverage_mean"] >= args.relaxed_backfill_min_valid_increase_cancel_coverage
    relaxed_cond &= merged["baseline_lo_cancellations_per_1m"] >= args.relaxed_backfill_min_baseline_lo_cancellations_per_1m
    relaxed_cond &= merged["baseline_mode_changes_per_1m"] <= args.relaxed_backfill_max_baseline_mode_changes_per_1m

    relaxed = merged[relaxed_cond].copy()
    relaxed["hybrid_lane"] = "relaxed_backfill"
    relaxed = relaxed.sort_values(
        [
            "direct_score",
            "valid_increase_cancel_coverage_mean",
            "baseline_lo_cancellations_per_1m",
            "baseline_mode_changes_per_1m",
            "candidate_seed",
        ],
        ascending=[False, False, False, True, True],
    ).copy()

    selected = _select_rows(direct, indirect, relaxed, args)

    candidates = pd.concat([direct, indirect, relaxed], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["candidate_seed"], keep="first")

    passed_seeds = set(candidates["candidate_seed"].astype(int).tolist())
    rejections = merged[~merged["candidate_seed"].astype(int).isin(passed_seeds)].copy()

    print("=== hybrid selector ===")
    print(f"summary_rows={len(summary)}")
    print(f"controllability_rows={len(cont)}")
    print(f"merged_rows={len(merged)}")
    print(f"direct_candidates={len(direct)}")
    print(f"indirect_candidates={len(indirect)}")
    print(f"relaxed_backfill_candidates={len(relaxed)}")
    print(f"hybrid_candidates_unique={len(candidates)}")
    print(f"selected_top_k={len(selected)}")
    if len(selected):
        print("selected_seeds=" + str(selected["candidate_seed"].astype(int).tolist()))

    show = [
        "hybrid_rank",
        "candidate_seed",
        "hybrid_lane",
        "direct_score",
        "indirect_score",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "task_level_top1_cancel_share_mean",
        "baseline_lo_cancellations_per_1m",
        "baseline_mode_changes_per_1m",
        "baseline_lo_cancellation_ratio_total",
        "baseline_deadline_misses_sum",
    ]
    show = [c for c in show if c in selected.columns]
    if len(selected):
        print("\n=== selected rows ===")
        print(selected[show].to_string(index=False))

    print("\n=== lane counts ===")
    if len(selected) and "hybrid_lane" in selected.columns:
        print(selected["hybrid_lane"].value_counts().to_string())
    else:
        print("<none>")

    if args.dry_run:
        print("\nDry-run only; no files written.")
        return

    if not args.output_summary or not args.output_manifest:
        raise ValueError("--output-summary and --output-manifest are required unless --dry-run is set")

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(args.output_summary, index=False)

    manifest_selected = manifest[manifest["candidate_seed"].astype(int).isin(selected["candidate_seed"].astype(int))].copy()
    manifest_selected = (
        manifest_selected.set_index("candidate_seed")
        .loc[selected["candidate_seed"].astype(int).tolist()]
        .reset_index()
    )
    manifest_selected.to_csv(args.output_manifest, index=False)

    if args.output_rejections:
        args.output_rejections.parent.mkdir(parents=True, exist_ok=True)
        rejections.to_csv(args.output_rejections, index=False)

    if args.output_candidates:
        args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.output_candidates, index=False)

    print("\nSaved:")
    print(f"summary: {args.output_summary}")
    print(f"manifest: {args.output_manifest}")
    if args.output_rejections:
        print(f"rejections: {args.output_rejections}")
    if args.output_candidates:
        print(f"candidates: {args.output_candidates}")


if __name__ == "__main__":
    main()
