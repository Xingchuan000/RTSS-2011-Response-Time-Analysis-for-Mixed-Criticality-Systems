#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_broad_stratified_tasksets.py

Create stratified broad taskset manifests for broader evaluation.

This script is intentionally different from strict / hybrid top-K selectors:
it does NOT try to select the strongest tasksets. Instead, it constructs
low / medium / high-opportunity buckets from a broad candidate pool and samples
approximately evenly across each bucket by opportunity_score.

Typical usage from AMC project root:

python scripts/select_broad_stratified_tasksets.py ^
  --manifest-csv outputs/tasksets/controlled_medium_scale500/prescreen_top300_0_3000.csv ^
  --summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/headroom_quick_top300_summary_0_3000_e3m_s200_206.csv ^
  --controllability-summary-csv outputs/taskset_controllability/controlled_medium_scale500/prescreen_top300_controllability_summary_e1m_s200_210.csv ^
  --preset broad30 ^
  --output-manifest outputs/tasksets/controlled_medium_scale500/broad30_stratified_e1m_s200_210.csv ^
  --output-summary outputs/tasksets/controlled_medium_scale500/broad30_stratified_summary_e1m_s200_210.csv ^
  --output-candidates outputs/tasksets/controlled_medium_scale500/broad30_stratified_candidates_e1m_s200_210.csv ^
  --output-rejections outputs/tasksets/controlled_medium_scale500/broad30_stratified_rejections_e1m_s200_210.csv

Presets:
  broad30  -> low=10, medium=10, high_not_top20=10
  broad60  -> low=20, medium=20, high_not_top20=20
  broad100 -> low=30, medium=40, high_not_top20=30

The output manifest includes at least:
  candidate_seed, broad_bucket, bucket_rank, bucket_quantile, opportunity_score
plus key diagnostics/pressure/action-space fields.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_MAIN_TOP20 = [
    29, 185, 358, 397, 639, 671, 735, 861, 962, 1169,
    1264, 1502, 1511, 1896, 2221, 2301, 2429, 2452, 2535, 2829,
]

DEFAULT_OLD_OR_HYBRID_SEEDS = [
    # old final top20 extras / hybrid extras frequently used in earlier experiments
    1484, 2202, 870, 2784, 2247, 2155, 226, 693, 57, 90, 2574, 1881,
]


def _read_csv(path: Optional[str], name: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"{name} is empty: {p}")
    return df


def _find_seed_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["candidate_seed", "fixed_taskset_seed", "taskset_seed", "seed"]:
        if c in df.columns:
            return c
    raise KeyError(
        "Could not find seed column. Expected one of: "
        "candidate_seed, fixed_taskset_seed, taskset_seed, seed"
    )


def _coerce_seed_col(df: pd.DataFrame, seed_col: str) -> pd.DataFrame:
    out = df.copy()
    out[seed_col] = pd.to_numeric(out[seed_col], errors="coerce")
    out = out.dropna(subset=[seed_col])
    out[seed_col] = out[seed_col].astype(int)
    return out


def _rename_with_suffix_except_seed(df: pd.DataFrame, seed_col: str, suffix: str) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        if c != seed_col:
            mapping[c] = f"{c}{suffix}"
    return df.rename(columns=mapping)


def _merge_sources(
    manifest: pd.DataFrame,
    summary: Optional[pd.DataFrame],
    ctrl: Optional[pd.DataFrame],
    seed_col: Optional[str],
) -> pd.DataFrame:
    man_seed = _find_seed_col(manifest, seed_col)
    manifest = _coerce_seed_col(manifest, man_seed)
    if man_seed != "candidate_seed":
        manifest = manifest.rename(columns={man_seed: "candidate_seed"})

    # Drop duplicate columns cautiously; keep first row per seed in manifest.
    base = manifest.drop_duplicates(subset=["candidate_seed"]).copy()

    if summary is not None:
        sum_seed = _find_seed_col(summary, seed_col)
        summary = _coerce_seed_col(summary, sum_seed)
        if sum_seed != "candidate_seed":
            summary = summary.rename(columns={sum_seed: "candidate_seed"})
        # Keep unsuffixed names. If duplicate columns exist, pandas merge will suffix.
        summary = summary.drop_duplicates(subset=["candidate_seed"])
        base = base.merge(summary, on="candidate_seed", how="left", suffixes=("", "_summary"))

    if ctrl is not None:
        ctrl_seed = _find_seed_col(ctrl, seed_col)
        ctrl = _coerce_seed_col(ctrl, ctrl_seed)
        if ctrl_seed != "candidate_seed":
            ctrl = ctrl.rename(columns={ctrl_seed: "candidate_seed"})
        ctrl = ctrl.drop_duplicates(subset=["candidate_seed"])
        base = base.merge(ctrl, on="candidate_seed", how="left", suffixes=("", "_ctrl"))

    return base


def _first_existing_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _ensure_numeric(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _ensure_required_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common metric names into canonical columns used by the broad selector.
    The source files may differ slightly depending on earlier script versions.
    """
    out = df.copy()

    alias_groups = {
        "valid_increase_cancel_coverage_mean": [
            "valid_increase_cancel_coverage_mean",
            "valid_increase_cancel_coverage",
            "valid_increase_cancel_coverage_mean_ctrl",
        ],
        "valid_increase_top2_cancel_hit_count_mean": [
            "valid_increase_top2_cancel_hit_count_mean",
            "valid_increase_top2_hit_count_mean",
            "valid_increase_top2_cancel_hit_count",
        ],
        "valid_increase_top3_cancel_hit_count_mean": [
            "valid_increase_top3_cancel_hit_count_mean",
            "valid_increase_top3_hit_count_mean",
            "valid_increase_top3_cancel_hit_count",
        ],
        "task_level_top1_cancel_share_mean": [
            "task_level_top1_cancel_share_mean",
            "top1_cancel_share_mean",
            "task_level_top1_cancel_share",
        ],
        "baseline_lo_cancellations_per_1m": [
            "baseline_lo_cancellations_per_1m",
            "baseline_lo_cancellations_mean",
            "fast_baseline_lo_cancellations_mean",
        ],
        "baseline_lo_cancellation_ratio_total": [
            "baseline_lo_cancellation_ratio_total",
            "fast_baseline_lo_cancellation_ratio_total",
        ],
        "baseline_mode_changes_per_1m": [
            "baseline_mode_changes_per_1m",
            "baseline_mode_changes_mean",
            "fast_baseline_mode_changes_mean",
        ],
        "baseline_total_events_per_1m": [
            "baseline_total_events_per_1m",
            "baseline_total_events_mean",
            "fast_baseline_total_events_mean",
        ],
        "baseline_deadline_misses_sum": [
            "baseline_deadline_misses_sum",
            "baseline_deadline_misses",
            "fast_baseline_deadline_misses_sum",
        ],
        "valid_increase_count_mean": [
            "valid_increase_count_mean",
            "fast_valid_increase_count_mean",
        ],
        "valid_decrease_count_mean": [
            "valid_decrease_count_mean",
            "fast_valid_decrease_count_mean",
        ],
    }

    for canonical, aliases in alias_groups.items():
        if canonical not in out.columns:
            found = _first_existing_col(out, aliases)
            if found is not None:
                out[canonical] = out[found]
        out[canonical] = _ensure_numeric(out, canonical)

    # Safe defaults for optional hit/share fields. These are not hard filters by default.
    for c in [
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
    ]:
        out[c] = out[c].fillna(0.0)
    out["task_level_top1_cancel_share_mean"] = out["task_level_top1_cancel_share_mean"].fillna(1.0)

    return out


def _parse_int_list(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    items: List[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _bucket_sizes_from_args(args: argparse.Namespace) -> Dict[str, int]:
    if args.preset == "broad30":
        defaults = {"low": 10, "medium": 10, "high_not_top20": 10}
    elif args.preset == "broad60":
        defaults = {"low": 20, "medium": 20, "high_not_top20": 20}
    elif args.preset == "broad100":
        defaults = {"low": 30, "medium": 40, "high_not_top20": 30}
    else:
        defaults = {"low": args.low_n, "medium": args.medium_n, "high_not_top20": args.high_n}

    # Explicit values override preset defaults when provided.
    if args.low_n is not None:
        defaults["low"] = args.low_n
    if args.medium_n is not None:
        defaults["medium"] = args.medium_n
    if args.high_n is not None:
        defaults["high_not_top20"] = args.high_n

    if any(v < 0 for v in defaults.values()):
        raise ValueError(f"Bucket sizes must be non-negative: {defaults}")
    if sum(defaults.values()) <= 0:
        raise ValueError(f"Total selected seeds must be positive: {defaults}")
    return defaults


def _apply_common_filters(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reasons = []
    keep = pd.Series(True, index=df.index)

    def fail(mask: pd.Series, reason: str) -> None:
        nonlocal keep, reasons
        mask = mask.fillna(False)
        failed_idx = df.index[keep & ~mask]
        if len(failed_idx) > 0:
            reasons.extend((idx, reason) for idx in failed_idx)
        keep &= mask

    fail(df["baseline_deadline_misses_sum"].fillna(math.inf) <= args.max_deadline_misses, "deadline_miss")
    fail(df["valid_increase_count_mean"].fillna(-math.inf) >= args.common_min_valid_increase, "valid_increase_low")
    fail(df["valid_decrease_count_mean"].fillna(-math.inf) >= args.common_min_valid_decrease, "valid_decrease_low")
    fail(df["baseline_lo_cancellations_per_1m"].fillna(-math.inf) >= args.common_min_lo_cancellations_per_1m, "lo_cancellations_low")
    fail(df["baseline_mode_changes_per_1m"].fillna(math.inf) <= args.common_max_mode_changes_per_1m, "mode_changes_high")

    if args.min_total_events_per_1m is not None:
        fail(df["baseline_total_events_per_1m"].fillna(-math.inf) >= args.min_total_events_per_1m, "events_low")
    if args.max_total_events_per_1m is not None:
        fail(df["baseline_total_events_per_1m"].fillna(math.inf) <= args.max_total_events_per_1m, "events_high")
    if args.min_lo_cancellation_ratio is not None:
        fail(df["baseline_lo_cancellation_ratio_total"].fillna(-math.inf) >= args.min_lo_cancellation_ratio, "lo_ratio_low")
    if args.max_lo_cancellation_ratio is not None:
        fail(df["baseline_lo_cancellation_ratio_total"].fillna(math.inf) <= args.max_lo_cancellation_ratio, "lo_ratio_high")

    filtered = df[keep].copy()
    rej_rows = []
    for idx, reason in reasons:
        row = df.loc[idx, ["candidate_seed"]].to_dict()
        row["reject_reason"] = reason
        rej_rows.append(row)
    rejected = pd.DataFrame(rej_rows)
    return filtered, rejected


def _assign_buckets(df: pd.DataFrame, args: argparse.Namespace, main_top20: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["is_main_top20"] = out["candidate_seed"].isin(main_top20)

    cov = out["valid_increase_cancel_coverage_mean"]
    lo = out["baseline_lo_cancellations_per_1m"]
    mode = out["baseline_mode_changes_per_1m"]
    inc = out["valid_increase_count_mean"]
    dec = out["valid_decrease_count_mean"]
    dl = out["baseline_deadline_misses_sum"]

    bucket = pd.Series(pd.NA, index=out.index, dtype="object")

    low_mask = (
        (cov < args.low_max_coverage)
        & (lo >= args.low_min_lo_cancellations_per_1m)
        & (mode <= args.low_max_mode_changes_per_1m)
        & (inc >= args.low_min_valid_increase)
        & (dec >= args.low_min_valid_decrease)
        & (dl <= args.max_deadline_misses)
        & (~out["is_main_top20"])
    )

    medium_mask = (
        (cov >= args.medium_min_coverage)
        & (cov < args.medium_max_coverage)
        & (lo >= args.medium_min_lo_cancellations_per_1m)
        & (mode <= args.medium_max_mode_changes_per_1m)
        & (inc >= args.medium_min_valid_increase)
        & (dec >= args.medium_min_valid_decrease)
        & (dl <= args.max_deadline_misses)
        & (~out["is_main_top20"])
    )

    high_mask = (
        (cov >= args.high_min_coverage)
        & (lo >= args.high_min_lo_cancellations_per_1m)
        & (mode <= args.high_max_mode_changes_per_1m)
        & (inc >= args.high_min_valid_increase)
        & (dec >= args.high_min_valid_decrease)
        & (dl <= args.max_deadline_misses)
        & (~out["is_main_top20"])
    )

    bucket.loc[low_mask] = "low"
    bucket.loc[medium_mask] = "medium"
    bucket.loc[high_mask] = "high_not_top20"

    out["broad_bucket"] = bucket

    bucketed = out[out["broad_bucket"].notna()].copy()
    rejected = out[out["broad_bucket"].isna()].copy()
    if not rejected.empty:
        rejected["reject_reason"] = rejected["is_main_top20"].map({True: "main_top20_excluded", False: "no_bucket_match"})
    return bucketed, rejected


def _select_even_quantiles(bucket_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        return bucket_df.iloc[0:0].copy()
    if bucket_df.empty:
        return bucket_df.copy()

    work = bucket_df.sort_values(
        ["opportunity_score", "candidate_seed"], ascending=[True, True]
    ).reset_index(drop=True)
    m = len(work)
    if m <= n:
        selected = work.copy()
        selected["bucket_quantile"] = [(i + 0.5) / max(m, 1) for i in range(m)]
        selected["bucket_rank"] = list(range(1, m + 1))
        selected["selection_note"] = "all_available_bucket_smaller_than_requested"
        return selected

    # Target midpoints: 0.5/n, 1.5/n, ..., (n-0.5)/n
    qs = [(i + 0.5) / n for i in range(n)]
    idxs = []
    used = set()
    for q in qs:
        idx = int(round(q * (m - 1)))
        # Resolve collisions by nearest unused index.
        if idx in used:
            for delta in range(1, m):
                left = idx - delta
                right = idx + delta
                if left >= 0 and left not in used:
                    idx = left
                    break
                if right < m and right not in used:
                    idx = right
                    break
        used.add(idx)
        idxs.append(idx)

    selected = work.iloc[sorted(idxs)].copy()
    selected["bucket_rank"] = list(range(1, len(selected) + 1))
    selected["bucket_quantile"] = [(idx + 0.5) / m for idx in sorted(idxs)]
    selected["selection_note"] = "even_quantile_sample"
    return selected


def _make_outputs(df: pd.DataFrame, bucket_sizes: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    candidates_parts = []

    for bucket_name, requested_n in bucket_sizes.items():
        bucket_df = df[df["broad_bucket"] == bucket_name].copy()
        bucket_df = bucket_df.sort_values(
            ["opportunity_score", "candidate_seed"], ascending=[True, True]
        ).reset_index(drop=True)
        bucket_df["bucket_candidate_rank"] = list(range(1, len(bucket_df) + 1))
        bucket_df["bucket_candidate_quantile"] = [
            (i + 0.5) / max(len(bucket_df), 1) for i in range(len(bucket_df))
        ]
        bucket_df["bucket_requested_n"] = requested_n
        candidates_parts.append(bucket_df)

        selected = _select_even_quantiles(bucket_df, requested_n)
        selected["bucket_requested_n"] = requested_n
        selected_parts.append(selected)

    selected_all = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    candidates_all = pd.concat(candidates_parts, ignore_index=True) if candidates_parts else pd.DataFrame()

    selected_all = selected_all.sort_values(["broad_bucket", "bucket_rank", "candidate_seed"]).reset_index(drop=True)
    return selected_all, candidates_all


def _build_summary(selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def summarize(label: str, df: pd.DataFrame, candidate_count: Optional[int] = None) -> None:
        row = {"bucket": label, "selected_n": len(df)}
        if candidate_count is not None:
            row["candidate_n"] = candidate_count
        for c in [
            "valid_increase_cancel_coverage_mean",
            "baseline_lo_cancellations_per_1m",
            "baseline_lo_cancellation_ratio_total",
            "baseline_mode_changes_per_1m",
            "baseline_total_events_per_1m",
            "valid_increase_count_mean",
            "valid_decrease_count_mean",
            "opportunity_score",
        ]:
            if c in df.columns and not df.empty:
                row[f"{c}_mean"] = float(pd.to_numeric(df[c], errors="coerce").mean())
                row[f"{c}_median"] = float(pd.to_numeric(df[c], errors="coerce").median())
                row[f"{c}_min"] = float(pd.to_numeric(df[c], errors="coerce").min())
                row[f"{c}_max"] = float(pd.to_numeric(df[c], errors="coerce").max())
        if "strict_like_flag" in df.columns and not df.empty:
            row["strict_like_count"] = int(df["strict_like_flag"].fillna(False).sum())
        rows.append(row)

    summarize("ALL", selected, len(candidates))
    for b in ["low", "medium", "high_not_top20"]:
        summarize(
            b,
            selected[selected["broad_bucket"] == b],
            int((candidates["broad_bucket"] == b).sum()) if not candidates.empty else 0,
        )

    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select broad stratified tasksets by low/medium/high opportunity buckets."
    )

    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--controllability-summary-csv", required=True)
    ap.add_argument("--seed-column", default=None)

    ap.add_argument("--preset", choices=["broad30", "broad60", "broad100", "custom"], default="broad30")
    ap.add_argument("--low-n", type=int, default=None)
    ap.add_argument("--medium-n", type=int, default=None)
    ap.add_argument("--high-n", type=int, default=None)

    ap.add_argument("--main-top20-seeds", default=",".join(str(x) for x in DEFAULT_MAIN_TOP20))
    ap.add_argument("--extra-exclude-seeds", default="")
    ap.add_argument("--exclude-old-or-hybrid-known-seeds", action="store_true")

    # Common broad-pool filters.
    ap.add_argument("--max-deadline-misses", type=float, default=0.0)
    ap.add_argument("--common-min-valid-increase", type=float, default=2.0)
    ap.add_argument("--common-min-valid-decrease", type=float, default=2.0)
    ap.add_argument("--common-min-lo-cancellations-per-1m", type=float, default=8.0)
    ap.add_argument("--common-max-mode-changes-per-1m", type=float, default=50.0)
    ap.add_argument("--min-total-events-per-1m", type=float, default=None)
    ap.add_argument("--max-total-events-per-1m", type=float, default=None)
    ap.add_argument("--min-lo-cancellation-ratio", type=float, default=None)
    ap.add_argument("--max-lo-cancellation-ratio", type=float, default=None)

    # Bucket definitions.
    ap.add_argument("--low-max-coverage", type=float, default=0.20)
    ap.add_argument("--low-min-lo-cancellations-per-1m", type=float, default=10.0)
    ap.add_argument("--low-max-mode-changes-per-1m", type=float, default=50.0)
    ap.add_argument("--low-min-valid-increase", type=float, default=2.0)
    ap.add_argument("--low-min-valid-decrease", type=float, default=2.0)

    ap.add_argument("--medium-min-coverage", type=float, default=0.20)
    ap.add_argument("--medium-max-coverage", type=float, default=0.32)
    ap.add_argument("--medium-min-lo-cancellations-per-1m", type=float, default=15.0)
    ap.add_argument("--medium-max-mode-changes-per-1m", type=float, default=40.0)
    ap.add_argument("--medium-min-valid-increase", type=float, default=2.0)
    ap.add_argument("--medium-min-valid-decrease", type=float, default=2.0)

    ap.add_argument("--high-min-coverage", type=float, default=0.32)
    ap.add_argument("--high-min-lo-cancellations-per-1m", type=float, default=20.0)
    ap.add_argument("--high-max-mode-changes-per-1m", type=float, default=35.0)
    ap.add_argument("--high-min-valid-increase", type=float, default=3.0)
    ap.add_argument("--high-min-valid-decrease", type=float, default=3.0)

    # Output files.
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--output-summary", required=True)
    ap.add_argument("--output-candidates", default=None)
    ap.add_argument("--output-rejections", default=None)

    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bucket_sizes = _bucket_sizes_from_args(args)

    manifest = _read_csv(args.manifest_csv, "manifest-csv")
    summary = _read_csv(args.summary_csv, "summary-csv") if args.summary_csv else None
    ctrl = _read_csv(args.controllability_summary_csv, "controllability-summary-csv")

    df = _merge_sources(manifest, summary, ctrl, args.seed_column)
    df = _ensure_required_metrics(df)

    main_top20 = _parse_int_list(args.main_top20_seeds)
    extra_exclude = set(_parse_int_list(args.extra_exclude_seeds))
    if args.exclude_old_or_hybrid_known_seeds:
        extra_exclude.update(DEFAULT_OLD_OR_HYBRID_SEEDS)

    df["is_extra_excluded"] = df["candidate_seed"].isin(extra_exclude)
    if extra_exclude:
        df_before = len(df)
        df = df[~df["is_extra_excluded"]].copy()
        print(f"Excluded {df_before - len(df)} extra known/explicit seeds.", file=sys.stderr)

    # Opportunity score for stratified sampling, not for top-K selection.
    denom = df["baseline_mode_changes_per_1m"].clip(lower=1.0)
    df["opportunity_score"] = (
        df["valid_increase_cancel_coverage_mean"].fillna(0.0)
        * df["baseline_lo_cancellations_per_1m"].fillna(0.0)
        / denom.fillna(1.0)
    )

    df["strict_like_flag"] = (
        (df["valid_increase_top2_cancel_hit_count_mean"].fillna(0.0) >= 1.0)
        & (df["valid_increase_top3_cancel_hit_count_mean"].fillna(0.0) >= 2.0)
        & (df["task_level_top1_cancel_share_mean"].fillna(1.0) <= 0.75)
    )

    filtered, common_rej = _apply_common_filters(df, args)
    bucketed, bucket_rej = _assign_buckets(filtered, args, main_top20)
    selected, candidates = _make_outputs(bucketed, bucket_sizes)

    # Stable, useful output columns first, then keep remaining columns after.
    preferred_cols = [
        "candidate_seed",
        "broad_bucket",
        "bucket_rank",
        "bucket_quantile",
        "bucket_candidate_rank",
        "bucket_candidate_quantile",
        "bucket_requested_n",
        "selection_note",
        "opportunity_score",
        "strict_like_flag",
        "is_main_top20",
        "is_extra_excluded",
        "valid_increase_cancel_coverage_mean",
        "valid_increase_top2_cancel_hit_count_mean",
        "valid_increase_top3_cancel_hit_count_mean",
        "task_level_top1_cancel_share_mean",
        "baseline_lo_cancellations_per_1m",
        "baseline_lo_cancellation_ratio_total",
        "baseline_mode_changes_per_1m",
        "baseline_total_events_per_1m",
        "baseline_deadline_misses_sum",
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
    ]
    selected_cols = [c for c in preferred_cols if c in selected.columns] + [
        c for c in selected.columns if c not in preferred_cols
    ]
    candidates_cols = [c for c in preferred_cols if c in candidates.columns] + [
        c for c in candidates.columns if c not in preferred_cols
    ]

    selected = selected[selected_cols] if not selected.empty else selected
    candidates = candidates[candidates_cols] if not candidates.empty else candidates
    summary_df = _build_summary(selected, candidates)

    out_manifest = Path(args.output_manifest)
    out_summary = Path(args.output_summary)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(out_manifest, index=False)
    summary_df.to_csv(out_summary, index=False)

    if args.output_candidates:
        out_candidates = Path(args.output_candidates)
        out_candidates.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(out_candidates, index=False)

    if args.output_rejections:
        rejections = []
        if not common_rej.empty:
            rejections.append(common_rej)
        if not bucket_rej.empty:
            rejections.append(bucket_rej[["candidate_seed", "reject_reason"]].copy())
        rej = pd.concat(rejections, ignore_index=True) if rejections else pd.DataFrame(columns=["candidate_seed", "reject_reason"])
        out_rej = Path(args.output_rejections)
        out_rej.parent.mkdir(parents=True, exist_ok=True)
        rej.to_csv(out_rej, index=False)

    print("=== Broad stratified selection complete ===")
    print(f"Preset: {args.preset}")
    print(f"Requested bucket sizes: {bucket_sizes}")
    print(f"Input candidates after merge: {len(df)}")
    print(f"After common filters: {len(filtered)}")
    print(f"Bucket candidates: {len(candidates)}")
    print(f"Selected seeds: {len(selected)}")
    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Wrote manifest: {out_manifest}")
    print(f"Wrote summary:  {out_summary}")
    if args.output_candidates:
        print(f"Wrote candidates: {args.output_candidates}")
    if args.output_rejections:
        print(f"Wrote rejections: {args.output_rejections}")

    # Warn but do not fail if a bucket has fewer seeds than requested.
    for b, n in bucket_sizes.items():
        got = int((selected["broad_bucket"] == b).sum()) if not selected.empty else 0
        if got < n:
            print(
                f"WARNING: bucket {b!r} selected only {got}/{n}; "
                "consider relaxing that bucket's thresholds or using broad60/100 after more candidates.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
