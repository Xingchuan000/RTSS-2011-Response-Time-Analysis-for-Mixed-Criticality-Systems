#!/usr/bin/env python3
"""Opportunity-oriented selector for MC-FairGen single-action tasksets.

This selector intentionally does *not* rank primarily by valid action count.
It searches for tasksets that look like seed1484-style opportunities:

  high LO cancellation pressure
  low / controlled mode-change pressure
  enough but not necessarily maximal valid single-action space
  no baseline deadline misses

The main derived metrics are:

  cancel_to_mode = baseline_lo_cancellations_per_1m / max(baseline_mode_changes_per_1m, eps)
  opportunity_score = baseline_lo_cancellation_ratio_total
                      * baseline_lo_cancellations_per_1m
                      / max(baseline_mode_changes_per_1m, eps)

Typical use:

  python scripts/select_opportunity_pressure_tasksets.py \
    --summary-csv outputs/taskset_slack_scan/controlled_medium_scale500/headroom_quick_top300_summary_0_3000_e3m_s200_206.csv \
    --manifest-csv outputs/tasksets/controlled_medium_scale500/prescreen_top300_0_3000.csv \
    --top-k 40 \
    --output-summary outputs/taskset_slack_scan/controlled_medium_scale500/opportunity_top40_quick_summary.csv \
    --output-manifest outputs/tasksets/controlled_medium_scale500/opportunity_top40_quick.csv \
    --output-rejections outputs/taskset_slack_scan/controlled_medium_scale500/opportunity_top40_quick_rejections.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--output-rejections", type=Path, required=True)

    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed-column", type=str, default="candidate_seed")

    # Action-space requirements. These are deliberately looser than the old
    # learnability-first selector, because valid action *usefulness* matters
    # more than simply maximizing valid action count.
    p.add_argument("--min-valid-increase", type=float, default=3.0)
    p.add_argument("--min-valid-decrease", type=float, default=10.0)
    p.add_argument("--min-balance", type=float, default=0.20)

    # Pressure window. Defaults target seed1484-like candidates.
    p.add_argument("--min-events-per-1m", type=float, default=35.0)
    p.add_argument("--max-events-per-1m", type=float, default=100.0)
    p.add_argument("--min-lo-cancellations-per-1m", type=float, default=25.0)
    p.add_argument("--min-lo-cancellation-ratio", type=float, default=0.60)
    p.add_argument("--max-lo-cancellation-ratio", type=float, default=0.90)
    p.add_argument("--min-mode-changes-per-1m", type=float, default=1.0)
    p.add_argument("--max-mode-changes-per-1m", type=float, default=25.0)

    p.add_argument("--max-baseline-deadline-misses", type=float, default=0.0)
    p.add_argument(
        "--require-recommended-for-dqn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, reject rows whose recommended_for_dqn column is false. Default false because some older scans do not include the column.",
    )

    # Numerical guard for cancel_to_mode / opportunity_score.
    p.add_argument("--mode-eps", type=float, default=1e-9)

    # Optional: keep very high opportunity rows even if mode is slightly above
    # the strict threshold. Useful for exploration, but disabled by default.
    p.add_argument(
        "--soft-mode-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, do not reject by max mode; mode is only used in ranking.",
    )
    return p.parse_args()


def _bool_value(x: object) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Missing required column. Tried: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def _num(row: pd.Series, col: str | None, default: float = 0.0) -> float:
    if col is None:
        return default
    value = row.get(col, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _add_derived_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str | None]]:
    cols: dict[str, str | None] = {}

    cols["events"] = _find_col(df, ["baseline_total_events_per_1m", "baseline_total_events_mean"])
    cols["mode"] = _find_col(df, ["baseline_mode_changes_per_1m", "baseline_mode_changes_mean"])
    cols["lo_cancel"] = _find_col(df, ["baseline_lo_cancellations_per_1m", "baseline_lo_cancellations_mean"])
    cols["ratio"] = _find_col(df, ["baseline_lo_cancellation_ratio_total", "baseline_lo_cancellation_ratio_mean"])
    cols["deadline"] = _find_col(df, ["baseline_deadline_misses_sum", "deadline_misses_sum"], required=False)
    cols["recommended"] = _find_col(df, ["recommended_for_dqn", "recommended_fast"], required=False)

    required_action_cols = [
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "increase_decrease_balance",
    ]
    missing = [c for c in required_action_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required action-space columns: {missing}")

    out = df.copy()
    events = pd.to_numeric(out[cols["events"]], errors="coerce")
    mode = pd.to_numeric(out[cols["mode"]], errors="coerce")
    cancels = pd.to_numeric(out[cols["lo_cancel"]], errors="coerce")
    ratio = pd.to_numeric(out[cols["ratio"]], errors="coerce")

    denom = mode.clip(lower=args.mode_eps)
    out["opportunity_events_per_1m"] = events
    out["opportunity_mode_changes_per_1m"] = mode
    out["opportunity_lo_cancellations_per_1m"] = cancels
    out["opportunity_cancel_to_mode"] = cancels / denom
    out["opportunity_score"] = ratio * cancels / denom
    out["opportunity_valid_min_mean"] = out[["valid_increase_count_mean", "valid_decrease_count_mean"]].min(axis=1)
    out["opportunity_valid_total_mean"] = out["valid_increase_count_mean"] + out["valid_decrease_count_mean"]

    # Helpful diagnostic: high ratio + high cancellation + low event count can mean
    # a concentrated, controllable cancellation pattern.
    out["opportunity_cancel_to_event"] = cancels / events.clip(lower=args.mode_eps)
    return out, cols


def _reject_reasons(df: pd.DataFrame, cols: dict[str, str | None], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    for _, row in df.iterrows():
        r: list[str] = []

        if args.require_recommended_for_dqn and cols["recommended"] is not None:
            if not _bool_value(row.get(cols["recommended"], False)):
                r.append("not_recommended_for_dqn")

        deadline = _num(row, cols["deadline"], 0.0)
        if deadline > args.max_baseline_deadline_misses:
            r.append("baseline_deadline_misses")

        if _num(row, "valid_increase_count_mean") < args.min_valid_increase:
            r.append("low_valid_increase")
        if _num(row, "valid_decrease_count_mean") < args.min_valid_decrease:
            r.append("low_valid_decrease")
        if _num(row, "increase_decrease_balance") < args.min_balance:
            r.append("low_increase_decrease_balance")

        events = _num(row, cols["events"])
        if events < args.min_events_per_1m:
            r.append("too_few_events")
        if events > args.max_events_per_1m:
            r.append("too_many_events")

        cancels = _num(row, cols["lo_cancel"])
        if cancels < args.min_lo_cancellations_per_1m:
            r.append("too_few_lo_cancellations")

        ratio = _num(row, cols["ratio"])
        if ratio < args.min_lo_cancellation_ratio:
            r.append("low_lo_cancellation_ratio")
        if ratio > args.max_lo_cancellation_ratio:
            r.append("too_high_lo_cancellation_ratio")

        mode = _num(row, cols["mode"])
        if mode < args.min_mode_changes_per_1m:
            r.append("too_few_mode_changes")
        if not args.soft_mode_cap and mode > args.max_mode_changes_per_1m:
            r.append("too_many_mode_changes")

        reasons.append(";".join(r))
    return reasons


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    manifest = pd.read_csv(args.manifest_csv)

    if args.seed_column not in summary.columns:
        raise SystemExit(f"Missing seed column in summary: {args.seed_column}")
    if args.seed_column not in manifest.columns:
        raise SystemExit(f"Missing seed column in manifest: {args.seed_column}")

    df, cols = _add_derived_columns(summary, args)
    df["select_reject_reason"] = _reject_reasons(df, cols, args)

    selected = df[df["select_reject_reason"].eq("")].copy()
    rejected = df[~df["select_reject_reason"].eq("")].copy()

    sort_cols = [
        "opportunity_score",
        "opportunity_cancel_to_mode",
        cols["ratio"],
        cols["lo_cancel"],
        cols["mode"],
        cols["events"],
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        args.seed_column,
    ]
    ascending = [False, False, False, False, True, True, False, False, True]

    if not selected.empty:
        selected = selected.sort_values(sort_cols, ascending=ascending)
        top = selected.head(args.top_k).copy()
    else:
        top = selected.copy()

    seeds = top[args.seed_column].astype(int).tolist() if not top.empty else []

    manifest = manifest.copy()
    manifest[args.seed_column] = manifest[args.seed_column].astype(int)
    if seeds:
        top_manifest = manifest[manifest[args.seed_column].isin(seeds)].set_index(args.seed_column).loc[seeds].reset_index()
    else:
        top_manifest = manifest.iloc[0:0].copy()

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejections.parent.mkdir(parents=True, exist_ok=True)

    top.to_csv(args.output_summary, index=False)
    top_manifest.to_csv(args.output_manifest, index=False)
    rejected.to_csv(args.output_rejections, index=False)

    print("[Opportunity-pressure Select Summary]")
    print(f"summary_rows={len(summary)}")
    print(f"passed_filters={len(selected)}")
    print(f"selected_top_k={len(top)}")
    print(f"rejected={len(rejected)}")
    print("selected_seeds=" + ",".join(map(str, seeds)))

    show_cols = [
        args.seed_column,
        "opportunity_score",
        "opportunity_cancel_to_mode",
        cols["ratio"],
        cols["lo_cancel"],
        cols["mode"],
        cols["events"],
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "increase_decrease_balance",
    ]
    show_cols = [c for c in show_cols if c is not None and c in top.columns]
    print("\nTop selected:")
    if top.empty:
        print("<none>")
    else:
        print(top[show_cols].to_string(index=False))

    print("\nTop rejection reasons:")
    if rejected.empty:
        print("<none>")
    else:
        print(rejected["select_reject_reason"].value_counts(dropna=False).head(30).to_string())

    print("\nOutput files:")
    print(f"summary:   {args.output_summary}")
    print(f"manifest:  {args.output_manifest}")
    print(f"rejection: {args.output_rejections}")


if __name__ == "__main__":
    main()
