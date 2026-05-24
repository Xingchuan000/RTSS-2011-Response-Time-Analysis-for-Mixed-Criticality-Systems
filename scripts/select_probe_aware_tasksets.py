#!/usr/bin/env python3
"""基于 headroom + stable probe 的候选筛选器。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--headroom-summary", type=Path, required=True)
    p.add_argument("--probe-summary", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--output-rejections", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-valid-increase", type=float, default=3)
    p.add_argument("--min-valid-decrease", type=float, default=3)
    p.add_argument("--min-balance", type=float, default=0.20)
    p.add_argument("--min-events-per-1m", type=float, default=80)
    p.add_argument("--max-events-per-1m", type=float, default=260)
    p.add_argument("--min-mode-per-1m", type=float, default=1)
    p.add_argument("--max-mode-per-1m", type=float, default=120)
    p.add_argument("--min-lo-cancellation-ratio", type=float, default=0.30)
    p.add_argument("--max-lo-cancellation-ratio", type=float, default=0.85)
    p.add_argument("--min-probe-stable-reduction", type=float, default=0.01)
    p.add_argument("--exclude-probe-tradeoff-risk", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed-column", default="candidate_seed")
    return p.parse_args()


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main() -> None:
    args = parse_args()
    h = pd.read_csv(args.headroom_summary)
    p = pd.read_csv(args.probe_summary)
    m = pd.read_csv(args.manifest_csv)
    df = h.merge(p, on=args.seed_column, how="inner")

    events = "baseline_total_events_per_1m" if "baseline_total_events_per_1m" in df.columns else "baseline_total_events_mean"
    modes = "baseline_mode_changes_per_1m" if "baseline_mode_changes_per_1m" in df.columns else "baseline_mode_changes_mean"

    reasons = []
    for _, row in df.iterrows():
        r = []
        if float(row.get("valid_increase_count_mean", 0)) < args.min_valid_increase:
            r.append("low_valid_increase")
        if float(row.get("valid_decrease_count_mean", 0)) < args.min_valid_decrease:
            r.append("low_valid_decrease")
        if float(row.get("increase_decrease_balance", 0)) < args.min_balance:
            r.append("low_balance")
        if float(row.get(events, 0)) < args.min_events_per_1m or float(row.get(events, 0)) > args.max_events_per_1m:
            r.append("events_out_of_range")
        if float(row.get(modes, 0)) < args.min_mode_per_1m or float(row.get(modes, 0)) > args.max_mode_per_1m:
            r.append("mode_out_of_range")
        ratio = float(row.get("baseline_lo_cancellation_ratio_total", 0))
        if ratio < args.min_lo_cancellation_ratio or ratio > args.max_lo_cancellation_ratio:
            r.append("lo_cancel_ratio_out_of_range")
        if str(row.get("probe_stable_best_found", "")).lower() not in ["true", "1", "yes", "y"]:
            r.append("probe_stable_not_found")
        if float(row.get("probe_stable_best_relative_lo_cancellation_reduction", 0) or 0) < args.min_probe_stable_reduction:
            r.append("probe_stable_reduction_too_low")
        if args.exclude_probe_tradeoff_risk and str(row.get("probe_tradeoff_risk", "")).lower() in ["true", "1", "yes", "y"]:
            r.append("probe_tradeoff_risk")
        reasons.append(";".join(r))

    df["select_reject_reason"] = reasons
    selected = df[df["select_reject_reason"].eq("")].copy()
    rejected = df[~df["select_reject_reason"].eq("")].copy()
    selected["valid_min_mean"] = selected[["valid_increase_count_mean", "valid_decrease_count_mean"]].min(axis=1)
    selected["valid_total_mean"] = selected["valid_increase_count_mean"] + selected["valid_decrease_count_mean"]
    selected = selected.sort_values(
        [
            "probe_stable_best_relative_lo_cancellation_reduction",
            "valid_min_mean",
            "valid_total_mean",
            "baseline_lo_cancellation_ratio_total",
            modes,
            events,
            args.seed_column,
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    top = selected.head(args.top_k).copy()
    seeds = top[args.seed_column].astype(int).tolist()
    m[args.seed_column] = m[args.seed_column].astype(int)
    top_manifest = m[m[args.seed_column].isin(seeds)].set_index(args.seed_column).loc[seeds].reset_index()

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejections.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.output_summary, index=False)
    top_manifest.to_csv(args.output_manifest, index=False)
    rejected.to_csv(args.output_rejections, index=False)


if __name__ == "__main__":
    main()
