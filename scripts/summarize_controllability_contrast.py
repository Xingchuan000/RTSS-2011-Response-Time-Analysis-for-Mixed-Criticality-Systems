"""strong/weak controllability 对照汇总脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def _reduction_group(value: float) -> str:
    """按相对降幅分组。

    兼容两种输入：
    - 0.205 表示 20.5%
    - 20.5 表示 20.5%
    """

    if value <= 1.0:
        value = value * 100.0
    if value >= 7.0:
        return "strong"
    if value >= 3.0:
        return "medium"
    return "weak"


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"无法在结果表中找到任一候选列: {candidates}")


def main() -> None:
    args = build_parser().parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    result_df = pd.read_csv(args.result_csv)

    seed_col = _first_existing_col(result_df, ["candidate_seed", "fixed_taskset_seed", "seed"])
    reduction_col = _first_existing_col(
        result_df,
        ["lo_cancellation_rel_reduction", "lo_cancellation_relative_reduction_pct", "relative_reduction_pct"],
    )
    qos_col = _first_existing_col(result_df, ["qos_delta_percentage_points", "qos_delta_pp", "qos_delta"])

    work = result_df.copy()
    work["candidate_seed"] = work[seed_col].astype(int)
    joined = summary_df.merge(
        work[["candidate_seed", reduction_col, qos_col]],
        on="candidate_seed",
        how="left",
    ).rename(columns={reduction_col: "lo_cancellation_rel_reduction", qos_col: "qos_delta_percentage_points"})

    joined["reduction_group"] = joined["lo_cancellation_rel_reduction"].map(_reduction_group)
    joined = joined.sort_values(by=["controllability_score", "lo_cancellation_rel_reduction"], ascending=False)

    group_stats = (
        joined.groupby("reduction_group", as_index=False)[
            [
                "valid_increase_cancel_coverage_mean",
                "task_level_top2_cancel_share_mean",
                "task_level_cancel_concentration_hhi_mean",
                "controllability_score",
            ]
        ]
        .mean()
        .sort_values(by="reduction_group")
    )
    corr_pearson = joined[
        [
            "lo_cancellation_rel_reduction",
            "valid_increase_cancel_coverage_mean",
            "task_level_top2_cancel_share_mean",
            "controllability_score",
        ]
    ].corr(method="pearson")
    corr_spearman = joined[
        [
            "lo_cancellation_rel_reduction",
            "valid_increase_cancel_coverage_mean",
            "task_level_top2_cancel_share_mean",
            "controllability_score",
        ]
    ].corr(method="spearman")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_csv, index=False)

    lines: list[str] = []
    lines.append("# Controllability Contrast Report")
    lines.append("")
    lines.append("## Group Means")
    lines.append("")
    lines.append(group_stats.to_markdown(index=False))
    lines.append("")
    lines.append("## Pearson Correlation")
    lines.append("")
    lines.append(corr_pearson.to_markdown())
    lines.append("")
    lines.append("## Spearman Correlation")
    lines.append("")
    lines.append(corr_spearman.to_markdown())
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
