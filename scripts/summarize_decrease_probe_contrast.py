#!/usr/bin/env python3
"""汇总 controllability 与 static decrease probe 对照结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


MANUAL_GROUPS: dict[str, set[int]] = {
    "strong": {1484, 2429, 2829, 2221, 1502},
    "medium": set(),
    "weak": {90, 2574, 57, 185, 2784},
    "opportunity_or_lightprobe_weak": {395, 2505, 367, 1563, 343, 559},
}


def parse_args() -> argparse.Namespace:
    """解析参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-level-summary", type=Path, required=True)
    parser.add_argument("--decrease-probe-summary", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 行。"""

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全浮点解析。"""

    raw = row.get(key, "")
    if raw in (None, ""):
        return default
    return float(raw)


def _to_int(row: dict[str, Any], key: str, default: int = -1) -> int:
    """安全整型解析。"""

    raw = row.get(key, "")
    if raw in (None, ""):
        return default
    return int(float(raw))


def _manual_group(seed: int) -> str:
    """按预设 seed 集合分组。"""

    for group_name, seed_set in MANUAL_GROUPS.items():
        if seed in seed_set:
            return group_name
    return "unassigned"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    """按字段求均值，空列表返回 0.0。"""

    if not rows:
        return 0.0
    return mean(_to_float(row, key, 0.0) for row in rows)


def _rate_true(rows: list[dict[str, Any]], key: str) -> float:
    """统计布尔字段为真比例。"""

    if not rows:
        return 0.0
    true_count = 0
    for row in rows:
        raw = str(row.get(key, "")).strip().lower()
        if raw in {"true", "1"}:
            true_count += 1
    return float(true_count) / float(len(rows))


def _distribution(rows: list[dict[str, Any]], key: str) -> str:
    """将类别分布压成可读文本。"""

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key, "")).strip() or "<empty>"
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name}:{count}" for name, count in ordered)


def main() -> None:
    """执行汇总。"""

    args = parse_args()
    task_rows = _read_csv(args.task_level_summary)
    probe_rows = _read_csv(args.decrease_probe_summary)
    result_rows = _read_csv(args.result_csv) if args.result_csv is not None and args.result_csv.exists() else []

    task_by_seed = {_to_int(row, "candidate_seed"): row for row in task_rows}
    probe_by_seed = {_to_int(row, "candidate_seed"): row for row in probe_rows}
    result_by_seed = {_to_int(row, "candidate_seed"): row for row in result_rows}

    all_seeds = sorted(set(task_by_seed.keys()) | set(probe_by_seed.keys()) | set(result_by_seed.keys()))
    joined_rows: list[dict[str, Any]] = []
    for seed in all_seeds:
        row: dict[str, Any] = {"candidate_seed": int(seed), "manual_group": _manual_group(seed)}
        row.update(task_by_seed.get(seed, {}))
        row.update(probe_by_seed.get(seed, {}))
        if seed in result_by_seed:
            row.update({f"result_{k}": v for k, v in result_by_seed[seed].items()})
        joined_rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if joined_rows:
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(joined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(joined_rows)
    else:
        args.output_csv.write_text("", encoding="utf-8")

    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in joined_rows:
        group = str(row.get("manual_group", "unassigned"))
        rows_by_group.setdefault(group, []).append(row)

    md_lines: list[str] = []
    md_lines.append("# Decrease Probe Contrast Report")
    md_lines.append("")
    md_lines.append("## Group Summary")
    md_lines.append("")
    md_lines.append("| group | count | valid_increase_cancel_coverage_mean | valid_decrease_cancel_coverage_mean | decrease_source_score | best_decrease_rel_lo_reduction | best_decrease_lc_loss_reduction | best_decrease_mode_delta_ratio | stable_positive_rate | best_decrease_probe_type_distribution |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for group_name in ["strong", "medium", "weak", "opportunity_or_lightprobe_weak", "unassigned"]:
        rows = rows_by_group.get(group_name, [])
        if not rows:
            continue
        md_lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(len(rows)),
                    f"{_mean(rows, 'valid_increase_cancel_coverage_mean'):.6f}",
                    f"{_mean(rows, 'valid_decrease_cancel_coverage_mean'):.6f}",
                    f"{_mean(rows, 'decrease_source_score'):.6f}",
                    f"{_mean(rows, 'best_decrease_rel_lo_reduction'):.6f}",
                    f"{_mean(rows, 'best_decrease_lc_loss_reduction'):.6f}",
                    f"{_mean(rows, 'best_decrease_mode_delta_ratio'):.6f}",
                    f"{_rate_true(rows, 'decrease_probe_has_stable_positive'):.6f}",
                    _distribution(rows, "best_decrease_probe_type"),
                ]
            )
            + " |"
        )

    md_lines.append("")
    md_lines.append("## Key Answers")
    md_lines.append("")

    weak_rows = rows_by_group.get("weak", [])
    opp_rows = rows_by_group.get("opportunity_or_lightprobe_weak", [])
    top_rows = [row for row in joined_rows if str(row.get("best_decrease_probe_type", "")) == "decrease_top_cancelled_lo"]
    low_rows = [row for row in joined_rows if str(row.get("best_decrease_probe_type", "")) == "decrease_low_cancel_high_budget"]

    weak_beats_increase = _rate_true(weak_rows, "decrease_probe_best_beats_increase_reference")
    opp_stable_positive = _rate_true(opp_rows, "decrease_probe_has_stable_positive")
    top_mode_tradeoff = _mean(top_rows, "best_decrease_mode_delta_ratio")
    low_mode_tradeoff = _mean(low_rows, "best_decrease_mode_delta_ratio")

    md_lines.append(f"- weak seed 中，decrease 优于 increase reference 的比例：{weak_beats_increase:.6f}。")
    md_lines.append(f"- opportunity/lightprobe weak seed 中，decrease 出现 stable positive 的比例：{opp_stable_positive:.6f}。")
    md_lines.append(
        f"- 两类 decrease 对比：top_cancelled 的平均相对 mode 变化比率为 {top_mode_tradeoff:.6f}，"
        f"low_cancel_high_budget 为 {low_mode_tradeoff:.6f}。"
    )
    md_lines.append(
        "- mode tradeoff 判断建议：若 `best_decrease_mode_delta_ratio` 在组内均值明显高于 `stable_mode_delta`（通常 0.05），"
        "则说明 decrease 改善可能伴随模式切换代价。"
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
