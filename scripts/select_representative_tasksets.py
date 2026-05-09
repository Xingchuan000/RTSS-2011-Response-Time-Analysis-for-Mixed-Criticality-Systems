"""阶段 B：从扫描结果中按研究角色选择代表性 taskset。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_bool(value: str) -> bool:
    """把 CSV 中的布尔文本统一转换为 bool。"""

    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_float(row: dict[str, str], key: str) -> float:
    """把 CSV 数值字段解析为 float。"""

    text = str(row.get(key, "")).strip()
    return float(text) if text else 0.0


def _eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """筛选可用于代表性选择的候选集合。"""

    return [
        row
        for row in rows
        if _to_bool(row.get("schedulable", "false"))
        and _to_float(row, "baseline_deadline_misses_sum") == 0.0
    ]


def _select_current_seed(rows: list[dict[str, str]], seed: int) -> dict[str, str] | None:
    """选择当前研究基准 seed。"""

    for row in rows:
        if int(float(row["fixed_taskset_seed"])) == seed:
            return row
    return None


def _select_high_balanced_event_candidate(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """选择高事件且 mode/lo 相对均衡的候选。"""

    candidates = _eligible_rows(rows)
    if not candidates:
        return None

    def score(row: dict[str, str]) -> float:
        mode = _to_float(row, "baseline_mode_changes_mean")
        lo = _to_float(row, "baseline_lo_cancellations_mean")
        total = _to_float(row, "baseline_total_events_mean")
        imbalance = abs(mode - lo)
        return total - imbalance

    return max(candidates, key=score)


def _select_high_lo_candidate(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """选择 LO cancellation 高点候选。"""

    candidates = _eligible_rows(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row, "baseline_lo_cancellations_mean"))


def _select_high_mode_candidate(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """选择 mode change 高点候选。"""

    candidates = _eligible_rows(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row, "baseline_mode_changes_mean"))


def _select_best_increase_headroom_candidate(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """选择 increase headroom 最大的候选。"""

    candidates = _eligible_rows(rows)
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row, "valid_increase_count_mean"))


def _add_selection(
    selections: dict[int, dict[str, str]],
    *,
    row: dict[str, str] | None,
    role: str,
    reason: str,
) -> None:
    """把某个角色的选择结果写入去重容器。

    去重规则：
    - 同一个 seed 被多个角色命中时，不重复新增行；
    - 在原有行上追加 `selection_roles` 与 `selection_reasons`。
    """

    if row is None:
        return

    seed = int(float(row["fixed_taskset_seed"]))
    if seed in selections:
        selections[seed]["selection_roles"] = f"{selections[seed]['selection_roles']};{role}"
        selections[seed]["selection_reasons"] = f"{selections[seed]['selection_reasons']};{reason}"
        return

    result = dict(row)
    result["selection_roles"] = role
    result["selection_reasons"] = reason
    selections[seed] = result


def _row_to_output(rank: int, row: dict[str, str]) -> dict[str, str | int | float | bool]:
    """把扫描结果行映射为输出 CSV 行。"""

    return {
        "rank": rank,
        "selection_roles": row.get("selection_roles", ""),
        "selection_reasons": row.get("selection_reasons", ""),
        "fixed_taskset_seed": int(float(row.get("fixed_taskset_seed", "0") or "0")),
        "schedulable": _to_bool(row.get("schedulable", "false")),
        "num_tasks": _to_float(row, "num_tasks"),
        "num_hi_tasks": _to_float(row, "num_hi_tasks"),
        "num_lo_tasks": _to_float(row, "num_lo_tasks"),
        "baseline_mode_changes_mean": _to_float(row, "baseline_mode_changes_mean"),
        "baseline_lo_cancellations_mean": _to_float(row, "baseline_lo_cancellations_mean"),
        "baseline_total_events_mean": _to_float(row, "baseline_total_events_mean"),
        "baseline_deadline_misses_sum": _to_float(row, "baseline_deadline_misses_sum"),
        "valid_action_count_mean": _to_float(row, "valid_action_count_mean"),
        "valid_increase_count_mean": _to_float(row, "valid_increase_count_mean"),
        "valid_decrease_count_mean": _to_float(row, "valid_decrease_count_mean"),
        "increase_headroom_group": row.get("increase_headroom_group", ""),
        "decrease_headroom_group": row.get("decrease_headroom_group", ""),
        "balanced_headroom_group": row.get("balanced_headroom_group", ""),
        "increase_decrease_balance": _to_float(row, "increase_decrease_balance"),
        "safety_margin_min_mean": _to_float(row, "safety_margin_min_mean"),
        "safety_margin_min_p05": _to_float(row, "safety_margin_min_p05"),
        "safety_margin_min_fraction_zero": _to_float(row, "safety_margin_min_fraction_zero"),
        "recommended_for_dqn": _to_bool(row.get("recommended_for_dqn", "false")),
        "not_recommended_reason": row.get("not_recommended_reason", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-seed", type=int, default=1)
    parser.add_argument("--max-tasksets", type=int, default=5)
    parser.add_argument("--min-tasksets", type=int, default=3)
    return parser


def main() -> None:
    """执行角色驱动的代表性 taskset 选择。"""

    args = build_parser().parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    selections_by_seed: dict[int, dict[str, str]] = {}

    _add_selection(
        selections_by_seed,
        row=_select_current_seed(rows, args.current_seed),
        role="current_baseline_seed",
        reason="当前研究基准 taskset，用于与历史结果保持可比性。",
    )
    _add_selection(
        selections_by_seed,
        row=_select_high_balanced_event_candidate(rows),
        role="high_balanced_event",
        reason="事件总量高且 mode change 与 LO cancellation 相对均衡。",
    )
    _add_selection(
        selections_by_seed,
        row=_select_high_lo_candidate(rows),
        role="high_lo_cancellation",
        reason="LO cancellation 强信号 taskset，用于验证 LO 侧改善空间。",
    )

    if args.max_tasksets >= 4:
        _add_selection(
            selections_by_seed,
            row=_select_high_mode_candidate(rows),
            role="high_mode_change",
            reason="mode change 强信号 taskset，用于验证模式切换优化空间。",
        )

    if args.max_tasksets >= 5:
        _add_selection(
            selections_by_seed,
            row=_select_best_increase_headroom_candidate(rows),
            role="best_increase_headroom",
            reason="increase headroom 最大 taskset，用于验证可增预算空间影响。",
        )

    # 固定角色顺序，保证输出稳定可读。
    ordered_role_priority = [
        "current_baseline_seed",
        "high_balanced_event",
        "high_lo_cancellation",
        "high_mode_change",
        "best_increase_headroom",
    ]

    def first_role_priority(row: dict[str, str]) -> int:
        roles = row.get("selection_roles", "").split(";")
        for index, role in enumerate(ordered_role_priority):
            if role in roles:
                return index
        return len(ordered_role_priority)

    selected_rows_sorted = sorted(selections_by_seed.values(), key=first_role_priority)

    # 阶段要求是“至少 3 个”，因此输出数量下界受 min_tasksets 控制，上界受 max_tasksets 控制。
    selected_rows_sorted = selected_rows_sorted[: args.max_tasksets]
    if len(selected_rows_sorted) < args.min_tasksets:
        eligible = _eligible_rows(rows)
        chosen_seeds = {int(float(row["fixed_taskset_seed"])) for row in selected_rows_sorted}
        for row in eligible:
            seed = int(float(row["fixed_taskset_seed"]))
            if seed in chosen_seeds:
                continue
            row_copy = dict(row)
            row_copy["selection_roles"] = "supplemental_candidate"
            row_copy["selection_reasons"] = "补足最小输出数量。"
            selected_rows_sorted.append(row_copy)
            chosen_seeds.add(seed)
            if len(selected_rows_sorted) >= args.min_tasksets:
                break

    output_rows = [_row_to_output(rank=i + 1, row=row) for i, row in enumerate(selected_rows_sorted)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "selection_roles",
        "selection_reasons",
        "fixed_taskset_seed",
        "schedulable",
        "num_tasks",
        "num_hi_tasks",
        "num_lo_tasks",
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "baseline_total_events_mean",
        "baseline_deadline_misses_sum",
        "valid_action_count_mean",
        "valid_increase_count_mean",
        "valid_decrease_count_mean",
        "increase_headroom_group",
        "decrease_headroom_group",
        "balanced_headroom_group",
        "increase_decrease_balance",
        "safety_margin_min_mean",
        "safety_margin_min_p05",
        "safety_margin_min_fraction_zero",
        "recommended_for_dqn",
        "not_recommended_reason",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Selected tasksets:")
    for row in output_rows:
        print(
            f"seed={row['fixed_taskset_seed']} "
            f"roles={row['selection_roles']} "
            f"events={row['baseline_total_events_mean']} "
            f"inc={row['valid_increase_count_mean']} "
            f"balance={row['balanced_headroom_group']}"
        )


if __name__ == "__main__":
    main()
