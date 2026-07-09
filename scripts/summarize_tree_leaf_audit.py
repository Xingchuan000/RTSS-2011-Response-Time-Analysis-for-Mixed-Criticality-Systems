"""Leaf-level execution audit 后处理汇总脚本。

把多个 seed/method 的 leaf_audit.jsonl 汇总为跨 seed 的审计表。

所有聚合 key 统一包含 taskset_seed，避免 formal10 等多 taskset 场景下
不同 taskset 的同名 tree/leaf/seed 被错误合并。

用法：

    python scripts/summarize_tree_leaf_audit.py \
        --audit-dir outputs/.../h5/tree_audit \
        --output-dir outputs/.../h5/tree_audit_summary

输出文件：
- leaf_summary_all.csv：按 (taskset_seed, method, tree_id, tree_leaf_id) 聚合。
- leaf_teacher_disagreement.csv：teacher disagreement 高的叶子排行。
- leaf_fallback_summary.csv：fallback 率高的叶子排行。
- leaf_high_regret_cases.csv：step-level 高 regret 明细（top 1000）。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np


def _load_all_jsonl(audit_dir: Path, pattern: str) -> list[dict[str, object]]:
    """加载 audit_dir 下所有匹配 pattern 的 JSONL 文件，合并为单一行列表。"""

    rows: list[dict[str, object]] = []
    for filepath in sorted(audit_dir.glob(pattern)):
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _safe_float(value: object, default: float = 0.0) -> float:
    """尝试转为 float，失败则返回默认值。"""

    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _taskset_seed_of(row: dict[str, object]) -> str:
    """统一提取 taskset_seed 并转为字符串，避免 None / 空值飘移。"""

    value = row.get("taskset_seed", "")
    return str(value) if value is not None else ""


def _leaf_group_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    """返回跨 taskset 安全的 leaf 聚合 key。

    key = (taskset_seed, method, tree_id, tree_leaf_id)
    确保不同 taskset 的同名 tree/leaf 不会被错误合并。
    """

    taskset_seed = _taskset_seed_of(row)
    method = str(row.get("method", ""))
    tree_id = str(row.get("tree_id", ""))
    leaf_id = int(row.get("tree_leaf_id", -1))
    return taskset_seed, method, tree_id, leaf_id


def _seed_total_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    """返回计算 per-seed denominator 的 key。

    key = (taskset_seed, method, tree_id, seed)
    用于按 (taskset_seed, method, tree_id, seed) 统计每组的总 audit 行数，
    确保 hit_rate denominator 不会被其他 taskset/method/tree 稀释。
    """

    taskset_seed = _taskset_seed_of(row)
    method = str(row.get("method", ""))
    tree_id = str(row.get("tree_id", ""))
    seed = int(row.get("seed", 0))
    return taskset_seed, method, tree_id, seed


def _build_leaf_summary_all(
    rows: list[dict[str, object]],
    output_dir: Path,
    min_hit_count: int = 1,
) -> None:
    """按 (taskset_seed, method, tree_id, tree_leaf_id) 聚合，输出 leaf_summary_all.csv。"""

    # 按 (taskset_seed, method, tree_id, leaf_id) 分组
    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    seed_sets: dict[tuple[str, str, str, int], set[int]] = {}
    for row in rows:
        leaf_id = int(row.get("tree_leaf_id", -1))
        seed = int(row.get("seed", 0))
        key = _leaf_group_key(row)
        groups.setdefault(key, []).append(row)
        seed_sets.setdefault(key, set()).add(seed)

    # 整个数据集的 denominator: 按 (_seed_total_key) 统计每组总 audit 行数
    seed_total: dict[tuple[str, str, str, int], int] = {}
    for r in rows:
        key = _seed_total_key(r)
        seed_total[key] = seed_total.get(key, 0) + 1

    summary_rows: list[dict[str, object]] = []
    for (taskset_seed, method, tree_id, leaf_id), leaf_rows in sorted(groups.items()):
        hit_count = len(leaf_rows)
        if hit_count < min_hit_count:
            continue
        seed_count = len(seed_sets.get((taskset_seed, method, tree_id, leaf_id), set()))

        # 取第一行的静态信息
        first = leaf_rows[0]
        path_depth = first.get("tree_path_depth")
        leaf_n_node_samples = first.get("tree_leaf_n_node_samples")
        leaf_impurity = first.get("tree_leaf_impurity")
        path_predicates_json = first.get("tree_path_predicates_json")

        # 计算每个 seed 的 hit_rate，再平均
        seed_hit_rates: list[float] = []
        seed_groups: dict[int, list[dict[str, object]]] = {}
        for r in leaf_rows:
            s = int(r.get("seed", 0))
            seed_groups.setdefault(s, []).append(r)
        for s, s_rows in seed_groups.items():
            total = max(seed_total.get((taskset_seed, method, tree_id, s), 0), 1)
            seed_hit_rates.append(len(s_rows) / total)
        hit_rate_mean_per_seed = mean(seed_hit_rates) if seed_hit_rates else 0.0

        # 动作众数
        raw_top1_ids = [r.get("tree_raw_top1_action_id") for r in leaf_rows]
        raw_top1_counter = Counter(r for r in raw_top1_ids if r is not None)
        raw_top1_mode = raw_top1_counter.most_common(1)[0][0] if raw_top1_counter else None

        sel_ids = [r.get("tree_selected_action_id") for r in leaf_rows]
        sel_counter = Counter(s for s in sel_ids if s is not None)
        sel_mode = sel_counter.most_common(1)[0][0] if sel_counter else None

        # fallback 统计
        fallback_count = sum(int(bool(r.get("tree_fallback_used", False))) for r in leaf_rows)
        fallback_rate = fallback_count / hit_count if hit_count > 0 else 0.0

        # raw_invalid 统计
        raw_invalid_count = sum(int(bool(r.get("tree_raw_top1_invalid", False))) for r in leaf_rows)
        raw_invalid_rate = raw_invalid_count / hit_count if hit_count > 0 else 0.0

        # teacher match 统计
        match_vals = [r.get("teacher_selected_action_match") for r in leaf_rows]
        match_count = sum(1 for v in match_vals if v is True)
        match_rate = match_count / hit_count if hit_count > 0 else 0.0

        raw_match_vals = [r.get("teacher_raw_action_match") for r in leaf_rows]
        raw_match_count = sum(1 for v in raw_match_vals if v is True)
        raw_match_rate = raw_match_count / hit_count if hit_count > 0 else 0.0

        # q_regret 统计
        q_regrets = [
            _safe_float(r.get("teacher_q_regret_selected"))
            for r in leaf_rows
            if r.get("teacher_q_regret_selected") is not None
        ]
        q_regret_mean = float(np.mean(q_regrets)) if q_regrets else None
        q_regret_p95 = float(np.percentile(q_regrets, 95)) if q_regrets else None

        # reward 统计
        rewards = [_safe_float(r.get("reward")) for r in leaf_rows]
        reward_sum = float(np.sum(rewards))
        reward_mean = float(np.mean(rewards)) if rewards else 0.0

        # accepted 统计
        accepted_count = sum(int(bool(r.get("accepted", False))) for r in leaf_rows)
        accepted_rate = accepted_count / hit_count if hit_count > 0 else 0.0

        # outcome delta 统计
        delta_deadline_misses_sum = sum(
            int(r.get("delta_deadline_misses", 0) or 0) for r in leaf_rows
        )
        delta_mode_changes_sum = sum(
            int(r.get("delta_mode_changes", 0) or 0) for r in leaf_rows
        )
        delta_lo_cancellations_sum = sum(
            int(r.get("delta_lo_cancellations", 0) or 0) for r in leaf_rows
        )

        summary_rows.append({
            "taskset_seed": taskset_seed,
            "method": method,
            "tree_id": tree_id,
            "tree_leaf_id": leaf_id,
            "hit_count": hit_count,
            "seed_count": seed_count,
            "hit_rate_mean_per_seed": hit_rate_mean_per_seed,
            "path_depth": path_depth,
            "leaf_n_node_samples": leaf_n_node_samples,
            "leaf_impurity": leaf_impurity,
            "selected_action_id_mode": sel_mode,
            "raw_top1_action_id_mode": raw_top1_mode,
            "fallback_rate": fallback_rate,
            "raw_invalid_rate": raw_invalid_rate,
            "teacher_match_rate": match_rate,
            "raw_teacher_match_rate": raw_match_rate,
            "q_regret_selected_mean": q_regret_mean,
            "q_regret_selected_p95": q_regret_p95,
            "reward_mean": reward_mean,
            "reward_sum": reward_sum,
            "accepted_rate": accepted_rate,
            "delta_deadline_misses_sum": delta_deadline_misses_sum,
            "delta_mode_changes_sum": delta_mode_changes_sum,
            "delta_lo_cancellations_sum": delta_lo_cancellations_sum,
            "path_predicates_json": path_predicates_json if path_predicates_json is not None else "",
        })

    csv_path = output_dir / "leaf_summary_all.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


def _build_leaf_teacher_disagreement(
    rows: list[dict[str, object]],
    output_dir: Path,
    min_hit_count: int = 1,
) -> None:
    """筛选 teacher_selected_action_match == False 的行，按 leaf 聚合，输出 leaf_teacher_disagreement.csv。"""

    disagree_rows = [
        r for r in rows
        if r.get("teacher_selected_action_match") is False
    ]
    if not disagree_rows:
        return

    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in disagree_rows:
        key = _leaf_group_key(row)
        groups.setdefault(key, []).append(row)

    # total_hits per leaf for rate calculation
    total_hits: dict[tuple[str, str, str, int], int] = {}
    for row in rows:
        key = _leaf_group_key(row)
        total_hits[key] = total_hits.get(key, 0) + 1

    summary_rows: list[dict[str, object]] = []
    for (taskset_seed, method, tree_id, leaf_id), d_rows in groups.items():
        hit_count = total_hits.get((taskset_seed, method, tree_id, leaf_id), len(d_rows))
        if hit_count < min_hit_count:
            continue
        disagreement_count = len(d_rows)
        disagreement_rate = disagreement_count / hit_count if hit_count > 0 else 0.0

        teacher_best_ids = [r.get("teacher_best_action_id") for r in d_rows]
        teacher_best_counter = Counter(t for t in teacher_best_ids if t is not None)
        teacher_best_mode = teacher_best_counter.most_common(1)[0][0] if teacher_best_counter else None

        sel_ids = [r.get("tree_selected_action_id") for r in d_rows]
        sel_counter = Counter(s for s in sel_ids if s is not None)
        sel_mode = sel_counter.most_common(1)[0][0] if sel_counter else None

        q_regrets = [
            _safe_float(r.get("teacher_q_regret_selected"))
            for r in d_rows
            if r.get("teacher_q_regret_selected") is not None
        ]
        q_regret_mean = float(np.mean(q_regrets)) if q_regrets else None
        q_regret_p95 = float(np.percentile(q_regrets, 95)) if q_regrets else None

        first = d_rows[0]
        path_predicates_json = first.get("tree_path_predicates_json")

        summary_rows.append({
            "taskset_seed": taskset_seed,
            "method": method,
            "tree_id": tree_id,
            "tree_leaf_id": leaf_id,
            "disagreement_count": disagreement_count,
            "disagreement_rate": disagreement_rate,
            "teacher_best_action_id_mode": teacher_best_mode,
            "selected_action_id_mode": sel_mode,
            "q_regret_selected_mean": q_regret_mean,
            "q_regret_selected_p95": q_regret_p95,
            "path_predicates_json": path_predicates_json if path_predicates_json is not None else "",
        })

    # 按 disagreement_rate 和 q_regret_selected_mean 降序排序
    summary_rows.sort(
        key=lambda r: (
            -float(r.get("disagreement_rate", 0) or 0),
            -(float(r.get("q_regret_selected_mean", 0) or 0)),
        )
    )

    csv_path = output_dir / "leaf_teacher_disagreement.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


def _build_leaf_fallback_summary(
    rows: list[dict[str, object]],
    output_dir: Path,
    min_hit_count: int = 1,
) -> None:
    """按 leaf 聚合 fallback 信息，输出 leaf_fallback_summary.csv。"""

    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = _leaf_group_key(row)
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (taskset_seed, method, tree_id, leaf_id), leaf_rows in sorted(groups.items()):
        hit_count = len(leaf_rows)
        if hit_count < min_hit_count:
            continue
        fallback_count = sum(int(bool(r.get("tree_fallback_used", False))) for r in leaf_rows)
        fallback_rate = fallback_count / hit_count if hit_count > 0 else 0.0

        raw_invalid_count = sum(int(bool(r.get("tree_raw_top1_invalid", False))) for r in leaf_rows)
        raw_invalid_rate = raw_invalid_count / hit_count if hit_count > 0 else 0.0

        raw_top1_ids = [r.get("tree_raw_top1_action_id") for r in leaf_rows]
        raw_top1_counter = Counter(r for r in raw_top1_ids if r is not None)
        raw_top1_mode = raw_top1_counter.most_common(1)[0][0] if raw_top1_counter else None

        sel_ids = [r.get("tree_selected_action_id") for r in leaf_rows]
        sel_counter = Counter(s for s in sel_ids if s is not None)
        sel_mode = sel_counter.most_common(1)[0][0] if sel_counter else None

        first = leaf_rows[0]
        path_predicates_json = first.get("tree_path_predicates_json")

        summary_rows.append({
            "taskset_seed": taskset_seed,
            "method": method,
            "tree_id": tree_id,
            "tree_leaf_id": leaf_id,
            "hit_count": hit_count,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "raw_invalid_count": raw_invalid_count,
            "raw_invalid_rate": raw_invalid_rate,
            "raw_top1_action_id_mode": raw_top1_mode,
            "selected_action_id_mode": sel_mode,
            "path_predicates_json": path_predicates_json if path_predicates_json is not None else "",
        })

    # 按 fallback_rate 降序排序
    summary_rows.sort(
        key=lambda r: -float(r.get("fallback_rate", 0) or 0)
    )

    csv_path = output_dir / "leaf_fallback_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


def _build_leaf_high_regret_cases(
    rows: list[dict[str, object]],
    output_dir: Path,
    min_hit_count: int,
) -> None:
    """输出 step-level 高 regret 明细，按 q_regret 降序排序保留 top 1000。

    只保留总 hit_count >= min_hit_count 的叶子上的行。
    """

    # 先计算每个 leaf 的总 hit_count（用于 min_hit_count 过滤）
    leaf_hits: dict[tuple[str, str, str, int], int] = {}
    for row in rows:
        key = _leaf_group_key(row)
        leaf_hits[key] = leaf_hits.get(key, 0) + 1

    # 筛选有 teacher_q_regret_selected 且 leaf hit >= min_hit_count 的行
    regret_rows = [
        r for r in rows
        if r.get("teacher_q_regret_selected") is not None
        and leaf_hits.get(_leaf_group_key(r), 0) >= min_hit_count
    ]
    if not regret_rows:
        return

    regret_rows.sort(
        key=lambda r: -float(r.get("teacher_q_regret_selected", 0.0))
    )

    # 保留 top 1000
    top_rows = regret_rows[:1000]

    detail_rows: list[dict[str, object]] = []
    for row in top_rows:
        detail_rows.append({
            "taskset_seed": row.get("taskset_seed"),
            "seed": row.get("seed"),
            "scenario_seed": row.get("scenario_seed"),
            "method": row.get("method"),
            "tree_id": row.get("tree_id"),
            "time": row.get("time"),
            "tree_leaf_id": row.get("tree_leaf_id"),
            "tree_path_depth": row.get("tree_path_depth"),
            "tree_raw_top1_action_id": row.get("tree_raw_top1_action_id"),
            "tree_selected_action_id": row.get("tree_selected_action_id"),
            "teacher_best_action_id": row.get("teacher_best_action_id"),
            "teacher_q_best": row.get("teacher_q_best"),
            "teacher_q_selected": row.get("teacher_q_selected"),
            "teacher_q_regret_selected": row.get("teacher_q_regret_selected"),
            "reward": row.get("reward"),
            "accepted": row.get("accepted"),
            "delta_deadline_misses": row.get("delta_deadline_misses"),
            "delta_lo_cancellations": row.get("delta_lo_cancellations"),
            "path_predicates_json": row.get("tree_path_predicates_json", ""),
            "tree_path_feature_values_json": row.get("tree_path_feature_values_json", ""),
        })

    csv_path = output_dir / "leaf_high_regret_cases.csv"
    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)


def main() -> None:
    """leaf audit 汇总入口。"""

    parser = argparse.ArgumentParser(description="Leaf-level execution audit 汇总")
    parser.add_argument("--audit-dir", type=Path, required=True, help="leaf_audit.jsonl 所在目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_leaf_audit.jsonl",
        help="JSONL 文件匹配模式",
    )
    parser.add_argument(
        "--min-hit-count",
        type=int,
        default=1,
        help="最小 hit count 过滤阈值",
    )
    args = parser.parse_args()

    if not args.audit_dir.is_dir():
        raise ValueError(f"audit-dir 不存在或不是目录: {args.audit_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 JSONL
    rows = _load_all_jsonl(args.audit_dir, args.pattern)
    if not rows:
        print(f"未找到匹配模式的 JSONL 文件: {args.audit_dir}/{args.pattern}")
        return

    print(f"加载 {len(rows)} 条 audit 记录")

    # 生成各汇总文件
    _build_leaf_summary_all(rows, args.output_dir, args.min_hit_count)
    print(f"  -> leaf_summary_all.csv")

    _build_leaf_teacher_disagreement(rows, args.output_dir, args.min_hit_count)
    print(f"  -> leaf_teacher_disagreement.csv")

    _build_leaf_fallback_summary(rows, args.output_dir, args.min_hit_count)
    print(f"  -> leaf_fallback_summary.csv")

    _build_leaf_high_regret_cases(rows, args.output_dir, args.min_hit_count)
    print(f"  -> leaf_high_regret_cases.csv")

    print("汇总完成。")


if __name__ == "__main__":
    main()
