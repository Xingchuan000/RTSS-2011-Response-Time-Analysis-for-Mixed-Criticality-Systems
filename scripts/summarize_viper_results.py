"""汇总 tree retention 与复杂度结果。"""

from __future__ import annotations

import argparse
import csv
from statistics import median
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.viper.metrics import retention_higher_is_better, retention_lower_is_better


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _summary_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return (sum(values) / len(values), median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--parent-method", type=str, required=True)
    parser.add_argument("--teacher-method", type=str, required=True)
    parser.add_argument("--tree-method", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-empty", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    rows = _read_rows(args.eval_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, object]] = []
    by_seed: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["method"]] = row
    for seed, method_rows in by_seed.items():
        if args.parent_method not in method_rows or args.teacher_method not in method_rows or args.tree_method not in method_rows:
            continue
        parent = method_rows[args.parent_method]
        teacher = method_rows[args.teacher_method]
        tree = method_rows[args.tree_method]
        teacher_no_positive_qos_gain = float(teacher["lo_quality_qos"]) <= float(parent["lo_quality_qos"])
        teacher_no_positive_zero_service_gain = (
            float(teacher["lo_zero_service_ratio"]) >= float(parent["lo_zero_service_ratio"])
        )
        teacher_no_positive_equiv_jne_gain = float(teacher["lo_equiv_jne"]) >= float(parent["lo_equiv_jne"])
        qos_retention = retention_higher_is_better(
            float(parent["lo_quality_qos"]),
            float(teacher["lo_quality_qos"]),
            float(tree["lo_quality_qos"]),
        )
        zero_service_retention = retention_lower_is_better(
            float(parent["lo_zero_service_ratio"]),
            float(teacher["lo_zero_service_ratio"]),
            float(tree["lo_zero_service_ratio"]),
        )
        equiv_jne_retention = retention_lower_is_better(
            float(parent["lo_equiv_jne"]),
            float(teacher["lo_equiv_jne"]),
            float(tree["lo_equiv_jne"]),
        )
        per_seed.append(
            {
                "seed": seed,
                "parent_lo_quality_qos": parent["lo_quality_qos"],
                "teacher_lo_quality_qos": teacher["lo_quality_qos"],
                "qos_retention": qos_retention,
                "parent_lo_zero_service_ratio": parent["lo_zero_service_ratio"],
                "teacher_lo_zero_service_ratio": teacher["lo_zero_service_ratio"],
                "zero_service_retention": zero_service_retention,
                "parent_lo_equiv_jne": parent["lo_equiv_jne"],
                "teacher_lo_equiv_jne": teacher["lo_equiv_jne"],
                "tree_lo_quality_qos": tree["lo_quality_qos"],
                "tree_lo_zero_service_ratio": tree["lo_zero_service_ratio"],
                "tree_lo_equiv_jne": tree["lo_equiv_jne"],
                "equiv_jne_retention": equiv_jne_retention,
                "tree_depth": tree.get("tree_depth", ""),
                "tree_node_count": tree.get("tree_node_count", ""),
                "tree_leaf_count": tree.get("tree_leaf_count", ""),
                "tree_raw_top1_invalid_rate": tree.get("tree_raw_top1_invalid_rate", ""),
                "tree_fallback_rate": tree.get("tree_fallback_rate", ""),
                "tree_deadline_misses": tree["deadline_misses"],
                "tree_hi_deadline_misses": tree["hi_deadline_misses"],
                "tree_lo_deadline_misses": tree["lo_deadline_misses"],
                "teacher_no_positive_qos_gain": teacher_no_positive_qos_gain,
                "teacher_no_positive_zero_service_gain": teacher_no_positive_zero_service_gain,
                "teacher_no_positive_equiv_jne_gain": teacher_no_positive_equiv_jne_gain,
            }
        )
    if not per_seed and not args.allow_empty:
        available_methods = sorted({row["method"] for row in rows})
        raise SystemExit(
            "没有找到同时包含 parent/teacher/tree 的 seed；"
            f"parent={args.parent_method}, teacher={args.teacher_method}, tree={args.tree_method}, "
            f"available_methods={available_methods}"
        )
    with (args.output_dir / "per_seed_tree_vs_parent_vs_dqn.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(per_seed[0].keys()) if per_seed else [
            "seed",
            "parent_lo_quality_qos",
            "teacher_lo_quality_qos",
            "tree_lo_quality_qos",
            "qos_retention",
            "parent_lo_zero_service_ratio",
            "teacher_lo_zero_service_ratio",
            "tree_lo_zero_service_ratio",
            "zero_service_retention",
            "parent_lo_equiv_jne",
            "teacher_lo_equiv_jne",
            "tree_lo_equiv_jne",
            "equiv_jne_retention",
            "tree_depth",
            "tree_node_count",
            "tree_leaf_count",
            "tree_raw_top1_invalid_rate",
            "tree_fallback_rate",
            "tree_deadline_misses",
            "tree_hi_deadline_misses",
            "tree_lo_deadline_misses",
            "teacher_no_positive_qos_gain",
            "teacher_no_positive_zero_service_gain",
            "teacher_no_positive_equiv_jne_gain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if per_seed:
            writer.writerows(per_seed)
    qos_values = [float(row["qos_retention"]) for row in per_seed if row["qos_retention"] is not None]
    zero_values = [float(row["zero_service_retention"]) for row in per_seed if row["zero_service_retention"] is not None]
    equiv_values = [float(row["equiv_jne_retention"]) for row in per_seed if row["equiv_jne_retention"] is not None]
    qos_mean, qos_median = _summary_stats(qos_values)
    zero_mean, zero_median = _summary_stats(zero_values)
    equiv_mean, equiv_median = _summary_stats(equiv_values)
    with (args.output_dir / "retention_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "qos_retention_mean", "value": qos_mean},
                {"metric": "qos_retention_median", "value": qos_median},
                {"metric": "zero_service_retention_mean", "value": zero_mean},
                {"metric": "zero_service_retention_median", "value": zero_median},
                {"metric": "equiv_jne_retention_mean", "value": equiv_mean},
                {"metric": "equiv_jne_retention_median", "value": equiv_median},
                {"metric": "valid_seed_count", "value": len(per_seed)},
                {
                    "metric": "teacher_no_positive_gain_seed_count",
                    "value": sum(
                        int(
                            bool(row["teacher_no_positive_qos_gain"])
                            or bool(row["teacher_no_positive_zero_service_gain"])
                            or bool(row["teacher_no_positive_equiv_jne_gain"])
                        )
                        for row in per_seed
                    ),
                },
                {"metric": "qos_retention_missing_count", "value": len(per_seed) - len(qos_values)},
                {"metric": "zero_service_retention_missing_count", "value": len(per_seed) - len(zero_values)},
                {"metric": "equiv_jne_retention_missing_count", "value": len(per_seed) - len(equiv_values)},
            ]
        )
    with (args.output_dir / "tree_complexity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        if per_seed:
            depth_values = [float(row["tree_depth"]) for row in per_seed]
            node_values = [float(row["tree_node_count"]) for row in per_seed]
            leaf_values = [float(row["tree_leaf_count"]) for row in per_seed]
            writer.writerows(
                [
                    {"metric": "tree_depth_mean", "value": _summary_stats(depth_values)[0]},
                    {"metric": "tree_depth_median", "value": _summary_stats(depth_values)[1]},
                    {"metric": "tree_node_count_mean", "value": _summary_stats(node_values)[0]},
                    {"metric": "tree_node_count_median", "value": _summary_stats(node_values)[1]},
                    {"metric": "tree_leaf_count_mean", "value": _summary_stats(leaf_values)[0]},
                    {"metric": "tree_leaf_count_median", "value": _summary_stats(leaf_values)[1]},
                    {"metric": "max_tree_depth", "value": max(depth_values)},
                    {"metric": "max_tree_node_count", "value": max(node_values)},
                    {"metric": "max_tree_leaf_count", "value": max(leaf_values)},
                ]
            )
    with (args.output_dir / "tree_safety_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        if per_seed:
            invalid_values = [float(row["tree_raw_top1_invalid_rate"]) for row in per_seed]
            fallback_values = [float(row["tree_fallback_rate"]) for row in per_seed]
            writer.writerows(
                [
                    {"metric": "deadline_misses_sum", "value": sum(int(row["tree_deadline_misses"]) for row in per_seed)},
                    {"metric": "hi_deadline_misses_sum", "value": sum(int(row["tree_hi_deadline_misses"]) for row in per_seed)},
                    {"metric": "lo_deadline_misses_sum", "value": sum(int(row["tree_lo_deadline_misses"]) for row in per_seed)},
                    {"metric": "max_tree_raw_top1_invalid_rate", "value": max(invalid_values)},
                    {"metric": "mean_tree_raw_top1_invalid_rate", "value": _summary_stats(invalid_values)[0]},
                    {"metric": "max_tree_fallback_rate", "value": max(fallback_values)},
                    {"metric": "mean_tree_fallback_rate", "value": _summary_stats(fallback_values)[0]},
                ]
            )


if __name__ == "__main__":
    main()
