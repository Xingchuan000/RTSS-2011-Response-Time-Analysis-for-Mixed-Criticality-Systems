"""Normalize tree/HOUT leaf audit rows and rank mutation candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def build_leaf_candidate_table(
    *,
    tree_path: Path,
    audit_paths: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    leaves = {
        int(leaf["node_id"]): leaf
        for leaf in tree.get("leaves", ())
        if isinstance(leaf, dict) and "node_id" in leaf
    }
    aggregates: dict[int, dict[str, Any]] = {
        leaf_id: {
            "leaf_id": leaf_id,
            "action_ranking": list(leaf.get("action_ranking", ())),
            "raw_action_id": leaf.get("raw_action_id"),
            "training_samples": leaf.get("n_node_samples", 0),
            "hout_hit_count": 0,
            "scenario_coverage": set(),
            "raw_top1_invalid_count": 0,
            "fallback_count": 0,
            "all_invalid_count": 0,
            "noop_count": 0,
            "reject_reasons": {},
        }
        for leaf_id, leaf in leaves.items()
    }
    for path in audit_paths:
        for row in _read_rows(path):
            if "leaf_id" not in row:
                continue
            leaf_id = int(row["leaf_id"])
            target = aggregates.setdefault(leaf_id, {"leaf_id": leaf_id})
            hits = int(row.get("hit_count", row.get("hout_hit_count", row.get("count", 1))))
            target["hout_hit_count"] = int(target.get("hout_hit_count", 0)) + hits
            scenario = row.get("scenario_id", row.get("scenario"))
            target.setdefault("scenario_coverage", set())
            if scenario is not None:
                target["scenario_coverage"].add(str(scenario))
            for source, destination in (
                ("raw_top1_invalid_count", "raw_top1_invalid_count"),
                ("fallback_count", "fallback_count"),
                ("all_invalid_count", "all_invalid_count"),
                ("noop_count", "noop_count"),
            ):
                target[destination] = int(target.get(destination, 0)) + int(row.get(source, 0))
            reason = row.get("reject_reason")
            if reason:
                histogram = target.setdefault("reject_reasons", {})
                histogram[str(reason)] = histogram.get(str(reason), 0) + hits
    result = []
    for row in aggregates.values():
        scenarios = row.get("scenario_coverage", set())
        row["scenario_coverage"] = sorted(scenarios)
        row["scenario_coverage_count"] = len(scenarios)
        row["candidate_score"] = [
            int(row.get("hout_hit_count", 0)),
            int(row.get("raw_top1_invalid_count", 0) > 0),
            int(row.get("fallback_count", 0)),
            int(row.get("scenario_coverage_count", 0)),
            int(row.get("training_samples", 0)),
        ]
        result.append(row)
    return sorted(result, key=lambda item: tuple(item["candidate_score"]), reverse=True)


def choose_high_frequency_guard(rows: Iterable[Mapping[str, Any]]) -> str | None:
    histogram: dict[str, int] = {}
    for row in rows:
        for reason, count in dict(row.get("reject_reasons", {})).items():
            histogram[str(reason)] = histogram.get(str(reason), 0) + int(count)
    return max(histogram, key=histogram.get) if histogram else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("leaves", raw.get("rows", raw.get("events", [])))
    if not isinstance(raw, list):
        raise ValueError(f"leaf audit 必须为 row array: {path}")
    return [dict(row) for row in raw if isinstance(row, dict)]
