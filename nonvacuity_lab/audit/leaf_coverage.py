"""Build seed/variant-specific leaf coverage and action-risk audit records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..activation.event_normalizer import normalize_hout_event
from ..canonical import file_hash
from .action_risk import classify_actions
from .tree_reader import iter_leaves, leaf_guards, load_tree

LeafKey = tuple[int, str, int]


def aggregate_hout_events(paths: Iterable[Path]) -> dict[LeafKey, dict[str, Any]]:
    rows: dict[LeafKey, dict[str, Any]] = defaultdict(_empty_coverage)
    for path in paths:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        events = raw.get("events", raw) if isinstance(raw, dict) else raw
        if not isinstance(events, list):
            continue
        path_seed = _infer_seed(path)
        path_variant = _infer_variant(path)
        for event in events:
            if not isinstance(event, Mapping):
                continue
            normalized = normalize_hout_event(event)
            leaf_id = normalized.get("leaf_id")
            if leaf_id is None:
                continue
            seed = _as_int(normalized.get("seed"), path_seed)
            variant = str(normalized.get("tree_variant") or path_variant)
            row = rows[(seed, variant, int(leaf_id))]
            row["hout_hit_count"] += 1
            row["scenario_coverage"].add(str(normalized.get("scenario_id", 0)))
            row["raw_top1_invalid_count"] += int(bool(normalized.get("raw_top1_invalid")))
            selected_rank = normalized.get("selected_rank")
            row["fallback_count"] += int(selected_rank is not None and int(selected_rank) > 1)
            row["all_invalid_count"] += int(bool(normalized.get("all_invalid")))
            row["noop_count"] += int(bool(normalized.get("implicit_noop")))
            if selected_rank is not None:
                row["selected_rank_histogram"][str(selected_rank)] += 1
            reason = normalized.get("reject_reason")
            if reason:
                row["reject_reason_histogram"][str(reason)] += 1
            rejected_action = normalized.get("rejected_action_id")
            if rejected_action is not None:
                row["rejected_action_histogram"][str(rejected_action)] += 1
    return rows


def audit_all_leaves(*, seed_root: Path, hout_root: Path, source_root: Path) -> list[dict[str, Any]]:
    del source_root  # Kept in the public signature for CLI compatibility.
    rows: list[dict[str, Any]] = []
    hout_by_leaf = aggregate_hout_events(sorted(Path(hout_root).rglob("*.json")))
    for tree_path in sorted(Path(seed_root).rglob("integer_tree.json")):
        tree = load_tree(tree_path)
        seed = _infer_seed(tree_path)
        variant = _infer_variant(tree_path)
        action_path = tree_path.with_name("action_definitions.json")
        actions = _load_actions(action_path)
        tasks = _load_tasks(tree_path)
        risks = classify_actions(action_definitions=actions, tasks=tasks)
        risk_by_action = {int(record.action_id): record for record in risks}
        guards = leaf_guards(tree)
        digest = file_hash(tree_path)
        for leaf in iter_leaves(tree):
            leaf_id = int(leaf.get("leaf_id", leaf.get("node_id", leaf.get("id", -1))))
            coverage = _serializable_coverage(hout_by_leaf.get((seed, variant, leaf_id)))
            ranking = tuple(int(item) for item in leaf.get("action_ranking", leaf.get("ranking", ())))
            action_risks = []
            rejected_hist = coverage["rejected_action_histogram"]
            for action_id in ranking:
                record = risk_by_action.get(action_id)
                if record is None:
                    continue
                payload = dict(record.__dict__)
                payload["observed_reject_count"] = int(rejected_hist.get(str(action_id), 0))
                action_risks.append(payload)
            rows.append(
                {
                    "seed": seed,
                    "tree_variant": variant,
                    "tree_path": str(tree_path.resolve()),
                    "tree_hash": digest,
                    "leaf_id": leaf_id,
                    "guard": guards.get(leaf_id, []),
                    "action_ranking": list(ranking),
                    "training_samples": int(leaf.get("n_node_samples", 0)),
                    "action_risks": action_risks,
                    "selected_region_status": "UNKNOWN",
                    "symbolic_witness_ref": None,
                    **coverage,
                }
            )
    return sorted(rows, key=lambda row: (row["seed"], row["tree_variant"], row["leaf_id"]))


def _empty_coverage() -> dict[str, Any]:
    return {
        "hout_hit_count": 0,
        "scenario_coverage": set(),
        "raw_top1_invalid_count": 0,
        "fallback_count": 0,
        "all_invalid_count": 0,
        "noop_count": 0,
        "selected_rank_histogram": Counter(),
        "reject_reason_histogram": Counter(),
        "rejected_action_histogram": Counter(),
    }


def _serializable_coverage(value: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(value or _empty_coverage())
    row["scenario_coverage"] = sorted(row.get("scenario_coverage", ()))
    for key in (
        "selected_rank_histogram",
        "reject_reason_histogram",
        "rejected_action_histogram",
    ):
        row[key] = dict(row.get(key, {}))
    for key in (
        "hout_hit_count",
        "raw_top1_invalid_count",
        "fallback_count",
        "all_invalid_count",
        "noop_count",
    ):
        row[key] = int(row.get(key, 0))
    return row


def _load_actions(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        actions = raw.get("actions", [])
        if isinstance(actions, list):
            return [item for item in actions if isinstance(item, Mapping)]
    return []


def _load_tasks(tree_path: Path) -> list[Mapping[str, Any]]:
    candidates = [
        tree_path.parent / "formal_inputs" / "code_taskset_canonical.json",
        tree_path.parent.parent / "formal_inputs" / "code_taskset_canonical.json",
        tree_path.parent / "taskset.json",
        tree_path.parent.parent / "taskset.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            items = raw.get("ordered_tasks", raw.get("tasks", []))
        else:
            items = raw
        if not isinstance(items, list):
            continue
        result: list[Mapping[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            normalized = dict(item)
            normalized.setdefault("priority", normalized.get("priority_index", index))
            normalized.setdefault("task_id", normalized.get("id", normalized.get("name", index)))
            result.append(normalized)
        return result
    return []


def _infer_seed(path: Path) -> int:
    for part in reversed(path.parts):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return -1


def _infer_variant(path: Path) -> str:
    aliases = {
        "compact": "best_overall",
        "balanced": "best_balanced",
        "performance": "best_performance",
    }
    for part in reversed(path.parts):
        if part in {"best_overall", "best_balanced", "best_performance"}:
            return part
        if part in aliases:
            return aliases[part]
    return path.parent.name


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
