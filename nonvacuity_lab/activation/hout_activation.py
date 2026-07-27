"""Activation extraction from paired replay event JSON/JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..schema import ActivationStatus
from .event_normalizer import normalize_hout_event
from .schema import ActivationResult


def evaluate_hout_activation(
    *,
    mutation_id: str,
    base_events: Iterable[Mapping[str, Any]],
    mutated_events: Iterable[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> ActivationResult:
    base = [normalize_hout_event(row) for row in base_events]
    mutated = [normalize_hout_event(row) for row in mutated_events]
    leaf_id = _optional_int(rule.get("required_leaf_id"))
    action_id = _optional_int(rule.get("required_action_id"))
    base_target = [row for row in base if _matches(row, leaf_id, action_id)]
    mutated_target = [row for row in mutated if _matches(row, leaf_id, action_id)]
    baseline_rejects = [
        row
        for row in base_target
        if bool(row.get("raw_top1_invalid", row.get("baseline_reject", False)))
    ]
    selected_after = [
        row
        for row in mutated_target
        if (
            row.get("selected_action_id") is not None
            and action_id is not None
            and int(row["selected_action_id"]) == action_id
        )
        or bool(row.get("selected_after_mutation", False))
    ]
    all_invalid = [
        row
        for row in base_target + mutated_target
        if bool(row.get("all_invalid", False))
    ]
    require_reject = bool(rule.get("require_baseline_reject", False))
    require_selected = bool(rule.get("require_selected_after_mutation", False))
    require_all_invalid = bool(rule.get("require_all_invalid", False))
    activated = bool(base_target or mutated_target)
    if require_reject:
        activated = activated and bool(baseline_rejects)
    if require_selected:
        activated = activated and bool(selected_after)
    if require_all_invalid:
        activated = activated and bool(all_invalid)
    first_intervention = min(
        (
            row.get("time", row.get("timestamp", row.get("step")))
            for row in mutated_target
            if row.get("selected_after_mutation") or row.get("selected_action_id") == action_id
        ),
        default=None,
    )
    return ActivationResult(
        mutation_id=mutation_id,
        status=ActivationStatus.ACTIVATED if activated else ActivationStatus.NOT_ACTIVATED,
        evidence_modes=("HOUT",),
        leaf_id=leaf_id,
        action_id=action_id,
        hout_hit_count=len(base_target),
        baseline_reject_count=len(baseline_rejects),
        selected_after_mutation_count=len(selected_after),
        all_invalid_count=len(all_invalid),
        details={
            "first_intervention_time": first_intervention,
            "reject_reasons": _histogram(row.get("reject_reason") for row in baseline_rejects),
            "base_event_count": len(base),
            "mutated_event_count": len(mutated),
        },
    )


def load_events(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("events", raw.get("records", []))
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(f"HOUT event 文件必须为 object array/JSONL: {path}")
    return [dict(item) for item in raw]


def _matches(row: Mapping[str, Any], leaf_id: int | None, action_id: int | None) -> bool:
    leaf_value = row.get("leaf_id")
    raw_action = row.get("raw_top1_action_id")
    leaf_match = leaf_id is None or (
        leaf_value is not None and int(leaf_value) == leaf_id
    )
    action_match = action_id is None or (
        raw_action is not None and int(raw_action) == action_id
    )
    return leaf_match and action_match


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
