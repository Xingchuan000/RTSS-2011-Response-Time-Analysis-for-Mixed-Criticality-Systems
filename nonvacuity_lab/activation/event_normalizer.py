"""Normalize runtime/HOUT records to the lab's canonical event vocabulary."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_hout_event(event: Mapping[str, Any]) -> dict[str, Any]:
    raw_action = _first(event, "raw_top1_action_id", "raw_action_id")
    raw_invalid = bool(
        _first(
            event,
            "raw_top1_invalid",
            "tree_raw_top1_invalid",
            "baseline_reject",
            default=False,
        )
    )
    selected = _first(event, "selected_action_id", "tree_selected_action_id")
    return {
        "scenario_id": _first(event, "scenario_id", "scenario", default=0),
        "timestamp": _first(event, "timestamp", "time", "step", default=0),
        "leaf_id": _first(event, "leaf_id", "tree_leaf_id"),
        "raw_top1_action_id": raw_action,
        "raw_top1_valid": bool(
            _first(event, "raw_top1_valid", default=not raw_invalid)
        ),
        "raw_top1_invalid": raw_invalid,
        "selected_action_id": selected,
        "selected_rank": _first(event, "selected_rank", "tree_selected_rank"),
        "implicit_noop": bool(_first(event, "implicit_noop", default=False)),
        "all_invalid": bool(_first(event, "all_invalid", default=False)),
        "reject_reason": _first(event, "reject_reason", "tree_reject_reason"),
        "target_task": _first(event, "target_task"),
        "budget_before": _first(event, "budget_before"),
        "budget_after": _first(event, "budget_after"),
        "demand_trace_fingerprint": _first(event, "demand_trace_fingerprint"),
        "selected_after_mutation": bool(
            _first(event, "selected_after_mutation", default=False)
        ),
    }


def _first(event: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in event and event[key] is not None:
            return event[key]
    return default
