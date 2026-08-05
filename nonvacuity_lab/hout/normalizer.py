from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class HoutSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedDecisionEvent:
    time: int
    scenario_seed: int
    controller_decision_index: int
    leaf_id: int | None = None
    raw_top1_action_id: int | None = None
    raw_top1_valid: bool | None = None
    rejected_action_id: int | None = None
    selected_action_id: int | None = None
    selected_rank: int | None = None
    all_invalid: bool = False
    implicit_noop: bool = False
    reject_reason: str | None = None
    budget_before: dict[str, int] | None = None
    candidate_budget: dict[str, int] | None = None
    budget_after: dict[str, int] | None = None
    hi_miss: bool = False
    lo_miss: bool = False
    lo_qos: float | None = None
    zero_service: bool | None = None
    cancellation_count: int | None = None
    retention: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIELD_ALIASES = {
    "scenario": "scenario_seed", "raw_action": "raw_top1_action_id",
    "action": "selected_action_id", "fallback_rank": "selected_rank",
    "noop": "implicit_noop",
}


def _optional_int(value):
    return None if value is None else int(value)


def _optional_bool(value):
    return None if value is None else bool(value)


def _optional_float(value):
    return None if value is None else float(value)


def _budget(value):
    return {str(k): int(v) for k, v in (value or {}).items()}


def normalize_event(raw: dict[str, Any]) -> NormalizedDecisionEvent:
    if not isinstance(raw, dict):
        raise HoutSchemaError("event must be an object")
    canonical = dict(raw)
    for old, new in FIELD_ALIASES.items():
        if new not in canonical and old in canonical:
            canonical[new] = canonical[old]
    required = {"time", "scenario_seed", "all_invalid", "implicit_noop"}
    missing = required - canonical.keys()
    if missing:
        raise HoutSchemaError(f"missing fields: {sorted(missing)}")
    return NormalizedDecisionEvent(
        time=int(canonical["time"]), scenario_seed=int(canonical["scenario_seed"]),
        controller_decision_index=int(canonical.get("controller_decision_index", 0)),
        leaf_id=_optional_int(canonical.get("leaf_id")),
        raw_top1_action_id=_optional_int(canonical.get("raw_top1_action_id")),
        raw_top1_valid=_optional_bool(canonical.get("raw_top1_valid")),
        rejected_action_id=_optional_int(canonical.get("rejected_action_id")),
        selected_action_id=_optional_int(canonical.get("selected_action_id")),
        selected_rank=_optional_int(canonical.get("selected_rank")),
        all_invalid=bool(canonical["all_invalid"]),
        implicit_noop=bool(canonical["implicit_noop"]),
        reject_reason=canonical.get("reject_reason"),
        budget_before=_budget(canonical.get("budget_before")),
        candidate_budget=_budget(canonical.get("candidate_budget")),
        budget_after=_budget(canonical.get("budget_after")),
        hi_miss=bool(canonical.get("hi_miss", False)), lo_miss=bool(canonical.get("lo_miss", False)),
        lo_qos=_optional_float(canonical.get("lo_qos")), zero_service=_optional_bool(canonical.get("zero_service")),
        cancellation_count=_optional_int(canonical.get("cancellation_count")), retention=_optional_float(canonical.get("retention")),
    )


def load_events(path):
    import json
    events = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                events.append(normalize_event(json.loads(line)))
    return events
