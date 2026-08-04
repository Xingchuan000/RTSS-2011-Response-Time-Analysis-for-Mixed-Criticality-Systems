from __future__ import annotations

from collections import Counter

from .normalizer import NormalizedDecisionEvent


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def aggregate_hout_events(events: list[NormalizedDecisionEvent]) -> dict:
    count = len(events)
    fallback = [e for e in events if (e.selected_rank or 0) > 1]
    noops = [e for e in events if e.implicit_noop]
    invalid = [e for e in events if e.raw_top1_valid is False]
    return {
        "decision_count": count,
        "fallback_count": len(fallback), "fallback_rate": _ratio(len(fallback), count),
        "implicit_noop_count": len(noops), "implicit_noop_rate": _ratio(len(noops), count),
        "all_invalid_count": sum(e.all_invalid for e in events),
        "raw_top1_invalid_count": len(invalid), "raw_top1_invalid_rate": _ratio(len(invalid), count),
        "selected_rank_mean": _mean(e.selected_rank for e in events if e.selected_rank is not None),
        "selected_rank_max": max((e.selected_rank for e in events if e.selected_rank is not None), default=None),
        "reject_reason_counts": dict(Counter(e.reject_reason for e in events if e.reject_reason)),
        "leaf_counts": dict(Counter(e.leaf_id for e in events if e.leaf_id is not None)),
        "hi_miss_count": sum(e.hi_miss for e in events), "lo_miss_count": sum(e.lo_miss for e in events),
        "lo_qos_mean": _mean(e.lo_qos for e in events if e.lo_qos is not None),
        "retention_mean": _mean(e.retention for e in events if e.retention is not None),
    }


def compare_paired_hout(base_events, mutated_events, profile=None) -> dict:
    base_keys = {(e.scenario_seed, e.controller_decision_index) for e in base_events}
    mutated_keys = {(e.scenario_seed, e.controller_decision_index) for e in mutated_events}
    if base_keys != mutated_keys:
        raise ValueError("paired HOUT decision keys differ")
    if profile is not None:
        required = set(profile.required_scenarios)
        present = {e.scenario_seed for e in base_events} & {e.scenario_seed for e in mutated_events}
        if not required <= present:
            raise ValueError("required scenario missing from paired outputs")
    base, mutated = aggregate_hout_events(base_events), aggregate_hout_events(mutated_events)
    def delta(key):
        if base[key] is None or mutated[key] is None:
            return None
        return mutated[key] - base[key]
    return {"base": base, "mutated": mutated, "delta": {
        "fallback_rate": delta("fallback_rate"), "implicit_noop_rate": delta("implicit_noop_rate"),
        "lo_qos_mean": delta("lo_qos_mean"), "retention_mean": delta("retention_mean"),
        "hi_miss_count": mutated["hi_miss_count"] - base["hi_miss_count"],
    }}
