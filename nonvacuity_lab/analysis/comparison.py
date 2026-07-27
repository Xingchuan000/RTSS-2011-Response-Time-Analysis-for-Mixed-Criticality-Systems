from __future__ import annotations

from typing import Any, Mapping


PROOF_FIELDS = (
    "workflow_status",
    "result_status",
    "failure_route",
    "failure_code",
    "violated_obligation_id",
    "outer_bundle_root",
)


def compare_proofs(
    base: Mapping[str, Any] | None,
    mutated: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "proof_comparison_v1",
        "fields": {
            field: {
                "base": (base or {}).get(field),
                "mutated": (mutated or {}).get(field),
                "changed": (base or {}).get(field) != (mutated or {}).get(field),
            }
            for field in PROOF_FIELDS
        },
    }


def compare_metrics(
    base: Mapping[str, Any],
    mutated: Mapping[str, Any],
    *,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    rows = {}
    for metric in metrics:
        left, right = base.get(metric), mutated.get(metric)
        delta = None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = right - left
        rows[metric] = {"base": left, "mutated": right, "delta": delta}
    return {"schema_version": "hout_comparison_v1", "metrics": rows}
