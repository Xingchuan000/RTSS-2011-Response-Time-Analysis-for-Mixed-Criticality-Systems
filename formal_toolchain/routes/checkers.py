"""Fail-closed route checker helpers used by the plugin catalog."""

from __future__ import annotations

from typing import Any, Mapping


def unresolved_derived_checker(obligation_id: str, *, expected_predecessors: tuple[str, ...]):
    expected = tuple(expected_predecessors)
    if len(expected) != len(set(expected)):
        raise ValueError(f"DUPLICATE_EXPECTED_PREDECESSOR:{obligation_id}")

    def check(*, predecessors: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        predecessors = predecessors or {}
        expected_set = set(expected)
        actual_set = {str(item) for item in predecessors}
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "PREDECESSOR_SET_MISMATCH",
                    "witness": {"obligation_id": obligation_id,
                                "missing": missing, "extra": extra}}
        if any(item.get("status", item.get("obligation_status")) != "PASS"
               for item in predecessors.values() if isinstance(item, Mapping)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "PREDECESSOR_NOT_PASS", "witness": {"obligation_id": obligation_id}}
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": f"{obligation_id}_NOT_IMPLEMENTED",
                "witness": {"obligation_id": obligation_id, "derived": False}}
    return check
