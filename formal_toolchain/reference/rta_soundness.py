"""Single-source soundness receipt for independently replayed all-task RTA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_object


def derive_all_task_rta_soundness(*, replay: Mapping[str, Any], taskset: Any,
                                  theorem_id: str) -> dict[str, Any]:
    """Validate the complete all-task replay and bind it to one taskset.

    The fresh verifier and route checker both consume this helper so the RTA
    arithmetic obligation cannot PASS while the later mathematical-conformance
    node looks for a differently shaped receipt.
    """

    witness = replay.get("witness", replay.get("replay", {}).get("witness", {}))
    expected_names = [str(task.name) for task in taskset.tasks]
    expected_fingerprint = taskset.to_dict()["fingerprint"]
    actual_names = list(witness.get("task_order", ())) if isinstance(witness, Mapping) else []
    formula_version = witness.get("schema_version") if isinstance(witness, Mapping) else None
    rows = witness.get("tasks", ()) if isinstance(witness, Mapping) else ()
    rows_complete = (
        isinstance(rows, list)
        and [str(row.get("task", {}).get("name")) for row in rows] == expected_names
        and all(isinstance(row, Mapping) and row.get("status") == "PASS" for row in rows)
    )
    ok = (
        replay.get("status") == "PASS"
        and isinstance(witness, Mapping)
        and witness.get("reference_taskset_fingerprint") == expected_fingerprint
        and formula_version == "all_task_rta_v3"
        and actual_names == expected_names
        and witness.get("all_tasks_covered") is True
        and witness.get("all_deadlines_met") is True
        and witness.get("complete_integer_candidate_domains") is True
        and rows_complete
    )
    receipt = {
        "theorem_id": str(theorem_id),
        "status": "PASS" if ok else "UNRESOLVED",
        "prefix_taskset_fingerprint": expected_fingerprint,
        "formula_version": formula_version,
        "all_task_name_set": expected_names,
        "all_tasks_covered": bool(
            isinstance(witness, Mapping) and witness.get("all_tasks_covered") is True
        ),
        "complete_integer_candidate_domains": bool(
            isinstance(witness, Mapping)
            and witness.get("complete_integer_candidate_domains") is True
        ),
        "rows_complete": bool(rows_complete),
    }
    receipt["receipt_hash"] = sha256_object(receipt)
    return {
        "status": "PASS" if ok else "UNRESOLVED",
        "witness": dict(witness) if isinstance(witness, Mapping) else {},
        "soundness_receipt": receipt,
    }


__all__ = ["derive_all_task_rta_soundness"]
