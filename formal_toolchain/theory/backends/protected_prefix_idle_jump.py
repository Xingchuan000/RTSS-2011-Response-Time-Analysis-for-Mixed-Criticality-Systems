"""Backend for the parameterized CloseAt idle-jump theorem.

A receipt is accepted only when it records a source-bound proof over all legal
idle jumps.  Boolean summaries from finite traces are deliberately insufficient.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def verify_idle_jump_expansion_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required_true = (
        "parameterized",
        "source_bound_transition_relation",
        "all_integer_times_observable",
        "time_indexed_closed_observation_defined",
        "protected_observable_stutters_on_expanded_idle_ticks",
        "independent_of_complete_execution_witness",
    )
    ok = (
        receipt.get("theorem_id") == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and receipt.get("status") == "PASS"
        and receipt.get("proof_scope") == "ALL_LEGAL_CLOSED_IDLE_JUMPS"
        and all(receipt.get(field) is True for field in required_true)
        and isinstance(receipt.get("receipt_hash"), str)
    )
    payload = {
        "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "status": "PASS" if ok else "UNRESOLVED",
        "code": None if ok else "IDLE_JUMP_SOURCE_BOUND_THEOREM_NOT_VERIFIED",
        "required_fields": list(required_true),
        "source_receipt_hash": receipt.get("receipt_hash"),
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
