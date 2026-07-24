"""Backend boundary for the parameterized CloseAt idle-jump theorem.

The current repository has the theorem statement and finite diagnostic helpers,
but no source-bound relational proof kernel.  This backend therefore validates
the claimed scope only for diagnostics and remains fail-closed.  A future
backend may return PASS only after checking a receipt emitted by the executable
transition/SMT kernel itself.
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
    structurally_complete = (
        receipt.get("theorem_id")
            == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and receipt.get("proof_scope") == "ALL_LEGAL_CLOSED_IDLE_JUMPS"
        and all(receipt.get(field) is True for field in required_true)
    )
    payload = {
        "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "status": "UNRESOLVED",
        "code": "IDLE_JUMP_EXECUTABLE_PROOF_KERNEL_NOT_IMPLEMENTED",
        "required_fields": list(required_true),
        "structurally_complete_claim": structurally_complete,
        "source_receipt_hash": receipt.get("receipt_hash"),
        "reason": (
            "Self-declared Boolean fields and finite traces do not prove the "
            "parameterized CloseAt idle-jump frame theorem.  PASS requires a "
            "receipt produced and independently checked by the executable "
            "transition/SMT kernel."
        ),
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
