"""Fresh verifier backend for the CloseAt idle-jump expansion theorem."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def verify_idle_jump_expansion_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "all_integer_times_observable",
        "time_indexed_closed_observation_defined",
        "protected_observable_stutters_on_expanded_idle_ticks",
    )
    ok = (
        receipt.get("theorem_id") == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and receipt.get("status") == "PASS"
        and all(receipt.get(field) is True for field in required)
    )
    payload = {
        "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "status": "PASS" if ok else "UNRESOLVED",
        "required_fields": list(required),
        "source_receipt_hash": receipt.get("receipt_hash"),
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
