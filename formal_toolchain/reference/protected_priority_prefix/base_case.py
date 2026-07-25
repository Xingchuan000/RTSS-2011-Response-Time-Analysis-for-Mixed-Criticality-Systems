"""Executable base relation for the protected-prefix simulation."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def _empty(value: Any) -> bool:
    return value in (None, (), [], {}, frozenset(), set())


def _receipt_pass(receipt: Mapping[str, Any] | None) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get("status", receipt.get("obligation_status")) == "PASS":
        return True
    witness = receipt.get("witness")
    return isinstance(witness, Mapping) and witness.get("status") == "PASS"


def prove_standard_initial_relation(
    full_initial_state: Any,
    prefix_initial_state: Any,
    protected_partition_receipt: Mapping[str, Any] | None,
    observable_schema_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Construct the t=0 relation from actual initial states and predecessors."""
    partition_ok = _receipt_pass(protected_partition_receipt)
    schema_ok = _receipt_pass(observable_schema_receipt)
    states_present = full_initial_state is not None and prefix_initial_state is not None

    def attr(state: Any, name: str, default: Any = None) -> Any:
        if isinstance(state, Mapping):
            return state.get(name, default)
        return getattr(state, name, default)

    same_time_zero = states_present and all(attr(state, "time") == 0 for state in (full_initial_state, prefix_initial_state))
    same_lo_mode = states_present and all(str(attr(state, "mode")) in {"LO", "Mode.LO"} for state in (full_initial_state, prefix_initial_state))
    empty_observables = states_present and all(
        _empty(attr(state, field))
        for state in (full_initial_state, prefix_initial_state)
        for field in (
            "jobs", "released", "terminal", "misses",
            "pending_releases", "ready_order",
        )
    )
    no_running = states_present and all(
        attr(state, "running", attr(state, "running_job_key")) is None
        for state in (full_initial_state, prefix_initial_state)
    )
    proved = all((partition_ok, schema_ok, same_time_zero, same_lo_mode, empty_observables, no_running))
    receipt = {
        "theorem_id": "PROTECTED_PREFIX_INITIAL_RELATION",
        "status": "PASS" if proved else "UNRESOLVED",
        "base_relation_proved": proved,
        "both_time_zero": same_time_zero,
        "both_mode_lo": same_lo_mode,
        "protected_job_key_sets_empty": empty_observables,
        "ready_running_empty": no_running,
        "relation": "Rel_pp_close(full_init, prefix_init)",
        "partition_receipt_hash": (
            protected_partition_receipt.get("receipt_hash")
            if isinstance(protected_partition_receipt, Mapping) else None
        ),
        "observable_schema_receipt_hash": (
            observable_schema_receipt.get("receipt_hash")
            if isinstance(observable_schema_receipt, Mapping) else None
        ),
    }
    receipt["receipt_hash"] = sha256_object(receipt)
    return receipt
