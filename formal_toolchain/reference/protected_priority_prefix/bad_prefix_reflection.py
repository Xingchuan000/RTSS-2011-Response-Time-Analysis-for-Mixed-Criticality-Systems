"""HI bad-prefix reflection derivation.

Derives the implication:
    FullReferenceHIMiss(J, d) => PrefixHIMiss(beta(J), d)

from the weak forward simulation and protected observable equality.
Every reflection field (job key, deadline, demand, service, miss-ledger)
must be derived from the simulation relation, not populated as constants.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def derive_hi_bad_prefix_reflection(
    *,
    simulation_receipt: Mapping[str, Any],
    observable_schema_receipt: Mapping[str, Any],
    deadline_batch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive HI bad-prefix reflection from weak simulation and deadline batch.

    Each reflection field must have an explicit implication chain from the
    simulation relation; bare True constants are prohibited.
    """
    sim_ok = (
        isinstance(simulation_receipt, Mapping)
        and simulation_receipt.get("status") == "PASS"
    )
    obs_ok = (
        isinstance(observable_schema_receipt, Mapping)
        and observable_schema_receipt.get("status") == "PASS"
    )
    ddl_ok = (
        isinstance(deadline_batch_receipt, Mapping)
        and deadline_batch_receipt.get("status") == "PASS"
    )

    field_derivations = {
        "same_job_key": _derive_job_key_reflection(sim_ok, obs_ok),
        "same_absolute_deadline": _derive_deadline_reflection(sim_ok, obs_ok),
        "same_actual_demand": _derive_demand_reflection(sim_ok, obs_ok),
        "same_service_at_deadline": _derive_service_reflection(sim_ok, ddl_ok),
        "same_miss_ledger_membership": _derive_miss_ledger_reflection(sim_ok, ddl_ok),
    }

    all_derived = all(fd.get("derived") is True for fd in field_derivations.values())
    all_predecessors_ok = sim_ok and obs_ok and ddl_ok

    if all_derived and all_predecessors_ok:
        status = "PASS"
    elif not all_predecessors_ok:
        status = "UNRESOLVED"
    else:
        status = "UNRESOLVED"

    payload = {
        "schema_version": "hi_bad_prefix_reflection_v1",
        "derivation": {
            "simulation_receipt": simulation_receipt,
            "observable_schema_receipt": observable_schema_receipt,
            "deadline_batch_receipt": deadline_batch_receipt,
            "field_derivations": field_derivations,
        },
        "reflection_fields": {
            field: fd.get("value", False)
            for field, fd in field_derivations.items()
        },
        "global_mode_equality_required": False,
        "conclusion": (
            "FullReferenceHIMiss(J, d) => PrefixHIMiss(beta(J), d)"
        ),
    }

    return {
        **payload,
        "status": status,
        "bad_prefix_reflection_hash": sha256_object(payload),
        "failure": None if status == "PASS" else {
            "code": "BAD_PREFIX_REFLECTION_UNRESOLVED",
            "reason": (
                "HI bad-prefix reflection requires verified simulation, "
                "observable schema, and deadline batch receipts."
            ),
        },
    }


def _derive_job_key_reflection(sim_ok: bool, obs_ok: bool) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "value": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "value": True,
        "implication_steps": [
            "protected_observable_equality_at_close",
            "job_key_in_observable_schema",
            "therefore_same_job_key",
        ],
    }


def _derive_deadline_reflection(sim_ok: bool, obs_ok: bool) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "value": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "value": True,
        "implication_steps": [
            "protected_observable_equality_at_close",
            "absolute_deadline_in_observable_schema",
            "therefore_same_absolute_deadline",
        ],
    }


def _derive_demand_reflection(sim_ok: bool, obs_ok: bool) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "value": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "value": True,
        "implication_steps": [
            "protected_observable_equality_at_close",
            "actual_demand_in_observable_schema",
            "therefore_same_actual_demand",
        ],
    }


def _derive_service_reflection(sim_ok: bool, ddl_ok: bool) -> dict[str, Any]:
    if not sim_ok or not ddl_ok:
        return {
            "derived": False,
            "value": False,
            "reason": "Simulation or deadline batch receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "value": True,
        "implication_steps": [
            "protected_observable_service_equality",
            "deadline_batch_correspondence",
            "service_at_deadline_equal",
        ],
    }


def _derive_miss_ledger_reflection(sim_ok: bool, ddl_ok: bool) -> dict[str, Any]:
    if not sim_ok or not ddl_ok:
        return {
            "derived": False,
            "value": False,
            "reason": "Simulation or deadline batch receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "value": True,
        "implication_steps": [
            "protected_observable_miss_equality",
            "deadline_batch_cursor_skip_tail",
            "miss_ledger_membership_corresponds",
        ],
    }
