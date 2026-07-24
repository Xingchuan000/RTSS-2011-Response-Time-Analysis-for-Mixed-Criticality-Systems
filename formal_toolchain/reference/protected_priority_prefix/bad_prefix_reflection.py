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
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive HI bad-prefix reflection from weak simulation and deadline batch.

    Section 10.2 requirements:
    Each reflection field must be derived from the simulation relation:
    - same_job_key: from simulation relation job_key equality
    - same_absolute_deadline: from simulation relation deadline equality
    - same_actual_demand: from simulation relation demand equality
    - same_service_at_deadline: from deadline batch correspondence
    - same_completion_state: from simulation relation completion equality
    - same_miss_ledger_membership: from deadline batch and miss-ledger equality

    Bare True constants are prohibited.  Every field must have an explicit
    implication chain rooted in a PASSed simulation/observable/deadline receipt.
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
        "same_job_key": _derive_job_key_reflection(sim_ok, obs_ok, simulation_receipt, observable_schema_receipt),
        "same_absolute_deadline": _derive_deadline_reflection(sim_ok, obs_ok, simulation_receipt, observable_schema_receipt),
        "same_actual_demand": _derive_demand_reflection(sim_ok, obs_ok, simulation_receipt, observable_schema_receipt),
        "same_service_at_deadline": _derive_service_reflection(sim_ok, ddl_ok, simulation_receipt, deadline_batch_receipt),
        "same_completion_state": _derive_completion_reflection(sim_ok, obs_ok, simulation_receipt, observable_schema_receipt),
        "same_miss_ledger_membership": _derive_miss_ledger_reflection(sim_ok, ddl_ok, simulation_receipt, deadline_batch_receipt),
    }

    all_derived = all(fd.get("derived") is True for fd in field_derivations.values())
    all_predecessors_ok = sim_ok and obs_ok and ddl_ok
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"
        and proof_kernel_receipt.get("all_reflection_fields_derived") is True
    )
    # Narrative implication steps are explanatory evidence, not a proof kernel.
    status = "PASS" if (all_derived and all_predecessors_ok and kernel_ok) else "UNRESOLVED"

    payload = {
        "schema_version": "hi_bad_prefix_reflection_v2",
        "derivation": {
            "simulation_receipt_id": simulation_receipt.get("obligation_id") if sim_ok else None,
            "observable_schema_receipt_id": observable_schema_receipt.get("obligation_id") if obs_ok else None,
            "deadline_batch_receipt_id": deadline_batch_receipt.get("obligation_id") if ddl_ok else None,
            "proof_kernel_receipt": proof_kernel_receipt,
            "field_derivations": field_derivations,
        },
        "reflection_fields": {
            field: fd.get("derived", False)
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
                "HI bad-prefix reflection requires verified simulation, observable "
                "schema, deadline-batch correspondence, and a proof-kernel receipt; "
                "narrative implication strings cannot establish the theorem."
            ),
        },
    }


def _derive_job_key_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_equality_at_close (from simulation relation)",
            "job_key_preserved_in_observable_schema",
            "full_job_key_beta_mapped_to_prefix_job_key",
            "therefore_same_job_key",
        ],
        "requires_implications": [
            "RelPP_Close => same protected job keys",
            "Observable schema includes job_key",
            "beta: FullJobKey -> PrefixJobKey is identity on protected keys",
        ],
    }


def _derive_deadline_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_equality_at_close (from simulation relation)",
            "absolute_deadline_preserved_in_observable_schema",
            "same protected job => same absolute_deadline",
            "therefore_same_absolute_deadline",
        ],
        "requires_implications": [
            "RelPP_Close => same absolute_deadline for same job_key",
            "Observable schema includes absolute_deadline",
        ],
    }


def _derive_demand_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_equality_at_close (from simulation relation)",
            "actual_demand_preserved_in_observable_schema",
            "same protected job => same actual_demand",
            "therefore_same_actual_demand",
        ],
        "requires_implications": [
            "RelPP_Close => same actual_demand for same job_key",
            "Observable schema includes actual_demand",
        ],
    }


def _derive_service_reflection(
    sim_ok: bool, ddl_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    ddl_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not ddl_ok:
        return {
            "derived": False,
            "reason": "Simulation or deadline batch receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_service_equality (from simulation relation)",
            "deadline_batch_correspondence (DDL observe at same time)",
            "same_service_at_deadline_observation_point",
            "therefore_service_at_deadline_equal",
        ],
        "requires_implications": [
            "RelPP_Close => same executed_service for same job_key",
            "Deadline batch cursor => same DDL observation time",
        ],
    }


def _derive_completion_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not obs_ok:
        return {
            "derived": False,
            "reason": "Simulation or observable schema receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_equality_at_close (from simulation relation)",
            "completion_state_preserved_in_observable_schema",
            "full_job_completed => prefix_job_completed",
            "therefore_same_completion_state",
        ],
        "requires_implications": [
            "RelPP_Close => same completed flag",
            "Observable schema includes completed field",
        ],
    }


def _derive_miss_ledger_reflection(
    sim_ok: bool, ddl_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    ddl_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sim_ok or not ddl_ok:
        return {
            "derived": False,
            "reason": "Simulation or deadline batch receipt not PASS.",
            "implication_steps": [],
        }
    return {
        "derived": True,
        "implication_steps": [
            "protected_observable_miss_equality (from simulation relation)",
            "deadline_batch_cursor_skip_tail (protected DDL observe is no-op or miss-append)",
            "same_miss_ledger_membership_at_corresponding_deadlines",
            "therefore_miss_ledger_membership_corresponds",
        ],
        "requires_implications": [
            "RelPP_Close => same miss_ledger for protected jobs",
            "Deadline batch cursor => same DDL observation and miss decision",
        ],
    }
