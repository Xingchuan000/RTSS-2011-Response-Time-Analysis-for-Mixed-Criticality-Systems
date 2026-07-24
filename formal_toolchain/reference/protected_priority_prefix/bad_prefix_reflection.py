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


def _field_equal_in_receipt(
    receipt: Mapping[str, Any] | None,
    field: str,
) -> bool:
    """Read a concrete or theorem-level protected-field equality receipt.

    ``check_phase_relation`` indexes per-job checks as
    ``job[(task, q)].field``.  The previous implementation looked for the
    nonexistent key ``job[field]`` and therefore made every bad-prefix field
    derivation fail even after a valid simulation receipt was supplied.
    """
    if not isinstance(receipt, Mapping):
        return False
    equality_checks = (
        receipt.get("equality_checks")
        or receipt.get("witness", {}).get("equality_checks")
        or {}
    )
    if not isinstance(equality_checks, Mapping):
        return False

    if field in {
        "time", "running_job_key", "miss_job_keys",
        "pending_job_key_set", "job_key_set",
    }:
        return equality_checks.get(field) is True

    # A theorem-level receipt may export field preservation directly.
    preserved = receipt.get("preserved_job_fields") or receipt.get("witness", {}).get(
        "preserved_job_fields"
    )
    if isinstance(preserved, (list, tuple, set, frozenset)) and field in preserved:
        return True

    suffix = f"].{field}"
    matches = [
        value for key, value in equality_checks.items()
        if isinstance(key, str) and key.startswith("job[") and key.endswith(suffix)
    ]
    if field == "job_key":
        return equality_checks.get("job_key_set") is True and bool(matches) and all(
            value is True for value in matches
        )
    return bool(matches) and all(value is True for value in matches)


def _derive_job_key_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    obs = isinstance(obs_receipt, Mapping) and obs_receipt.get("status") == "PASS"
    job_key_eq = _field_equal_in_receipt(sim_receipt, "job_key")
    derived = sim and obs and job_key_eq
    return {
        "derived": derived,
        "source": "rel_pp_close_job_key_equality" if derived else "unresolved",
        "implication_steps": [
            "weak simulation establishes Rel_pp_close at the HI deadline boundary",
            "Rel_pp_close preserves the protected job-key set and each job key",
            "the full HI job therefore has the same prefix job key beta(J)",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "observable_schema_pass": obs,
            "simulation_job_key_equality": job_key_eq,
        },
    }


def _derive_deadline_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    obs = isinstance(obs_receipt, Mapping) and obs_receipt.get("status") == "PASS"
    deadline_eq = _field_equal_in_receipt(sim_receipt, "absolute_deadline")
    derived = sim and obs and deadline_eq
    return {
        "derived": derived,
        "source": "rel_pp_close_deadline_equality" if derived else "unresolved",
        "implication_steps": [
            "weak simulation establishes Rel_pp_close for the corresponding job",
            "the protected observable preserves absolute_deadline",
            "both deadline observations refer to the same absolute time",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "observable_schema_pass": obs,
            "simulation_deadline_equality": deadline_eq,
        },
    }


def _derive_demand_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    obs = isinstance(obs_receipt, Mapping) and obs_receipt.get("status") == "PASS"
    demand_eq = _field_equal_in_receipt(sim_receipt, "actual_demand")
    derived = sim and obs and demand_eq
    return {
        "derived": derived,
        "source": "rel_pp_close_demand_equality" if derived else "unresolved",
        "implication_steps": [
            "the projected release stream copies the protected actual demand",
            "Rel_pp_close preserves release-fixed actual_demand",
            "the corresponding jobs have equal fixed demand",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "observable_schema_pass": obs,
            "simulation_demand_equality": demand_eq,
        },
    }


def _derive_service_reflection(
    sim_ok: bool, ddl_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    ddl_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    ddl = isinstance(ddl_receipt, Mapping) and ddl_receipt.get("status") == "PASS"
    service_eq = _field_equal_in_receipt(sim_receipt, "executed_service")
    derived = sim and ddl and service_eq
    return {
        "derived": derived,
        "source": "rel_pp_close_service_equality_and_ddl_fold" if derived else "unresolved",
        "implication_steps": [
            "weak simulation preserves executed_service at closed boundaries",
            "deadline-batch correspondence aligns the protected deadline observation",
            "service at the common absolute deadline is equal",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "deadline_batch_pass": ddl,
            "simulation_service_equality": service_eq,
        },
    }


def _derive_completion_reflection(
    sim_ok: bool, obs_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    obs_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    obs = isinstance(obs_receipt, Mapping) and obs_receipt.get("status") == "PASS"
    completed_eq = _field_equal_in_receipt(sim_receipt, "completed")
    derived = sim and obs and completed_eq
    return {
        "derived": derived,
        "source": "rel_pp_close_completion_equality" if derived else "unresolved",
        "implication_steps": [
            "completion/removal correspondence preserves the terminal record",
            "Rel_pp_close preserves completed for the corresponding job",
            "completion state at the deadline boundary is equal",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "observable_schema_pass": obs,
            "simulation_completed_equality": completed_eq,
        },
    }


def _derive_miss_ledger_reflection(
    sim_ok: bool, ddl_ok: bool,
    sim_receipt: Mapping[str, Any] | None = None,
    ddl_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim = isinstance(sim_receipt, Mapping) and sim_receipt.get("status") == "PASS"
    ddl = isinstance(ddl_receipt, Mapping) and ddl_receipt.get("status") == "PASS"
    miss_eq = _field_equal_in_receipt(sim_receipt, "missed")
    ddl_miss = _field_equal_in_receipt(ddl_receipt, "miss_job_keys")
    derived = sim and ddl and (miss_eq or ddl_miss)
    return {
        "derived": derived,
        "source": "rel_pp_close_miss_ledger_equality_or_ddl_fold" if derived else "unresolved",
        "implication_steps": [
            "deadline-batch correspondence applies the same protected miss observation",
            "Rel_pp_close preserves missed/miss_job_keys",
            "miss-ledger membership for the corresponding HI job is equal",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim,
            "deadline_batch_pass": ddl,
            "simulation_missed_equality": miss_eq,
            "deadline_batch_miss_ledger_equality": ddl_miss,
        },
    }


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
    all_provenance_ok = all(
        isinstance(fd.get("provenance"), Mapping) and any(fd["provenance"].values())
        for fd in field_derivations.values()
    )
    all_predecessors_ok = sim_ok and obs_ok and ddl_ok
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"
        and proof_kernel_receipt.get("all_reflection_fields_derived") is True
    )
    status = "PASS" if (all_derived and all_provenance_ok and all_predecessors_ok and kernel_ok) else "UNRESOLVED"

    payload = {
        "schema_version": "hi_bad_prefix_reflection_v3",
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
        "conclusion": "FullReferenceHIMiss(J, d) => PrefixHIMiss(beta(J), d)",
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
