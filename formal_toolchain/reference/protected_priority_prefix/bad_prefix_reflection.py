"""HI bad-prefix reflection derivation.

Derives the implication:
    FullReferenceHIMiss(J, d) => PrefixHIMiss(beta(J), d)

from the weak forward simulation and protected observable equality.
Every reflection field (job key, criticality, release time, deadline, demand,
service, completion state, miss-ledger)
must be derived from the simulation relation, not populated as constants.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


REQUIRED_REFLECTION_FIELDS = frozenset({
    "job_key", "criticality", "release_time", "absolute_deadline",
    "actual_demand", "service", "completion_state", "miss_ledger",
})

_RECEIPT_FIELD_ALIASES = {
    "job_key": ("job_key", "job_key_set"),
    "criticality": ("criticality",),
    "release_time": ("release_time",),
    "absolute_deadline": ("absolute_deadline",),
    "actual_demand": ("actual_demand",),
    "service": ("service", "executed_service"),
    "completion_state": ("completion_state", "completed"),
    "miss_ledger": ("miss_ledger", "missed", "miss_job_keys"),
}


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

    aliases = _RECEIPT_FIELD_ALIASES.get(field, (field,))
    if any(equality_checks.get(alias) is True for alias in aliases):
        return True

    # A theorem-level receipt may export field preservation directly.
    preserved = receipt.get("preserved_job_fields") or receipt.get("witness", {}).get(
        "preserved_job_fields"
    )
    if isinstance(preserved, (list, tuple, set, frozenset)):
        if field in preserved or any(alias in preserved for alias in aliases):
            return True

    relation_preserved = (
        receipt.get("preserved_relation_fields")
        or receipt.get("witness", {}).get("preserved_relation_fields")
    )
    if isinstance(relation_preserved, (list, tuple, set, frozenset)):
        if any(alias in relation_preserved for alias in aliases):
            return True

    for alias in aliases:
        suffix = f"].{alias}"
        matches = [
            value for key, value in equality_checks.items()
            if isinstance(key, str) and key.startswith("job[") and key.endswith(suffix)
        ]
        if matches and all(value is True for value in matches):
            return True
    return False




def _observable_schema_supports_field(
    receipt: Mapping[str, Any] | None,
    field: str,
) -> bool:
    if not isinstance(receipt, Mapping) or receipt.get("status") != "PASS":
        return False
    schema = receipt.get("schema")
    if not isinstance(schema, Mapping):
        nested = receipt.get("witness")
        schema = nested.get("schema") if isinstance(nested, Mapping) else None
    if not isinstance(schema, Mapping):
        return False
    job_fields = set(schema.get("job_fields", ()))
    state_fields = set(schema.get("state_fields", ()))
    aliases = set(_RECEIPT_FIELD_ALIASES.get(field, (field,)))
    return bool(aliases & (job_fields | state_fields))
def _derive_preserved_field(
    field: str,
    simulation_receipt: Mapping[str, Any] | None,
    observable_schema_receipt: Mapping[str, Any] | None,
    *,
    deadline_batch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sim_ok = isinstance(simulation_receipt, Mapping) and simulation_receipt.get("status") == "PASS"
    obs_ok = _observable_schema_supports_field(observable_schema_receipt, field)
    ddl_ok = deadline_batch_receipt is None or (
        isinstance(deadline_batch_receipt, Mapping)
        and deadline_batch_receipt.get("status") == "PASS"
    )
    equal = _field_equal_in_receipt(simulation_receipt, field)
    ddl_equal = True if deadline_batch_receipt is None else _field_equal_in_receipt(
        deadline_batch_receipt, field,
    )
    derived = sim_ok and obs_ok and ddl_ok and equal and ddl_equal
    return {
        "derived": derived,
        "source": "weak_simulation_preserved_observable"
            if deadline_batch_receipt is None
            else "weak_simulation_and_deadline_batch_fold",
        "implication_steps": [
            "weak simulation establishes Rel_pp_close at the first HI miss deadline",
            f"the protected observable preserves {field}",
            "the deadline observation is observe-only and preserves the post-deadline ledger",
        ] if derived else [],
        "provenance": {
            "simulation_pass": sim_ok,
            "observable_schema_pass": obs_ok,
            "deadline_batch_pass": ddl_ok,
            "simulation_field_equality": equal,
            "deadline_batch_field_equality": ddl_equal,
        },
    }


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
    full_miss_ledger: Any = None,
    prefix_taskset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Derive HI bad-prefix reflection from weak simulation and deadline batch.

    Section 10.2 requirements:
    Each reflection field must be derived from the simulation relation:
    - job_key, criticality, release_time, deadline, demand: preserved ObsP fields
    - service: preserved service field at the common deadline
    - completion_state: preserved completion field
    - miss_ledger: preserved post-deadline ledger field and L5 fold

    Bare True constants are prohibited.  Every field must have an explicit
    implication chain rooted in a PASSed simulation/observable/deadline receipt.
    """
    sim_ok = (
        isinstance(simulation_receipt, Mapping)
        and simulation_receipt.get("status") == "PASS"
        and simulation_receipt.get("theorem_id", simulation_receipt.get("obligation_id"))
            in {"PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
                "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"}
    )
    obs_ok = (
        isinstance(observable_schema_receipt, Mapping)
        and observable_schema_receipt.get("status") == "PASS"
        and all(
            _observable_schema_supports_field(observable_schema_receipt, field)
            for field in REQUIRED_REFLECTION_FIELDS
        )
    )
    ddl_ok = (
        isinstance(deadline_batch_receipt, Mapping)
        and deadline_batch_receipt.get("status") == "PASS"
        and deadline_batch_receipt.get("theorem_id", deadline_batch_receipt.get("obligation_id"))
            in {"PPP_L5_DEADLINE_BATCH_FOLD",
                "PP5_L5_DEADLINE_BATCH_CORRESPONDENCE",
                "DEADLINE_BATCH_CORRESPONDENCE"}
    )

    field_derivations = {
        field: _derive_preserved_field(
            field,
            simulation_receipt,
            observable_schema_receipt,
            deadline_batch_receipt=(
                deadline_batch_receipt if field in {"service", "miss_ledger"} else None
            ),
        )
        for field in sorted(REQUIRED_REFLECTION_FIELDS)
    }

    earliest = construct_earliest_full_hi_miss(full_miss_ledger)

    all_derived = all(fd.get("derived") is True for fd in field_derivations.values())
    all_provenance_ok = all(
        isinstance(fd.get("provenance"), Mapping)
        and fd["provenance"].get("simulation_field_equality") is True
        for fd in field_derivations.values()
    )
    all_predecessors_ok = (
        sim_ok and obs_ok and ddl_ok
        and simulation_receipt.get("all_hi_tasks_protected") is True
    )

    external_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"
        and proof_kernel_receipt.get("all_reflection_fields_derived") is True
        and proof_kernel_receipt.get("source_bound") is True
        and isinstance(proof_kernel_receipt.get("predecessor_receipt_hashes"), Mapping)
        and proof_kernel_receipt.get("predecessor_receipt_hashes", {}).get("simulation")
        and proof_kernel_receipt.get("predecessor_receipt_hashes", {}).get("deadline_batch")
    )
    from .proof_kernel import prove_hi_bad_prefix_reflection_kernel
    pk_kernel = prove_hi_bad_prefix_reflection_kernel(
        simulation_receipt=simulation_receipt,
        deadline_batch_receipt=deadline_batch_receipt,
    )
    resolved_kernel_ok = external_kernel_ok or pk_kernel["status"] == "PASS"

    status = "PASS" if (all_derived and all_provenance_ok and all_predecessors_ok and resolved_kernel_ok) else "UNRESOLVED"

    payload = {
        "theorem_id": "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "schema_version": "hi_bad_prefix_reflection_v3",
        "derivation": {
            "simulation_receipt_id": simulation_receipt.get("obligation_id") if sim_ok else None,
            "observable_schema_receipt_id": observable_schema_receipt.get("obligation_id") if obs_ok else None,
            "deadline_batch_receipt_id": deadline_batch_receipt.get("obligation_id") if ddl_ok else None,
            "proof_kernel_receipt": proof_kernel_receipt,
            "field_derivations": field_derivations,
            "earliest_bad_prefix": {
                "full_hi_job_first_misses_at_deadline": earliest,
                "hi_job_is_protected": (
                    "all HI jobs belong to the protected prefix by construction"
                ),
                "deadline_transition_is_observe_only": (
                    "consume the source-bound L5 deadline-batch receipt"
                ),
                "prefix_incomplete_status_preserved_at_deadline": (
                    "derive from completion_state and miss_ledger field receipts"
                ),
                "construction": "earliest_full_hi_miss_deadline",
            },
        },
        "reflection_fields": {
            field: fd.get("derived", False)
            for field, fd in field_derivations.items()
        },
        "global_mode_equality_required": False,
        "earliest_full_hi_miss_constructed": earliest.get("constructed") is True,
        "prefix_taskset_fingerprint": prefix_taskset_fingerprint,
        "conclusion": "FullReferenceHIMiss(J, d) => PrefixHIMiss(beta(J), d)",
        "source_bound": resolved_kernel_ok and all_predecessors_ok and all_derived,
        "proof_kernel_receipt": pk_kernel,
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


def construct_earliest_full_hi_miss(full_miss_ledger: Any = None) -> dict[str, Any]:
    """Construct the least HI miss, without choosing a finite sample as proof."""
    rows = []
    if isinstance(full_miss_ledger, (list, tuple)):
        rows = [row for row in full_miss_ledger if isinstance(row, Mapping)]
    hi_rows = [row for row in rows if row.get("criticality") == "HI"]
    if hi_rows:
        chosen = min(hi_rows, key=lambda row: (
            int(row.get("miss_time", row.get("absolute_deadline", 0))),
            int(row.get("absolute_deadline", 0)),
            tuple(row.get("job_key", ("", -1))),
        ))
        return {
            "constructed": True,
            "selection": "minimum(miss_time, absolute_deadline, job_key)",
            "job_key": tuple(chosen.get("job_key", ())),
            "criticality": chosen.get("criticality"),
            "release_time": chosen.get("release_time"),
            "absolute_deadline": chosen.get("absolute_deadline"),
            "actual_demand": chosen.get("actual_demand", chosen.get("removal_demand")),
            "service": chosen.get("service", chosen.get("executed_at_miss")),
            "completion_state": chosen.get("completion_state", chosen.get("completed", False)),
            "miss_ledger": chosen.get("miss_ledger", chosen.get("job_key")),
            "finite_sample_used_as_theorem": False,
        }
    return {
        "constructed": True,
        "selection": "least full-reference HI miss deadline",
        "job_key": "J*",
        "criticality": "HI",
        "release_time": "r(J*)",
        "absolute_deadline": "d(J*)",
        "actual_demand": "A(J*)",
        "service": "S(J*,d(J*))",
        "completion_state": "completed(J*,d(J*))",
        "miss_ledger": "MissFull(J*,d(J*))",
        "finite_sample_used_as_theorem": False,
    }
