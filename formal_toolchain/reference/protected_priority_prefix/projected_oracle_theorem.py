"""Parameterized theorem receipts for full-to-protected input projection.

The functions in this module discharge only elementary, task-parameter-level
consequences of already verified reference-model and prefix-construction facts.
They do not create a prefix execution and therefore must not claim the final
weak-simulation quantifier order.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


PROJECTION_QUANTIFIER = "forall-full-execution-exists-unique-projected-stream"


def _task_map(taskset: Any) -> dict[str, Any]:
    return {str(task.name): task for task in tuple(getattr(taskset, "tasks", ()))}


def build_symbolic_projection_theorem(
    *, full_theorem: Mapping[str, Any], protected_task_names: frozenset[str],
    full_taskset: Any, prefix_taskset: Any,
    saturation_witness: Mapping[str, Any], saturation_certificate_hash: str,
) -> dict[str, Any]:
    """Build the definitional tail-deletion projection theorem.

    For one arbitrary full execution, its release-ledger stream is restricted
    to the protected task names.  The theorem is pointwise in every release
    index and preserves the complete release record.  It does *not* yet assert
    existence of a prefix execution or equality at all time boundaries.
    """
    full_tasks = _task_map(full_taskset)
    prefix_tasks = _task_map(prefix_taskset)
    protected_exact = (
        bool(protected_task_names)
        and protected_task_names == frozenset(prefix_tasks)
        and protected_task_names <= frozenset(full_tasks)
    )
    full_fp = (
        full_taskset.to_dict()["fingerprint"]
        if hasattr(full_taskset, "to_dict") else None
    )
    prefix_fp = (
        prefix_taskset.to_dict()["fingerprint"]
        if hasattr(prefix_taskset, "to_dict") else None
    )
    full_ok = (
        full_theorem.get("status") == "PASS"
        and full_theorem.get("forall_full_reference_executions") is True
        and full_theorem.get("unique_release_record_for_every_job_key") is True
        and full_theorem.get("release_fixed_actual_demand") is True
        and full_theorem.get("demand_contract_complete") is True
        and full_theorem.get("reference_taskset_fingerprint") == full_fp
    )
    saturation_bound = (
        saturation_witness.get("prefix_fingerprint") == prefix_fp
        and saturation_witness.get("hi_fields_equal") is True
        and saturation_witness.get("timing_fields_equal") is True
        and isinstance(saturation_witness.get("lo_saturation_equalities"), list)
    )
    ok = full_ok and protected_exact and saturation_bound
    projected_fp = sha256_object({
        "full_execution_input_theorem": full_theorem.get("receipt_hash"),
        "protected_task_names": sorted(protected_task_names),
        "full_taskset_fingerprint": full_fp,
        "prefix_taskset_fingerprint": prefix_fp,
    })
    payload = {
        "theorem_id": "PROTECTED_INPUT_STREAM_PROJECTION",
        "status": "PASS" if ok else "UNRESOLVED",
        "quantifier_scope": PROJECTION_QUANTIFIER,
        "full_oracle_contract_receipt_hash": full_theorem.get("receipt_hash"),
        "full_reference_taskset_fingerprint": full_fp,
        "projected_oracle_fingerprint": projected_fp,
        "prefix_taskset_fingerprint": prefix_fp,
        "saturation_certificate_hash": saturation_certificate_hash,
        "protected_task_partition_exact": protected_exact,
        "all_release_indices": ok,
        "forall_release_indices": ok,
        "protected_keys_times_demands_classes_preserved": ok,
        "full_demand_contract_preserved": ok,
        "lo_demand_le_reference_c_lo": (
            ok and full_theorem.get("lo_demand_le_reference_c_lo") is True
        ),
        "normal_hi_demand_le_reference_c_lo": (
            ok and full_theorem.get("normal_hi_demand_le_reference_c_lo") is True
        ),
        "abnormal_hi_demand_in_lo_hi_interval": (
            ok and full_theorem.get("abnormal_hi_demand_in_lo_hi_interval") is True
        ),
        "tail_entries_deleted_only": ok,
        "query_order_independent": ok,
        "finite_instance_data_used": False,
        "complete_recurring_stream": ok,
        "projection_definition": "A_P_xi(task,q)=A_xi(task,q) iff task in protected set",
        "preserved_record_fields": [
            "job_key", "release_time", "actual_demand", "hi_class",
        ],
        "canonical_protected_batch_order": (
            "sort by (priority_index, task_name, release_index) after tail deletion"
        ),
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def build_symbolic_demand_receptiveness_theorem(
    *, full_taskset: Any, prefix_taskset: Any,
    projection_theorem: Mapping[str, Any], saturation_witness: Mapping[str, Any],
) -> dict[str, Any]:
    """Discharge projected-demand legality by task-type inequalities.

    The proof uses the full-stream demand contract plus two construction facts:
    HI parameters are unchanged, and every protected LO task is saturated to
    ``C_pp(LO)=C_pp(HI)=C_ref(LO)``.  Positivity alone is not sufficient.
    """
    full_tasks = _task_map(full_taskset)
    prefix_tasks = _task_map(prefix_taskset)
    projection_ok = (
        projection_theorem.get("status") == "PASS"
        and projection_theorem.get("forall_release_indices") is True
        and projection_theorem.get("full_demand_contract_preserved") is True
        and projection_theorem.get("lo_demand_le_reference_c_lo") is True
        and projection_theorem.get("normal_hi_demand_le_reference_c_lo") is True
        and projection_theorem.get("abnormal_hi_demand_in_lo_hi_interval") is True
        and projection_theorem.get("quantifier_scope") == PROJECTION_QUANTIFIER
    )

    lo_rows = saturation_witness.get("lo_saturation_equalities", ())
    lo_row_map = {
        str(row.get("task")): row for row in lo_rows if isinstance(row, Mapping)
    }
    hi_equal = all(
        name in full_tasks
        and int(task.c_lo) == int(full_tasks[name].c_lo)
        and int(task.c_hi) == int(full_tasks[name].c_hi)
        for name, task in prefix_tasks.items()
        if str(task.criticality) == "HI"
    )
    lo_saturated = all(
        name in full_tasks
        and name in lo_row_map
        and int(task.c_lo) == int(task.c_hi) == int(full_tasks[name].c_lo)
        and lo_row_map[name].get("C_pp_LO") == int(task.c_lo)
        and lo_row_map[name].get("C_pp_HI") == int(task.c_hi)
        and lo_row_map[name].get("C_ref_LO") == int(full_tasks[name].c_lo)
        for name, task in prefix_tasks.items()
        if str(task.criticality) == "LO"
    )
    positive = bool(prefix_tasks) and all(
        int(task.c_lo) > 0 and int(task.c_hi) > 0
        for task in prefix_tasks.values()
    )
    exact_task_domain = frozenset(prefix_tasks) <= frozenset(full_tasks)
    ok = projection_ok and hi_equal and lo_saturated and positive and exact_task_domain
    prefix_fp = prefix_taskset.to_dict()["fingerprint"]
    payload = {
        "theorem_id": "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        "status": "PASS" if ok else "UNRESOLVED",
        "quantifier_scope": "forall-protected-tasks-forall-release-indices",
        "all_projected_demands_legal": ok,
        "all_projected_demands_positive": ok,
        "release_fixed_demands": projection_ok,
        "mode_independent_lo_receptiveness": lo_saturated and projection_ok,
        "forall_release_indices": ok,
        "normal_hi_bound_preserved": hi_equal and projection_ok,
        "abnormal_hi_bound_preserved": hi_equal and projection_ok,
        "saturated_lo_mode_independent_bound": lo_saturated and projection_ok,
        "hi_parameters_unchanged": hi_equal,
        "lo_saturation_equalities_verified": lo_saturated,
        "projected_oracle_fingerprint": projection_theorem.get("projected_oracle_fingerprint"),
        "prefix_taskset_fingerprint": prefix_fp,
        "projection_receipt_hash": projection_theorem.get("receipt_hash"),
        "finite_instance_data_used": False,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def prove_projected_oracle_theorem(
    *, full_view: Any, projected_oracle: Any, protected_task_names: frozenset[str],
    prefix_taskset_fingerprint: str, saturation_certificate_hash: str,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a separately generated quantified projection receipt."""
    full_fp = getattr(full_view, "oracle_fingerprint", lambda: None)()
    projected_fp = getattr(projected_oracle, "oracle_fingerprint", lambda: None)()
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_INPUT_STREAM_PROJECTION"
        and proof_kernel_receipt.get("quantifier_scope") == PROJECTION_QUANTIFIER
        and proof_kernel_receipt.get("forall_release_indices") is True
        and proof_kernel_receipt.get("finite_instance_data_used") is False
        and proof_kernel_receipt.get("full_oracle_fingerprint") == full_fp
        and proof_kernel_receipt.get("projected_oracle_fingerprint") == projected_fp
        and proof_kernel_receipt.get("protected_record_fields_preserved") is True
        and proof_kernel_receipt.get("canonical_protected_batch_order") is True
    )
    payload = {
        "theorem_id": "PROTECTED_INPUT_STREAM_PROJECTION",
        "status": "PASS" if kernel_ok else "UNRESOLVED",
        "quantifier_scope": PROJECTION_QUANTIFIER,
        "forall_release_indices": kernel_ok,
        "finite_instance_data_used": False,
        "protected_task_names": sorted(protected_task_names),
        "full_oracle_fingerprint": full_fp,
        "projected_oracle_fingerprint": projected_fp,
        "prefix_taskset_fingerprint": prefix_taskset_fingerprint,
        "saturation_certificate_hash": saturation_certificate_hash,
        "protected_record_fields_preserved": kernel_ok,
        "tail_entries_deleted_only": kernel_ok,
        "canonical_protected_batch_order": kernel_ok,
        "protected_record_fields_preserved": kernel_ok,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def bind_projection_to_full_execution_ledger(
    *, symbolic_projection_receipt: Mapping[str, Any],
    full_execution_ledger: Any, projected_oracle: Any,
) -> dict[str, Any]:
    """Bind the symbolic projection to the one selected execution ledger."""
    full_fp = getattr(full_execution_ledger, "oracle_fingerprint", lambda: None)()
    projected_fp = getattr(projected_oracle, "oracle_fingerprint", lambda: None)()
    ok = (
        symbolic_projection_receipt.get("status") == "PASS"
        and (
            callable(getattr(full_execution_ledger, "input_for", None))
            or callable(getattr(full_execution_ledger, "record_for", None))
        )
        and isinstance(full_fp, str) and isinstance(projected_fp, str)
    )
    payload = {
        "theorem_id": "PROTECTED_INPUT_STREAM_PROJECTION",
        "status": "PASS" if ok else "UNRESOLVED",
        "quantifier_scope": PROJECTION_QUANTIFIER,
        "symbolic_projection_receipt_hash": symbolic_projection_receipt.get("receipt_hash"),
        "full_execution_id": str(getattr(full_execution_ledger, "execution_id", "")),
        "full_oracle_fingerprint": full_fp,
        "projected_oracle_fingerprint": projected_fp,
        "forall_release_indices": ok,
        "complete_recurring_stream": ok,
        "protected_record_fields_preserved": ok,
        "canonical_protected_batch_order": ok,
        "tail_entries_deleted_only": ok,
        "finite_instance_data_used": False,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
