"""Prefix-specific reference model conformance derivation.

Proves that the saturated protected-prefix taskset satisfies all model
assumptions required to import the C-AMC-sem all-task schedulability
theorem.  This is NOT a copy of the full-reference conformance; each
proof step is recomputed from the prefix construction witness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.taskset import ReferenceTaskset

from .types import ProtectedPrefixBuildResult


@dataclass(frozen=True, slots=True)
class PrefixModelConformanceWitness:
    finite_nonempty_taskset: bool
    constrained_deadlines: bool
    positive_integer_parameters: bool
    strict_total_priority_order: bool
    lo_wcet_relation: bool
    hi_wcet_relation: bool
    all_hi_tasks_preserved: bool
    executable_semantics_shared: bool
    release_fixed_demands: bool
    abnormal_classification_at_arrival: bool
    abnormal_hi_only_switch_trigger: bool
    quiescent_idle_only_recovery: bool
    lo_version_selected_at_release: bool
    recurring_history_preserved: bool
    candidate_enumeration_complete: bool


def derive_prefix_model_conformance(
    *,
    full_taskset: ReferenceTaskset,
    prefix_taskset: ReferenceTaskset,
    construction: ProtectedPrefixBuildResult,
    runtime_schema_certificate: Mapping[str, Any],
    execution_existence_receipt: Mapping[str, Any] | None = None,
    candidate_enumeration_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prefix_tasks_list = list(prefix_taskset.tasks)
    full_by_name = {str(task.name): task for task in full_taskset.tasks}

    finite_nonempty = len(prefix_tasks_list) > 0

    constrained_deadlines = all(
        0 < task.deadline <= task.period
        for task in prefix_tasks_list
    )

    positive_integer_parameters = all(
        isinstance(task.period, int) and task.period > 0
        and isinstance(task.deadline, int) and task.deadline > 0
        and isinstance(task.c_lo, int) and task.c_lo > 0
        and isinstance(task.c_hi, int) and task.c_hi > 0
        for task in prefix_tasks_list
    )

    n_prefix = len(prefix_tasks_list)
    strict_total_priority_order = (
        tuple(int(task.priority_index) for task in prefix_tasks_list)
        == tuple(range(n_prefix))
    )

    lo_wcet_relation = all(
        task.c_hi == task.c_lo and task.c_lo == full_by_name[task.name].c_lo
        for task in prefix_tasks_list
        if task.criticality == "LO"
    )

    hi_wcet_relation = all(
        task.c_hi == full_by_name[task.name].c_hi
        and task.c_lo == full_by_name[task.name].c_lo
        for task in prefix_tasks_list
        if task.criticality == "HI"
    )

    protected_set = frozenset(construction.protected_task_names)
    hi_task_names_in_full = {
        str(task.name) for task in full_taskset.tasks
        if task.criticality == "HI"
    }
    all_hi_tasks_preserved = hi_task_names_in_full <= protected_set

    executable_semantics_shared = (
        isinstance(runtime_schema_certificate, Mapping)
        and runtime_schema_certificate.get("status") == "PASS"
        and runtime_schema_certificate.get("pp0_transition_status") == "PASS"
    )

    # These are semantic, quantified obligations.  They may be discharged only
    # by an explicit PP0 receipt; source-shape or task-count facts are not enough.
    pp0_witness = runtime_schema_certificate.get("pp0_witness", {})
    if not isinstance(pp0_witness, Mapping):
        pp0_witness = {}
    release_fixed_demands = executable_semantics_shared and pp0_witness.get("release_fixed_demands") is True
    abnormal_classification_at_arrival = executable_semantics_shared and pp0_witness.get("abnormal_classification_at_arrival") is True
    abnormal_hi_only_switch_trigger = executable_semantics_shared and pp0_witness.get("abnormal_hi_only_switch_trigger") is True
    quiescent_idle_only_recovery = executable_semantics_shared and pp0_witness.get("quiescent_idle_only_recovery") is True
    lo_version_selected_at_release = executable_semantics_shared and pp0_witness.get("lo_version_selected_at_release") is True

    execution_receipt = execution_existence_receipt or {}
    recurring_history_preserved = (
        isinstance(execution_receipt, Mapping)
        and execution_receipt.get("status") == "PASS"
        and execution_receipt.get("recurring_history_preserved") is True
    )
    enumeration_receipt = candidate_enumeration_receipt or {}
    candidate_enumeration_complete = (
        isinstance(enumeration_receipt, Mapping)
        and enumeration_receipt.get("status") == "PASS"
        and enumeration_receipt.get("complete_integer_candidate_domains") is True
    )

    witness = PrefixModelConformanceWitness(
        finite_nonempty_taskset=finite_nonempty,
        constrained_deadlines=constrained_deadlines,
        positive_integer_parameters=positive_integer_parameters,
        strict_total_priority_order=strict_total_priority_order,
        lo_wcet_relation=lo_wcet_relation,
        hi_wcet_relation=hi_wcet_relation,
        all_hi_tasks_preserved=all_hi_tasks_preserved,
        executable_semantics_shared=executable_semantics_shared,
        release_fixed_demands=release_fixed_demands,
        abnormal_classification_at_arrival=abnormal_classification_at_arrival,
        abnormal_hi_only_switch_trigger=abnormal_hi_only_switch_trigger,
        quiescent_idle_only_recovery=quiescent_idle_only_recovery,
        lo_version_selected_at_release=lo_version_selected_at_release,
        recurring_history_preserved=recurring_history_preserved,
        candidate_enumeration_complete=candidate_enumeration_complete,
    )

    all_flags = asdict(witness)
    static_fields = {
        "finite_nonempty_taskset", "constrained_deadlines",
        "positive_integer_parameters", "strict_total_priority_order",
        "lo_wcet_relation", "hi_wcet_relation", "all_hi_tasks_preserved",
    }
    static_ok = all(all_flags[name] for name in static_fields)
    ok = all(all_flags.values())

    payload = {
        "schema_version": "prefix_model_conformance_v1",
        "full_taskset_fingerprint": full_taskset.to_dict().get("fingerprint"),
        "prefix_taskset_fingerprint": prefix_taskset.to_dict().get("fingerprint"),
        "cutoff": construction.cutoff_task_name,
        "cutoff_priority_index": construction.cutoff_priority_index,
        "saturation_witness_hash": sha256_object(construction.saturation_witness),
        "runtime_schema_receipt_hash": runtime_schema_certificate.get(
            "certificate_hash",
            sha256_object(runtime_schema_certificate),
        ),
        "rta_formula_version": "C-AMC-sem-all-task-v3",
        "witness": all_flags,
    }

    return {
        **payload,
        "status": "PASS" if ok else ("FAIL" if not static_ok else "UNRESOLVED"),
        "conformance_hash": sha256_object(payload),
        "failure": None if ok else {
            "code": ("PREFIX_MODEL_CONFORMANCE_STATIC_FAILURE"
                     if not static_ok else "PREFIX_MODEL_CONFORMANCE_SEMANTIC_PROOF_MISSING"),
            "failed_flags": [k for k, v in all_flags.items() if not v],
        },
    }
