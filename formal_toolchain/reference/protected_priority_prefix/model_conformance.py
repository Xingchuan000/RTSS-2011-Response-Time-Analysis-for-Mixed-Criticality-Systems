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
    valid_periodic_offsets: bool
    strict_total_priority_order: bool
    lo_wcet_relation: bool
    hi_wcet_relation: bool
    all_hi_tasks_preserved: bool
    executable_semantics_shared: bool
    single_processor_preemptive_work_conserving_fp: bool
    no_blocking_self_suspension_or_nonpreemptive_segments: bool
    fixed_processor_supply_and_mode_independent_priority: bool
    release_fixed_demands: bool
    projected_demands_positive_and_within_prefix_wcet: bool
    abnormal_classification_at_arrival: bool
    abnormal_hi_only_switch_trigger: bool
    quiescent_idle_only_recovery: bool
    lo_version_selected_at_release: bool
    standard_empty_lo_initial_state: bool
    recurring_history_preserved: bool
    candidate_enumeration_complete: bool
    zero_relative_start_lemma_required: bool


def derive_prefix_model_conformance(
    *,
    full_taskset: ReferenceTaskset,
    prefix_taskset: ReferenceTaskset,
    construction: ProtectedPrefixBuildResult,
    runtime_schema_certificate: Mapping[str, Any],
    execution_existence_receipt: Mapping[str, Any] | None = None,
    demand_receptiveness_receipt: Mapping[str, Any] | None = None,
    candidate_enumeration_receipt: Mapping[str, Any] | None = None,
    zero_relative_start_receipt: Mapping[str, Any] | None = None,
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

    valid_periodic_offsets = all(
        isinstance(task.offset, int) and 0 <= task.offset < task.period
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
    single_processor_preemptive_work_conserving_fp = (
        executable_semantics_shared
        and pp0_witness.get("single_processor_preemptive_work_conserving_fp") is True
    )
    no_blocking_self_suspension_or_nonpreemptive_segments = (
        executable_semantics_shared
        and pp0_witness.get("no_blocking_self_suspension_or_nonpreemptive_segments") is True
    )
    fixed_processor_supply_and_mode_independent_priority = (
        executable_semantics_shared
        and pp0_witness.get("fixed_processor_supply_and_mode_independent_priority") is True
    )
    release_fixed_demands = executable_semantics_shared and pp0_witness.get("release_fixed_demands") is True
    abnormal_classification_at_arrival = executable_semantics_shared and pp0_witness.get("abnormal_classification_at_arrival") is True
    abnormal_hi_only_switch_trigger = executable_semantics_shared and pp0_witness.get("abnormal_hi_only_switch_trigger") is True
    quiescent_idle_only_recovery = executable_semantics_shared and pp0_witness.get("quiescent_idle_only_recovery") is True
    lo_version_selected_at_release = executable_semantics_shared and pp0_witness.get("lo_version_selected_at_release") is True

    execution_receipt = execution_existence_receipt or {}
    execution_payload = (
        execution_receipt.get("witness", execution_receipt)
        if isinstance(execution_receipt, Mapping) else {}
    )
    execution_status = (
        execution_receipt.get("status", execution_receipt.get("obligation_status"))
        if isinstance(execution_receipt, Mapping) else None
    )
    if execution_status is None and isinstance(execution_payload, Mapping):
        execution_status = execution_payload.get("status")
    recurring_history_preserved = (
        execution_status == "PASS"
        and isinstance(execution_payload, Mapping)
        and execution_payload.get("recurring_history_preserved") is True
    )

    standard_empty_lo_initial_state = (
        execution_status == "PASS"
        and isinstance(execution_payload, Mapping)
        and execution_payload.get("standard_empty_lo_initial_state") is True
    )

    demand_receipt = demand_receptiveness_receipt or {}
    demand_payload = (
        demand_receipt.get("witness", demand_receipt)
        if isinstance(demand_receipt, Mapping) else {}
    )
    demand_status = (
        demand_receipt.get("status", demand_receipt.get("obligation_status"))
        if isinstance(demand_receipt, Mapping) else None
    )
    if demand_status is None and isinstance(demand_payload, Mapping):
        demand_status = demand_payload.get("status")
    projected_demands_positive_and_within_prefix_wcet = (
        demand_status == "PASS"
        and isinstance(demand_payload, Mapping)
        and demand_payload.get("all_projected_demands_legal") is True
        and demand_payload.get("all_projected_demands_positive") is True
        and demand_payload.get("release_fixed_demands") is True
        and demand_payload.get("mode_independent_lo_receptiveness") is True
    )

    enumeration_receipt = candidate_enumeration_receipt or {}
    enumeration_payload = (
        enumeration_receipt.get("witness", enumeration_receipt)
        if isinstance(enumeration_receipt, Mapping) else {}
    )
    enumeration_status = (
        enumeration_receipt.get("status", enumeration_receipt.get("obligation_status"))
        if isinstance(enumeration_receipt, Mapping) else None
    )
    if enumeration_status is None and isinstance(enumeration_payload, Mapping):
        enumeration_status = enumeration_payload.get("status")
    candidate_enumeration_complete = (
        enumeration_status == "PASS"
        and isinstance(enumeration_payload, Mapping)
        and enumeration_payload.get("complete_integer_candidate_domains") is True
    )

    # ZERO_RELATIVE_START is already a common mathematical predecessor of the
    # selected all-task RTA chain.  Prefix conformance must consume that verified
    # predecessor explicitly; it must not look for an unrelated receipt hidden
    # inside the runtime-schema certificate.
    zero_receipt = zero_relative_start_receipt or {}
    zero_status = (
        zero_receipt.get("status", zero_receipt.get("obligation_status"))
        if isinstance(zero_receipt, Mapping) else None
    )
    zero_payload = (
        zero_receipt.get("witness", zero_receipt)
        if isinstance(zero_receipt, Mapping) else {}
    )
    if zero_status is None and isinstance(zero_payload, Mapping):
        zero_status = zero_payload.get("status")
    zero_relative_start_lemma_required = zero_status == "PASS"

    witness = PrefixModelConformanceWitness(
        finite_nonempty_taskset=finite_nonempty,
        constrained_deadlines=constrained_deadlines,
        positive_integer_parameters=positive_integer_parameters,
        valid_periodic_offsets=valid_periodic_offsets,
        strict_total_priority_order=strict_total_priority_order,
        lo_wcet_relation=lo_wcet_relation,
        hi_wcet_relation=hi_wcet_relation,
        all_hi_tasks_preserved=all_hi_tasks_preserved,
        executable_semantics_shared=executable_semantics_shared,
        single_processor_preemptive_work_conserving_fp=single_processor_preemptive_work_conserving_fp,
        no_blocking_self_suspension_or_nonpreemptive_segments=no_blocking_self_suspension_or_nonpreemptive_segments,
        fixed_processor_supply_and_mode_independent_priority=fixed_processor_supply_and_mode_independent_priority,
        release_fixed_demands=release_fixed_demands,
        projected_demands_positive_and_within_prefix_wcet=projected_demands_positive_and_within_prefix_wcet,
        abnormal_classification_at_arrival=abnormal_classification_at_arrival,
        abnormal_hi_only_switch_trigger=abnormal_hi_only_switch_trigger,
        quiescent_idle_only_recovery=quiescent_idle_only_recovery,
        lo_version_selected_at_release=lo_version_selected_at_release,
        standard_empty_lo_initial_state=standard_empty_lo_initial_state,
        recurring_history_preserved=recurring_history_preserved,
        candidate_enumeration_complete=candidate_enumeration_complete,
        zero_relative_start_lemma_required=zero_relative_start_lemma_required,
    )

    all_flags = asdict(witness)
    static_fields = {
        "finite_nonempty_taskset", "constrained_deadlines",
        "positive_integer_parameters", "valid_periodic_offsets",
        "strict_total_priority_order",
        "lo_wcet_relation", "hi_wcet_relation", "all_hi_tasks_preserved",
    }
    static_ok = all(all_flags[name] for name in static_fields)

    # Zero-relative-start lemma absence means the conformance is semantically
    # incomplete (Section 6.3).  Missing ZERO_RELATIVE_START_LEMMA produces
    # UNRESOLVED, not FAIL, because the lemma is disclosed as external TCB.
    if not zero_relative_start_lemma_required:
        semantic_flags = {
            k for k, v in all_flags.items()
            if k not in static_fields and k != "zero_relative_start_lemma_required" and not v
        }
        ok = False
    else:
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
