"""The quantified input/state domain for full-to-prefix simulation.

Phase H updates (Section 10):
  - Quantifier order fixed: forall-full-exists-one-prefix-forall-boundaries
  - Completeness checks for recurring input stream
  - One internally constructed complete-execution witness
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.executable_semantics import close_timestamp, initial_reference_state

from .input_projection import check_projected_demands_legal, project_protected_release_stream
from .observable import observable_schema
from .runtime_schema import build_runtime_schema_certificate
from .types import ProtectedPrefixBuildResult


@dataclass(frozen=True, slots=True)
class FullToPrefixSimulationDomainWitness:
    reference_model_conformance: bool
    partition_valid: bool
    saturation_valid: bool
    runtime_schema_valid: bool
    protected_input_independence: bool
    projected_demands_legal: bool
    complete_prefix_execution_exists: bool
    quantifier_order_correct: bool
    same_fixed_oracle: bool
    complete_execution_witness_constructed: bool
    idle_jump_expansion_verified: bool
    time_indexed_closed_observation_defined: bool


def _predecessor_witness(predecessors: dict[str, Any], name: str) -> dict[str, Any]:
    """Unwrap the obligation envelope and one nested theorem witness.

    Route checkers commonly return ``witness={theorem result}``, while theorem
    results such as complete-execution existence themselves contain a nested
    ``witness``.  Reading only one level makes all future PASS receipts appear
    to lack quantifier/oracle fields and prevents the domain from closing.
    """
    value: Any = predecessors.get(name, {})
    for _ in range(3):
        if not isinstance(value, dict):
            return {}
        nested = value.get("witness")
        if not isinstance(nested, dict):
            break
        value = nested
    return value if isinstance(value, dict) else {}


def check_full_to_prefix_simulation_domain(**kwargs: Any) -> dict[str, Any]:
    """Compose the simulation domain from independently verified predecessors.

    This node proves only a conjunction.  It must not regenerate initial-only
    traces or accept side-channel receipts that are absent from the resolved DAG.

    Phase H: Requires quantifier order fixed as
    'forall-full-exists-one-prefix-forall-boundaries' and same_fixed_oracle.
    """
    state = getattr(kwargs.get("context"), "fresh_state", None) or kwargs.get("fresh_state")
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "REFERENCE_MODEL_CONFORMANCE", "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "PROTECTED_PREFIX_LO_SATURATION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_INPUT_STREAM_PROJECTION", "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
    }
    if set(predecessors) != required:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PREDECESSOR_SET_MISMATCH",
                "witness": {"expected": sorted(required), "actual": sorted(predecessors)}}
    if state is None or state.prepared_route is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PROTECTED_PREFIX_TASKSET_MISSING", "witness": {}}
    if any(predecessors[name].get("obligation_status") != "PASS" for name in required):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FULL_TO_PREFIX_DOMAIN_PREDECESSOR_NOT_PASS", "witness": {}}

    construction: ProtectedPrefixBuildResult = state.prepared_route.construction_witnesses["build_result"]
    full = state.full_reference_taskset
    prefix_taskset = state.analysis_taskset
    projection = _predecessor_witness(predecessors, "PROTECTED_INPUT_STREAM_PROJECTION")
    receptiveness = _predecessor_witness(predecessors, "PROTECTED_INPUT_DEMAND_RECEPTIVENESS")
    execution = _predecessor_witness(predecessors, "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS")
    model = _predecessor_witness(predecessors, "REFERENCE_MODEL_CONFORMANCE")
    partition = _predecessor_witness(predecessors, "PROTECTED_PRIORITY_PREFIX_PARTITION")
    saturation = _predecessor_witness(predecessors, "PROTECTED_PREFIX_LO_SATURATION")
    runtime_schema = _predecessor_witness(
        predecessors, "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"
    )

    model_conditions = model.get("condition_results", ())
    model_conformance = bool(model_conditions) and all(
        isinstance(row, dict) and row.get("passed") is True
        for row in model_conditions
    )
    partition_valid = all(
        partition.get(name) is True
        for name in ("tail_all_lo", "all_hi_protected", "partition_complete", "order_preserved")
    )
    saturation_valid = bool(
        saturation.get("hi_fields_equal") is True
        and saturation.get("timing_fields_equal") is True
        and saturation.get("lo_saturation_equalities")
        and all(
            row.get("C_pp_LO") == row.get("C_pp_HI") == row.get("C_ref_LO")
            for row in saturation.get("lo_saturation_equalities", ())
            if isinstance(row, dict)
        )
    )
    runtime_schema_valid = all(
        runtime_schema.get(name) is True
        for name in (
            "single_processor_preemptive_work_conserving_fp",
            "no_blocking_self_suspension_or_nonpreemptive_segments",
            "fixed_processor_supply_and_mode_independent_priority",
            "release_fixed_demands", "abnormal_classification_at_arrival",
            "abnormal_hi_only_switch_trigger", "quiescent_idle_only_recovery",
            "lo_version_selected_at_release", "deadline_observe_only",
            "protected_input_independence",
        )
    )

    quantifier_order_correct = (
        projection.get("quantifier_scope")
            == "forall-full-execution-exists-unique-projected-stream"
        and execution.get("quantifier_order")
            == "forall-full-exists-one-prefix-forall-boundaries"
    )
    projection_oracle_fp = projection.get("projected_oracle_fingerprint")
    execution_oracle_fp = execution.get("projected_oracle_fingerprint")
    same_fixed_oracle = (
        execution.get("same_fixed_oracle") is True
        and projection.get("complete_recurring_stream") is True
        and isinstance(projection_oracle_fp, str)
        and projection_oracle_fp == execution_oracle_fp
    )
    complete_execution_witness_constructed = (
        execution.get("complete_execution_witness_constructed") is True
        and execution.get("finite_prefix_compatibility_proved") is True
    )

    witness = FullToPrefixSimulationDomainWitness(
        reference_model_conformance=model_conformance,
        partition_valid=partition_valid,
        saturation_valid=saturation_valid,
        runtime_schema_valid=runtime_schema_valid,
        protected_input_independence=(
            projection.get("complete_recurring_stream") is True
            and projection.get("protected_input_independence") is True
        ),
        projected_demands_legal=(
            receptiveness.get("all_projected_demands_legal") is True
            and receptiveness.get("mode_independent_lo_receptiveness") is True
        ),
        complete_prefix_execution_exists=(
            execution.get("complete_execution_exists") is True
            and execution.get("time_divergent") is True
            and execution.get("same_fixed_oracle") is True
        ),
        quantifier_order_correct=quantifier_order_correct,
        same_fixed_oracle=same_fixed_oracle,
        complete_execution_witness_constructed=complete_execution_witness_constructed,
        idle_jump_expansion_verified=(
            execution.get("idle_jump_expansion_verified") is True
        ),
        time_indexed_closed_observation_defined=(
            execution.get("time_indexed_closed_observation_defined") is True
        ),
    )
    flags = asdict(witness)
    payload = {
        **flags,
        "full_taskset_fingerprint": full.to_dict()["fingerprint"],
        "prefix_taskset_fingerprint": prefix_taskset.to_dict()["fingerprint"],
        "cutoff_task_name": construction.cutoff_task_name,
        "protected_task_names": list(construction.protected_task_names),
        "tail_task_names": list(construction.tail_task_names),
        "protected_observable_schema_hash": sha256_object(observable_schema()),
        "predecessor_artifact_hashes": {
            name: predecessors[name].get("artifact_hash") for name in sorted(required)
        },
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
    }
    if all(flags.values()):
        return {"status": "PASS", "route": None, "code": None, "witness": payload}
    return {
        "status": "UNRESOLVED", "route": "UNRESOLVED",
        "code": "FULL_TO_PREFIX_SIMULATION_DOMAIN_PROOF_MISSING",
        "witness": {**payload, "missing_flags": [k for k, v in flags.items() if not v]},
    }
