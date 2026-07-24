"""The quantified input/state domain for full-to-prefix simulation.

Phase H updates (Section 10):
  - Quantifier order fixed: forall-full-exists-one-prefix-forall-boundaries
  - Completeness checks for recurring input stream
  - Single-witness binding verification
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
from .execution_builder import build_complete_prefix_execution_witness


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
    single_witness_verified: bool
    idle_jump_expansion_verified: bool
    time_indexed_closed_observation_defined: bool


def _predecessor_witness(predecessors: dict[str, Any], name: str) -> dict[str, Any]:
    receipt = predecessors.get(name, {})
    witness = receipt.get("witness", {}) if isinstance(receipt, dict) else {}
    return witness if isinstance(witness, dict) else {}


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

    quantifier_order_correct = (
        projection.get("quantifier_order") == "forall-full-exists-one-prefix-forall-boundaries"
        and execution.get("quantifier_order") == "forall-full-exists-one-prefix-forall-boundaries"
    )
    projection_oracle_fp = projection.get("projected_oracle_fingerprint")
    execution_oracle_fp = execution.get("projected_oracle_fingerprint")
    same_fixed_oracle = (
        execution.get("same_fixed_oracle") is True
        and projection.get("complete_recurring_stream") is True
        and isinstance(projection_oracle_fp, str)
        and projection_oracle_fp == execution_oracle_fp
    )
    single_witness_verified = execution.get("single_witness_receipt_verified") is True

    witness = FullToPrefixSimulationDomainWitness(
        reference_model_conformance=True,
        partition_valid=True,
        saturation_valid=True,
        runtime_schema_valid=True,
        protected_input_independence=(
            projection.get("complete_recurring_stream") is True
            and projection.get("protected_input_independence") is True
        ),
        projected_demands_legal=(
            receptiveness.get("all_projected_demands_legal") is True
            and receptiveness.get("mode_independent_lo_receptiveness") is True
        ),
        complete_prefix_execution_exists=(
            execution.get("single_complete_execution") is True
            and execution.get("time_divergent") is True
            and execution.get("same_fixed_oracle") is True
        ),
        quantifier_order_correct=quantifier_order_correct,
        same_fixed_oracle=same_fixed_oracle,
        single_witness_verified=single_witness_verified,
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
