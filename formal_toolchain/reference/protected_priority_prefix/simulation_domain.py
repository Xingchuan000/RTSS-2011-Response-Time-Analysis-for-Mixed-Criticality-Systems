"""The quantified input/state domain for full-to-prefix simulation."""

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


def check_full_to_prefix_simulation_domain(**kwargs: Any) -> dict[str, Any]:
    state = getattr(kwargs.get("context"), "fresh_state", None) or kwargs.get("fresh_state")
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "REFERENCE_MODEL_CONFORMANCE", "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "PROTECTED_PREFIX_LO_SATURATION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
    }
    if set(predecessors) != required:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PREDECESSOR_SET_MISMATCH",
                "witness": {"expected": sorted(required), "actual": sorted(predecessors)}}
    if state is None or state.prepared_route is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PROTECTED_PREFIX_TASKSET_MISSING", "witness": {}}
    construction: ProtectedPrefixBuildResult = state.prepared_route.construction_witnesses["build_result"]
    full = state.full_reference_taskset
    prefix = state.analysis_taskset
    full_initial = initial_reference_state(full)
    projected = project_protected_release_stream(
        full_initial, protected_task_names=frozenset(construction.protected_task_names))
    legality = check_projected_demands_legal(projected, prefix)
    partition = construction.partition_witness
    saturation = construction.saturation_witness
    runtime = build_runtime_schema_certificate()
    try:
        prefix_initial = initial_reference_state(prefix)
        close_timestamp(prefix_initial, prefix)
        execution_exists = True
    except (TypeError, ValueError, RuntimeError):
        execution_exists = False
    witness = FullToPrefixSimulationDomainWitness(
        reference_model_conformance=predecessors["REFERENCE_MODEL_CONFORMANCE"].get("obligation_status") == "PASS",
        partition_valid=all(partition.get(k) is True for k in ("tail_all_lo", "all_hi_protected", "partition_complete", "order_preserved")),
        saturation_valid=saturation.get("hi_fields_equal") is True and saturation.get("timing_fields_equal") is True,
        runtime_schema_valid=runtime["status"] == "PASS",
        protected_input_independence=all(item.job_key[0] in construction.protected_task_names for item in projected),
        projected_demands_legal=legality["status"] == "PASS",
        complete_prefix_execution_exists=execution_exists,
    )
    payload = asdict(witness)
    payload.update({
        "full_taskset_fingerprint": full.to_dict()["fingerprint"],
        "prefix_taskset_fingerprint": prefix.to_dict()["fingerprint"],
        "cutoff_task_name": construction.cutoff_task_name,
        "protected_task_names": list(construction.protected_task_names),
        "tail_task_names": list(construction.tail_task_names),
        "input_projection_schema_hash": sha256_object({"version": "protected-release-input-v1", "fields": list(ProtectedInputFields)}),
        "protected_observable_schema_hash": sha256_object(observable_schema()),
        "runtime_schema_certificate_hash": runtime["certificate_hash"],
        "projected_inputs": [asdict(item) for item in projected],
        "demand_legality": legality,
    })
    ok = all(asdict(witness).values())
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "UNRESOLVED",
            "code": None if ok else "FULL_TO_PREFIX_SIMULATION_DOMAIN_FAILED", "witness": payload}


ProtectedInputFields = ("job_key", "task_name", "release_time", "actual_demand", "hi_class")
