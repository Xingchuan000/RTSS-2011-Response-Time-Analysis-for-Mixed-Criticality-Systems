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
    input_projection_receipt = kwargs.get("input_projection_receipt") or {}
    execution_existence_receipt = kwargs.get("execution_existence_receipt") or {}

    witness = FullToPrefixSimulationDomainWitness(
        reference_model_conformance=(
            predecessors["REFERENCE_MODEL_CONFORMANCE"].get("obligation_status") == "PASS"
        ),
        partition_valid=(
            predecessors["PROTECTED_PRIORITY_PREFIX_PARTITION"].get("obligation_status") == "PASS"
        ),
        saturation_valid=(
            predecessors["PROTECTED_PREFIX_LO_SATURATION"].get("obligation_status") == "PASS"
        ),
        runtime_schema_valid=(
            predecessors["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"].get("obligation_status") == "PASS"
        ),
        protected_input_independence=(
            isinstance(input_projection_receipt, dict)
            and input_projection_receipt.get("status") == "PASS"
            and input_projection_receipt.get("complete_recurring_stream") is True
            and input_projection_receipt.get("protected_input_independence") is True
        ),
        projected_demands_legal=(
            isinstance(input_projection_receipt, dict)
            and input_projection_receipt.get("status") == "PASS"
            and input_projection_receipt.get("all_projected_demands_legal") is True
        ),
        complete_prefix_execution_exists=(
            isinstance(execution_existence_receipt, dict)
            and execution_existence_receipt.get("status") == "PASS"
            and execution_existence_receipt.get("single_complete_execution") is True
            and execution_existence_receipt.get("time_divergent") is True
        ),
    )
    flags = asdict(witness)
    payload = {
        **flags,
        "full_taskset_fingerprint": full.to_dict()["fingerprint"],
        "prefix_taskset_fingerprint": prefix.to_dict()["fingerprint"],
        "cutoff_task_name": construction.cutoff_task_name,
        "protected_task_names": list(construction.protected_task_names),
        "tail_task_names": list(construction.tail_task_names),
        "protected_observable_schema_hash": sha256_object(observable_schema()),
        "input_projection_receipt_hash": sha256_object(input_projection_receipt),
        "execution_existence_receipt_hash": sha256_object(execution_existence_receipt),
    }
    if all(flags.values()):
        return {"status": "PASS", "route": None, "code": None, "witness": payload}
    if any(
        predecessors[name].get("obligation_status") == "FAIL"
        for name in required
    ):
        status, code = "FAIL", "FULL_TO_PREFIX_SIMULATION_DOMAIN_PREDECESSOR_FAILED"
    else:
        status, code = "UNRESOLVED", "FULL_TO_PREFIX_SIMULATION_DOMAIN_PROOF_MISSING"
    return {"status": status, "route": "UNRESOLVED", "code": code,
            "witness": {**payload, "missing_flags": [k for k, v in flags.items() if not v]}}
