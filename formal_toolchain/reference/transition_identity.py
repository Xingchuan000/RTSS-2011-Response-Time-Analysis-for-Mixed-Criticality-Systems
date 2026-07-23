from __future__ import annotations

import ast
import inspect
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.reference.p0_transition_contract import REFERENCE_P0_CASE_IDS
from formal_toolchain.reference.p0_transition_contract import validate_reference_p0_contract
from formal_toolchain.core.predecessor_contract import validate_verified_predecessor
from formal_toolchain.reference.executable_semantics import step_reference_p0


def _step_enforces_canonical_validator() -> bool:
    tree = ast.parse(inspect.getsource(step_reference_p0))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_executable_reference_p0_step"
        for node in ast.walk(tree)
    )


def build_reference_transition_identity_certificate(
    *,
    reference_taskset: Mapping[str, Any],
    model_bounds: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    context_hash: str,
) -> dict[str, Any]:
    expected = {
        "REFERENCE_TASKSET",
        "REFERENCE_SEMANTICS_CONTRACT",
        "REFERENCE_PREFIX_EXTENSION",
        "EFFECTIVE_EVENT_ORDER",
        "EFFECTIVE_EVENT_FRONTIER_RELATION",
    }
    actual = set(verified_predecessors.keys()) if isinstance(verified_predecessors, Mapping) else set()
    if actual != expected:
        raise ValueError(
            f"REFERENCE_TRANSITION_IDENTITY_PREDECESSOR_SET_MISMATCH: "
            f"expected {expected}, got {actual}"
        )

    for oid in expected:
        cert = verified_predecessors.get(oid, {})
        validate_verified_predecessor(
            predecessors=verified_predecessors,
            obligation_id=oid,
            contexts=contexts,
        )

    if not _step_enforces_canonical_validator():
        raise ValueError("REFERENCE_P0_VALIDATOR_NOT_ENFORCED_BY_STEP")

    contract = validate_reference_p0_contract(model_bounds)
    if contract.get("status") != "PASS":
        raise ValueError("REFERENCE_P0_CONTRACT_INVALID")

    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "p0_transition_contract.py": root / "reference" / "p0_transition_contract.py",
        "p0_projection.py": root / "reference" / "p0_projection.py",
        "executable_semantics.py": root / "reference" / "executable_semantics.py",
        "case_templates.py": root / "bridge" / "case_templates.py",
        "transition_cases.py": root / "bridge" / "transition_cases.py",
    }
    source_hashes = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }

    reference_taskset_fingerprint = ""
    if isinstance(reference_taskset, Mapping):
        reference_taskset_fingerprint = str(reference_taskset.get("fingerprint", ""))

    model_bounds_hash = ""
    if hasattr(model_bounds, "fingerprint"):
        model_bounds_hash = model_bounds.fingerprint
    elif isinstance(model_bounds, Mapping):
        model_bounds_hash = str(model_bounds.get("fingerprint", ""))

    witness = {
        "schema_version": "reference_transition_identity_v3",
        "transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
        "case_ids": list(REFERENCE_P0_CASE_IDS),
        "contract": contract,
        "identity_scope": {
            "core_timing_state": "SHARED_CANONICAL_TRANSITION_CONTRACT",
            "event_frontier": "EFFECTIVE_EVENT_FRONTIER_RELATION",
            "prefix_totality": "REFERENCE_PREFIX_EXTENSION",
        },
        "identity_by_construction": {
            "single_semantics_source": "render_reference_p0_delta",
            "n5_symbolic_consumer": "compile_case_template",
            "executable_consumer": "validate_executable_reference_p0_step",
            "validator_enforced_by_step": True,
            "duplicate_numeric_delta": False,
        },
        "source_bindings": source_hashes,
        "reference_taskset_fingerprint": reference_taskset_fingerprint,
        "model_bounds_hash": model_bounds_hash,
    }

    return obligation_certificate(
        obligation_id="REFERENCE_TRANSITION_SYSTEM_IDENTITY",
        status="PASS",
        context_hash=context_hash,
        inputs={
            "transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
            "reference_taskset_fingerprint": reference_taskset_fingerprint,
            "model_bounds_hash": model_bounds_hash,
        },
        witness=witness,
        direct_predecessor_hashes={
            key: value.get("artifact_hash", sha256_object(value))
            for key, value in verified_predecessors.items()
        },
        checker_id="formal_toolchain.reference.transition_identity",
        checker_version="reference-transition-identity-v3",
    )
