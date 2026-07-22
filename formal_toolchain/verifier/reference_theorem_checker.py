from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.reference.theorem_invocation import (
    build_reference_hi_subset_safety_certificate,
    build_reference_schedulability_certificate,
)
from formal_toolchain.theory.loader import load_verified_theory_statement

from .predecessor_contract import require_exact_predecessor_set, require_verified_predecessor


def _theory(theorem_id: str) -> dict[str, Any]:
    from pathlib import Path
    return load_verified_theory_statement(Path(__file__).resolve().parents[1] / "theory", theorem_id)


def verify_reference_taskset_schedulable(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_reference: Any,
    **_: Any,
) -> dict[str, Any]:
    expected_ids = {"ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_MODEL_CONFORMANCE", "THEORY_LIBRARY_VERSION"}
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
        contexts = getattr(raw_inputs, "contexts", {})
        for obligation_id in ("ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_MODEL_CONFORMANCE"):
            require_verified_predecessor(predecessors=verified_predecessors, obligation_id=obligation_id, contexts=contexts)
    except Exception as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_TASKSET_SCHEDULABLE_PREDECESSOR_INVALID", "failure": str(exc)}
    if fresh_reference is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    theorem = _theory("C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY")
    rebuilt = build_reference_schedulability_certificate(
        all_task_rta_certificate=verified_predecessors["ALL_TASK_REFERENCE_RTA_ARITHMETIC"],
        model_conformance_certificate=verified_predecessors["REFERENCE_MODEL_CONFORMANCE"],
        theorem={**theorem, "library_version_hash": verified_predecessors["THEORY_LIBRARY_VERSION"]["artifact_hash"]},
        reference_context_hash=expected_context_hash,
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "route": "REFERENCE_CERTIFICATE_FAILED", "code": "ALL_TASK_SUFFICIENT_CERTIFICATE_NOT_ESTABLISHED", "witness": rebuilt}
    if candidate_certificate.get("obligation_status") == "PASS":
        if candidate_certificate.get("witness") != rebuilt.get("witness") or candidate_certificate.get("inputs") != rebuilt.get("inputs"):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_TASKSET_SCHEDULABLE_REPLAY_MISMATCH"}
    if not verify_obligation_certificate(rebuilt):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_TASKSET_SCHEDULABLE_REBUILD_INVALID"}
    return {"status": "PASS", "route": None, "code": None, "witness": rebuilt}


def verify_reference_hi_subset_safety(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_reference: Any,
    **_: Any,
) -> dict[str, Any]:
    expected_ids = {"REFERENCE_TASKSET_SCHEDULABLE"}
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
    except Exception as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_HI_SUBSET_PREDECESSOR_INVALID", "failure": str(exc)}
    if fresh_reference is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    theorem = _theory("REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY")
    hi_task_names = [task.name for task in fresh_reference.tasks if task.criticality == "HI"]
    rebuilt = build_reference_hi_subset_safety_certificate(
        schedulability_certificate=verified_predecessors["REFERENCE_TASKSET_SCHEDULABLE"],
        theorem=theorem,
        reference_context_hash=expected_context_hash,
        hi_task_names=hi_task_names,
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "route": "REFERENCE_CERTIFICATE_FAILED", "code": "REFERENCE_SCHEDULABILITY_NOT_ESTABLISHED", "witness": rebuilt}
    if candidate_certificate.get("obligation_status") == "PASS":
        if candidate_certificate.get("witness") != rebuilt.get("witness") or candidate_certificate.get("inputs") != rebuilt.get("inputs"):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_HI_SUBSET_SAFETY_REPLAY_MISMATCH"}
    if not verify_obligation_certificate(rebuilt):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_HI_SUBSET_SAFETY_REBUILD_INVALID"}
    return {"status": "PASS", "route": None, "code": None, "witness": rebuilt}
