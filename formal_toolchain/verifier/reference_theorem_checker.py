from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.reference.theorem_invocation import (
    build_reference_hi_subset_safety_certificate,
    build_reference_schedulability_certificate,
)
from formal_toolchain.theory.loader import load_verified_theory_statement
from formal_toolchain.theory.loader import registry_dependencies_for_theorem

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
    theorem = _theory("C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY")
    expected_ids = set(theorem["premise_obligation_ids"]) | (
        registry_dependencies_for_theorem(
            __import__("pathlib").Path(__file__).resolve().parents[1] / "theory",
            theorem["theorem_id"],
        ) - set(theorem["premise_obligation_ids"])
    )
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
        contexts = getattr(raw_inputs, "contexts", {})
        for obligation_id in ("ALL_TASK_REFERENCE_RTA_ARITHMETIC", "REFERENCE_MODEL_CONFORMANCE"):
            require_verified_predecessor(predecessors=verified_predecessors, obligation_id=obligation_id, contexts=contexts)
    except Exception as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_TASKSET_SCHEDULABLE_PREDECESSOR_INVALID", "failure": str(exc)}
    if fresh_reference is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}

    rta = verified_predecessors.get("ALL_TASK_REFERENCE_RTA_ARITHMETIC", {})
    rta_witness = rta.get("witness", {})
    conformance = verified_predecessors.get("REFERENCE_MODEL_CONFORMANCE", {})
    conformance_witness = conformance.get("witness", {})

    rta_schema = rta_witness.get("schema_version")
    if rta_schema != "all_task_rta_v3":
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "RTA_SCHEMA_VERSION_MISMATCH",
                "witness": {"expected": "all_task_rta_v3", "actual": rta_schema}}

    expected_task_count = len(fresh_reference.tasks) if hasattr(fresh_reference, "tasks") else 0
    if rta_witness.get("task_count_expected") != expected_task_count:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "RTA_TASK_COUNT_MISMATCH",
                "witness": {"expected": expected_task_count, "actual": rta_witness.get("task_count_expected")}}

    conformance_contract_hash = conformance_witness.get("contract_hash")
    theorem_premises = theorem.get("premise_obligation_ids")
    if not isinstance(theorem_premises, list) or conformance_witness.get("imported_theorem_id") != theorem.get("theorem_id"):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "CONFORMANCE_THEOREM_PREMISE_MISMATCH",
                "witness": {"conformance_contract_hash": conformance_contract_hash,
                            "theorem_premises": theorem_premises}}

    if rta_witness.get("all_tasks_covered") is not True:
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "RTA_NOT_ALL_TASKS_COVERED"}
    if rta_witness.get("all_deadlines_met") is not True:
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "RTA_NOT_ALL_DEADLINES_MET"}

    rebuilt = build_reference_schedulability_certificate(
        all_task_rta_certificate=rta,
        model_conformance_certificate=conformance,
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
    schedulability = verified_predecessors["REFERENCE_TASKSET_SCHEDULABLE"]
    schedulability_witness = schedulability.get("witness", {})
    if schedulability.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "SCHEDULABILITY_NOT_PASS", "witness": schedulability}
    theorem = _theory("REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY")
    hi_task_names = [task.name for task in fresh_reference.tasks if task.criticality == "HI"]
    if not hi_task_names:
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "NO_HI_TASKS_IN_REFERENCE_TASKSET"}
    if schedulability_witness.get("hi_task_names"):
        expected_hi = set(hi_task_names)
        actual_hi = set(schedulability_witness["hi_task_names"])
        if expected_hi != actual_hi:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "HI_TASK_NAME_MISMATCH",
                    "witness": {"expected": sorted(expected_hi), "actual": sorted(actual_hi)}}
    rebuilt = build_reference_hi_subset_safety_certificate(
        schedulability_certificate=schedulability,
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
