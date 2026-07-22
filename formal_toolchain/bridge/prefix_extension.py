from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.core.predecessor_contract import validate_verified_predecessor
from formal_toolchain.theory.backends.reference_prefix_extension import EXPECTED_CASE_IDS, EXPECTED_SOLVER_OBLIGATIONS


def _receipt_is_valid(
    receipt: Mapping[str, Any], theorem: Mapping[str, Any], proof_path: Path,
) -> bool:
    if receipt.get("status") != "PASS" or receipt.get("backend_id") != "reference-prefix-extension-z3-v3":
        return False
    required = {
        "backend_id", "proof_object_hash", "theorem_statement_hash",
        "theorem_assumption_hash", "source_bindings", "case_ids",
        "solver_obligations", "z3_version",
    }
    if not required <= set(receipt):
        return False
    if receipt.get("proof_object_hash") != sha256_file(proof_path):
        return False
    if receipt.get("proof_object_hash") != theorem.get("proof_object", {}).get("sha256"):
        return False
    if receipt.get("theorem_statement_hash") != theorem.get("statement_hash"):
        return False
    if receipt.get("theorem_assumption_hash") != theorem.get("assumption_hash"):
        return False
    if receipt.get("case_ids") != list(EXPECTED_CASE_IDS):
        return False
    if set(receipt.get("solver_obligations", {})) != set(EXPECTED_SOLVER_OBLIGATIONS):
        return False
    body = {key: receipt[key] for key in required}
    return receipt.get("receipt_hash") == sha256_object(body)


def build_parameterized_prefix_extension_certificate(
    *, reference_taskset: Mapping[str, Any],
    reference_taskset_certificate: Mapping[str, Any],
    time_progress_certificate: Mapping[str, Any],
    event_order_certificate: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]],
    context_hash: str,
    theorem_statement: Mapping[str, Any],
    theorem_proof_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    predecessors = {
        "REFERENCE_TASKSET": reference_taskset_certificate,
        "TIME_PROGRESS": time_progress_certificate,
        "EFFECTIVE_EVENT_ORDER": event_order_certificate,
    }
    if set(predecessors) != {"REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER"}:
        raise ValueError("REFERENCE_PREFIX_PREDECESSOR_SET_MISMATCH")
    for obligation_id in predecessors:
        validate_verified_predecessor(
            predecessors=predecessors, obligation_id=obligation_id, contexts=contexts,
        )
    fresh_fingerprint = reference_taskset.get("fingerprint")
    certificate_fingerprint = reference_taskset_certificate.get("witness", {}).get("reference_taskset", {}).get("fingerprint")
    if certificate_fingerprint != fresh_fingerprint:
        raise ValueError("REFERENCE_PREFIX_REFERENCE_TASKSET_FINGERPRINT_MISMATCH")
    if theorem_statement.get("theorem_id") != "REFERENCE_PREFIX_EXTENSION":
        raise ValueError("THEOREM_MUST_BE_REFERENCE_PREFIX_EXTENSION")
    proof_object = theorem_statement.get("proof_object", {})
    proof_path = (Path(__file__).resolve().parents[1] / "theory" / proof_object.get("path", "")).resolve()
    if not proof_path.is_file() or not _receipt_is_valid(theorem_proof_receipt, theorem_statement, proof_path):
        raise ValueError("REFERENCE_PREFIX_THEOREM_RECEIPT_INVALID")
    tasks = reference_taskset.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("REFERENCE_PREFIX_REFERENCE_TASKSET_EMPTY")
    for task in tasks:
        period = int(task.get("period", 0))
        deadline = int(task.get("deadline", -1))
        offset = int(task.get("offset", 0))
        if period <= 0 or not 0 < deadline <= period or not 0 <= offset < period:
            raise ValueError("REFERENCE_PREFIX_TASK_PERIOD_DEADLINE_OFFSET_INVALID")
    ref_state_source_hash = sha256_file(Path(__file__).resolve().parents[1] / "reference" / "reference_state.py")
    exec_source_hash = sha256_file(Path(__file__).resolve().parents[1] / "reference" / "executable_semantics.py")
    receipt_hash = theorem_proof_receipt["receipt_hash"]
    return obligation_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION", status="PASS", context_hash=context_hash,
        inputs={
            "theorem_id": "REFERENCE_PREFIX_EXTENSION",
            "reference_taskset_fingerprint": fresh_fingerprint,
            "theorem_statement_hash": theorem_statement["statement_hash"],
            "theorem_assumption_hash": theorem_statement["assumption_hash"],
            "theorem_proof_object_hash": proof_object["sha256"],
            "reference_state_source_hash": ref_state_source_hash,
            "executable_semantics_source_hash": exec_source_hash,
        },
        witness={
            "schema_version": "reference_prefix_extension_v4",
            "closed_state_predicate": "is_closed_reference_state",
            "case_ids": list(EXPECTED_CASE_IDS),
            "backend_receipt_hash": receipt_hash,
            "theorem_proof_receipt": dict(theorem_proof_receipt),
            "reference_context_hash": context_hash,
            "predecessor_context_layers": {
                obligation_id: contexts.get({
                    "REFERENCE_TASKSET": "reference_context",
                    "TIME_PROGRESS": "semantic_context",
                    "EFFECTIVE_EVENT_ORDER": "semantic_context",
                }[obligation_id], {}).get("hash")
                for obligation_id in predecessors
            },
        },
        direct_predecessor_hashes={key: value["artifact_hash"] for key, value in predecessors.items()},
        checker_id=__name__, checker_version="prefix-extension-v5",
    )
