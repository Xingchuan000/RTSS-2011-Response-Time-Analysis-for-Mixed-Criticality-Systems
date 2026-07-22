from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def build_reference_schedulability_certificate(
    *,
    all_task_rta_certificate: Mapping[str, Any],
    model_conformance_certificate: Mapping[str, Any],
    theorem: Mapping[str, Any],
    reference_context_hash: str,
) -> dict[str, Any]:
    rta_witness = all_task_rta_certificate.get("witness", {})
    status = "PASS" if (
        all_task_rta_certificate.get("obligation_status") == "PASS"
        and model_conformance_certificate.get("obligation_status") == "PASS"
        and rta_witness.get("all_tasks_covered") is True
        and rta_witness.get("all_deadlines_met") is True
    ) else "UNRESOLVED"

    direct = {
        "ALL_TASK_REFERENCE_RTA_ARITHMETIC": all_task_rta_certificate["artifact_hash"],
        "REFERENCE_MODEL_CONFORMANCE": model_conformance_certificate["artifact_hash"],
        "THEORY_LIBRARY_VERSION": theorem["library_version_hash"],
    }
    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "assumption_hash": theorem["assumption_hash"],
        "all_task_rta_artifact_hash": all_task_rta_certificate["artifact_hash"],
        "model_conformance_artifact_hash": model_conformance_certificate["artifact_hash"],
        "rta_schema_version": rta_witness.get("schema_version"),
        "conclusion": "REFERENCE_TASKSET_SCHEDULABLE",
    }
    witness["invocation_hash"] = sha256_object(witness)

    return obligation_certificate(
        obligation_id="REFERENCE_TASKSET_SCHEDULABLE",
        status=status,
        context_hash=reference_context_hash,
        inputs={"theorem": theorem},
        witness=witness,
        direct_predecessor_hashes=direct,
        checker_id=__name__,
        checker_version="reference-theorem-invocation-v1",
        failure=None if status == "PASS" else {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "ALL_TASK_SUFFICIENT_CERTIFICATE_NOT_ESTABLISHED",
        },
    )


def build_reference_hi_subset_safety_certificate(
    *,
    schedulability_certificate: Mapping[str, Any],
    theorem: Mapping[str, Any],
    reference_context_hash: str,
    hi_task_names: list[str],
) -> dict[str, Any]:
    status = "PASS" if (
        schedulability_certificate.get("obligation_status") == "PASS"
        and bool(hi_task_names)
    ) else "UNRESOLVED"
    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "schedulability_artifact_hash": schedulability_certificate.get("artifact_hash"),
        "hi_task_names": sorted(hi_task_names),
        "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
    }
    witness["invocation_hash"] = sha256_object(witness)
    return obligation_certificate(
        obligation_id="REFERENCE_HI_SUBSET_SAFETY",
        status=status,
        context_hash=reference_context_hash,
        inputs={"theorem": theorem},
        witness=witness,
        direct_predecessor_hashes={
            "REFERENCE_TASKSET_SCHEDULABLE": schedulability_certificate["artifact_hash"]
        },
        checker_id=__name__,
        checker_version="reference-hi-subset-v1",
        failure=None if status == "PASS" else {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "REFERENCE_SCHEDULABILITY_NOT_ESTABLISHED",
        },
    )
