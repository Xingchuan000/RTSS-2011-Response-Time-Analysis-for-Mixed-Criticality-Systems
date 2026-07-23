from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


REFERENCE_TRANSITION_SYSTEM_ID = "FIXED_EXECUTABLE_REFERENCE_P0_V3"


def _reference_taskset_fingerprint(certificate: Mapping[str, Any]) -> str:
    """读取证书已经绑定的 reference taskset fingerprint。"""
    witness = certificate.get("witness", {})
    if isinstance(witness, Mapping):
        value = witness.get("reference_taskset_fingerprint")
        if isinstance(value, str) and value:
            return value
    inputs = certificate.get("inputs", {})
    if isinstance(inputs, Mapping):
        value = inputs.get("reference_taskset_fingerprint")
        if isinstance(value, str) and value:
            return value
    return ""


def build_reference_schedulability_certificate(
    *,
    all_task_rta_certificate: Mapping[str, Any],
    model_conformance_certificate: Mapping[str, Any],
    theorem: Mapping[str, Any],
    reference_context_hash: str,
) -> dict[str, Any]:
    rta_witness = all_task_rta_certificate.get("witness", {})
    rta_fingerprint = _reference_taskset_fingerprint(
        all_task_rta_certificate
    )
    conformance_fingerprint = _reference_taskset_fingerprint(
        model_conformance_certificate
    )
    same_reference_taskset = (
        bool(rta_fingerprint)
        and rta_fingerprint == conformance_fingerprint
    )
    theorem_bound = (
        theorem.get("theorem_id")
        == "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY"
    )
    conformance_witness = model_conformance_certificate.get("witness", {})
    conformance_bound = (
        isinstance(conformance_witness, Mapping)
        and conformance_witness.get("conclusion")
        in {None, "REFERENCE_MODEL_CONFORMS_TO_IMPORTED_THEOREM"}
    )
    status = "PASS" if (
        theorem_bound
        and conformance_bound
        and all_task_rta_certificate.get("obligation_status") == "PASS"
        and model_conformance_certificate.get("obligation_status") == "PASS"
        and rta_witness.get("all_tasks_covered") is True
        and rta_witness.get("all_deadlines_met") is True
        and same_reference_taskset
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
        "reference_taskset_fingerprint": rta_fingerprint,
        "reference_transition_system_id": REFERENCE_TRANSITION_SYSTEM_ID,
        "reference_binding_checked": same_reference_taskset,
        "theorem_binding_checked": theorem_bound,
        "model_conformance_binding_checked": conformance_bound,
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
    schedulability_witness = schedulability_certificate.get(
        "witness", {}
    )
    reference_taskset_fingerprint = (
        _reference_taskset_fingerprint(
            schedulability_certificate
        )
    )
    reference_transition_system_id = (
        schedulability_witness.get(
            "reference_transition_system_id"
        )
        if isinstance(schedulability_witness, Mapping)
        else None
    )
    theorem_bound = (
        theorem.get("theorem_id")
        == "REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY"
    )
    schedulability_conclusion_bound = (
        isinstance(schedulability_witness, Mapping)
        and schedulability_witness.get("conclusion")
        == "REFERENCE_TASKSET_SCHEDULABLE"
    )
    status = "PASS" if (
        theorem_bound
        and schedulability_conclusion_bound
        and schedulability_certificate.get("obligation_status") == "PASS"
        and bool(hi_task_names)
        and bool(reference_taskset_fingerprint)
        and reference_transition_system_id
        == REFERENCE_TRANSITION_SYSTEM_ID
    ) else "UNRESOLVED"
    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "schedulability_artifact_hash": schedulability_certificate.get("artifact_hash"),
        "hi_task_names": sorted(hi_task_names),
        "reference_taskset_fingerprint": reference_taskset_fingerprint,
        "reference_transition_system_id": reference_transition_system_id,
        "theorem_binding_checked": theorem_bound,
        "schedulability_conclusion_bound": schedulability_conclusion_bound,
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
