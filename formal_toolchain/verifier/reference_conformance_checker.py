from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.reference.model_conformance import (
    build_reference_model_conformance_certificate,
    load_reference_model_conformance_contract,
)
from formal_toolchain.theory.loader import load_verified_theory_statement

from .predecessor_contract import require_exact_predecessor_set, require_verified_predecessor


def verify_reference_model_conformance(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_reference: Any,
    **_: Any,
) -> dict[str, Any]:
    if fresh_reference is None:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "code": "FRESH_REFERENCE_TASKSET_MISSING",
        }

    theorem = load_verified_theory_statement(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "theory",
        "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
    )
    contract = load_reference_model_conformance_contract()
    expected_ids = {
        predecessor_id
        for condition in contract["conditions"]
        for predecessor_id in condition["predecessor_obligation_ids"]
    }
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
        contexts = getattr(raw_inputs, "contexts", {})
        for obligation_id in expected_ids:
            require_verified_predecessor(
                predecessors=verified_predecessors,
                obligation_id=obligation_id,
                contexts=contexts,
            )
    except Exception as exc:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "REFERENCE_MODEL_CONFORMANCE_PREDECESSOR_INVALID",
            "failure": str(exc),
        }

    fresh_fingerprint = fresh_reference.to_dict().get("fingerprint")
    reference_certificate = verified_predecessors["REFERENCE_TASKSET"]
    reference_witness = reference_certificate.get("witness", {})
    reference_taskset = reference_witness.get("reference_taskset", {})
    certificate_fingerprint = reference_taskset.get("fingerprint")
    if certificate_fingerprint != fresh_fingerprint:
        return {
            "status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
            "code": "REFERENCE_TASKSET_FINGERPRINT_MISMATCH",
            "failure": {"certificate": certificate_fingerprint, "fresh": fresh_fingerprint},
        }
    rebuilt = build_reference_model_conformance_certificate(
        reference_taskset=fresh_reference.to_dict(),
        context_hash=expected_context_hash,
        verified_predecessors=verified_predecessors,
        contexts=getattr(raw_inputs, "contexts", {}),
        conformance_contract=contract,
        imported_theorem=theorem,
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {
            "status": "UNRESOLVED",
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "REFERENCE_MODEL_CONFORMANCE_FAILED",
            "witness": rebuilt,
        }

    if candidate_certificate.get("obligation_status") == "PASS":
        from formal_toolchain.core.artifact import verify_obligation_certificate
        if (not verify_obligation_certificate(candidate_certificate)
                or candidate_certificate != rebuilt):
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "REFERENCE_CONFORMANCE_REPLAY_MISMATCH",
            }
    return {
        "status": "PASS",
        "route": None,
        "code": None,
        "witness": rebuilt,
    }
