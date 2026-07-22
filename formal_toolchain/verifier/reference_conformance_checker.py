from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.reference.model_conformance import (
    REQUIRED_CHECK_IDS,
    build_reference_model_conformance_certificate,
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

    expected_ids = set(REQUIRED_CHECK_IDS) | {"REFERENCE_TASKSET"}
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
        contexts = getattr(raw_inputs, "contexts", {})
        for obligation_id in expected_ids:
            if obligation_id == "REFERENCE_TASKSET":
                continue
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

    rebuilt = build_reference_model_conformance_certificate(
        reference_taskset=fresh_reference.to_dict(),
        context_hash=expected_context_hash,
        predecessor_summaries={
            obligation_id: {
                "obligation_id": obligation_id,
                "artifact_hash": cert.get("artifact_hash"),
                "context_hash": cert.get("certificate_context_hash"),
                "verified": True,
                "obligation_status": cert.get("obligation_status"),
            }
            for obligation_id, cert in verified_predecessors.items()
        },
        imported_theorem=load_verified_theory_statement(
            __import__("pathlib").Path(__file__).resolve().parents[1] / "theory",
            "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
        ),
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {
            "status": "UNRESOLVED",
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "REFERENCE_MODEL_CONFORMANCE_FAILED",
            "witness": rebuilt,
        }

    candidate_witness = candidate_certificate.get("witness", {})
    rebuilt_witness = rebuilt.get("witness", {})
    if candidate_certificate.get("obligation_status") == "PASS":
        if candidate_witness.get("check_ids") != list(REQUIRED_CHECK_IDS):
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "REFERENCE_CONFORMANCE_CHECK_SET_MISMATCH",
            }
        if candidate_witness != rebuilt_witness:
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
