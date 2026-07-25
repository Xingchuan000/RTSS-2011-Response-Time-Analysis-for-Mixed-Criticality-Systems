from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.budget_domination import build_budget_to_reference_domination

from .predecessor_contract import require_exact_predecessor_set, require_verified_predecessor


def verify_budget_to_reference_domination(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_reference: Any,
    certified_envelope: Mapping[str, Any],
    **_: Any,
) -> dict[str, Any]:
    expected_ids = {
        "CERTIFIED_ENVELOPE",
        "REFERENCE_TASKSET",
        "ACTIVE_RELEASE_BUDGET_INVARIANT",
        "CODE_REFERENCE_UPPER_BOUND_MAPPING",
        "RELEASE_FIXED_REMOVAL_MAPPING",
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
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "BUDGET_REFERENCE_DOMINATION_PREDECESSOR_INVALID", "failure": str(exc)}

    if fresh_reference is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}

    rebuilt = build_budget_to_reference_domination(
        reference_taskset=fresh_reference.to_dict(),
        certified_envelope=certified_envelope,
        reference_context_hash=expected_context_hash,
        direct_predecessor_hashes={
            key: verified_predecessors[key]["artifact_hash"]
            for key in expected_ids
        },
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {
            "status": "UNRESOLVED",
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "BUDGET_REFERENCE_DOMINATION_FAILED",
            "witness": rebuilt,
        }

    candidate_pass = candidate_certificate.get("obligation_status") == "PASS"
    if candidate_pass:
        if candidate_certificate.get("inputs") != rebuilt.get("inputs"):
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "BUDGET_REFERENCE_DOMINATION_INPUT_MISMATCH",
            }
        if candidate_certificate.get("witness") != rebuilt.get("witness"):
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "BUDGET_REFERENCE_DOMINATION_REPLAY_MISMATCH",
            }
        if candidate_certificate.get("direct_predecessor_hashes") != rebuilt.get("direct_predecessor_hashes"):
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "BUDGET_REFERENCE_DOMINATION_PREDECESSOR_MISMATCH",
            }
    if not verify_obligation_certificate(rebuilt):
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "BUDGET_REFERENCE_DOMINATION_REBUILD_INVALID",
        }
    return {"status": "PASS", "route": None, "code": None, "witness": rebuilt}
