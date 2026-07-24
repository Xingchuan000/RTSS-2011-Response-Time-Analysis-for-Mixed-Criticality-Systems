from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.reference.finite_bad_prefix_contradiction import build_finite_bad_prefix_contradiction
from formal_toolchain.theory.loader import load_verified_theory_statement

from .predecessor_contract import require_exact_predecessor_set, require_verified_predecessor


def verify_finite_bad_prefix_contradiction(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_reference: Any,
    **_: Any,
) -> dict[str, Any]:
    expected_ids = {"SELECTED_REFERENCE_HI_SAFETY", "HI_BAD_CLOSED_PREFIX_REFLECTION"}
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=expected_ids)
        contexts = getattr(raw_inputs, "contexts", {})
        for obligation_id in expected_ids:
            require_verified_predecessor(predecessors=verified_predecessors, obligation_id=obligation_id, contexts=contexts)
    except Exception as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "FINITE_BAD_PREFIX_PREDECESSOR_INVALID", "failure": str(exc)}
    theorem = load_verified_theory_statement(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "theory",
        "FINITE_BAD_PREFIX_CONTRADICTION",
    )
    rebuilt = build_finite_bad_prefix_contradiction(
        reference_hi_safety_certificate=verified_predecessors["SELECTED_REFERENCE_HI_SAFETY"],
        bad_prefix_reflection_certificate=verified_predecessors["HI_BAD_CLOSED_PREFIX_REFLECTION"],
        theorem=theorem,
        composition_context_hash=expected_context_hash,
    )
    if rebuilt.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "route": "REFERENCE_CERTIFICATE_FAILED", "code": "FINITE_BAD_PREFIX_CONTRADICTION_NOT_ESTABLISHED", "witness": rebuilt}
    if candidate_certificate.get("obligation_status") == "PASS":
        if candidate_certificate.get("witness") != rebuilt.get("witness") or candidate_certificate.get("inputs") != rebuilt.get("inputs"):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "FINITE_BAD_PREFIX_CONTRADICTION_REPLAY_MISMATCH"}
    if not verify_obligation_certificate(rebuilt):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "FINITE_BAD_PREFIX_CONTRADICTION_REBUILD_INVALID"}
    return {"status": "PASS", "route": None, "code": None, "witness": rebuilt}
