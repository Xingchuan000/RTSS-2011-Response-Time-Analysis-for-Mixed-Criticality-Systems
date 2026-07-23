from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def build_final_claim_composition(
    *,
    finite_bad_prefix_contradiction_certificate: Mapping[str, Any],
    theorem: Mapping[str, Any],
    composition_context_hash: str,
    claim: str = "DEPLOYED_HI_SAFETY",
) -> dict[str, Any]:
    contradiction_witness = finite_bad_prefix_contradiction_certificate.get(
        "witness", {}
    )
    theorem_bound = (
        theorem.get("theorem_id")
        == "FINAL_DEPLOYED_HI_SAFETY_COMPOSITION"
    )
    contradiction_bound = (
        isinstance(contradiction_witness, Mapping)
        and contradiction_witness.get("theorem_id")
        == "FINITE_BAD_PREFIX_CONTRADICTION"
        and contradiction_witness.get("conclusion")
        == "NO_CONCRETE_HI_DEADLINE_MISS"
    )
    claim_bound = claim == "DEPLOYED_HI_SAFETY"
    status = "PASS" if (
        theorem_bound
        and contradiction_bound
        and claim_bound
        and finite_bad_prefix_contradiction_certificate.get("obligation_status") == "PASS"
    ) else "UNRESOLVED"

    witness = {
        "schema_version": "final_claim_composition_v1",
        "mathematical_root_id": "FINAL_CLAIM_COMPOSITION",
        "claim": claim,
        "theorem_id": theorem.get("theorem_id"),
        "statement_hash": theorem.get("statement_hash"),
        "finite_contradiction_hash": finite_bad_prefix_contradiction_certificate["artifact_hash"],
        "composition": "FINAL_CLAIM_COMPOSITION <= FINITE_BAD_PREFIX_CONTRADICTION",
        "theorem_binding_checked": theorem_bound,
        "contradiction_conclusion_bound": contradiction_bound,
        "claim_binding_checked": claim_bound,
    }

    return obligation_certificate(
        obligation_id="FINAL_CLAIM_COMPOSITION",
        status=status,
        context_hash=composition_context_hash,
        inputs={"theorem": theorem, "claim": claim},
        witness=witness,
        direct_predecessor_hashes={
            "FINITE_BAD_PREFIX_CONTRADICTION": finite_bad_prefix_contradiction_certificate["artifact_hash"],
        },
        checker_id=__name__,
        checker_version="final-claim-composition-v1",
        failure=None if status == "PASS" else {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "FINAL_CLAIM_COMPOSITION_NOT_ESTABLISHED",
        },
    )
