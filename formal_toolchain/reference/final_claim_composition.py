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
    status = "PASS" if (
        finite_bad_prefix_contradiction_certificate.get("obligation_status") == "PASS"
    ) else "UNRESOLVED"

    witness = {
        "schema_version": "final_claim_composition_v1",
        "mathematical_root_id": "FINAL_CLAIM_COMPOSITION",
        "claim": claim,
        "theorem_id": theorem.get("theorem_id"),
        "statement_hash": theorem.get("statement_hash"),
        "finite_contradiction_hash": finite_bad_prefix_contradiction_certificate["artifact_hash"],
        "composition": "FINAL_CLAIM_COMPOSITION <= FINITE_BAD_PREFIX_CONTRADICTION",
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
