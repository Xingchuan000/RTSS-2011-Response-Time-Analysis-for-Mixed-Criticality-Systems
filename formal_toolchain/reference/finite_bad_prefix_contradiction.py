from __future__ import annotations

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def build_finite_bad_prefix_contradiction(
    *,
    reference_hi_safety_certificate,
    bad_prefix_reflection_certificate,
    theorem,
    composition_context_hash,
):
    status = "PASS" if (
        reference_hi_safety_certificate.get("obligation_status") == "PASS"
        and bad_prefix_reflection_certificate.get("obligation_status") == "PASS"
    ) else "UNRESOLVED"

    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "premise_reference_hi_safety": reference_hi_safety_certificate["artifact_hash"],
        "premise_bad_prefix_reflection": bad_prefix_reflection_certificate["artifact_hash"],
        "contradiction": (
            "ConcreteHIMiss -> ReferenceFiniteHIMissPrefix "
            "and ReferenceHISafety -> not ReferenceFiniteHIMissPrefix"
        ),
        "conclusion": "NO_CONCRETE_HI_DEADLINE_MISS",
    }
    witness["composition_hash"] = sha256_object(witness)

    return obligation_certificate(
        obligation_id="FINITE_BAD_PREFIX_CONTRADICTION",
        status=status,
        context_hash=composition_context_hash,
        inputs={"theorem": theorem},
        witness=witness,
        direct_predecessor_hashes={
            "REFERENCE_HI_SUBSET_SAFETY": reference_hi_safety_certificate["artifact_hash"],
            "HI_BAD_CLOSED_PREFIX_REFLECTION": bad_prefix_reflection_certificate["artifact_hash"],
        },
        checker_id=__name__,
        checker_version="finite-bad-prefix-contradiction-v1",
        failure=None if status == "PASS" else {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "FINITE_BAD_PREFIX_CONTRADICTION_NOT_ESTABLISHED",
        },
    )
