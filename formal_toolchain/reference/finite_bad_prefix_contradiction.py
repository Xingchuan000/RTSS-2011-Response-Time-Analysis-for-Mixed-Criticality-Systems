from __future__ import annotations

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def build_finite_bad_prefix_contradiction(
    *,
    reference_hi_safety_certificate,
    bad_prefix_reflection_certificate,
    theorem,
    composition_context_hash,
    first_miss_set=None,
    reflected_jobs=None,
):
    status = "PASS" if (
        reference_hi_safety_certificate.get("obligation_status") == "PASS"
        and bad_prefix_reflection_certificate.get("obligation_status") == "PASS"
    ) else "UNRESOLVED"

    bad_witness = bad_prefix_reflection_certificate.get("witness", {})
    first_miss = bad_witness.get("first_miss_set")
    if first_miss is not None:
        miss_time = first_miss.get("miss_time", 0)
        hi_jobs = first_miss.get("jobs", ())
    else:
        miss_time = None
        hi_jobs = ()

    contradiction_preimage = {
        "first_miss_time": miss_time,
        "reflected_hi_jobs": tuple(
            (str(j.get("job_key", j)), j.get("miss_time", 0)) for j in hi_jobs
        ),
        "reference_safety_theorem": reference_hi_safety_certificate.get("inputs", {}).get("theorem", {}).get("theorem_id"),
        "reference_taskset_fingerprint": reference_hi_safety_certificate.get("witness", {}).get("reference_taskset_fingerprint"),
        "reflection_artifact_hash": bad_prefix_reflection_certificate["artifact_hash"],
        "reference_safety_artifact_hash": reference_hi_safety_certificate["artifact_hash"],
    }

    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "premise_reference_hi_safety": reference_hi_safety_certificate["artifact_hash"],
        "premise_bad_prefix_reflection": bad_prefix_reflection_certificate["artifact_hash"],
        "contradiction_preimage": contradiction_preimage,
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
        inputs={"theorem": theorem, "contradiction_preimage": contradiction_preimage},
        witness=witness,
        direct_predecessor_hashes={
            "REFERENCE_HI_SUBSET_SAFETY": reference_hi_safety_certificate["artifact_hash"],
            "HI_BAD_CLOSED_PREFIX_REFLECTION": bad_prefix_reflection_certificate["artifact_hash"],
        },
        checker_id=__name__,
        checker_version="finite-bad-prefix-contradiction-v2",
        failure=None if status == "PASS" else {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "FINITE_BAD_PREFIX_CONTRADICTION_NOT_ESTABLISHED",
        },
    )
