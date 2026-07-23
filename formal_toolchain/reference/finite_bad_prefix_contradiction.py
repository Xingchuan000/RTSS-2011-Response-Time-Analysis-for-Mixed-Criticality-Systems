from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


REFERENCE_TRANSITION_SYSTEM_ID = "FIXED_EXECUTABLE_REFERENCE_P0_V3"


def _witness(certificate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = certificate.get("witness", {})
    return value if isinstance(value, Mapping) else {}


def build_finite_bad_prefix_contradiction(
    *,
    reference_hi_safety_certificate,
    bad_prefix_reflection_certificate,
    theorem,
    composition_context_hash,
    first_miss_set=None,
    reflected_jobs=None,
):
    del first_miss_set, reflected_jobs

    safety_pass = (
        reference_hi_safety_certificate.get(
            "obligation_status"
        )
        == "PASS"
    )
    reflection_pass = (
        bad_prefix_reflection_certificate.get(
            "obligation_status"
        )
        == "PASS"
    )

    safety_witness = _witness(
        reference_hi_safety_certificate
    )
    reflection_witness = _witness(
        bad_prefix_reflection_certificate
    )

    theorem_bound = theorem.get("theorem_id") == "FINITE_BAD_PREFIX_CONTRADICTION"
    safety_theorem_bound = (
        safety_witness.get("theorem_id")
        == "REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY"
        and safety_witness.get("conclusion")
        == "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES"
    )
    reflection_theorem_bound = (
        reflection_witness.get("theorem_id")
        == "FINITE_HI_BAD_PREFIX_REFLECTION"
        and reflection_witness.get("quantification")
        == "FOR_ALL_FIRST_FINITE_CONCRETE_HI_BAD_CLOSED_PREFIXES"
    )

    safety_fingerprint = safety_witness.get(
        "reference_taskset_fingerprint"
    )
    reflection_fingerprint = reflection_witness.get(
        "reference_taskset_fingerprint"
    )
    safety_system_id = safety_witness.get(
        "reference_transition_system_id"
    )
    reflection_system_id = reflection_witness.get(
        "reference_transition_system_id"
    )

    reference_binding_complete = all(
        isinstance(value, str) and bool(value)
        for value in (
            safety_fingerprint,
            reflection_fingerprint,
            safety_system_id,
            reflection_system_id,
        )
    )
    same_reference_taskset = (
        reference_binding_complete
        and safety_fingerprint
        == reflection_fingerprint
    )
    same_reference_system = (
        reference_binding_complete
        and safety_system_id
        == reflection_system_id
        == REFERENCE_TRANSITION_SYSTEM_ID
    )

    if safety_pass and reflection_pass:
        status = (
            "PASS"
            if theorem_bound
            and safety_theorem_bound
            and reflection_theorem_bound
            and same_reference_taskset
            and same_reference_system
            else "FAIL"
        )
    else:
        status = "UNRESOLVED"

    first_miss = reflection_witness.get(
        "first_miss_set"
    )
    if isinstance(first_miss, Mapping):
        miss_time = first_miss.get("miss_time")
        hi_jobs = first_miss.get("jobs", ())
    else:
        miss_time = None
        hi_jobs = ()

    contradiction_preimage = {
        "first_miss_time": miss_time,
        "reflected_hi_jobs": tuple(
            (
                str(job.get("job_key", job)),
                job.get("miss_time", 0),
            )
            for job in hi_jobs
            if isinstance(job, Mapping)
        ),
        "reference_safety_theorem": (
            reference_hi_safety_certificate.get(
                "inputs", {}
            )
            .get("theorem", {})
            .get("theorem_id")
        ),
        "reference_taskset_fingerprint": (
            safety_fingerprint
        ),
        "reflection_reference_taskset_fingerprint": (
            reflection_fingerprint
        ),
        "reference_transition_system_id": (
            safety_system_id
        ),
        "reflection_reference_transition_system_id": (
            reflection_system_id
        ),
        "same_reference_taskset": (
            same_reference_taskset
        ),
        "same_reference_system": (
            same_reference_system
        ),
        "reflection_artifact_hash": (
            bad_prefix_reflection_certificate[
                "artifact_hash"
            ]
        ),
        "reference_safety_artifact_hash": (
            reference_hi_safety_certificate[
                "artifact_hash"
            ]
        ),
    }

    witness = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "premise_reference_hi_safety": (
            reference_hi_safety_certificate[
                "artifact_hash"
            ]
        ),
        "premise_bad_prefix_reflection": (
            bad_prefix_reflection_certificate[
                "artifact_hash"
            ]
        ),
        "reference_binding": {
            "taskset_fingerprint": safety_fingerprint,
            "transition_system_id": safety_system_id,
            "same_reference_taskset": same_reference_taskset,
            "same_reference_system": same_reference_system,
        },
        "theorem_binding": {
            "composition_theorem": theorem_bound,
            "reference_safety_theorem": safety_theorem_bound,
            "bad_prefix_reflection_theorem": reflection_theorem_bound,
        },
        "contradiction_preimage": contradiction_preimage,
        "contradiction": (
            "ConcreteHIMiss -> ReferenceFiniteHIMissPrefix "
            "and ReferenceHISafety -> "
            "not ReferenceFiniteHIMissPrefix"
        ),
        "conclusion": "NO_CONCRETE_HI_DEADLINE_MISS",
    }
    witness["composition_hash"] = sha256_object(
        witness
    )

    if status == "PASS":
        failure = None
    elif status == "FAIL":
        binding_failure = not (
            same_reference_taskset and same_reference_system
        )
        failure = {
            "route": (
                "MODEL_CONFORMANCE_FAILED"
                if binding_failure
                else "REFERENCE_CERTIFICATE_FAILED"
            ),
            "code": (
                "FINITE_BAD_PREFIX_REFERENCE_BINDING_MISMATCH"
                if binding_failure
                else "FINITE_BAD_PREFIX_THEOREM_BINDING_MISMATCH"
            ),
        }
    else:
        failure = {
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": (
                "FINITE_BAD_PREFIX_CONTRADICTION_"
                "NOT_ESTABLISHED"
            ),
        }

    return obligation_certificate(
        obligation_id="FINITE_BAD_PREFIX_CONTRADICTION",
        status=status,
        context_hash=composition_context_hash,
        inputs={
            "theorem": theorem,
            "contradiction_preimage": (
                contradiction_preimage
            ),
        },
        witness=witness,
        direct_predecessor_hashes={
            "REFERENCE_HI_SUBSET_SAFETY": (
                reference_hi_safety_certificate[
                    "artifact_hash"
                ]
            ),
            "HI_BAD_CLOSED_PREFIX_REFLECTION": (
                bad_prefix_reflection_certificate[
                    "artifact_hash"
                ]
            ),
        },
        checker_id=__name__,
        checker_version=(
            "finite-bad-prefix-contradiction-v3"
        ),
        failure=failure,
    )
