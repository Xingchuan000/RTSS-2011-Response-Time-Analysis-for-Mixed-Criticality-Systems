from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.reference.finite_bad_prefix_contradiction import (
    build_finite_bad_prefix_contradiction,
)


SYSTEM_ID = "FIXED_EXECUTABLE_REFERENCE_P0_V3"


def _certificate(obligation_id: str, *, fingerprint: str):
    if obligation_id == "SELECTED_REFERENCE_HI_SAFETY":
        witness = {
            "theorem_id": "SELECTED_REFERENCE_HI_SAFETY",
            "route_id": "strict_full",
            "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
        }
    else:
        witness = {
            "theorem_id": "FINITE_HI_BAD_PREFIX_REFLECTION",
            "quantification": "FOR_ALL_FIRST_FINITE_CONCRETE_HI_BAD_CLOSED_PREFIXES",
            "first_miss_set": None,
        }
    witness.update({
        "reference_taskset_fingerprint": fingerprint,
        "reference_transition_system_id": SYSTEM_ID,
    })
    return obligation_certificate(
        obligation_id=obligation_id,
        status="PASS",
        context_hash="a" * 64,
        inputs={"theorem": {"theorem_id": witness["theorem_id"]}},
        witness=witness,
        checker_id="test",
        checker_version="1",
    )


def _theorem():
    return {
        "theorem_id": "FINITE_BAD_PREFIX_CONTRADICTION",
        "statement_hash": "b" * 64,
    }


def test_contradiction_requires_same_reference_taskset():
    result = build_finite_bad_prefix_contradiction(
        reference_hi_safety_certificate=_certificate(
            "SELECTED_REFERENCE_HI_SAFETY",
            fingerprint="1" * 64,
        ),
        bad_prefix_reflection_certificate=_certificate(
            "HI_BAD_CLOSED_PREFIX_REFLECTION",
            fingerprint="2" * 64,
        ),
        theorem=_theorem(),
        composition_context_hash="c" * 64,
    )

    assert result["obligation_status"] == "FAIL"
    assert result["failure"]["code"] == (
        "FINITE_BAD_PREFIX_REFERENCE_BINDING_MISMATCH"
    )


def test_contradiction_passes_for_same_reference_model():
    result = build_finite_bad_prefix_contradiction(
        reference_hi_safety_certificate=_certificate(
            "SELECTED_REFERENCE_HI_SAFETY",
            fingerprint="1" * 64,
        ),
        bad_prefix_reflection_certificate=_certificate(
            "HI_BAD_CLOSED_PREFIX_REFLECTION",
            fingerprint="1" * 64,
        ),
        theorem=_theorem(),
        composition_context_hash="c" * 64,
    )

    assert result["obligation_status"] == "PASS"
    binding = result["witness"]["reference_binding"]
    assert binding["same_reference_taskset"] is True
    assert binding["same_reference_system"] is True


def test_contradiction_rejects_wrong_composition_theorem():
    theorem = _theorem()
    theorem["theorem_id"] = "WRONG_THEOREM"
    result = build_finite_bad_prefix_contradiction(
        reference_hi_safety_certificate=_certificate(
            "SELECTED_REFERENCE_HI_SAFETY", fingerprint="1" * 64
        ),
        bad_prefix_reflection_certificate=_certificate(
            "HI_BAD_CLOSED_PREFIX_REFLECTION", fingerprint="1" * 64
        ),
        theorem=theorem,
        composition_context_hash="c" * 64,
    )
    assert result["obligation_status"] == "FAIL"
    assert result["failure"]["code"] == "FINITE_BAD_PREFIX_THEOREM_BINDING_MISMATCH"
