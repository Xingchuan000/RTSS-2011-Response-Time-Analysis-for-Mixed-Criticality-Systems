from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.reference.final_claim_composition import build_final_claim_composition


def _contradiction(conclusion="NO_CONCRETE_HI_DEADLINE_MISS"):
    return obligation_certificate(
        obligation_id="FINITE_BAD_PREFIX_CONTRADICTION",
        status="PASS",
        context_hash="a" * 64,
        inputs={},
        witness={
            "theorem_id": "FINITE_BAD_PREFIX_CONTRADICTION",
            "conclusion": conclusion,
        },
        checker_id="test",
        checker_version="1",
    )


def _theorem(theorem_id="FINAL_DEPLOYED_HI_SAFETY_COMPOSITION"):
    return {"theorem_id": theorem_id, "statement_hash": "b" * 64}


def test_final_claim_requires_exact_contradiction_conclusion():
    result = build_final_claim_composition(
        finite_bad_prefix_contradiction_certificate=_contradiction("OTHER"),
        theorem=_theorem(),
        composition_context_hash="c" * 64,
    )
    assert result["obligation_status"] == "UNRESOLVED"


def test_final_claim_requires_exact_theorem():
    result = build_final_claim_composition(
        finite_bad_prefix_contradiction_certificate=_contradiction(),
        theorem=_theorem("WRONG"),
        composition_context_hash="c" * 64,
    )
    assert result["obligation_status"] == "UNRESOLVED"


def test_final_claim_passes_with_exact_chain():
    result = build_final_claim_composition(
        finite_bad_prefix_contradiction_certificate=_contradiction(),
        theorem=_theorem(),
        composition_context_hash="c" * 64,
    )
    assert result["obligation_status"] == "PASS"
