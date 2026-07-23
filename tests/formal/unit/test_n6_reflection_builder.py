from copy import deepcopy

import pytest

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.bridge.bad_prefix import build_hi_bad_prefix_reflection_certificate
from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.bridge.state_relation import build_n6_relation_interface


IDS = {
    "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION", "RELEASE_FIXED_REMOVAL_MAPPING",
    "DEADLINE_OBSERVATION", "HI_NONTRUNCATION", "EFFECTIVE_EVENT_FRONTIER_RELATION",
    "EARLY_STOP_CONFIGURATION_GATE",
}


CONTEXTS = {
    "bridge_context": {"hash": "a" * 64},
    "semantic_context": {"hash": "b" * 64},
}
BOUNDS = P0ModelBounds(1, 1, 1, 1)


def _predecessors(*, missing_relation_field=False,
                  invalid_deadline=False, invalid_hi_nontruncation=False):
    relation_interface = build_n6_relation_interface(BOUNDS)
    if missing_relation_field:
        relation_interface["required_fields"] = relation_interface["required_fields"][:-1]

    predecessors = {}
    for obligation_id in IDS:
        layer = (
            "semantic_context"
            if obligation_id in {"DEADLINE_OBSERVATION", "HI_NONTRUNCATION"}
            else "bridge_context"
        )
        witness = {}
        if obligation_id == "CLOSED_PREFIX_REFINEMENT":
            witness = {
                "pointwise_closed_prefix_relation": True,
                "n6_relation_interface": relation_interface,
            }
        elif obligation_id == "DEADLINE_OBSERVATION":
            witness = {
                "deadline_is_observation_only": not invalid_deadline,
                "completion_precedes_equal_deadline": not invalid_deadline,
            }
        elif obligation_id == "HI_NONTRUNCATION":
            witness = {
                "contract": {
                    "hi_nontruncation": not invalid_hi_nontruncation,
                }
            }
        predecessors[obligation_id] = obligation_certificate(
            obligation_id=obligation_id,
            status="PASS",
            context_hash=CONTEXTS[layer]["hash"],
            inputs={},
            witness=witness,
            checker_id="test",
            checker_version="1",
        )
    return predecessors


def _theorem_and_receipt():
    return (
        {
            "theorem_id": "FINITE_HI_BAD_PREFIX_REFLECTION",
            "statement_hash": "c" * 64,
            "assumption_hash": "d" * 64,
            "proof_object": {"sha256": "e" * 64},
        },
        {"status": "PASS", "receipt_hash": "f" * 64},
    )


def test_universal_n6_certificate_has_diagnostic_not_proof_trace():
    predecessors = _predecessors()
    theorem, receipt = _theorem_and_receipt()
    result = build_hi_bad_prefix_reflection_certificate(
        verified_predecessors=predecessors, contexts=CONTEXTS, context_hash="a" * 64,
        theorem_statement=theorem, theorem_proof_receipt=receipt,
    )
    assert result["obligation_status"] == "PASS"
    assert result["witness"]["trace_diagnostic"]["status"] == "NOT_RUN"
    assert "no_hi_miss_available" not in result["witness"]


def test_n6_rejects_missing_relation_field():
    theorem, receipt = _theorem_and_receipt()
    with pytest.raises(ValueError, match="N6_RELATION_INTERFACE_FIELDS_INVALID"):
        build_hi_bad_prefix_reflection_certificate(
            verified_predecessors=_predecessors(missing_relation_field=True),
            contexts=CONTEXTS,
            context_hash="a" * 64,
            theorem_statement=theorem,
            theorem_proof_receipt=receipt,
        )


def test_n6_rejects_invalid_deadline_observation_interface():
    theorem, receipt = _theorem_and_receipt()
    with pytest.raises(ValueError, match="N6_DEADLINE_OBSERVATION_INTERFACE_INVALID"):
        build_hi_bad_prefix_reflection_certificate(
            verified_predecessors=_predecessors(invalid_deadline=True),
            contexts=CONTEXTS,
            context_hash="a" * 64,
            theorem_statement=theorem,
            theorem_proof_receipt=receipt,
        )


def test_n6_rejects_invalid_hi_nontruncation_interface():
    theorem, receipt = _theorem_and_receipt()
    with pytest.raises(ValueError, match="N6_HI_NONTRUNCATION_INTERFACE_INVALID"):
        build_hi_bad_prefix_reflection_certificate(
            verified_predecessors=_predecessors(invalid_hi_nontruncation=True),
            contexts=CONTEXTS,
            context_hash="a" * 64,
            theorem_statement=theorem,
            theorem_proof_receipt=receipt,
        )
