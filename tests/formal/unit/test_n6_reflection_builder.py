from copy import deepcopy

import pytest

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.bad_prefix import build_hi_bad_prefix_reflection_certificate
from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.bridge.state_relation import (
    N6_REQUIRED_QUANTITIES,
    build_n6_relation_interface,
    parameterized_state_relation_schema_hash,
)


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
        relation_interface["required_quantities"] = relation_interface["required_quantities"][:-1]

    predecessors = {}
    for obligation_id in IDS:
        layer = (
            "semantic_context"
            if obligation_id in {"DEADLINE_OBSERVATION", "HI_NONTRUNCATION"}
            else "bridge_context"
        )
        witness = {}
        inputs = {}
        if obligation_id == "CLOSED_PREFIX_REFINEMENT":
            witness = {
                "pointwise_closed_prefix_relation": True,
                "n6_relation_interface": relation_interface,
                "reference_transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
            }
        elif obligation_id == "REFERENCE_PREFIX_EXTENSION":
            inputs = {"reference_taskset_fingerprint": "9" * 64}
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
            inputs=inputs,
            witness=witness,
            checker_id="test",
            checker_version="1",
        )
    return predecessors


def _theorem_and_receipt():
    theorem = {
        "theorem_id": "FINITE_HI_BAD_PREFIX_REFLECTION",
        "statement_hash": "c" * 64,
        "assumption_hash": "d" * 64,
        "proof_object": {
            "sha256": "e" * 64,
            "backend": "finite-hi-bad-prefix-z3-v1",
        },
    }
    body = {
        "backend_id": "finite-hi-bad-prefix-z3-v1",
        "proof_object_hash": theorem["proof_object"]["sha256"],
        "theorem_statement_hash": theorem["statement_hash"],
        "theorem_assumption_hash": theorem["assumption_hash"],
        "source_bindings": {"source": "f" * 64},
        "relation_interface": "n6_closed_prefix_relation_interface_v2",
        "parameterized_relation_schema_hash": parameterized_state_relation_schema_hash(),
        "required_quantities": list(N6_REQUIRED_QUANTITIES),
        "solver_obligations": {"obligation": {"result": "UNSAT"}},
        "z3_version": "test",
    }
    return theorem, {"status": "PASS", **body, "receipt_hash": sha256_object(body)}


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


def test_n6_rejects_receipt_for_wrong_interface():
    theorem, receipt = _theorem_and_receipt()
    receipt = deepcopy(receipt)
    receipt["relation_interface"] = "n6_closed_prefix_relation_interface_v1"
    with pytest.raises(ValueError, match="N6_THEOREM_RECEIPT_BINDING_MISMATCH"):
        build_hi_bad_prefix_reflection_certificate(
            verified_predecessors=_predecessors(),
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


def test_fresh_verified_predecessor_shapes_remain_n6_consumable():
    from formal_toolchain.verifier.bridge_proof_checker import (
        _verified_closed_prefix_witness,
    )
    from formal_toolchain.verifier.recompute import _semantic_certificate

    predecessors = _predecessors()
    relation = predecessors["CLOSED_PREFIX_REFINEMENT"]["witness"][
        "n6_relation_interface"
    ]
    closed_witness = _verified_closed_prefix_witness(
        candidate_witness={
            "reference_transition_system_id":
                "FIXED_EXECUTABLE_REFERENCE_P0_V3",
            "n6_relation_interface": relation,
            "parameterized_relation_schema_hash":
                parameterized_state_relation_schema_hash(),
            "pointwise_closed_prefix_relation": True,
        },
        receipt_hash="1" * 64,
        replay_hash="2" * 64,
    )
    predecessors["CLOSED_PREFIX_REFINEMENT"] = obligation_certificate(
        obligation_id="CLOSED_PREFIX_REFINEMENT",
        status="PASS",
        context_hash=CONTEXTS["bridge_context"]["hash"],
        inputs={"candidate_artifact_hash": "3" * 64, "fresh_process": True},
        witness=closed_witness,
        checker_id="test",
        checker_version="1",
    )

    extension_candidate = obligation_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION",
        status="PASS",
        context_hash=CONTEXTS["bridge_context"]["hash"],
        inputs={"reference_taskset_fingerprint": "9" * 64},
        witness={"schema_version": "reference_prefix_extension_v4"},
        checker_id="test",
        checker_version="1",
    )
    predecessors["REFERENCE_PREFIX_EXTENSION"] = _semantic_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION",
        candidate=extension_candidate,
        status="PASS",
        context_hash=CONTEXTS["bridge_context"]["hash"],
        predecessors={},
        witness=extension_candidate["witness"],
        verified_inputs=extension_candidate["inputs"],
    )

    theorem, receipt = _theorem_and_receipt()
    result = build_hi_bad_prefix_reflection_certificate(
        verified_predecessors=predecessors,
        contexts=CONTEXTS,
        context_hash=CONTEXTS["bridge_context"]["hash"],
        theorem_statement=theorem,
        theorem_proof_receipt=receipt,
    )
    assert result["obligation_status"] == "PASS"
    assert result["witness"]["reference_taskset_fingerprint"] == "9" * 64
