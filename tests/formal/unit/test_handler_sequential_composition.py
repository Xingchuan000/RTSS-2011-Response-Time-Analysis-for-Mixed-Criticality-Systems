import pytest

pytest.importorskip("z3")

from formal_toolchain.bridge.handler_decomposition import compose_fixed_transition_sequence


SCHEMA_HASH = "parameterized-state-relation-v1"


def _step(
    case_id: str,
    *,
    precondition: str,
    concrete_delta: str,
    reference_delta: str,
    relation: str,
):
    return {
        "case_id": case_id,
        "status": "PASS",
        "concrete_feasibility": "SAT",
        "reference_totality": "PASS",
        "relation_preservation": "PASS",
        "parameterized_contract_status": "PASS",
        "parameterized_relation_schema_hash": SCHEMA_HASH,
        "precondition_formula": precondition,
        "concrete_delta": concrete_delta,
        "reference_delta": reference_delta,
        "relation_preservation_formula": relation,
    }


def test_sequence_uses_distinct_intermediate_states_and_relation_witnesses():
    result = compose_fixed_transition_sequence([
        _step(
            "STEP_1",
            precondition="(= c_time r_time)",
            concrete_delta="(= c_time_post (+ c_time 1))",
            reference_delta="(= r_time_post (+ r_time 1))",
            relation="(= c_time_post r_time_post)",
        ),
        _step(
            "STEP_2",
            precondition="(= c_time r_time)",
            concrete_delta="(= c_time_post (+ c_time 1))",
            reference_delta="(= r_time_post (+ r_time 1))",
            relation="(= c_time_post r_time_post)",
        ),
    ])

    assert result.status == "PASS"
    assert result.feasibility_result == "SAT"
    assert result.precondition_chain_result == "PASS"
    assert result.relation_chain_result == "PASS"
    assert result.state_ids == ("s0", "s1", "s2")
    assert "c_s0_time" in result.formula
    assert "c_s1_time" in result.formula
    assert "c_s2_time" in result.formula
    assert "(= c_s1_time r_s1_time)" in result.formula
    assert "(= c_s2_time r_s2_time)" in result.formula


def test_empty_or_malformed_sequence_is_rejected():
    malformed = _step(
        "BROKEN",
        precondition="(= c_time r_time)",
        concrete_delta="(= c_time c_time_post)",
        reference_delta="(= r_time r_time)",
        relation="(= c_time_post r_time_post)",
    )
    malformed["reference_delta"] = ""

    result = compose_fixed_transition_sequence([malformed])

    assert result.status == "UNRESOLVED"
    assert result.failure == "SEQUENCE_CHILD_FORMULA_MISSING:BROKEN:reference_delta"


def test_next_precondition_must_be_entailed_from_previous_post_state():
    result = compose_fixed_transition_sequence([
        _step(
            "STEP_1",
            precondition="(= c_x r_x)",
            concrete_delta="(= c_x_post 0)",
            reference_delta="(= r_x_post 0)",
            relation="(= c_x_post r_x_post)",
        ),
        _step(
            "STEP_2",
            precondition="(and (= c_x r_x) (= c_x 1))",
            concrete_delta="(= c_x_post (+ c_x 1))",
            reference_delta="(= r_x_post (+ r_x 1))",
            relation="(= c_x_post r_x_post)",
        ),
    ])

    assert result.status == "UNRESOLVED"
    assert result.precondition_chain_result == "FAIL"
    assert result.failure == "SEQUENCE_NEXT_PRECONDITION_NOT_ENTAILED:STEP_2"


def test_child_relation_proof_components_are_required():
    step = _step(
        "STEP_1",
        precondition="(= c_x r_x)",
        concrete_delta="(= c_x_post c_x)",
        reference_delta="(= r_x_post r_x)",
        relation="(= c_x_post r_x_post)",
    )
    step["relation_preservation"] = "UNRESOLVED"

    result = compose_fixed_transition_sequence([step])

    assert result.status == "UNRESOLVED"
    assert result.failure == "SEQUENCE_CHILD_RELATION_PRESERVATION_MISSING:STEP_1"
