from formal_toolchain.bridge.handler_decomposition import compose_fixed_transition_sequence


SCHEMA_HASH = "parameterized-state-relation-v1"


def test_sequence_uses_distinct_intermediate_states():
    result = compose_fixed_transition_sequence([
        {"status": "PASS", "parameterized_contract_status": "PASS", "parameterized_relation_schema_hash": SCHEMA_HASH, "precondition_formula": "(= c_time c_time)", "concrete_delta": "(= c_time_post (+ c_time 1))", "reference_delta": "(= r_time_post (+ r_time 1))"},
        {"status": "PASS", "parameterized_contract_status": "PASS", "parameterized_relation_schema_hash": SCHEMA_HASH, "precondition_formula": "(= c_time c_time)", "concrete_delta": "(= c_time_post (+ c_time 1))", "reference_delta": "(= r_time_post (+ r_time 1))"},
    ])
    assert result.status == "PASS"
    assert result.state_ids == ("s0", "s1", "s2")
    assert "c_s0_time" in result.formula and "c_s1_time" in result.formula and "c_s2_time" in result.formula


def test_empty_or_malformed_sequence_is_rejected():
    result = compose_fixed_transition_sequence([{"status": "PASS", "parameterized_contract_status": "PASS", "parameterized_relation_schema_hash": SCHEMA_HASH, "concrete_delta": "(= c_time c_time_post)", "reference_delta": ""}])
    assert result.status == "UNRESOLVED"
    assert result.failure == "INTERMEDIATE_STATE_BINDING_MISSING"


def test_next_precondition_must_be_entailable_from_previous_post_state():
    result = compose_fixed_transition_sequence([
        {"case_id": "STEP_1", "status": "PASS", "parameterized_contract_status": "PASS", "parameterized_relation_schema_hash": SCHEMA_HASH, "precondition_formula": "(= c_x c_x)",
         "concrete_delta": "(= c_x_post 0)",
         "reference_delta": "(= r_x_post 0)"},
        {"case_id": "STEP_2", "status": "PASS", "parameterized_contract_status": "PASS", "parameterized_relation_schema_hash": SCHEMA_HASH, "precondition_formula": "(= c_x 1)",
         "concrete_delta": "(= c_x_post (+ c_x 1))",
         "reference_delta": "(= r_x_post (+ r_x 1))"},
    ])
    assert result.status == "UNRESOLVED"
    assert result.failure == "SEQUENCE_NEXT_PRECONDITION_NOT_ENTAILED:STEP_2"
