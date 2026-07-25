from __future__ import annotations

from dataclasses import replace

from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import (
    compile_all_transitions,
)
from formal_toolchain.reference.protected_priority_prefix.pp0_checker import (
    _solve_code_bound_smt2,
    build_pp0_transition_certificate,
)
from formal_toolchain.reference.protected_priority_prefix.pp0_smt_encoder import (
    audit_generated_query_congruence,
    generate_code_bound_queries,
)
from formal_toolchain.reference.protected_priority_prefix.transition_ir_validation import (
    validate_all_compiled_ir,
    validate_compiled_ir,
)


def test_phase1_compiles_all_nine_source_bound_transition_cases():
    irs = compile_all_transitions()
    assert [ir.case_id for ir in irs] == [
        "FINAL_DISPATCH",
        "REM_COMPLETION",
        "RECOVERY",
        "DEADLINE_OBSERVATION",
        "ARRIVAL_BATCH",
        "MODE_SWITCH",
        "RELEASE",
        "SERVICE_UNIT",
        "TAIL_ONLY_SERVICE",
    ]
    assert all(ir.is_compiled() for ir in irs)
    for ir in irs:
        receipt = ir.compilation_receipt
        assert receipt is not None
        assert receipt.total_semantic_coverage is True
        assert receipt.semantic_validator_passed is True
        assert receipt.unsupported_nodes == ()
        assert receipt.control_flow_coverage["all_paths_covered"] is True
        assert receipt.control_flow_coverage["all_helpers_covered"] is True
        assert ir.return_path_count == ir.covered_return_path_count
        assert ir.raise_path_count == ir.covered_raise_path_count
        assert len(ir.paths) == ir.return_path_count + ir.raise_path_count
        assert ir.semantic_effect is not None
        assert ir.semantic_effect.derivation_complete is True
        assert ir.semantic_effect.concrete_write_targets == ir.semantic_effect.covered_concrete_write_targets
        assert set(ir.semantic_effect.path_effect_hashes) == set(ir.semantic_effect.source_path_hashes)


def test_tail_service_is_bound_to_the_real_service_primitive():
    ir = {item.case_id: item for item in compile_all_transitions()}["TAIL_ONLY_SERVICE"]
    assert ir.source_function == "apply_service_tick"
    assert ir.semantic_effect is not None
    assert ir.semantic_effect.field_equations == (("time", "(+ time 1)"),)
    assert "RUNNING_JOB_IS_TAIL" in ir.semantic_effect.required_assumption_ids
    assert "NO_PROTECTED_READY_JOB" in ir.semantic_effect.required_assumption_ids


def test_fresh_translation_validation_passes_all_compiled_and_bound_cases():
    result = validate_all_compiled_ir()
    assert result["status"] == "PASS"
    assert result["executable_case_count"] == 9
    assert result["case_count"] == 18
    assert result["pass_count"] == 18
    assert result["unresolved_count"] == 0


def test_projection_effect_mutation_breaks_source_binding():
    ir = {item.case_id: item for item in compile_all_transitions()}["SERVICE_UNIT"]
    assert ir.semantic_effect is not None
    mutated_effect = replace(
        ir.semantic_effect,
        field_equations=tuple(
            (field, "(+ time 2)" if field == "time" else expression)
            for field, expression in ir.semantic_effect.field_equations
        ),
    )
    mutated = replace(ir, semantic_effect=mutated_effect)
    result = validate_compiled_ir(mutated)
    assert result["status"] == "UNRESOLVED"
    assert result["checks"]["complete_ir_hash"] is False
    assert result["checks"]["semantic_effect_hash"] is False


def test_phase2_generates_twelve_conditional_code_bound_queries():
    queries = generate_code_bound_queries()
    assert len(queries) == 12
    for query in queries.values():
        assert query["transition_equations_bound"] is True
        assert query["direct_executable_encoding"] is True
        assert query["projection_derivation_complete"] is True
        assert query["all_paths_covered"] is True
        assert query["required_assumption_ids"]
        diagnostic = audit_generated_query_congruence(query["smt2_source"])
        assert diagnostic["diagnostic_status"] == "PASS"
        assert diagnostic["authoritative_proof"] is False


def test_phase2_z3_proves_all_twelve_negated_relations_unsat():
    certificate = build_pp0_transition_certificate()
    assert certificate["status"] == "PASS"
    assert certificate["receipt_count"] == 12
    assert certificate["pass_count"] == 12
    assert certificate["fail_count"] == 0
    assert certificate["unresolved_count"] == 0
    assert certificate["code_bound_query_count"] == 12
    for row in certificate["receipt_results"]:
        assert row["solver_result"] == "UNSAT"
        assert row["projection_derivation_complete"] is True
        assert row["required_assumption_ids"]


def test_deleting_a_protected_service_update_makes_the_negated_query_sat():
    source = generate_code_bound_queries()["PP0_SERVICE_PROTECTED"]["smt2_source"]
    old = "(assert (= service_p_post (service_one_digest service_p_pre running_p_pre)))"
    new = "(assert (= service_p_post service_p_pre))"
    assert old in source
    mutated = source.replace(old, new, 1)
    result, detail = _solve_code_bound_smt2(mutated)
    assert result == "SAT", detail
