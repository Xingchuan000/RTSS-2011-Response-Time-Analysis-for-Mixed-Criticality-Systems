from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import compile_all_transitions
from formal_toolchain.reference.protected_priority_prefix.pp0_checker import build_pp0_transition_certificate
from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
    prove_same_time_closure_termination_kernel,
    prove_canonical_successor_total_kernel,
    prove_time_divergence_kernel,
    prove_idle_jump_stutter_kernel,
    prove_weak_forward_simulation_kernel,
    prove_hi_bad_prefix_reflection_kernel,
    prove_pp8_reference_hi_safety_from_prefix_kernel,
)
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import build_runtime_schema_certificate
from formal_toolchain.routes.registry import resolve_registry


def test_diagnostic_ast_ir_cannot_claim_total_executable_semantics():
    irs = compile_all_transitions()
    assert len(irs) == 9
    assert all(ir.compilation_status == "PARTIAL_AST_EXTRACTION" for ir in irs)
    assert all(ir.total_semantic_coverage is False for ir in irs)
    assert all(not ir.is_compiled() for ir in irs)


def test_handwritten_pp0_adapter_cannot_be_code_bound_without_refinement_receipt():
    certificate = build_pp0_transition_certificate()
    assert certificate["status"] == "UNRESOLVED"
    assert certificate["pass_count"] == 0
    assert certificate["unresolved_count"] == 12
    assert all(row.get("transition_equations_bound") is False
               for row in certificate["receipt_results"])


def test_dynamic_theorems_do_not_follow_from_compiler_presence_or_pass_labels():
    for result in (
        prove_same_time_closure_termination_kernel(),
        prove_canonical_successor_total_kernel(),
        prove_time_divergence_kernel(),
        prove_idle_jump_stutter_kernel(),
        prove_weak_forward_simulation_kernel(
            macro_step_receipt={"status": "PASS"},
            execution_receipt={"status": "PASS"},
            base_case_receipt={"status": "PASS"},
        ),
        prove_hi_bad_prefix_reflection_kernel(
            simulation_receipt={"status": "PASS"},
            deadline_batch_receipt={"status": "PASS"},
        ),
        prove_pp8_reference_hi_safety_from_prefix_kernel(
            bad_prefix_reflection_receipt={"status": "PASS"},
            mathematical_conformance_receipt={"status": "PASS"},
        ),
    ):
        assert result["status"] == "UNRESOLVED"


def test_reference_model_conformance_is_not_circularly_dependent_on_rta():
    registry = resolve_registry("protected_prefix")
    by_id = {entry["id"]: entry for entry in registry.entries}
    deps = set(by_id["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"])
    assert "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC" not in deps


def test_relational_pp0_receipts_do_not_self_prove_local_runtime_assumptions():
    cert = build_runtime_schema_certificate()
    witness = cert["pp0_witness"]
    assert witness["local_semantics_theorems_required"] is True
    for key, value in witness.items():
        if key not in {"source_receipt_ids", "local_semantics_theorems_required"}:
            assert value is False


def test_complete_execution_route_does_not_regenerate_a_default_wcet_oracle():
    from pathlib import Path
    source = Path("formal_toolchain/routes/protected_prefix_checkers.py").read_text()
    block = source[source.index("def check_prefix_complete_execution_exists"):source.index("def check_registered_parametric_lemma")]
    assert "full_oracle = FullReferenceRecurringInputOracle" not in block
    assert "ARBITRARY_FULL_EXECUTION_ORACLE_WITNESS_REQUIRED" in block


def test_pp8_cannot_compose_two_shape_compatible_pass_labels():
    from formal_toolchain.reference.protected_priority_prefix.prefix_safety_lift import (
        prove_reference_hi_safety_from_prefix,
    )
    result = prove_reference_hi_safety_from_prefix(
        bad_prefix_reflection_receipt={
            "status": "PASS", "bad_prefix_reflection_hash": "a" * 64,
            "artifact_hash": "b" * 64,
        },
        mathematical_conformance_receipt={
            "status": "PASS", "artifact_hash": "c" * 64,
            "theorem_id": "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
            "proof_partition": ["PP7-A1", "PP7-A2", "PP7-B"],
            "pp7_a1_model_conformance_hash": "d" * 64,
            "pp7_a2_imported_theorem_binding_hash": "e" * 64,
            "pp7_b_rta_soundness_hash": "f" * 64,
        },
        proof_kernel_receipt={"status": "PASS"},
    )
    assert result["status"] == "UNRESOLVED"
    assert result["source_bound_composition"] is False
