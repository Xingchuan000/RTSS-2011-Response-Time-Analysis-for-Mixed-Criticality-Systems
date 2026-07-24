from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.protected_priority_prefix.construction import (
    build_saturated_protected_prefix,
)
from formal_toolchain.reference.protected_priority_prefix.macro_step import (
    prove_arrival_batch_projection,
    prove_deadline_batch_correspondence,
)
from formal_toolchain.reference.protected_priority_prefix.pp0_smt_encoder import (
    generate_code_bound_queries,
)
from formal_toolchain.reference.protected_priority_prefix.pp_transition_binding import (
    compile_all_transitions,
)
from formal_toolchain.reference.protected_priority_prefix.weak_simulation_kernel import (
    prove_weak_forward_simulation,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.theory.backends.protected_prefix_bad_prefix import (
    ProtectedPrefixBadPrefixBackend,
)


def _construction():
    full = ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)
    return build_saturated_protected_prefix(full, source_context_hash="a" * 64)


def _fold_receipt(phase: str):
    return {
        "status": "PASS",
        "theorem_id": "BATCH_CURSOR_PARAMETERIZED_FOLD",
        "phase": phase,
        "base_case": True,
        "protected_step": True,
        "tail_step": True,
        "end_case": True,
        "all_batch_sizes": True,
        "finite_instance_data_used": False,
        "relation_schema_hash": sha256_object({"schema": "phase_relation_v4_close_at"}),
    }


def test_handwritten_pp0_queries_are_never_mislabeled_code_bound():
    queries = generate_code_bound_queries()
    assert queries
    assert all(q["transition_equations_bound"] is False for q in queries.values())
    assert all(q["proof_scope"] == "HAND_WRITTEN_SCHEMA_NOT_CODE_BOUND" for q in queries.values())
    assert all(case.binding_status == "UNRESOLVED" for case in compile_all_transitions().values())


def test_l8_batch_lemmas_consume_symbolic_fold_without_concrete_batch():
    construction = _construction()
    ddl = prove_deadline_batch_correspondence(
        construction=construction,
        fold_kernel_receipt=_fold_receipt("DDLCursor"),
    )
    arr = prove_arrival_batch_projection(
        construction=construction,
        fold_kernel_receipt=_fold_receipt("ARRCursor"),
    )
    assert ddl["status"] == "PASS"
    assert arr["status"] == "PASS"
    assert ddl["finite_instance_data_used"] is False
    assert arr["finite_instance_data_used"] is False


def test_weak_simulation_accepts_actual_l8_lemma_identifier():
    result = prove_weak_forward_simulation(
        macro_step_receipt={
            "status": "PASS",
            "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
        },
        execution_existence_receipt={
            "status": "PASS",
            "witness": {"single_complete_execution": True},
        },
        base_case_receipt={"status": "PASS", "base_relation_proved": True},
        proof_kernel_receipt={
            "status": "PASS",
            "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
            "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
            "induction_on_t_complete": True,
        },
    )
    assert result["status"] == "PASS"
    assert result["macro_step_L1_L8_proved"] is True


def test_bad_prefix_backend_requires_completion_state_reflection():
    assert "same_completion_state" in ProtectedPrefixBadPrefixBackend.REQUIRED_REFLECTION_FIELDS


def test_bad_prefix_field_lookup_matches_phase_relation_key_shape():
    from formal_toolchain.reference.protected_priority_prefix.bad_prefix_reflection import (
        _field_equal_in_receipt,
    )
    receipt = {
        "status": "PASS",
        "equality_checks": {
            "job_key_set": True,
            "job[('hi', 0)].job_key": True,
            "job[('hi', 0)].absolute_deadline": True,
            "job[('hi', 0)].actual_demand": True,
            "job[('hi', 0)].executed_service": True,
            "job[('hi', 0)].completed": True,
            "job[('hi', 0)].missed": True,
            "miss_job_keys": True,
        },
    }
    for field in (
        "job_key", "absolute_deadline", "actual_demand",
        "executed_service", "completed", "missed", "miss_job_keys",
    ):
        assert _field_equal_in_receipt(receipt, field) is True
