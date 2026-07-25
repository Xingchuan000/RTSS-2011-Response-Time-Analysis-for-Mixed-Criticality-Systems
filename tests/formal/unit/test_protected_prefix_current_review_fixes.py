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
from formal_toolchain.reference.protected_priority_prefix.batch_cursor_kernel import (
    build_parameterized_fold_receipt,
)
from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import (
    compile_all_transitions as compile_executable_transitions,
)
from formal_toolchain.reference.protected_priority_prefix.pp0_checker import (
    build_pp0_transition_certificate,
)
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import (
    build_runtime_schema_certificate,
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
    irs = {ir.case_id: ir for ir in compile_executable_transitions()}
    pp0_rows = build_pp0_transition_certificate()["receipt_results"]
    by_receipt = {row["receipt_id"]: row for row in pp0_rows}
    local = build_runtime_schema_certificate()["local_runtime_semantics"]["receipts"]
    if phase == "DDLCursor":
        return build_parameterized_fold_receipt(
            phase=phase,
            transition_ir=irs["DEADLINE_OBSERVATION"],
            pp0_receipt=by_receipt["PP0_DDL_OBSERVE_ONLY"],
            required_local_theorem_id="DEADLINE_OBSERVE_ONLY",
            local_theorem_receipts=local,
        )
    return build_parameterized_fold_receipt(
        phase=phase,
        transition_ir=irs["ARRIVAL_BATCH"],
        pp0_receipt=by_receipt["PP0_ARR_PENDING_PLAN_PROJECTION"],
        required_local_theorem_id="ABNORMAL_HI_CLASSIFIED_AT_ARRIVAL",
        local_theorem_receipts=local,
        additional_pp0_receipts={
            "MODE_SWITCH": by_receipt["PP0_SWITCH_STUTTER_FULL_ONLY"],
            "RELEASE": by_receipt["PP0_RELEASE_PROTECTED_PAYLOAD"],
        },
        projection_receipt={"status": "PASS", "forall_release_indices": True},
        demand_receptiveness_receipt={"status": "PASS"},
    )


def test_pp0_queries_are_bound_only_to_complete_projection_derivations():
    queries = generate_code_bound_queries()
    assert len(queries) == 12
    for query in queries.values():
        assert query["transition_equations_bound"] is True
        assert query["proof_scope"] == "CODE_BOUND_RELATIONAL"
        assert query["direct_executable_encoding"] is True
        assert query["projection_derivation_complete"] is True
        assert query["all_paths_covered"] is True
        assert query["required_assumption_ids"]
    bound = compile_all_transitions()
    assert len(bound) == 9
    assert all(case.binding_status == "CODE_BOUND" for case in bound.values())
    assert all(case.projection_derivation_complete for case in bound.values())


def test_descriptive_proof_kernel_outlines_are_not_authoritative_passes():
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_same_time_closure_termination_kernel,
        prove_canonical_successor_total_kernel,
        prove_time_divergence_kernel,
        prove_idle_jump_stutter_kernel,
        prove_complete_execution_exists_kernel,
        prove_weak_forward_simulation_kernel,
        prove_hi_bad_prefix_reflection_kernel,
        prove_pp8_reference_hi_safety_from_prefix_kernel,
    )

    construction = _construction()
    local_source_bound = [
        prove_same_time_closure_termination_kernel(),
        prove_canonical_successor_total_kernel(),
        prove_time_divergence_kernel(),
        prove_idle_jump_stutter_kernel(),
    ]
    assert all(item["status"] == "PASS" for item in local_source_bound)

    quantified = [
        prove_complete_execution_exists_kernel(
            prefix_taskset=construction.prefix_taskset,
        ),
        prove_weak_forward_simulation_kernel(),
        prove_hi_bad_prefix_reflection_kernel(),
        prove_pp8_reference_hi_safety_from_prefix_kernel(),
    ]
    assert all(item["status"] == "UNRESOLVED" for item in quantified)


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


def test_weak_simulation_does_not_accept_pass_labels_as_an_induction_proof():
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
    assert result["status"] == "UNRESOLVED"
    assert result["macro_step_L1_L8_proved"] is False


def test_bad_prefix_backend_requires_completion_state_reflection():
    assert "completion_state" in ProtectedPrefixBadPrefixBackend.REQUIRED_REFLECTION_FIELDS


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


def test_time_divergence_receipt_exposes_internal_kernel_facts():
    from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
        prove_time_divergence,
    )

    receipt = prove_time_divergence({"status": "PASS"})
    assert receipt["status"] == "PASS"
    assert receipt["service_branch_advances_by_1"] is True
    assert receipt["idle_branch_jumps_to_future_event"] is True
    assert receipt["unbounded_iteration_proved"] is True
