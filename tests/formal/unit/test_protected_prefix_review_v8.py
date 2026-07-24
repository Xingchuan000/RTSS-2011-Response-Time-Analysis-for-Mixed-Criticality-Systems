from __future__ import annotations

from formal_toolchain.reference.protected_priority_prefix.batch_cursor import (
    prove_parameterized_fold_kernel,
)
from formal_toolchain.reference.protected_priority_prefix.construction import (
    build_saturated_protected_prefix,
)
from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
    _LEXICOGRAPHIC_MEASURE_ORDER, prove_closure_measure_well_founded,
    prove_complete_execution_exists,
)
from formal_toolchain.reference.protected_priority_prefix.input_oracle import (
    FullReferenceRecurringInputOracle, LazyInfiniteProtectedInputOracle,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.registry import resolve_registry


def _construction():
    full = ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 3, 1, "LO", 0, 3, 3, 1, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 2, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)
    return full, build_saturated_protected_prefix(full, source_context_hash="a" * 64)


def test_finite_demand_checks_do_not_prove_universal_receptiveness():
    full, construction = _construction()
    oracle = LazyInfiniteProtectedInputOracle(
        FullReferenceRecurringInputOracle(full), construction,
    )
    result = oracle.demand_receptiveness_checks(0)
    assert result["finite_check_status"] == "PASS"
    assert result["status"] == "UNRESOLVED"
    assert result["parameterized"] is False
    assert result["code"] == "PARAMETRIC_DEMAND_RECEPTIVENESS_PROOF_MISSING"


def test_caller_booleans_are_not_a_batch_induction_kernel():
    result = prove_parameterized_fold_kernel(
        phase="ARRCursor",
        proof_inputs={
            "base_case": True,
            "protected_step": True,
            "tail_step": True,
            "end_case": True,
        },
    )
    assert result["status"] == "UNRESOLVED"
    assert result["code"] == "BATCH_CURSOR_PROOF_KERNEL_MISSING"
    assert result["all_batch_sizes"] is False


def test_closure_measure_dimension_matches_the_canonical_seven_phases():
    receipt = prove_closure_measure_well_founded()
    assert receipt["status"] == "PASS"
    assert len(_LEXICOGRAPHIC_MEASURE_ORDER) == 7
    assert len(receipt["lower_bound"]) == len(_LEXICOGRAPHIC_MEASURE_ORDER)


def test_complete_execution_requires_idle_jump_stutter_expansion():
    _, construction = _construction()
    result = prove_complete_execution_exists(
        time_divergence_receipt={"status": "PASS"},
        input_projection_receipt={"status": "PASS"},
        prefix_taskset=construction.prefix_taskset,
        single_witness_receipt={
            "status": "PASS",
            "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
            "same_fixed_oracle": True,
            "same_initial_state": True,
            "same_successor_function": True,
            "all_finite_prefixes_are_prefixes_of_one_execution": True,
        },
        proof_kernel_receipt={
            "status": "PASS",
            "theorem_id": "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "dependent_choice_construction_verified": True,
            "recurring_history_preserved": True,
        },
    )
    assert result["status"] == "UNRESOLVED"
    assert result["witness"]["idle_jump_expansion_verified"] is False
    assert result["witness"]["time_indexed_closed_observation_defined"] is False


def test_idle_jump_stutter_expansion_is_an_explicit_route_obligation():
    resolved = resolve_registry("protected_prefix")
    by_id = {entry["id"]: entry for entry in resolved.entries}
    assert set(by_id["PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"]["depends_on"]) == {
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
    }
    assert "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION" in (
        by_id["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"]["depends_on"]
    )
