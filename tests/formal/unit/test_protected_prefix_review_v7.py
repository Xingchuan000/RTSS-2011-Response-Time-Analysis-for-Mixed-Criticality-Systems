from __future__ import annotations

from formal_toolchain.reference.protected_priority_prefix.batch_cursor import (
    construct_fold_lemma, verify_fold_lemma,
)
from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import (
    compile_function,
)
from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
    define_next_closed_boundary, prove_canonical_successor_total,
    prove_same_time_closure_terminates, prove_time_divergence,
)
from formal_toolchain.reference.protected_priority_prefix.construction import (
    build_saturated_protected_prefix,
)
from formal_toolchain.reference.protected_priority_prefix.input_oracle import (
    FullReferenceRecurringInputOracle, ProtectedInputOracle,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.registry import resolve_registry


def _tasksets():
    full = ReferenceTaskset((
        ReferenceTask("hi", 20, 20, 2, 5, "HI", 0, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 2, 1, "LO", 1, 2, 2, 1, 0),
    ), "a" * 64)
    construction = build_saturated_protected_prefix(full, source_context_hash="a" * 64)
    return full, construction


def test_partial_ast_extraction_is_not_code_bound_compilation():
    removal = compile_function("apply_removal")
    recovery = compile_function("apply_recovery")
    assert removal.compilation_status == "PARTIAL_AST_EXTRACTION"
    assert recovery.compilation_status == "PARTIAL_AST_EXTRACTION"
    assert not removal.is_compiled()
    assert not recovery.is_compiled()


def test_concrete_batch_check_cannot_claim_parameterized_fold_theorem():
    full_entries = [{"task_name": "hi", "job_key": ("hi", 0), "release_time": 0, "actual_demand": 2, "hi_class": "NORMAL"}]
    prefix_entries = list(full_entries)
    lemma = construct_fold_lemma(full_entries, prefix_entries, frozenset({"hi"}), "ARRCursor")
    verified = verify_fold_lemma(lemma)
    assert lemma.end_case is True
    assert lemma.parameterized is False
    assert verified["status"] == "UNRESOLVED"


def test_execution_totality_is_not_inferred_from_callable_definition():
    full, construction = _tasksets()
    oracle = ProtectedInputOracle(
        FullReferenceRecurringInputOracle(construction.prefix_taskset),
        frozenset(construction.protected_task_names), construction,
    )
    definition = define_next_closed_boundary(construction.prefix_taskset, oracle)
    total = prove_canonical_successor_total(construction.prefix_taskset, definition)
    closure = prove_same_time_closure_terminates(total)
    divergence = prove_time_divergence(closure)
    assert definition["status"] == "UNRESOLVED"
    assert total["status"] == "UNRESOLVED"
    assert closure["status"] == "UNRESOLVED"
    assert divergence["status"] == "UNRESOLVED"


def test_complete_execution_registry_matches_checker_predecessors():
    resolved = resolve_registry("protected_prefix")
    by_id = {entry["id"]: entry for entry in resolved.entries}
    assert set(by_id["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"]["depends_on"]) == {
        "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "PROTECTED_INPUT_STREAM_PROJECTION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
    }


def test_prefix_model_conformance_receives_execution_and_candidate_domain_proofs():
    resolved = resolve_registry("protected_prefix")
    by_id = {entry["id"]: entry for entry in resolved.entries}
    deps = set(by_id["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"])
    assert "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS" in deps
    assert "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC" in deps


def test_prefix_model_conformance_consumes_zero_start_and_execution_proofs():
    registry = resolve_registry("protected_prefix")
    by_id = {item["id"]: item for item in registry.entries}
    deps = set(by_id["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"])
    assert "ZERO_RELATIVE_START" in deps
    assert "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS" in deps
    assert "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC" in deps
