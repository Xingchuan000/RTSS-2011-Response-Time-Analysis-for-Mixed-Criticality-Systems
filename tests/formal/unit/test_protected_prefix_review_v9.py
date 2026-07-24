from __future__ import annotations

from dataclasses import dataclass

import pytest

from formal_toolchain.reference.protected_priority_prefix.idle_jump_stutter import (
    prove_idle_jump_stutter_expansion,
)
from formal_toolchain.reference.protected_priority_prefix.time_indexed_close import close_at
from formal_toolchain.reference.protected_priority_prefix.macro_step import (
    prove_mode_tail_phase_join,
    prove_protected_macro_step_preservation,
)
from formal_toolchain.reference.protected_priority_prefix.construction import (
    build_saturated_protected_prefix,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.routes.registry import resolve_registry


@dataclass(frozen=True)
class _State:
    time: int
    jobs: tuple[str, ...] = ()
    running: str | None = None
    misses: tuple[str, ...] = ()


def _tasksets():
    full = ReferenceTaskset((
        ReferenceTask("hi", 20, 20, 2, 5, "HI", 0, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 2, 1, "LO", 1, 2, 2, 1, 0),
    ), "a" * 64)
    return full, build_saturated_protected_prefix(full, source_context_hash="a" * 64)


def test_finite_idle_jump_sample_cannot_prove_universal_theorem():
    execution = [_State(0), _State(5)]
    receipt = prove_idle_jump_stutter_expansion(execution=execution)
    assert receipt["status"] == "UNRESOLVED"
    assert receipt["parameterized"] is False
    assert receipt["finite_diagnostic"]["status"] == "PASS"
    assert receipt["finite_diagnostic"]["finite_horizon_only"] is True


def test_close_at_does_not_extrapolate_beyond_finite_execution():
    execution = [_State(0), _State(5)]
    with pytest.raises(ValueError, match="TIME_INDEXED_CLOSE_AFTER_FINITE_HORIZON"):
        close_at(execution, 6)


def test_execution_existence_dag_is_non_circular_and_consumes_input_legality():
    registry = resolve_registry("protected_prefix")
    by_id = {item["id"]: item for item in registry.entries}
    assert set(by_id["PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES"]["depends_on"]) == {
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
    }
    assert set(by_id["PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"]["depends_on"]) == {
        "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        "PROTECTED_INPUT_STREAM_PROJECTION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
    }
    assert by_id["PROTECTED_PREFIX_TIME_DIVERGENCE"]["depends_on"] == [
        "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"
    ]
    complete = set(by_id["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"]["depends_on"])
    assert "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL" in complete
    assert "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION" in complete


def test_prefix_model_conformance_directly_consumes_demand_receptiveness():
    registry = resolve_registry("protected_prefix")
    by_id = {item["id"]: item for item in registry.entries}
    assert "PROTECTED_INPUT_DEMAND_RECEPTIVENESS" in set(
        by_id["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]["depends_on"]
    )


def test_mode_join_has_symmetric_recovery_and_switch_cases():
    _, construction = _tasksets()
    receipt = prove_mode_tail_phase_join(construction=construction)
    cases = receipt["identity_skip_cases"]
    assert "FULL_ONLY_RECOVERY" in cases
    assert "PREFIX_ONLY_RECOVERY" in cases
    assert "FULL_ONLY_MODE_SWITCH" in cases
    assert "PREFIX_ONLY_MODE_SWITCH" in cases


def test_macro_step_theorem_uses_common_integer_close_at_domain():
    full, construction = _tasksets()
    receipt = prove_protected_macro_step_preservation(
        construction=construction,
        full_taskset=full,
        prefix_taskset=construction.prefix_taskset,
    )
    assert receipt["status"] == "UNRESOLVED"
    assert "CloseAt_full(t+1)" in receipt["conclusion"]
    assert receipt["integer_time_induction"] is True
    assert receipt["idle_jump_stutter_theorem_consumed"] is False


def test_simulation_domain_unwraps_nested_complete_execution_witness():
    from formal_toolchain.reference.protected_priority_prefix.simulation_domain import (
        _predecessor_witness,
    )
    predecessors = {
        "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS": {
            "obligation_status": "PASS",
            "witness": {
                "status": "PASS",
                "witness": {
                    "single_complete_execution": True,
                    "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
                },
            },
        }
    }
    payload = _predecessor_witness(
        predecessors, "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"
    )
    assert payload["single_complete_execution"] is True
    assert payload["quantifier_order"] == "forall-full-exists-one-prefix-forall-boundaries"
