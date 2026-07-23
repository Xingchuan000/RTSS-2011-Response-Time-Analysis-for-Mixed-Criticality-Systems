import pytest

from pathlib import Path

from formal_toolchain.bridge.effect_compiler import compile_reschedule_family_effect
from formal_toolchain.bridge.handler_decomposition import (
    prove_handler_reschedule_unreachability,
)
from formal_toolchain.bridge.model_bounds import _legacy_test_bounds
from formal_toolchain.bridge.runtime_branch_map import bind_reschedule_branch_families


ROOT = Path(__file__).parents[3]


def test_reschedule_effects_are_compiled_into_scheduler_state_and_frontier():
    bindings = bind_reschedule_branch_families(ROOT)
    bounds = _legacy_test_bounds()
    keep = compile_reschedule_family_effect(
        case_id="RESCHEDULE_KEEP_SAME", branch_binding=bindings["RESCHEDULE_KEEP_SAME"], bounds=bounds)
    idle = compile_reschedule_family_effect(
        case_id="RESCHEDULE_TO_IDLE", branch_binding=bindings["RESCHEDULE_TO_IDLE"], bounds=bounds)
    dispatch = compile_reschedule_family_effect(
        case_id="PREEMPTION_DISPATCH", branch_binding=bindings["PREEMPTION_DISPATCH"], bounds=bounds)
    assert "(= c_running_post 0)" in idle.to_smt()
    assert "(= c_running_job_key_post selected_job_key)" in dispatch.to_smt()
    assert "c_queue_token_epoch_post" in idle.to_smt()
    assert "c_queue_event_count_post (+ c_queue_event_count 2)" in dispatch.to_smt()
    assert keep.modified_components == ()


def test_handler_context_excluded_reschedule_alternatives_are_unsat():
    pytest.importorskip("z3")
    result = prove_handler_reschedule_unreachability()
    assert result["status"] == "PASS"
    assert result["proofs"] == {
        "completion_to_idle": "UNSAT",
        "controller_force_keep": "UNSAT",
    }
