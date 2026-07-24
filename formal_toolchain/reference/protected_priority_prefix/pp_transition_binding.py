from __future__ import annotations

import inspect
import textwrap
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference import executable_semantics

from .executable_transition_ir import (
    Assignment, BoolExpr, IntExpr, GeneratedEventRule,
    atomic_bool, cmp_expr, const_expr, var_expr,
    state_assignment, frame_assignment, time_assignment,
)
from .pp_transition_semantics import BoundTransitionCase, StateUpdate, skip_state_update


def _source_hash(fn: Any) -> str:
    source = textwrap.dedent(inspect.getsource(fn))
    return sha256_object({"source": source})


# These records currently bind only a source-function hash while both state
# updates are stutters.  They are provenance adapters, not semantic compiler
# output, and therefore must never satisfy a CODE_BOUND theorem premise.
_BOUND: str = "UNRESOLVED"


def _skip_full_update() -> StateUpdate:
    return skip_state_update()


def _skip_prefix_update() -> StateUpdate:
    return skip_state_update()


def _rem_completion() -> BoundTransitionCase:
    fn = executable_semantics.apply_removal
    fqn = "formal_toolchain.reference.executable_semantics.apply_removal"
    h = _source_hash(fn)
    guard = atomic_bool("removal_guard")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="REM_COMPLETION",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset(),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _recovery() -> BoundTransitionCase:
    fn = executable_semantics.apply_recovery
    fqn = "formal_toolchain.reference.executable_semantics.apply_recovery"
    h = _source_hash(fn)
    guard = atomic_bool("recovery_mode_HI_and_quiescent")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="RECOVERY",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"mode", "mode_switches"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _deadline_observation() -> BoundTransitionCase:
    fn = executable_semantics.apply_deadline_observation
    fqn = "formal_toolchain.reference.executable_semantics.apply_deadline_observation"
    h = _source_hash(fn)
    guard = atomic_bool("ddl_observe_guard")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="DEADLINE_OBSERVATION",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset(),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _arrival_batch() -> BoundTransitionCase:
    fn = executable_semantics.apply_arrival_batch
    fqn = "formal_toolchain.reference.executable_semantics.apply_arrival_batch"
    h = _source_hash(fn)
    guard = atomic_bool("arrival_batch_guard")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="ARRIVAL_BATCH",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"pending_releases", "frontier"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _mode_switch() -> BoundTransitionCase:
    fn = executable_semantics.apply_mode_switch
    fqn = "formal_toolchain.reference.executable_semantics.apply_mode_switch"
    h = _source_hash(fn)
    guard = atomic_bool("mode_switch_guard_abnormal_hi_arrival")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="MODE_SWITCH",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"mode", "mode_switches"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _release() -> BoundTransitionCase:
    fn = executable_semantics.apply_release
    fqn = "formal_toolchain.reference.executable_semantics.apply_release"
    h = _source_hash(fn)
    guard = atomic_bool("release_guard_pending_plan_exists")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="RELEASE",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"pending_releases", "released", "jobs", "ready_order"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _final_dispatch() -> BoundTransitionCase:
    fn = executable_semantics._normalize_dispatch
    fqn = "formal_toolchain.reference.executable_semantics._normalize_dispatch"
    h = _source_hash(fn)
    guard = atomic_bool("dispatch_guard_ready_set_nonempty")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="FINAL_DISPATCH",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"running", "ready_order"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _service_unit() -> BoundTransitionCase:
    fn = executable_semantics.apply_service_tick
    fqn = "formal_toolchain.reference.executable_semantics.apply_service_tick"
    h = _source_hash(fn)
    guard = atomic_bool("service_tick_guard_running_job_exists")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="SERVICE_UNIT",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"jobs", "terminal"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


def _tail_only_service() -> BoundTransitionCase:
    fn = executable_semantics.close_timestamp
    fqn = "formal_toolchain.reference.executable_semantics.close_timestamp"
    h = _source_hash(fn)
    guard = atomic_bool("close_timestamp_guard")
    full_up = skip_state_update()
    prefix_up = skip_state_update()
    return BoundTransitionCase(
        case_id="TAIL_ONLY_SERVICE",
        source_function=fqn,
        source_ast_hash=h,
        guard=guard,
        full_update=full_up,
        prefix_update=prefix_up,
        frame_fields=frozenset({"time", "frontier", "running", "ready_order"}),
        generated_event_phase_constraints=(),
        binding_status=_BOUND,
    )


BOUND_TRANSITIONS: dict[str, BoundTransitionCase] = {
    "REM_COMPLETION": _rem_completion(),
    "RECOVERY": _recovery(),
    "DEADLINE_OBSERVATION": _deadline_observation(),
    "ARRIVAL_BATCH": _arrival_batch(),
    "MODE_SWITCH": _mode_switch(),
    "RELEASE": _release(),
    "FINAL_DISPATCH": _final_dispatch(),
    "SERVICE_UNIT": _service_unit(),
    "TAIL_ONLY_SERVICE": _tail_only_service(),
}


def compile_all_transitions() -> dict[str, BoundTransitionCase]:
    return dict(BOUND_TRANSITIONS)


def bound_transition_for_case(case_id: str) -> BoundTransitionCase | None:
    return BOUND_TRANSITIONS.get(case_id)
