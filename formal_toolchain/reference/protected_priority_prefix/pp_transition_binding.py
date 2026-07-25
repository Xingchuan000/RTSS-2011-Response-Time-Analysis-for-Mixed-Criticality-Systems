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
from .executable_transition_compiler import compile_all_transitions as _compiler_compile_all
from .executable_transition_ir import CompiledTransitionIR

RELATION_SCHEMA_HASH = sha256_object({"schema": "phase_relation_v4_close_at"})


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
    """Bind transition cases only from a fresh total executable compilation."""
    return _compiled_bindings()


def bound_transition_for_case(case_id: str) -> BoundTransitionCase | None:
    return _compiled_bindings().get(_canonical_case_id(case_id))


def _canonical_case_id(case_id: str) -> str:
    return {"DDL_OBSERVE": "DEADLINE_OBSERVATION",
            "ARRIVAL_BATCH_OPEN": "ARRIVAL_BATCH"}.get(case_id, case_id)


def executable_binding_ok(ir: CompiledTransitionIR, fresh_source: Any, relation_schema: Any) -> bool:
    fresh_hash = getattr(fresh_source, "ast_hash", None) or getattr(fresh_source, "source_function_ast_hash", None)
    schema_hash = getattr(relation_schema, "hash", None) or (
        relation_schema.get("hash") if isinstance(relation_schema, Mapping) else None)
    return bool(
        ir.compilation_status == "COMPILED"
        and ir.total_semantic_coverage
        and ir.source_function_ast_hash == fresh_hash
        and ir.covered_return_path_count == ir.return_path_count
        and ir.covered_raise_path_count == ir.raise_path_count
        and schema_hash == RELATION_SCHEMA_HASH
    )


def _state_update(ir: CompiledTransitionIR) -> StateUpdate:
    assignments = tuple(a for a in ir.post_equations if a.kind == "state")
    frames = tuple(a for a in ir.post_equations if a.kind == "frame")
    return StateUpdate(assignments, frames,
                       time_assignment=next((a for a in ir.post_equations if a.kind == "time"), None),
                       generated_events=ir.generated_events)


def _bind_compiled_ir(ir: CompiledTransitionIR) -> BoundTransitionCase:
    fresh = _compiler_compile_all()
    fresh_ir = next((x for x in fresh if x.case_id == ir.case_id), None)
    fresh_source = fresh_ir or ir
    ok = executable_binding_ok(ir, fresh_source, {"hash": RELATION_SCHEMA_HASH})
    return BoundTransitionCase(
        case_id=_canonical_case_id(ir.case_id),
        source_function=f"{ir.source_module}.{ir.source_function}",
        source_ast_hash=ir.source_function_ast_hash,
        guard=ir.precondition,
        full_update=_state_update(ir),
        prefix_update=_state_update(ir),
        frame_fields=ir.frame_fields,
        generated_event_phase_constraints=tuple(
            e.condition for e in ir.generated_events if e.condition is not None),
        binding_status="CODE_BOUND" if ok else "UNRESOLVED",
    )


def _compiled_bindings() -> dict[str, BoundTransitionCase]:
    return {_canonical_case_id(ir.case_id): _bind_compiled_ir(ir)
            for ir in _compiler_compile_all()}
