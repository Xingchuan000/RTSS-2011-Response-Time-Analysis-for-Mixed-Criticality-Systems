"""Bind PP0 cases to the path-sensitive executable compiler output."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .executable_transition_compiler import compile_all_transitions as _compile_ir
from .executable_transition_ir import CompiledTransitionIR, raw_expr, state_assignment
from .pp_transition_semantics import BoundTransitionCase, StateUpdate

RELATION_SCHEMA_HASH = sha256_object({
    "schema": "phase_relation_v4_close_at",
    "binding": "compiled_semantic_effect_v1",
})


def _canonical_case_id(case_id: str) -> str:
    return {
        "DDL_OBSERVE": "DEADLINE_OBSERVATION",
        "ARRIVAL_BATCH_OPEN": "ARRIVAL_BATCH",
    }.get(case_id, case_id)


def executable_binding_ok(ir: CompiledTransitionIR, fresh_source: Any,
                          relation_schema: Any) -> bool:
    fresh_hash = (
        getattr(fresh_source, "source_function_ast_hash", None)
        or getattr(fresh_source, "ast_hash", None)
    )
    schema_hash = (
        relation_schema.get("hash") if isinstance(relation_schema, Mapping)
        else getattr(relation_schema, "hash", None)
    )
    return bool(
        ir.is_compiled()
        and ir.source_function_ast_hash == fresh_hash
        and schema_hash == RELATION_SCHEMA_HASH
        and ir.semantic_effect is not None
        and ir.semantic_effect.derivation_complete
        and ir.semantic_effect.concrete_write_targets
        == ir.semantic_effect.covered_concrete_write_targets
        and ir.semantic_effect.source_path_hashes
        == tuple(path.path_hash() for path in ir.paths)
    )


def _state_update(ir: CompiledTransitionIR) -> StateUpdate:
    effect = ir.semantic_effect
    assignments = tuple(
        state_assignment(f"{field}_post", raw_expr(expression))
        for field, expression in (effect.field_equations if effect else ())
    )
    return StateUpdate(
        state_assignments=assignments,
        frame_assignments=(),
        time_assignment=next((item for item in assignments if item.target == "time_post"), None),
        generated_events=ir.generated_events,
    )


def _bind(ir: CompiledTransitionIR, fresh_map: Mapping[str, CompiledTransitionIR]) -> BoundTransitionCase:
    fresh = fresh_map.get(ir.case_id)
    ok = fresh is not None and executable_binding_ok(
        ir, fresh, {"hash": RELATION_SCHEMA_HASH})
    effect = ir.semantic_effect
    return BoundTransitionCase(
        case_id=_canonical_case_id(ir.case_id),
        source_function=f"{ir.source_module}.{ir.source_function}",
        source_ast_hash=ir.source_function_ast_hash,
        guard=ir.precondition,
        full_update=_state_update(ir),
        prefix_update=_state_update(ir),
        frame_fields=effect.frame_fields if effect else frozenset(),
        generated_event_phase_constraints=tuple(
            event.condition for event in ir.generated_events
            if event.condition is not None
        ),
        binding_status="CODE_BOUND" if ok else "UNRESOLVED",
        compiled_ir_hash=ir.ir_hash(),
        semantic_effect_hash=effect.effect_hash() if effect else None,
        path_hashes=tuple(path.path_hash() for path in ir.paths),
        required_assumption_ids=effect.required_assumption_ids if effect else (),
        projection_derivation_complete=bool(effect and effect.derivation_complete),
        total_semantic_coverage=ir.total_semantic_coverage,
    )


def compile_all_transitions() -> dict[str, BoundTransitionCase]:
    first = tuple(_compile_ir())
    # Fresh independent recompile is part of translation validation.
    fresh = {item.case_id: item for item in _compile_ir(fresh=True)}
    return {item.case_id: _bind(item, fresh) for item in first}


def bound_transition_for_case(case_id: str) -> BoundTransitionCase | None:
    return compile_all_transitions().get(_canonical_case_id(case_id))


BOUND_TRANSITIONS = compile_all_transitions()
