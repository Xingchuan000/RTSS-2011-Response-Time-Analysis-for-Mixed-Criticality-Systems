from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .executable_transition_ir import (
    Assignment, BoolExpr, IntExpr, GeneratedEventRule,
    const_expr, var_expr,
)

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class StateUpdate:
    state_assignments: tuple[Assignment, ...]
    frame_assignments: tuple[Assignment, ...]
    time_assignment: Assignment | None = None
    generated_events: tuple[GeneratedEventRule, ...] = ()

    def is_empty(self) -> bool:
        return (not self.state_assignments
                and not self.frame_assignments
                and self.time_assignment is None
                and not self.generated_events)


@dataclass(frozen=True, slots=True)
class BoundTransitionCase:
    case_id: str
    source_function: str
    source_ast_hash: str
    guard: BoolExpr
    full_update: StateUpdate
    prefix_update: StateUpdate
    frame_fields: frozenset[str]
    generated_event_phase_constraints: tuple[BoolExpr, ...]
    binding_status: Literal["CODE_BOUND", "UNRESOLVED"]


def skip_state_update() -> StateUpdate:
    return StateUpdate(
        state_assignments=(),
        frame_assignments=(),
        time_assignment=None,
        generated_events=(),
    )
