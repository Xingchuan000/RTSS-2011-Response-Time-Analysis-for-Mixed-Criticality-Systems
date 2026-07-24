"""Canonical primitive transition schemas for the protected-prefix runtime.

Each schema defines the read/write/frame fields for one primitive case in the
C-AMC-sem reference executable semantics.  These are used to generate SMT2
queries that verify transition equations over all reachable states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PrimitiveTransitionSchema:
    case_id: str
    guard_fields: tuple[str, ...]
    read_fields: tuple[str, ...]
    write_fields: tuple[str, ...]
    protected_frame_fields: tuple[str, ...]
    time_delta: int


CANONICAL_CASES: tuple[PrimitiveTransitionSchema, ...] = (
    PrimitiveTransitionSchema(
        case_id="REM_COMPLETION",
        guard_fields=("service_ge_fixed_demand",),
        read_fields=("service", "fixed_demand", "job_key", "criticality"),
        write_fields=("active", "ready", "completed", "removed"),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality", "fixed_demand",
            "priority_index", "task_name", "hi_class",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="RECOVERY",
        guard_fields=("mode_is_hi", "no_active_jobs", "no_running_job", "no_pending_releases"),
        read_fields=("mode", "active_job_count", "running", "pending_release_count"),
        write_fields=("mode", "primary_on_switch_time"),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality",
            "fixed_demand", "service", "job_key", "priority_index",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="DDL_OBSERVE",
        guard_fields=("deadline_event_at_time",),
        read_fields=("absolute_deadline", "completion_time", "job_key"),
        write_fields=("missed", "miss_ledger"),
        protected_frame_fields=(
            "release_time", "criticality", "fixed_demand", "service",
            "active", "ready", "priority_index",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="ARRIVAL_BATCH_OPEN",
        guard_fields=("arrival_event_at_time",),
        read_fields=("release_demand_overrides", "abnormal_hi_releases", "ghost_future_budgets"),
        write_fields=("pending_releases", "frontier"),
        protected_frame_fields=(
            "key_identity", "release_time", "release_index", "criticality",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="MODE_SWITCH",
        guard_fields=("mode_is_lo", "pending_abnormal_switch_trigger"),
        read_fields=("mode", "pending_releases", "switch_trigger"),
        write_fields=("mode",),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality",
            "fixed_demand", "service", "active", "ready",
            "priority_index", "job_key",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="RELEASE",
        guard_fields=("release_event_at_time",),
        read_fields=("release_demand_overrides", "abnormal_hi_releases", "task_parameters"),
        write_fields=("active", "ready", "released_ledger", "arrival_demand"),
        protected_frame_fields=(
            "job_key", "release_time", "criticality", "priority_index",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="FINAL_DISPATCH",
        guard_fields=("active_set_nonempty",),
        read_fields=("active_set", "priority_index", "release_time", "job_key"),
        write_fields=("running",),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality",
            "fixed_demand", "service", "priority_index",
        ),
        time_delta=0,
    ),
    PrimitiveTransitionSchema(
        case_id="SERVICE_UNIT",
        guard_fields=("running_job_exists",),
        read_fields=("running_job", "fixed_demand", "service"),
        write_fields=("service", "state_time"),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality",
            "priority_index", "job_key", "active", "ready",
        ),
        time_delta=1,
    ),
    PrimitiveTransitionSchema(
        case_id="TAIL_ONLY_SERVICE",
        guard_fields=("protected_ready_empty", "tail_ready_nonempty"),
        read_fields=("tail_running_job", "tail_fixed_demand", "tail_service"),
        write_fields=("tail_service", "state_time"),
        protected_frame_fields=(
            "release_time", "absolute_deadline", "criticality",
            "fixed_demand", "service", "active", "ready",
            "priority_index", "job_key",
        ),
        time_delta=1,
    ),
)


def canonical_case_ids() -> tuple[str, ...]:
    return tuple(case.case_id for case in CANONICAL_CASES)


def schema_for_case(case_id: str) -> PrimitiveTransitionSchema | None:
    for case in CANONICAL_CASES:
        if case.case_id == case_id:
            return case
    return None


def transition_obligations() -> dict[str, str]:
    """Return a map from obligation key to its natural-language statement."""
    return {
        "FIXED_DEMAND_NOT_MODIFIED": (
            "For every primitive case, protected fixed demand is never modified "
            "by the transition; it is an input-only field."
        ),
        "PROTECTED_KEY_NOT_MODIFIED": (
            "Protected job key, release time, deadline, criticality, and "
            "priority are never modified by any transition."
        ),
        "MODE_ONLY_NOT_MODIFY_PROTECTED": (
            "Mode-only steps (RECOVERY, MODE_SWITCH) do not modify protected "
            "active, ready, running, service, or miss fields."
        ),
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": (
            "TAIL_ONLY_SERVICE does not modify any protected observable field."
        ),
        "DDL_READ_ONLY_DEADLINE_COMPLETION": (
            "DDL_OBSERVE only reads deadline/completion, monotonically "
            "appends to the miss ledger."
        ),
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": (
            "REM_COMPLETION guard is equivalent to service >= fixed_demand."
        ),
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": (
            "FINAL_DISPATCH selects one ready job by strict fixed-priority "
            "total order."
        ),
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": (
            "SERVICE_UNIT increases service by exactly one discrete unit "
            "for the running job only."
        ),
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": (
            "Protected entries in an arrival batch do not depend on the "
            "deletion of tail entries."
        ),
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": (
            "Closure phases within a same-timestamp loop have a fixed, "
            "finite, strictly decreasing measure."
        ),
    }
