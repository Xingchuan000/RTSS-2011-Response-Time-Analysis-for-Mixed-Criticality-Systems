"""Phase-indexed simulation relations for the full-to-prefix weak simulation.

Defines RelPP_* relations at each canonical phase boundary:
  RelPP_SvcEnd, RelPP_AfterREM, RelPP_AfterREC,
  RelPP_DDLCursor(k_full, k_prefix), RelPP_ARRCursor(k_full, k_prefix),
  RelPP_PreDisp, RelPP_Close

Each relation preserves:
  - time
  - protected released ledger
  - active/ready/running projection
  - release time and absolute deadline
  - static priority and criticality
  - actual demand and HI normal/abnormal class
  - accumulated service
  - completion and miss ledger

Each relation explicitly excludes:
  - global mode
  - protected LO primary/degraded label
  - tail jobs
  - full/prefix frontier tail-only events
  - trigger identity
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ProtectedJobRelationState:
    job_key: tuple[str, int]
    task_name: str
    criticality: str
    release_time: int
    absolute_deadline: int
    priority_index: int
    actual_demand: int
    hi_class: str | None
    executed_service: int
    active: bool
    ready: bool
    running: bool
    completed: bool
    missed: bool


@dataclass(frozen=True, slots=True)
class PhaseRelationState:
    phase: str
    time: int
    protected_jobs: tuple[ProtectedJobRelationState, ...]
    running_job_key: tuple[str, int] | None
    miss_job_keys: frozenset[tuple[str, int]]


PHASE_ORDER = (
    "SvcEnd",
    "AfterREM",
    "AfterREC",
    "DDLCursor",
    "ARRCursor",
    "PreDisp",
    "Close",
)


def phase_index_for(name: str) -> int:
    try:
        return PHASE_ORDER.index(name)
    except ValueError:
        return -1


def next_phase(current: str) -> str | None:
    idx = phase_index_for(current)
    if idx < 0 or idx >= len(PHASE_ORDER) - 1:
        return None
    return PHASE_ORDER[idx + 1]


def build_phase_relation(
    full_state: dict[str, Any],
    prefix_state: dict[str, Any],
    phase: str,
    k_full: int = 0,
    k_prefix: int = 0,
) -> Mapping[str, Any]:
    """Build a phase-specific relation between full and prefix states.

    The relation is phase-indexed; DDLCursor and ARRCursor use batch
    cursors (k_full, k_prefix) to track position within deadline and
    arrival batches respectively.
    """
    return {
        "phase": phase,
        "time_full": full_state.get("time"),
        "time_prefix": prefix_state.get("time"),
        "k_full": k_full if phase in ("DDLCursor", "ARRCursor") else None,
        "k_prefix": k_prefix if phase in ("DDLCursor", "ARRCursor") else None,
        "schema_version": "phase_relation_v1",
    }


def relation_includes_field(field_name: str) -> bool:
    """Check whether a field is included in the phase-indexed relation."""
    excluded = {
        "global_mode",
        "primary_degraded_label",
        "tail_jobs",
        "tail_only_event",
        "trigger_identity",
    }
    return field_name not in excluded


def relation_preserves_field(field_name: str) -> bool:
    """Check whether a field is preserved by all phase relations."""
    preserved = {
        "time",
        "release_time",
        "absolute_deadline",
        "priority_index",
        "criticality",
        "actual_demand",
        "hi_class",
        "executed_service",
        "completed",
        "missed",
    }
    return field_name in preserved
