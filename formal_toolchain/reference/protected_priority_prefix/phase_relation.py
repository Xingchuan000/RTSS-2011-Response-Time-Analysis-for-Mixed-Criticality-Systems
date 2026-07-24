"""Phase-indexed protected-observable relation.

This module performs an actual structural comparison of the protected
observable.  It deliberately excludes global mode, LO version labels and tail
state, but it never treats missing per-job fields as equality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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
    protected_pending_releases: tuple[Mapping[str, Any], ...]
    running_job_key: tuple[str, int] | None
    miss_job_keys: frozenset[tuple[str, int]]


PHASE_ORDER = (
    "SvcEnd", "AfterREM", "AfterREC", "DDLCursor", "ARRCursor", "PreDisp", "Close",
)

STATE_FIELDS = ("time", "pending_releases", "running_job_key", "miss_job_keys")
JOB_FIELDS = (
    "job_key", "task_name", "criticality", "release_time", "absolute_deadline",
    "priority_index", "actual_demand", "hi_class", "executed_service", "active",
    "ready", "running", "completed", "missed",
)
PENDING_RELEASE_FIELDS = (
    "job_key", "task_name", "criticality", "release_time", "absolute_deadline",
    "priority_index", "actual_demand", "hi_class",
)
EXCLUDED_FIELDS = (
    "global_mode", "primary_degraded_label", "protected_lo_primary_degraded_label",
    "tail_jobs", "tail_only_event", "switch_trigger_identity",
    "pending_effective_release_mode", "pending_lo_version_label",
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


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"PROTECTED_OBSERVABLE_NOT_MAPPING:{type(value).__name__}")


def _job_map(observable: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    raw_jobs = observable.get("jobs", observable.get("protected_jobs"))
    if not isinstance(raw_jobs, (tuple, list)):
        raise ValueError("PROTECTED_OBSERVABLE_JOBS_MISSING")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in raw_jobs:
        job = _mapping(raw)
        missing = [name for name in JOB_FIELDS if name not in job]
        if missing:
            raise ValueError(f"PROTECTED_JOB_FIELDS_MISSING:{','.join(missing)}")
        raw_key = job["job_key"]
        if not isinstance(raw_key, (tuple, list)) or len(raw_key) != 2:
            raise ValueError("PROTECTED_JOB_KEY_INVALID")
        key = (str(raw_key[0]), int(raw_key[1]))
        if key in result:
            raise ValueError(f"PROTECTED_JOB_KEY_DUPLICATE:{key[0]}:{key[1]}")
        job["job_key"] = key
        result[key] = job
    return result




def _pending_release_map(observable: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    raw_pending = observable.get("pending_releases", ())
    if not isinstance(raw_pending, (tuple, list)):
        raise ValueError("PROTECTED_PENDING_RELEASES_INVALID")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in raw_pending:
        pending = _mapping(raw)
        missing = [name for name in PENDING_RELEASE_FIELDS if name not in pending]
        if missing:
            raise ValueError(f"PROTECTED_PENDING_FIELDS_MISSING:{','.join(missing)}")
        raw_key = pending["job_key"]
        if not isinstance(raw_key, (tuple, list)) or len(raw_key) != 2:
            raise ValueError("PROTECTED_PENDING_JOB_KEY_INVALID")
        key = (str(raw_key[0]), int(raw_key[1]))
        if key in result:
            raise ValueError(f"PROTECTED_PENDING_JOB_KEY_DUPLICATE:{key[0]}:{key[1]}")
        pending["job_key"] = key
        result[key] = pending
    return result

def _normalize_key_set(value: Any) -> tuple[tuple[str, int], ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list, set, frozenset)):
        raise ValueError("PROTECTED_JOB_KEY_SET_INVALID")
    return tuple(sorted((str(k[0]), int(k[1])) for k in value))


def check_phase_relation(
    full_observable: Mapping[str, Any] | Any,
    prefix_observable: Mapping[str, Any] | Any,
    phase: str,
    k_full: int = 0,
    k_prefix: int = 0,
) -> dict[str, Any]:
    """Return PASS exactly when the protected observables are equal.

    Cursor values are metadata for DDL/arrival batch induction.  They must be
    non-negative integers, but they need not be equal because the full cursor
    may skip tail entries.
    """
    checks: dict[str, bool] = {}
    try:
        if phase_index_for(phase) < 0:
            raise ValueError("PROTECTED_PHASE_UNKNOWN")
        if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in (k_full, k_prefix)):
            raise ValueError("PROTECTED_PHASE_CURSOR_INVALID")
        full = _mapping(full_observable)
        prefix = _mapping(prefix_observable)

        for field in EXCLUDED_FIELDS:
            checks[f"excluded_{field}"] = field not in full and field not in prefix

        checks["time"] = full.get("time") == prefix.get("time")
        checks["running_job_key"] = (
            None if full.get("running_job_key") is None else tuple(full["running_job_key"])
        ) == (
            None if prefix.get("running_job_key") is None else tuple(prefix["running_job_key"])
        )
        checks["miss_job_keys"] = _normalize_key_set(full.get("miss_job_keys")) == _normalize_key_set(prefix.get("miss_job_keys"))

        full_jobs = _job_map(full)
        prefix_jobs = _job_map(prefix)
        checks["job_key_set"] = set(full_jobs) == set(prefix_jobs)
        for key in sorted(set(full_jobs) | set(prefix_jobs)):
            if key not in full_jobs or key not in prefix_jobs:
                checks[f"job[{key}].present"] = False
                continue
            for field in JOB_FIELDS:
                checks[f"job[{key}].{field}"] = full_jobs[key][field] == prefix_jobs[key][field]

        full_pending = _pending_release_map(full)
        prefix_pending = _pending_release_map(prefix)
        checks["pending_job_key_set"] = set(full_pending) == set(prefix_pending)
        for key in sorted(set(full_pending) | set(prefix_pending)):
            if key not in full_pending or key not in prefix_pending:
                checks[f"pending[{key}].present"] = False
                continue
            for field in PENDING_RELEASE_FIELDS:
                checks[f"pending[{key}].{field}"] = (
                    full_pending[key][field] == prefix_pending[key][field]
                )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return {
            "phase": phase, "k_full": k_full, "k_prefix": k_prefix,
            "schema_version": "phase_relation_v4_close_at", "status": "FAIL",
            "equality_checks": checks, "failed_fields": [str(exc)],
            "failure": {"code": "PROTECTED_PHASE_RELATION_INPUT_INVALID", "reason": str(exc)},
        }

    all_equal = bool(checks) and all(checks.values())
    return {
        "phase": phase,
        "k_full": k_full if phase in ("DDLCursor", "ARRCursor") else None,
        "k_prefix": k_prefix if phase in ("DDLCursor", "ARRCursor") else None,
        "schema_version": "phase_relation_v4_close_at",
        "status": "PASS" if all_equal else "FAIL",
        "equality_checks": checks,
        "failed_fields": [name for name, value in checks.items() if not value],
    }


def build_phase_relation(
    full_state: Mapping[str, Any] | Any,
    prefix_state: Mapping[str, Any] | Any,
    phase: str,
    k_full: int = 0,
    k_prefix: int = 0,
) -> Mapping[str, Any]:
    return check_phase_relation(full_state, prefix_state, phase, k_full, k_prefix)


def relation_includes_field(field_name: str) -> bool:
    return field_name not in EXCLUDED_FIELDS


def relation_preserves_field(field_name: str) -> bool:
    return (
        field_name in STATE_FIELDS
        or field_name in JOB_FIELDS
        or field_name in PENDING_RELEASE_FIELDS
    )
