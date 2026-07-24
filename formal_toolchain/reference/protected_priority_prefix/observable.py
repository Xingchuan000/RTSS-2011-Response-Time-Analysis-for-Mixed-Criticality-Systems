"""The observable state used by the protected-prefix simulation relation.

The projection deliberately omits global mode and the full-system tail.  Those
are implementation state, not protected demand state, and may legitimately
stutter differently after the tail is removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from formal_toolchain.adapters.formal_runtime_snapshot import MissRecord
from formal_toolchain.reference.reference_state import ReferenceState

from .types import ProtectedPrefixBuildResult

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ProtectedJobObservable:
    job_key: JobKey
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
class ProtectedPendingReleaseObservable:
    job_key: JobKey
    task_name: str
    criticality: str
    release_time: int
    absolute_deadline: int
    priority_index: int
    actual_demand: int
    hi_class: str | None


@dataclass(frozen=True, slots=True)
class ProtectedStateObservable:
    time: int
    jobs: tuple[ProtectedJobObservable, ...]
    pending_releases: tuple[ProtectedPendingReleaseObservable, ...]
    running_job_key: JobKey | None
    miss_job_keys: tuple[JobKey, ...]


def _hi_class(criticality: str, release_class: str | None) -> str | None:
    if criticality != "HI":
        return None
    if release_class == "HI_NORMAL":
        return "NORMAL"
    if release_class in {"HI_ABNORMAL", "HI_ABNORMAL_SWITCH_TRIGGER"}:
        return "ABNORMAL"
    raise ValueError(f"PROTECTED_HI_CLASS_INVALID:{release_class}")


def project_protected_state(
    state: ReferenceState,
    *,
    protected_task_names: frozenset[str],
    taskset: object,
) -> ProtectedStateObservable:
    """Project all released protected jobs, including terminal jobs.

    Keeping terminal records in the projection makes completion/removal and
    service equality observable without exposing mode, frontier, or tail state.
    """
    task_by_name = {str(getattr(task, "name")): task for task in taskset.tasks}
    keys = sorted(
        (key for key in state.released if key[0] in protected_task_names),
        key=lambda key: (key[1], key[0]),
    )
    miss_keys = tuple(sorted(
        {miss.job_key for miss in state.misses if miss.job_key[0] in protected_task_names}
    ))
    miss_set = set(miss_keys)
    jobs: list[ProtectedJobObservable] = []
    for key in keys:
        record = state.released[key]
        active_job = state.jobs.get(key)
        terminal = state.terminal.get(key)
        task = task_by_name[key[0]]
        service = active_job.executed if active_job is not None else (
            terminal.executed_service if terminal is not None else 0
        )
        completed = terminal is not None and terminal.terminal_kind == "COMPLETED"
        jobs.append(ProtectedJobObservable(
            job_key=key,
            task_name=key[0],
            criticality=record.criticality,
            release_time=record.release_time,
            absolute_deadline=record.absolute_deadline,
            priority_index=int(getattr(task, "priority_index")),
            actual_demand=int(record.removal_demand),
            hi_class=_hi_class(record.criticality, record.release_class),
            executed_service=int(service),
            active=active_job is not None,
            ready=key in state.ready_order,
            running=key == state.running,
            completed=completed,
            missed=key in miss_set,
        ))
    pending: list[ProtectedPendingReleaseObservable] = []
    for key in sorted(
        (key for key in state.pending_releases if key[0] in protected_task_names),
        key=lambda item: (item[1], item[0]),
    ):
        plan = state.pending_releases[key]
        pending.append(ProtectedPendingReleaseObservable(
            job_key=key,
            task_name=key[0],
            criticality=str(plan.criticality),
            release_time=int(plan.release_time),
            absolute_deadline=int(plan.absolute_deadline),
            priority_index=int(plan.priority_index),
            actual_demand=int(plan.removal_demand),
            hi_class=_hi_class(str(plan.criticality), str(plan.release_class)),
        ))

    running = state.running if state.running and state.running[0] in protected_task_names else None
    return ProtectedStateObservable(
        time=int(state.time), jobs=tuple(jobs), pending_releases=tuple(pending),
        running_job_key=running, miss_job_keys=miss_keys,
    )


def observable_schema() -> dict[str, object]:
    return {
        "version": "protected-observable-v2-pending-release",
        "state_fields": [
            "time", "jobs", "pending_releases", "running_job_key", "miss_job_keys"
        ],
        "job_fields": [
            "job_key", "task_name", "criticality", "release_time",
            "absolute_deadline", "priority_index", "actual_demand", "hi_class",
            "executed_service", "active", "ready", "running", "completed", "missed",
        ],
        "pending_release_fields": [
            "job_key", "task_name", "criticality", "release_time",
            "absolute_deadline", "priority_index", "actual_demand", "hi_class",
        ],
        "excluded": [
            "global_mode", "protected_lo_primary_degraded_label",
            "pending_effective_release_mode", "pending_lo_version_label", "tail_jobs",
        ],
    }
