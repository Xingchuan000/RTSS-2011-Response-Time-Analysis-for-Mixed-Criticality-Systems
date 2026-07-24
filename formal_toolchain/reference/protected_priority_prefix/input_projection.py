"""Projection of full-reference release inputs to a protected prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.bridge.logical_events import LogicalEventKind
from formal_toolchain.reference.executable_semantics import initial_reference_state
from formal_toolchain.reference.reference_state import ReferenceState

from .types import ProtectedPrefixBuildResult

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ProtectedReleaseInput:
    job_key: JobKey
    task_name: str
    release_time: int
    actual_demand: int
    hi_class: str | None


def _task_map(taskset: object) -> dict[str, Any]:
    return {str(task.name): task for task in taskset.tasks}


def project_protected_release_stream(
    full_initial_state: ReferenceState,
    *,
    protected_task_names: frozenset[str],
) -> tuple[ProtectedReleaseInput, ...]:
    """Read the authoritative initial arrival batch and preserve job indices."""
    result: list[ProtectedReleaseInput] = []
    # ReferenceState intentionally carries no taskset pointer.  Initial release
    # events contain the authoritative time/key; task metadata is recovered from
    # pending/released records when available and otherwise represented by the key.
    for event in full_initial_state.frontier:
        if event.kind is not LogicalEventKind.ARR_BATCH:
            continue
        for key in event.batch_jobs:
            if key[0] not in protected_task_names:
                continue
            if key not in full_initial_state.release_demand_overrides:
                raise ValueError(f"PROTECTED_INPUT_DEMAND_NOT_FIXED:{key[0]}:{key[1]}")
            result.append(ProtectedReleaseInput(
                job_key=key, task_name=key[0], release_time=int(event.time),
                actual_demand=int(full_initial_state.release_demand_overrides[key]),
                hi_class=("ABNORMAL" if key in full_initial_state.abnormal_hi_releases else None),
            ))
    return tuple(sorted(result, key=lambda item: (item.release_time, item.job_key[0], item.job_key[1])))


def build_prefix_initial_state_from_full_inputs(
    full_initial_state: ReferenceState,
    prefix_taskset: object,
    construction: ProtectedPrefixBuildResult,
) -> ReferenceState:
    """Construct prefix state without consuming or regenerating shared randomness."""
    protected = frozenset(construction.protected_task_names)
    overrides = {key: value for key, value in full_initial_state.release_demand_overrides.items()
                 if key[0] in protected}
    abnormal = frozenset(key for key in full_initial_state.abnormal_hi_releases if key[0] in protected)
    ghost = {name: value for name, value in full_initial_state.ghost_future_budgets.items()
             if name in protected}
    return initial_reference_state(
        prefix_taskset,
        abnormal_hi_releases=abnormal,
        primary_on_switch_time=full_initial_state.primary_on_switch_time,
        release_demand_overrides=overrides,
        ghost_future_budgets=ghost,
    )


def check_projected_demands_legal(
    projected_inputs: tuple[ProtectedReleaseInput, ...],
    prefix_taskset: object,
) -> dict[str, Any]:
    tasks = _task_map(prefix_taskset)
    checks: list[dict[str, Any]] = []
    for item in projected_inputs:
        task = tasks.get(item.task_name)
        if task is None:
            checks.append({"job_key": item.job_key, "legal": False, "code": "TASK_MISSING"})
            continue
        bound = int(task.c_hi if item.hi_class == "ABNORMAL" else task.c_lo)
        checks.append({"job_key": item.job_key, "actual_demand": item.actual_demand,
                       "bound": bound, "legal": 0 < item.actual_demand <= bound})
    return {"status": "PASS" if all(row["legal"] for row in checks) else "FAIL",
            "checks": checks}
