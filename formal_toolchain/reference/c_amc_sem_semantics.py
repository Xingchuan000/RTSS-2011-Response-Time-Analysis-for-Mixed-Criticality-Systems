from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from formal_toolchain.reference.reference_state import JobKey


def _field(task: Any, name: str) -> Any:
    return task[name] if isinstance(task, Mapping) else getattr(task, name)


def _tasks(taskset: Any) -> tuple[Any, ...]:
    return tuple(taskset.tasks if hasattr(taskset, "tasks") else taskset.get("tasks", ()))


def _task(taskset: Any, name: str) -> Any:
    matches = [task for task in _tasks(taskset) if str(_field(task, "name")) == name]
    if len(matches) != 1:
        raise ValueError(f"REFERENCE_TASK_NOT_UNIQUE:{name}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class ReferenceBatchClassification:
    mode_before: str
    mode_after: str
    switch_trigger: JobKey | None
    abnormal_hi_jobs: tuple[JobKey, ...]


def classify_arrival_batch(
    *,
    mode_before: str,
    batch_jobs: tuple[JobKey, ...],
    taskset: Any,
    abnormal_hi_releases: frozenset[JobKey],
) -> ReferenceBatchClassification:
    if mode_before not in {"LO", "HI"}:
        raise ValueError("REFERENCE_MODE_INVALID")
    abnormal = tuple(sorted(
        (jk for jk in batch_jobs
         if jk in abnormal_hi_releases
         and str(_field(_task(taskset, jk[0]), "criticality")) == "HI"),
        key=lambda jk: (int(_field(_task(taskset, jk[0]), "priority_index")), jk),
    ))
    trigger = abnormal[0] if mode_before == "LO" and abnormal else None
    return ReferenceBatchClassification(
        mode_before=mode_before,
        mode_after="HI" if trigger is not None else mode_before,
        switch_trigger=trigger,
        abnormal_hi_jobs=abnormal,
    )


def release_class_and_budget(
    *,
    task: Any,
    mode_before_batch: str,
    mode_after_batch: str,
    abnormal_hi: bool,
    switched_in_this_batch: bool,
    primary_on_switch_time: bool,
) -> tuple[str, str, int]:
    criticality = str(_field(task, "criticality"))
    c_lo = int(_field(task, "c_lo"))
    c_hi = int(_field(task, "c_hi"))
    if criticality == "HI":
        if abnormal_hi and mode_before_batch == "LO" and mode_after_batch == "HI":
            return "HI_ABNORMAL_SWITCH_TRIGGER", mode_after_batch, c_hi
        return "HI_NORMAL", mode_after_batch, c_hi if mode_after_batch == "HI" else c_lo
    if mode_after_batch == "HI":
        if switched_in_this_batch and primary_on_switch_time:
            return "LO_PRIMARY_SAME_BATCH_SWITCH_TIME", mode_after_batch, c_lo
        return "LO_DEGRADED_HI_MODE", mode_after_batch, c_hi
    return "LO_PRIMARY_NORMAL", mode_after_batch, c_lo
