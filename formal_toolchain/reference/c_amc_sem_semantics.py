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


@dataclass(frozen=True, slots=True)
class ReferenceReleaseDecision:
    release_class: str
    effective_release_mode: str
    release_budget: int


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


def decide_reference_release(
    *,
    task: Any,
    mode_before_batch: str,
    mode_after_batch: str,
    abnormal_hi: bool,
    is_switch_trigger: bool,
    switched_in_this_batch: bool,
    primary_on_switch_time: bool,
) -> ReferenceReleaseDecision:
    criticality = str(_field(task, "criticality"))
    c_lo = int(_field(task, "c_lo"))
    c_hi = int(_field(task, "c_hi"))

    same_switch_batch_uses_lo_mode = (
        switched_in_this_batch
        and primary_on_switch_time
        and mode_before_batch == "LO"
        and mode_after_batch == "HI"
    )

    effective_mode = (
        "LO"
        if same_switch_batch_uses_lo_mode
        else mode_after_batch
    )

    if criticality == "HI":
        if is_switch_trigger:
            if not (
                abnormal_hi
                and mode_before_batch == "LO"
                and mode_after_batch == "HI"
                and switched_in_this_batch
            ):
                raise ValueError(
                    "REFERENCE_SWITCH_TRIGGER_COMBINATION_INVALID"
                )

            return ReferenceReleaseDecision(
                release_class="HI_ABNORMAL_SWITCH_TRIGGER",
                effective_release_mode=effective_mode,
                release_budget=c_hi,
            )

        return ReferenceReleaseDecision(
            release_class="HI_NORMAL",
            effective_release_mode=effective_mode,
            release_budget=(
                c_hi
                if effective_mode == "HI"
                else c_lo
            ),
        )

    if criticality != "LO":
        raise ValueError(
            "REFERENCE_CRITICALITY_INVALID"
        )

    if abnormal_hi or is_switch_trigger:
        raise ValueError(
            "REFERENCE_LO_CANNOT_BE_ABNORMAL_HI"
        )

    if same_switch_batch_uses_lo_mode:
        return ReferenceReleaseDecision(
            release_class="LO_PRIMARY_SAME_BATCH_SWITCH_TIME",
            effective_release_mode="LO",
            release_budget=c_lo,
        )

    if effective_mode == "HI":
        return ReferenceReleaseDecision(
            release_class="LO_DEGRADED_HI_MODE",
            effective_release_mode="HI",
            release_budget=c_hi,
        )

    return ReferenceReleaseDecision(
        release_class="LO_PRIMARY_NORMAL",
        effective_release_mode="LO",
        release_budget=c_lo,
    )
