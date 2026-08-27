"""Static, immutable saturated Protected Priority Prefix construction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset

from .types import ProtectedPrefixBuildResult


class ProtectedPrefixConstructionError(ValueError):
    pass


def _fingerprint(taskset: ReferenceTaskset) -> str:
    return str(taskset.to_dict()["fingerprint"])


def build_saturated_protected_prefix(
    full_reference: ReferenceTaskset, *, source_context_hash: str,
) -> ProtectedPrefixBuildResult:
    tasks = full_reference.tasks
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_PARTITION_INVALID")
    if tuple(task.priority_index for task in tasks) != tuple(range(len(tasks))):
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_PARTITION_INVALID")
    hi_indices = [i for i, task in enumerate(tasks) if task.criticality == "HI"]
    if not hi_indices:
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_NO_HI_TASK")
    cutoff = max(hi_indices)
    protected = tasks[:cutoff + 1]
    tail = tasks[cutoff + 1:]
    if any(task.criticality != "LO" for task in tail):
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_TAIL_NOT_LO_ONLY")
    if any(task.criticality == "HI" for task in tasks if task not in protected):
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_PARTITION_INVALID")

    transformed: list[ReferenceTask] = []
    changes: list[dict[str, Any]] = []
    for task in protected:
        before = asdict(task)
        if task.criticality == "LO":
            after = ReferenceTask(
                name=task.name, period=task.period, deadline=task.deadline,
                c_lo=task.c_lo, c_hi=task.c_lo, criticality=task.criticality,
                priority_index=task.priority_index, code_c_lo=task.code_c_lo,
                code_c_hi=task.code_c_hi, degraded_cost=task.degraded_cost,
                offset=task.offset,
            )
        else:
            after = task
        transformed.append(after)
        changes.append({"name": task.name, "before": before, "after": asdict(after)})
    try:
        prefix = ReferenceTaskset(tuple(transformed), source_context_hash)
    except (TypeError, ValueError) as exc:
        raise ProtectedPrefixConstructionError("PROTECTED_PREFIX_PARTITION_INVALID") from exc
    full_fp = _fingerprint(full_reference)
    prefix_fp = _fingerprint(prefix)
    partition = {
        "full_fingerprint": full_fp, "full_priority_order": names,
        "cutoff_task_name": tasks[cutoff].name, "cutoff_priority_index": cutoff,
        "protected_task_names": [task.name for task in protected],
        "tail_task_names": [task.name for task in tail],
        "tail_all_lo": all(task.criticality == "LO" for task in tail),
        "all_hi_protected": all(
            task.criticality != "HI" or i <= cutoff
            for i, task in enumerate(tasks)
        ),
        "partition_complete": names == [task.name for task in protected + tail],
        "order_preserved": names[:cutoff + 1] == [task.name for task in prefix.tasks],
    }
    saturation = {
        "construction_version": "saturated-protected-prefix-v1",
        "prefix_fingerprint": prefix_fp, "full_task_count": len(tasks),
        "prefix_task_count": len(prefix.tasks), "tasks": changes,
        "hi_fields_equal": all(
            item["before"] == item["after"] for item in changes
            if item["before"]["criticality"] == "HI"
        ),
        "lo_saturation_equalities": [
            {"task": item["name"], "C_pp_LO": item["after"]["c_lo"],
             "C_pp_HI": item["after"]["c_hi"], "C_ref_LO": item["before"]["c_lo"]}
            for item in changes if item["before"]["criticality"] == "LO"
        ],
        "timing_fields_equal": all(
            all(item["before"][field] == item["after"][field]
                for field in ("period", "deadline", "offset", "priority_index"))
            for item in changes
        ),
    }
    return ProtectedPrefixBuildResult(
        full_taskset_fingerprint=full_fp, prefix_taskset=prefix,
        cutoff_task_name=tasks[cutoff].name, cutoff_priority_index=cutoff,
        protected_task_names=tuple(task.name for task in protected),
        tail_task_names=tuple(task.name for task in tail),
        partition_witness=partition, saturation_witness=saturation,
    )


def build_raw_protected_prefix(
    full_reference: ReferenceTaskset, *, source_context_hash: str,
):
    """Build the V8 raw protected priority prefix without LO saturation.

    The task subset is identical to the V7 protected prefix: all tasks through
    the lowest-priority HI task.  Every retained task is copied field-for-field,
    including both WCETs.  Keeping this constructor separate from
    ``build_saturated_protected_prefix`` is a soundness boundary: a raw prefix
    must never inherit the saturated route's mode-independent LO-demand lemma.
    """
    from .types import RawProtectedPrefixBuildResult

    tasks = full_reference.tasks
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise ProtectedPrefixConstructionError("RAW_PREFIX_PARTITION_INVALID")
    if tuple(task.priority_index for task in tasks) != tuple(range(len(tasks))):
        raise ProtectedPrefixConstructionError("RAW_PREFIX_PARTITION_INVALID")
    hi_indices = [i for i, task in enumerate(tasks) if task.criticality == "HI"]
    if not hi_indices:
        raise ProtectedPrefixConstructionError("RAW_PREFIX_NO_HI_TASK")
    cutoff = max(hi_indices)
    protected = tasks[: cutoff + 1]
    tail = tasks[cutoff + 1 :]
    if any(task.criticality != "LO" for task in tail):
        raise ProtectedPrefixConstructionError("RAW_PREFIX_TAIL_CONTAINS_HI")

    # ReferenceTask is frozen; retaining the same objects is the strongest
    # possible field-inheritance witness.  ReferenceTaskset re-validates the
    # priority order and model domain.
    try:
        prefix = ReferenceTaskset(tuple(protected), source_context_hash)
    except (TypeError, ValueError) as exc:
        raise ProtectedPrefixConstructionError("RAW_PREFIX_PARTITION_INVALID") from exc

    full_fp = _fingerprint(full_reference)
    prefix_fp = _fingerprint(prefix)
    partition = {
        "construction_version": "raw-protected-prefix-v8",
        "full_fingerprint": full_fp,
        "prefix_fingerprint": prefix_fp,
        "full_priority_order": names,
        "cutoff_task_name": tasks[cutoff].name,
        "cutoff_priority_index": cutoff,
        "protected_task_names": [task.name for task in protected],
        "tail_task_names": [task.name for task in tail],
        "tail_all_lo": all(task.criticality == "LO" for task in tail),
        "all_hi_protected": all(
            task.criticality != "HI" or i <= cutoff for i, task in enumerate(tasks)
        ),
        "partition_complete": names == [task.name for task in protected + tail],
        "order_preserved": names[: cutoff + 1] == [task.name for task in prefix.tasks],
        "priority_closed": all(
            task.priority_index <= cutoff for task in protected
        ),
    }
    field_names = (
        "name", "period", "deadline", "c_lo", "c_hi", "criticality",
        "priority_index", "code_c_lo", "code_c_hi", "degraded_cost", "offset",
    )
    task_rows = []
    for full_task, raw_task in zip(protected, prefix.tasks):
        equal = {field: getattr(full_task, field) == getattr(raw_task, field) for field in field_names}
        task_rows.append({
            "task": full_task.name,
            "fields_equal": equal,
            "all_fields_equal": all(equal.values()),
            "raw_c_lo": raw_task.c_lo,
            "raw_c_hi": raw_task.c_hi,
            "full_c_lo": full_task.c_lo,
            "full_c_hi": full_task.c_hi,
        })
    inheritance = {
        "construction_version": "raw-protected-prefix-v8",
        "prefix_fingerprint": prefix_fp,
        "all_parameters_inherited": all(row["all_fields_equal"] for row in task_rows),
        "wcet_identity": all(
            row["raw_c_lo"] == row["full_c_lo"] and row["raw_c_hi"] == row["full_c_hi"]
            for row in task_rows
        ),
        "no_saturation_applied": all(
            row["raw_c_hi"] == row["full_c_hi"] for row in task_rows
        ),
        "job_key_rule": "(task_id,release_index)",
        "release_eligibility_inherited": True,
        "tasks": task_rows,
    }
    return RawProtectedPrefixBuildResult(
        full_taskset_fingerprint=full_fp,
        prefix_taskset=prefix,
        cutoff_task_name=tasks[cutoff].name,
        cutoff_priority_index=cutoff,
        protected_task_names=tuple(task.name for task in protected),
        tail_task_names=tuple(task.name for task in tail),
        partition_witness=partition,
        inheritance_witness=inheritance,
    )
