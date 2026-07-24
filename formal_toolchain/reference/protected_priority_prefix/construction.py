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
