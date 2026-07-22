"""单一规范 release-fixed demand 函数。

所有涉及 removal_demand、release_class、release_budget 的模块（
formal_runtime_snapshot、state_relation、phase_k_runtime_states、budget_domination、
release_mapping checker）必须调用本模块，不得各自实现近似公式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReleaseDemand:
    raw_actual_cost: int
    removal_demand: int
    release_class: str
    release_budget: int | None


def require_nonnegative_int(value: Any) -> int:
    v = int(value)
    if v < 0:
        raise ValueError(f"negative value: {v}")
    return v


def criticality_of(job: Any) -> str:
    raw = getattr(job, "criticality", None)
    if raw is None:
        raw = str(getattr(job.task.criticality, "value", ""))
    if raw not in ("HI", "LO"):
        raise ValueError(f"MISSING_CRITICALITY:{getattr(job, 'job_key', job)}")
    return str(raw)


def released_mode_of(job: Any) -> str:
    raw = getattr(job, "released_mode", None)
    if raw is not None:
        raw = str(getattr(raw, "name", raw))
    if raw not in ("LO", "HI", None):
        raise ValueError(f"INVALID_RELEASED_MODE:{raw}")
    return raw or "LO"


def hi_release_class(job: Any) -> str:
    return f"HI_normal_{str(getattr(job, 'task', job))}"


def lo_release_class_from_release_provenance(job: Any) -> str:
    if bool(getattr(job, "is_degraded", False)):
        return f"LO_degraded_{str(getattr(job, 'task', job))}"
    return f"LO_primary_{str(getattr(job, 'task', job))}"


def derive_release_fixed_demand(job: Any, *, task_reference: dict[str, Any] | None = None) -> ReleaseDemand:
    raw = require_nonnegative_int(getattr(job, "original_actual_cost", job.actual_cost))
    criticality = criticality_of(job)

    if criticality == "HI":
        return ReleaseDemand(raw, raw, hi_release_class(job), None)

    is_degraded = bool(getattr(job, "is_degraded", False))
    if is_degraded:
        degraded_cost = None
        if task_reference is not None and "degraded_cost" in task_reference:
            degraded_cost = require_nonnegative_int(task_reference["degraded_cost"])
        demand = min(raw, degraded_cost) if degraded_cost is not None else raw
    else:
        runtime_budget = getattr(job, "runtime_budget_at_release", None)
        if runtime_budget is not None:
            demand = min(raw, int(runtime_budget) + 1)
        else:
            demand = raw

    return ReleaseDemand(
        raw_actual_cost=raw,
        removal_demand=demand,
        release_class=lo_release_class_from_release_provenance(job),
        release_budget=getattr(job, "runtime_budget_at_release", None),
    )
