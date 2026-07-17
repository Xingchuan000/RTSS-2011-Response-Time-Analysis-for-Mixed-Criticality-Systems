"""Phase F：数学整数时间、预算域和零 overhead 检查。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise ValueError(f"{label} 必须是{'正' if positive else ''}整数")
    return value


def build_budget_domain(tasks: Sequence[Any], metadata: Mapping[str, Mapping[str, Any]] | None = None,
                        *, runtime_config: Any | None = None) -> dict[str, Any]:
    """消费明确的预算来源，建立有限的 LO/HI/active release 整数域。"""
    if metadata is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "BUDGET_PROVENANCE_MISSING"}}
    domains: dict[str, Any] = {}
    for task in tasks:
        info = metadata.get(str(task.name))
        if info is None:
            raise ValueError(f"缺少 {task.name} 的 budget provenance")
        criticality = str(getattr(task.criticality, "value", task.criticality))
        initial = _int(info.get("initial_runtime_budget"), f"{task.name}.initial_runtime_budget", positive=True)
        floor = _int(info.get("budget_floor", 1), f"{task.name}.budget_floor")
        action_hard_upper = _int(
            info.get(
                "action_hard_upper",
                task.c_hi if criticality == "HI" else task.deadline,
            ),
            f"{task.name}.action_hard_upper",
            positive=True,
        )
        if not floor <= initial <= action_hard_upper:
            raise ValueError(f"{task.name} 的有限预算域非法")
        formal_lower = int(task.c_lo) if criticality == "HI" else 0
        candidate_positive_lower = int(task.c_lo) if criticality == "HI" else 1
        domains[str(task.name)] = {
            "initial": initial,
            "code_lower": int(task.c_lo),
            "code_upper": int(task.c_hi),
            "runtime_floor": floor,
            "formal_lower": formal_lower,
            "candidate_positive_lower": candidate_positive_lower,
            "action_hard_upper": action_hard_upper,
            "integer_interval": {
                "lower": formal_lower,
                "upper": action_hard_upper,
            },
            "active_release_budget_interval": {
                "lower": formal_lower,
                "upper": action_hard_upper,
            },
            "provenance": dict(info),
        }
    if runtime_config is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "EFFECTIVE_RUNTIME_CONFIG_MISSING"}}
    overhead = getattr(runtime_config, "processor_overhead", None)
    if overhead is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "PROCESSOR_OVERHEAD_UNVERIFIED"}}
    if overhead != 0:
        raise ValueError("P0 不接受非零 processor overhead")
    return {"status": "PASS", "schema_version": "budget_domain_v2", "integer_arithmetic": "python_unbounded_int",
            "processor_overhead": 0, "tasks": domains}


def materialize_finite_interval(
    row: Mapping[str, Any], *, max_values: int
) -> list[int] | None:
    interval = row["integer_interval"]
    lower = int(interval["lower"])
    upper = int(interval["upper"])
    count = upper - lower + 1
    if count > max_values:
        return None
    return list(range(lower, upper + 1))


def check_time_domain(tasks: Sequence[Any], *, overhead: int = 0,
                      scheduler_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """检查所有形式化 task 数值字段为 Python int（拒绝 bool）。"""
    from .scheduler import check_scheduler_model
    scheduler = check_scheduler_model(tasks, overhead=overhead, scheduler_facts=scheduler_facts)
    if scheduler.get("status") != "PASS":
        return scheduler
    return {"status": "PASS", "schema_version": "time_domain_v1", "arithmetic": "mathematical_integer",
            "python_integer_type": "unbounded", "numpy_fixed_width_allowed": False}
