"""Phase F：数学整数时间、预算域和零 overhead 检查。"""

from __future__ import annotations

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
    metadata = metadata
    domains: dict[str, Any] = {}
    for task in tasks:
        info = metadata.get(str(task.name))
        if info is None:
            raise ValueError(f"缺少 {task.name} 的 budget provenance")
        initial = _int(info.get("initial_runtime_budget"), f"{task.name}.initial_runtime_budget", positive=True)
        floor = _int(info.get("budget_floor", 1), f"{task.name}.budget_floor")
        upper = _int(info.get("budget_cap", task.c_hi), f"{task.name}.budget_cap", positive=True)
        if floor < 0 or not floor <= initial <= upper or upper < task.c_hi:
            raise ValueError(f"{task.name} 的有限预算域非法")
        domains[str(task.name)] = {"initial": initial, "code_lower": task.c_lo,
            "code_upper": task.c_hi, "runtime_floor": floor, "runtime_deploy_cap": upper,
            # 证明域按计划覆盖 LO 的 0..U；runtime floor 作为单独约束保留，
            # 不能把它偷偷当作 formal lower bound 从而放松/改变不变量。
            "finite_integer_domain": list(range(task.c_lo if str(getattr(task.criticality, "value", task.criticality)) == "HI" else 0,
                                                 upper + 1)),
            "active_release_budget_domain": list(range(task.c_lo if str(getattr(task.criticality, "value", task.criticality)) == "HI" else 0,
                                                       upper + 1)),
            "provenance": dict(info)}
    if runtime_config is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "EFFECTIVE_RUNTIME_CONFIG_MISSING"}}
    overhead = getattr(runtime_config, "processor_overhead", None)
    if overhead is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "PROCESSOR_OVERHEAD_UNVERIFIED"}}
    if overhead != 0:
        raise ValueError("P0 不接受非零 processor overhead")
    return {"status": "PASS", "schema_version": "budget_domain_v1", "integer_arithmetic": "python_unbounded_int",
            "processor_overhead": 0, "tasks": domains}


def check_time_domain(tasks: Sequence[Any], *, overhead: int = 0,
                      scheduler_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """检查所有形式化 task 数值字段为 Python int（拒绝 bool）。"""
    from .scheduler import check_scheduler_model
    scheduler = check_scheduler_model(tasks, overhead=overhead, scheduler_facts=scheduler_facts)
    if scheduler.get("status") != "PASS":
        return scheduler
    return {"status": "PASS", "schema_version": "time_domain_v1", "arithmetic": "mathematical_integer",
            "python_integer_type": "unbounded", "numpy_fixed_width_allowed": False}
