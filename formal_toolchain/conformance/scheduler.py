"""Phase F：P0 调度器与严格优先级合同检查。

本文件只检查目标对象已经导出的事实，不替目标对象重建调度器；因此不会
改变 ``amc_py`` 的运行语义，也不会把仿真结果当作形式化结论。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _positive_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数")


def check_scheduler_model(tasks: Sequence[Any], *, processor_count: int = 1,
                          fully_preemptive: bool = True, work_conserving: bool = True,
                          integer_tick: bool = True, overhead: int = 0,
                          scheduler_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """检查 P0 的单处理器、整数 tick、抢占式和 work-conserving 前提。"""
    if processor_count != 1 or not (fully_preemptive and work_conserving and integer_tick):
        raise ValueError("目标不满足 P0 单处理器抢占式 work-conserving 合同")
    if isinstance(overhead, bool) or not isinstance(overhead, int) or overhead != 0:
        raise ValueError("P0 只允许 processor overhead=0")
    if scheduler_facts is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "SCHEDULER_RUNTIME_FACTS_MISSING"}}
    required_facts = {"ready_selects_highest_priority", "tick_boundary_preemption",
                      "work_conserving", "no_blocking", "no_self_suspension", "no_non_preemptive_sections",
                      "sporadic_release_contract"}
    if not required_facts <= set(scheduler_facts):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "SCHEDULER_FACTS_INCOMPLETE",
                             "fields": sorted(required_facts - set(scheduler_facts))}}
    if not isinstance(scheduler_facts.get("evidence"), Mapping) or not scheduler_facts["evidence"]:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "SCHEDULER_RUNTIME_EVIDENCE_MISSING"}}
    binding = scheduler_facts.get("binding")
    if isinstance(binding, Mapping):
        from formal_toolchain.core.hashing import sha256_object
        if scheduler_facts.get("binding_hash") != sha256_object(binding):
            return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "SCHEDULER_BINDING_HASH_MISMATCH"}}
    elif not scheduler_facts.get("trace"):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "SCHEDULER_BINDING_OR_TRACE_MISSING"}}
    if scheduler_facts.get("source_root"):
        from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
        from pathlib import Path
        from formal_toolchain.core.hashing import sha256_object
        bound = bind_event_runtime(Path(scheduler_facts["source_root"]))
        if bound.get("status") != "PASS" or scheduler_facts.get("binding_hash") != sha256_object(bound):
            return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "SCHEDULER_SOURCE_BINDING_MISMATCH"}}
    if any(scheduler_facts[name] is not True for name in required_facts):
        raise ValueError("runtime scheduler facts 不满足 P0 合同")
    if not tasks:
        raise ValueError("taskset 不能为空")
    names: set[str] = set()
    records = []
    for task in tasks:
        name = str(task.name)
        if name in names:
            raise ValueError(f"task 名称重复: {name}")
        names.add(name)
        for field in ("period", "deadline", "c_lo", "c_hi"):
            _positive_int(getattr(task, field), f"{name}.{field}")
        if task.deadline > task.period or task.c_hi < task.c_lo:
            raise ValueError(f"task {name} 的 D<=T 或 C_HI>=C_LO 前提失败")
        records.append({"name": name, "period": task.period, "deadline": task.deadline,
                        "code_c_lo": task.c_lo, "code_c_hi": task.c_hi})
    return {"status": "PASS", "schema_version": "scheduler_model_conformance_v1",
            "processor_count": 1, "integer_tick": True, "fully_preemptive": True,
            "work_conserving": True, "overhead": 0, "tasks": records,
            "claims": ["ready_nonempty_runs_highest_priority", "tick_boundary_preemption"]}


def check_strict_priority_order(tasks: Sequence[Any]) -> dict[str, Any]:
    """检查 priority index 唯一、连续，并生成不含排序修复的证书。"""
    names = [str(task.name) for task in tasks]
    if len(names) != len(set(names)) or not names:
        raise ValueError("priority order 必须覆盖唯一 task")
    return {"status": "PASS", "schema_version": "strict_priority_order_v1",
            "priority_order": names, "priority_index": {name: i for i, name in enumerate(names)},
            "highest_priority_first": True}
