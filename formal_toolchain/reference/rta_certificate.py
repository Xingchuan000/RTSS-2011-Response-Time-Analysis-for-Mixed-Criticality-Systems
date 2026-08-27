"""Protected-HI RTA production/replay 复合证据。"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.rta_production import (
    all_task_protected_prefix_rta,
    all_task_raw_protected_prefix_rta,
    all_task_reference_rta,
)
from formal_toolchain.reference.rta_replay import replay_all_task_rta


def compare_rta_witnesses(production: Mapping[str, Any],
                          replay: Mapping[str, Any]) -> dict[str, Any]:
    """逐字段比较 production 与 replay，避免只比较顶层 status。"""

    if not isinstance(production.get("tasks"), list) or not isinstance(replay.get("replay_rows"), list):
        return {"status": "UNRESOLVED", "mismatches": [{"field": "tasks"}]}
    mismatches: list[dict[str, Any]] = []
    left_by_name = {str(row.get("task", {}).get("name")): row for row in production["tasks"]}
    right_by_name = {str(row.get("task", {}).get("name")): row for row in replay["replay_rows"]}
    for task_name in sorted(left_by_name):
        if task_name not in right_by_name:
            mismatches.append({"task": task_name, "field": "task_presence"})
            continue
        if right_by_name[task_name].get("status") != "PASS":
            mismatches.append({"task": task_name, "field": "replay_status",
                               "production": left_by_name[task_name].get("status"),
                               "replay": right_by_name[task_name].get("status")})
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def build_rta_composite(reference_taskset: Any, *, route_id: str = "strict_full") -> dict[str, Any]:
    """为当前选定 proof route 生成 production/replay 复合证据。

    ``protected_prefix`` 必须在饱和保护前缀上枚举 LO、Case1 和 Case2
    整数域。旧实现无条件对完整 reference taskset 运行 RTA；当被删除的
    低优先级 LO tail 自身不可调度时，fail-closed 分支会清空全部 Case1/
    Case2 witness，进而把本来可调度的 protected prefix 误报为整数域缺失。

    strict-full 路线的 analysis taskset 就是完整 reference taskset，因此
    保持原行为。
    """

    if route_id == "protected_prefix":
        production = all_task_protected_prefix_rta(reference_taskset)
    elif route_id == "raw_protected_prefix":
        production = all_task_raw_protected_prefix_rta(reference_taskset)
    elif route_id == "strict_full":
        production = all_task_reference_rta(reference_taskset)
    else:
        raise ValueError(f"UNKNOWN_RTA_ROUTE:{route_id}")
    replay = replay_all_task_rta(
        reference_taskset,
        production,
        expected_obligation_id=production.get("obligation_id"),
        expected_route_id=route_id,
    )
    consistency = compare_rta_witnesses(production, replay)
    statuses = (production.get("status"), replay.get("status"), consistency.get("status"))
    if "FAIL" in statuses:
        status = "FAIL"
    elif any(item == "UNRESOLVED" for item in statuses):
        status = "UNRESOLVED"
    else:
        status = "PASS"
    return {"status": status, "production": dict(production),
            "replay": dict(replay), "consistency": consistency,
            "production_hash": sha256_object(production),
            "replay_hash": sha256_object(replay),
            "consistency_hash": sha256_object(consistency)}
