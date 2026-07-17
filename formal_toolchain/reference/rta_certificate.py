"""Protected-HI RTA production/replay 复合证据。"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.rta_production import protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta


def compare_rta_witnesses(production: Mapping[str, Any],
                          replay: Mapping[str, Any]) -> dict[str, Any]:
    """逐字段比较 production 与 replay，避免只比较顶层 status。"""

    if not isinstance(production.get("tasks"), list) or not isinstance(replay.get("checks"), list):
        return {"status": "UNRESOLVED", "mismatches": [{"field": "tasks"}]}
    mismatches: list[dict[str, Any]] = []
    left_by_name = {str(row.get("task", {}).get("name")): row for row in production["tasks"]}
    right_by_name = {str(row.get("task")): row for row in replay["checks"]}
    for task_name in sorted(left_by_name):
        if task_name not in right_by_name:
            mismatches.append({"task": task_name, "field": "task_presence"})
            continue
        if right_by_name[task_name].get("status") != "PASS":
            mismatches.append({"task": task_name, "field": "replay_status",
                               "production": left_by_name[task_name].get("status"),
                               "replay": right_by_name[task_name].get("status")})
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def build_rta_composite(reference_taskset: Any) -> dict[str, Any]:
    """仅供 candidate 生成完整诊断；最终 gate 使用 verifier 侧 replay。"""

    production = protected_hi_rta(reference_taskset)
    replay = replay_rta(reference_taskset, production)
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
