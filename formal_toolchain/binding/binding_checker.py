"""目标绑定结果的统一 fail-closed 检查。"""

from __future__ import annotations

from typing import Any


def require_bound(ir: dict[str, Any], obligation_id: str) -> dict[str, Any]:
    if ir.get("status") != "PASS":
        failure = dict(ir.get("failure", {}))
        failure.setdefault("code", "UNSUPPORTED_AST_NODE")
        failure.setdefault("route", "UNRESOLVED")
        return {"obligation_id": obligation_id, "status": "UNRESOLVED", "failure": failure}
    return {"obligation_id": obligation_id, "status": "PASS", "ir": ir}


def mutation_detected(original: dict[str, Any], mutated: dict[str, Any], marker: str) -> bool:
    """负向测试辅助：语义相关 IR 必须因 mutation 发生可观察变化。"""
    return original != mutated and marker in repr(mutated)
