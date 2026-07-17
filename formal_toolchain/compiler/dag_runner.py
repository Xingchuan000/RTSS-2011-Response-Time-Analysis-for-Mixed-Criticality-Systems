"""按 registry 计算 obligation DAG 的确定性执行顺序。"""

from __future__ import annotations

from typing import Any, Mapping


def topological_order(entries: list[Mapping[str, Any]]) -> list[str]:
    """返回前驱在前、同层按 obligation id 排序的拓扑序。"""

    by_id = {str(entry["id"]): entry for entry in entries}
    state: dict[str, int] = {}
    result: list[str] = []

    def visit(obligation_id: str) -> None:
        mark = state.get(obligation_id, 0)
        if mark == 1:
            raise ValueError("obligation registry 存在环")
        if mark == 2:
            return
        state[obligation_id] = 1
        for predecessor in sorted(str(item) for item in by_id[obligation_id].get("depends_on", [])):
            visit(predecessor)
        state[obligation_id] = 2
        result.append(obligation_id)

    for obligation_id in sorted(by_id):
        visit(obligation_id)
    return result


def dependency_status(predecessors: Mapping[str, Mapping[str, Any]]) -> str:
    """按计划传播前驱状态；UNRESOLVED 不被改写为 FAIL。"""

    statuses = {str(item.get("obligation_status")) for item in predecessors.values()}
    if "FAIL" in statuses:
        return "NOT_RUN"
    if "UNRESOLVED" in statuses:
        return "NOT_RUN"
    if "NOT_APPLICABLE" in statuses:
        return "NOT_RUN"
    return "READY"


def claim_dependency_closure(entries: list[Mapping[str, Any]], claim: str) -> set[str]:
    """compiler 自己从 registry 计算 claim 闭包，不依赖 verifier 实现。"""

    by_id = {str(item["id"]): item for item in entries
              if item.get("activation") == "active" and item.get("required") is True}
    roots = {item_id for item_id, item in by_id.items()
             if item_id != "CLAIM_AGGREGATION_RESULT"
             and item.get("kind") != "derived_summary"
             and claim in {str(value) for value in item.get("gates_claims", [])}}
    closure: set[str] = set()
    stack = sorted(roots, reverse=True)
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        for dependency in sorted(str(value) for value in by_id[current].get("depends_on", [])):
            if dependency not in by_id:
                raise ValueError(f"claim 依赖 inactive/unknown obligation: {dependency}")
            stack.append(dependency)
    return closure
