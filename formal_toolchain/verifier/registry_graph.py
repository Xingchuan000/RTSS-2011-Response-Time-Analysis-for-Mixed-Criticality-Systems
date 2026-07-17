"""verifier 侧独立的 obligation DAG 实现。

verifier 不从 compiler 导入拓扑排序，避免 compiler 的 DAG 错误同时污染
candidate 生成和最终验证。
"""

from __future__ import annotations

from typing import Any, Mapping


def verifier_topological_order(entries: list[Mapping[str, Any]]) -> list[str]:
    """按 obligation 依赖返回确定性的拓扑序，并现场检查环和未知依赖。"""

    by_id = {str(row["id"]): row for row in entries}
    permanent: set[str] = set()
    temporary: set[str] = set()
    output: list[str] = []

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ValueError("registry dependency cycle")
        if node not in by_id:
            raise ValueError(f"unknown obligation: {node}")
        temporary.add(node)
        for dependency in sorted(str(item) for item in by_id[node].get("depends_on", [])):
            visit(dependency)
        temporary.remove(node)
        permanent.add(node)
        output.append(node)

    for node in sorted(by_id):
        visit(node)
    return output
