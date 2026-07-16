"""从整数树路径生成 leaf conjunction guard。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from amc_py.viper.integer_tree import IntegerTreeModel


def leaf_guards(model: IntegerTreeModel) -> dict[int, tuple[dict[str, Any], ...]]:
    guards: dict[int, tuple[dict[str, Any], ...]] = {}
    nodes = {node.node_id: node for node in model.nodes}
    leaves = {leaf.node_id for leaf in model.leaves}
    def walk(node_id: int, path: tuple[dict[str, Any], ...]) -> None:
        if node_id in leaves:
            guards[node_id] = path; return
        node = nodes[node_id]
        walk(node.left_child, path + ({"feature_index": node.feature_index, "operator": "<=", "threshold_int": node.threshold_int},))
        walk(node.right_child, path + ({"feature_index": node.feature_index, "operator": ">", "threshold_int": node.threshold_int},))
    walk(model.root_node_id, ())
    return guards
