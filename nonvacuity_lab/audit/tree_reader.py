from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping


def load_tree(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("integer_tree.json 顶层必须为 object")
    return raw


def iter_leaves(tree: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if isinstance(tree.get("leaves"), list):
        for leaf in tree["leaves"]:
            if isinstance(leaf, dict):
                yield dict(leaf)
        return
    nodes = tree.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and (
                node.get("is_leaf") is True or "action_ranking" in node
            ):
                yield dict(node)
        return
    yield from _walk(tree.get("root", tree))


def leaf_guards(tree: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Derive root-to-leaf guards for flat integer_tree_v1 artifacts."""
    internal = {
        int(node["node_id"]): dict(node)
        for node in tree.get("nodes", ())
        if isinstance(node, Mapping) and "node_id" in node
    }
    leaves = {
        int(leaf.get("node_id", leaf.get("leaf_id", leaf.get("id", -1))))
        for leaf in iter_leaves(tree)
    }
    root = tree.get("root_node_id")
    if root is None or not internal:
        return {leaf_id: [] for leaf_id in leaves}
    features = list(tree.get("feature_names", ()))
    result: dict[int, list[dict[str, Any]]] = {}

    def walk(node_id: int, guards: list[dict[str, Any]]) -> None:
        if node_id in leaves:
            result[node_id] = list(guards)
            return
        node = internal.get(node_id)
        if node is None:
            raise ValueError(f"tree child node 不存在: {node_id}")
        feature_index = int(node["feature_index"])
        feature_name = (
            str(features[feature_index])
            if 0 <= feature_index < len(features)
            else f"feature_{feature_index}"
        )
        threshold = int(node.get("threshold_int", node.get("threshold", 0)))
        left = int(node.get("left_child", node.get("left")))
        right = int(node.get("right_child", node.get("right")))
        walk(
            left,
            [*guards, {"feature_index": feature_index, "feature_name": feature_name, "op": "<=", "threshold_int": threshold}],
        )
        walk(
            right,
            [*guards, {"feature_index": feature_index, "feature_name": feature_name, "op": ">", "threshold_int": threshold}],
        )

    walk(int(root), [])
    return result


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(node, dict):
        return
    if "action_ranking" in node or node.get("is_leaf") is True:
        yield dict(node)
        return
    for key in ("left", "right", "children"):
        value = node.get(key)
        if isinstance(value, list):
            for child in value:
                yield from _walk(child)
        else:
            yield from _walk(value)
