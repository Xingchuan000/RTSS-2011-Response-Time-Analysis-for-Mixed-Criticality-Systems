"""sklearn CART 到纯 Python 整数树的编译与执行。

运行时 evaluator 只依赖本模块的数据类和 Python 整数比较，不导入 sklearn
或 NumPy；sklearn 仅在显式编译函数被调用时作为对象协议使用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from amc_py.viper.schema import INTEGER_TREE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class IntegerTreeNode:
    node_id: int
    feature_index: int
    threshold_int: int
    left_child: int
    right_child: int


@dataclass(frozen=True, slots=True)
class IntegerTreeLeaf:
    node_id: int
    raw_action_id: int
    class_counts: tuple[float, ...]
    weighted_class_counts: tuple[float, ...]
    sample_count: int
    impurity: float


@dataclass(frozen=True, slots=True)
class IntegerTreeModel:
    root_node_id: int
    state_dim: int
    action_dim: int
    nodes: tuple[IntegerTreeNode, ...]
    leaves: tuple[IntegerTreeLeaf, ...]
    feature_names: tuple[str, ...]
    fixed_point_config_hash: str
    schema_version: str = INTEGER_TREE_SCHEMA_VERSION


def compile_sklearn_tree_to_integer(classifier: Any, *, feature_names: tuple[str, ...], fixed_point_config_hash: str, state_dim: int, action_dim: int) -> IntegerTreeModel:
    """按 sklearn 的 ``<= threshold`` 分支规则编译成整数树。

    sklearn 阈值是浮点分裂点；部署语义明确规定使用 ``floor(threshold)``，
    因而整数输入在阈值处必然走左分支。
    """
    tree = classifier.tree_
    if len(feature_names) != state_dim:
        raise ValueError("feature_names 与 state_dim 不一致")
    nodes: list[IntegerTreeNode] = []
    leaves: list[IntegerTreeLeaf] = []
    reachable: set[int] = set()
    visiting: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in visiting:
            raise ValueError("integer tree 存在环")
        if node_id in reachable:
            return
        if node_id < 0 or node_id >= int(tree.node_count):
            raise ValueError("tree child 超出范围")
        visiting.add(node_id)
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        feature = int(tree.feature[node_id])
        if left == right:
            raw_index = max(range(len(tree.value[node_id][0])), key=lambda i: tree.value[node_id][0][i])
            raw_action_id = int(classifier.classes_[raw_index])
            leaves.append(IntegerTreeLeaf(node_id, raw_action_id,
                tuple(float(v) for v in tree.value[node_id].ravel()),
                tuple(float(v) for v in tree.value[node_id].ravel()),
                int(tree.n_node_samples[node_id]), float(tree.impurity[node_id])))
        else:
            if feature < 0 or feature >= state_dim:
                raise ValueError("内部节点 feature index 非法")
            if not math.isfinite(float(tree.threshold[node_id])):
                raise ValueError("内部节点 threshold 非法")
            visit(left); visit(right)
            nodes.append(IntegerTreeNode(node_id, feature, math.floor(float(tree.threshold[node_id])), left, right))
        visiting.remove(node_id)
        reachable.add(node_id)

    visit(0)
    if len(reachable) != int(tree.node_count):
        raise ValueError("存在不可达 tree node")
    leaf_ids = {leaf.node_id for leaf in leaves}
    if not leaf_ids or len(reachable) != len(nodes) + len(leaves):
        raise ValueError("tree 路径未完整终止于 leaf")
    return IntegerTreeModel(0, state_dim, action_dim, tuple(sorted(nodes, key=lambda x: x.node_id)), tuple(sorted(leaves, key=lambda x: x.node_id)), feature_names, fixed_point_config_hash)


def evaluate_integer_tree(model: IntegerTreeModel, state_int: tuple[int, ...]) -> tuple[int, int, list[dict[str, object]]]:
    if len(state_int) != model.state_dim:
        raise ValueError("state_int 维度不匹配")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in state_int):
        raise ValueError("state_int 必须只包含整数")
    nodes = {node.node_id: node for node in model.nodes}
    leaves = {leaf.node_id: leaf for leaf in model.leaves}
    node_id = model.root_node_id
    path: list[dict[str, object]] = []
    seen: set[int] = set()
    while node_id not in leaves:
        if node_id in seen or node_id not in nodes:
            raise ValueError("integer tree 图非法")
        seen.add(node_id)
        node = nodes[node_id]
        value = int(state_int[node.feature_index])
        left = value <= node.threshold_int
        next_node_id = node.left_child if left else node.right_child
        path.append({"node_id": node.node_id, "feature_index": node.feature_index, "feature_name": model.feature_names[node.feature_index], "value_int": value, "threshold_int": node.threshold_int, "operator": "<=" if left else ">", "decision": "left" if left else "right", "next_node_id": next_node_id})
        node_id = next_node_id
    leaf = leaves[node_id]
    return leaf.node_id, leaf.raw_action_id, path


def _model_dict(model: IntegerTreeModel) -> dict[str, object]:
    return {"schema_version": model.schema_version, "root_node_id": model.root_node_id, "state_dim": model.state_dim, "action_dim": model.action_dim, "nodes": [asdict(v) for v in model.nodes], "leaves": [asdict(v) for v in model.leaves], "feature_names": list(model.feature_names), "fixed_point_config_hash": model.fixed_point_config_hash}


def integer_tree_hash(model: IntegerTreeModel) -> str:
    data = json.dumps(_model_dict(model), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def save_integer_tree_json(model: IntegerTreeModel, path: Path) -> None:
    path.write_text(json.dumps(_model_dict(model), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def load_integer_tree_json(path: Path) -> IntegerTreeModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != INTEGER_TREE_SCHEMA_VERSION:
        raise ValueError("未知 integer tree schema version")
    return IntegerTreeModel(data["root_node_id"], data["state_dim"], data["action_dim"], tuple(IntegerTreeNode(**v) for v in data["nodes"]), tuple(IntegerTreeLeaf(node_id=v["node_id"], raw_action_id=v["raw_action_id"], class_counts=tuple(v["class_counts"]), weighted_class_counts=tuple(v["weighted_class_counts"]), sample_count=v["sample_count"], impurity=v["impurity"]) for v in data["leaves"]), tuple(data["feature_names"]), data["fixed_point_config_hash"], data["schema_version"])
