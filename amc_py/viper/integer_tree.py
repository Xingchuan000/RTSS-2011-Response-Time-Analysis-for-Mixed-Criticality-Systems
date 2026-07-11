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

from amc_py.viper.schema import INTEGER_TREE_SCHEMA_VERSION, INTEGER_TREE_SCHEMA_VERSION_RANKED


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
    # v1 叶子没有这两个字段；空 tuple 仅用于加载旧 artifact，不能被当作 ranked。
    action_ranking: tuple[int, ...] = ()
    full_action_counts: tuple[float, ...] = ()


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


def compile_sklearn_tree_to_integer(classifier: Any, *, feature_names: tuple[str, ...], fixed_point_config_hash: str, state_dim: int, action_dim: int, ranked: bool = True) -> IntegerTreeModel:
    """按 sklearn 的 ``<= threshold`` 分支规则编译成整数树。

    sklearn 阈值是浮点分裂点；部署语义明确规定使用 ``floor(threshold)``，
    因而整数输入在阈值处必然走左分支。

    ranked=True 时生成 integer_tree_ranked_v2 且叶子包含完整 action_ranking；
    ranked=False 时生成 integer_tree_v1 且叶子 action_ranking 为空。
    """
    tree = classifier.tree_
    schema_version = INTEGER_TREE_SCHEMA_VERSION_RANKED if ranked else INTEGER_TREE_SCHEMA_VERSION
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
            local_counts = tree.value[node_id][0]
            full_counts = [0.0] * action_dim
            for local_index, class_id in enumerate(classifier.classes_):
                action_id = int(class_id)
                if action_id < 0 or action_id >= action_dim:
                    raise ValueError("classifier.classes_ 中存在越界 action id")
                full_counts[action_id] = float(local_counts[local_index])
            if ranked:
                ranking = sorted(range(action_dim), key=lambda action_id: (-full_counts[action_id], action_id))
                action_ranking = tuple(ranking)
                full_action_counts = tuple(full_counts)
            else:
                # v1 schema：仅保留 raw_action_id，不存储完整 ranking/counts
                ranking = sorted(range(action_dim), key=lambda action_id: (-full_counts[action_id], action_id))
                action_ranking = ()  # 空 tuple，不能被当作 ranked
                full_action_counts = ()
            leaves.append(IntegerTreeLeaf(
                node_id=node_id,
                raw_action_id=(sorted(range(action_dim), key=lambda action_id: (-full_counts[action_id], action_id)))[0],
                class_counts=tuple(float(v) for v in tree.value[node_id].ravel()),
                weighted_class_counts=tuple(float(v) for v in tree.value[node_id].ravel()),
                sample_count=int(tree.n_node_samples[node_id]),
                impurity=float(tree.impurity[node_id]),
                action_ranking=action_ranking,
                full_action_counts=full_action_counts,
            ))
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
    return IntegerTreeModel(0, state_dim, action_dim, tuple(sorted(nodes, key=lambda x: x.node_id)), tuple(sorted(leaves, key=lambda x: x.node_id)), feature_names, fixed_point_config_hash, schema_version)


def evaluate_integer_tree(model: IntegerTreeModel, state_int: tuple[int, ...]) -> tuple[int, IntegerTreeLeaf, list[dict[str, object]]]:
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
    return leaf.node_id, leaf, path


def _model_dict(model: IntegerTreeModel) -> dict[str, object]:
    return {"schema_version": model.schema_version, "root_node_id": model.root_node_id, "state_dim": model.state_dim, "action_dim": model.action_dim, "nodes": [asdict(v) for v in model.nodes], "leaves": [asdict(v) for v in model.leaves], "feature_names": list(model.feature_names), "fixed_point_config_hash": model.fixed_point_config_hash}


def integer_tree_hash(model: IntegerTreeModel) -> str:
    data = json.dumps(_model_dict(model), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def save_integer_tree_json(model: IntegerTreeModel, path: Path) -> None:
    path.write_text(json.dumps(_model_dict(model), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def load_integer_tree_json(path: Path) -> IntegerTreeModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema_version = data.get("schema_version")
    if schema_version not in {INTEGER_TREE_SCHEMA_VERSION, INTEGER_TREE_SCHEMA_VERSION_RANKED}:
        raise ValueError("未知 integer tree schema version")
    leaves = []
    for value in data["leaves"]:
        if schema_version == INTEGER_TREE_SCHEMA_VERSION_RANKED:
            if "action_ranking" not in value or "full_action_counts" not in value:
                raise ValueError("ranked integer tree leaf 缺少完整 action ranking/counts")
            ranking = tuple(int(v) for v in value["action_ranking"])
            counts = tuple(float(v) for v in value["full_action_counts"])
            action_dim = int(data["action_dim"])
            if len(ranking) != action_dim or sorted(ranking) != list(range(action_dim)):
                raise ValueError("ranked integer tree leaf action_ranking 不是完整排列")
            if len(counts) != action_dim:
                raise ValueError("ranked integer tree leaf full_action_counts 维度不匹配")
            if int(value["raw_action_id"]) != ranking[0]:
                raise ValueError("ranked integer tree leaf raw_action_id 与 ranking 不一致")
        else:
            ranking = ()
            counts = ()
        leaves.append(IntegerTreeLeaf(node_id=value["node_id"], raw_action_id=value["raw_action_id"], class_counts=tuple(value["class_counts"]), weighted_class_counts=tuple(value["weighted_class_counts"]), sample_count=value["sample_count"], impurity=value["impurity"], action_ranking=ranking, full_action_counts=counts))
    return IntegerTreeModel(data["root_node_id"], data["state_dim"], data["action_dim"], tuple(IntegerTreeNode(**v) for v in data["nodes"]), tuple(leaves), tuple(data["feature_names"]), data["fixed_point_config_hash"], schema_version)
