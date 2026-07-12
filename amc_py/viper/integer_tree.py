"""不依赖 sklearn 的定点整数 CART 执行模型与编译器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from functools import lru_cache
from collections.abc import Sequence
from pathlib import Path

from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash
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
    action_ranking: tuple[int, ...]
    full_action_counts: tuple[float, ...]
    n_node_samples: int
    weighted_n_node_samples: float
    impurity: float


@dataclass(frozen=True, slots=True)
class IntegerTreeModel:
    schema_version: str
    root_node_id: int
    state_dim: int
    action_dim: int
    nodes: tuple[IntegerTreeNode, ...]
    leaves: tuple[IntegerTreeLeaf, ...]
    feature_names: tuple[str, ...]
    fixed_point_config_hash: str


@dataclass(frozen=True, slots=True)
class IntegerTreeEvaluation:
    leaf_id: int
    raw_action_id: int
    action_ranking: tuple[int, ...]
    path_node_ids: tuple[int, ...]
    path_predicates: tuple[dict[str, object], ...]


def _is_int_not_bool(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: object, field_name: str) -> int:
    if not _is_int_not_bool(value):
        raise ValueError(f"{field_name} 必须是 int")
    return int(value)


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是有限数值")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} 必须是有限数值")
    return numeric


@lru_cache(maxsize=128)
def _build_integer_tree_indexes(model: IntegerTreeModel) -> tuple[dict[int, IntegerTreeNode], dict[int, IntegerTreeLeaf]]:
    nodes = {node.node_id: node for node in model.nodes}
    leaves = {leaf.node_id: leaf for leaf in model.leaves}
    return nodes, leaves


def compile_sklearn_tree_to_integer(
    classifier,
    *,
    state_dim: int,
    action_dim: int,
    feature_names: tuple[str, ...],
    fixed_point_config: FixedPointConfig,
    verification_states: Sequence[tuple[int, ...]] | None = None,
) -> IntegerTreeModel:
    """将 sklearn CART 的结构与叶子类别计数编译到整数模型。"""

    tree = classifier.tree_
    if len(feature_names) != state_dim:
        raise ValueError("feature_names 与 state_dim 不一致")
    if int(tree.n_features) != state_dim:
        raise ValueError("sklearn tree 特征维度与 state_dim 不一致")
    nodes: list[IntegerTreeNode] = []
    leaves: list[IntegerTreeLeaf] = []
    classes = [int(value) for value in classifier.classes_]
    if len(set(classes)) != len(classes):
        raise ValueError("sklearn classes_ 中存在重复动作编号")
    if any(action_id < 0 or action_id >= action_dim for action_id in classes):
        raise ValueError("sklearn classes_ 中存在越界动作编号")
    for node_id in range(int(tree.node_count)):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == right:
            counts = [0.0] * action_dim
            for class_index, action_id in enumerate(classes):
                counts[action_id] = float(tree.value[node_id].ravel()[class_index])
            ranking = tuple(sorted(range(action_dim), key=lambda aid: (-counts[aid], aid)))
            leaves.append(IntegerTreeLeaf(
                node_id=node_id,
                raw_action_id=ranking[0],
                action_ranking=ranking,
                full_action_counts=tuple(counts),
                n_node_samples=int(tree.n_node_samples[node_id]),
                weighted_n_node_samples=float(tree.weighted_n_node_samples[node_id]),
                impurity=float(tree.impurity[node_id]),
            ))
        else:
            threshold = float(tree.threshold[node_id])
            if not math.isfinite(threshold):
                raise ValueError("CART 内部节点 threshold 必须有限")
            nodes.append(IntegerTreeNode(
                node_id=node_id,
                feature_index=int(tree.feature[node_id]),
                threshold_int=math.floor(threshold),
                left_child=left,
                right_child=right,
            ))
    model = IntegerTreeModel(
        schema_version=INTEGER_TREE_SCHEMA_VERSION,
        root_node_id=0,
        state_dim=state_dim,
        action_dim=action_dim,
        nodes=tuple(nodes),
        leaves=tuple(leaves),
        feature_names=tuple(feature_names),
        fixed_point_config_hash=fixed_point_config_hash(fixed_point_config),
    )
    _validate_integer_tree_model(model)
    if verification_states is not None:
        import numpy as np

        for state in verification_states:
            if len(state) != state_dim or any(not _is_int_not_bool(value) for value in state):
                raise ValueError("verification_states 必须是合法的整数状态向量")
            integer_evaluation = evaluate_integer_tree(model, state)
            # 验证仅用于训练/构建阶段，部署 evaluator 本身仍不依赖 sklearn。
            x = np.asarray([state], dtype=np.int32)
            sklearn_leaf = int(classifier.apply(x)[0])
            if sklearn_leaf != integer_evaluation.leaf_id:
                raise ValueError("sklearn 与 integer tree 的 leaf 路径不一致")
            probabilities = classifier.predict_proba(x)[0]
            counts = [0.0] * action_dim
            for class_index, action_id in enumerate(classes):
                counts[action_id] = float(probabilities[class_index])
            expected_ranking = tuple(sorted(range(action_dim), key=lambda aid: (-counts[aid], aid)))
            if expected_ranking != integer_evaluation.action_ranking:
                raise ValueError("sklearn 与 integer tree 的 action ranking 不一致")
    return model


def _validate_integer_tree_model(model: IntegerTreeModel) -> None:
    if not isinstance(model.schema_version, str) or model.schema_version != INTEGER_TREE_SCHEMA_VERSION:
        raise ValueError(f"未知 integer tree schema: {model.schema_version}")
    if not _is_int_not_bool(model.root_node_id) or model.root_node_id < 0:
        raise ValueError("integer tree root_node_id 非法")
    if not _is_int_not_bool(model.state_dim) or not _is_int_not_bool(model.action_dim):
        raise ValueError("integer tree state_dim/action_dim 类型非法")
    if model.state_dim != len(model.feature_names) or model.action_dim <= 0:
        raise ValueError("integer tree 维度非法")
    nodes: dict[int, IntegerTreeNode] = {}
    leaves: dict[int, IntegerTreeLeaf] = {}
    for node in model.nodes:
        if not _is_int_not_bool(node.node_id):
            raise ValueError("integer tree node_id 类型非法")
        if node.node_id in nodes:
            raise ValueError("integer tree 存在重复 node ID")
        if not _is_int_not_bool(node.feature_index) or not _is_int_not_bool(node.threshold_int):
            raise ValueError("integer tree node 字段类型非法")
        if not _is_int_not_bool(node.left_child) or not _is_int_not_bool(node.right_child):
            raise ValueError("integer tree child ID 类型非法")
        if not math.isfinite(float(node.threshold_int)):
            raise ValueError("integer tree threshold_int 必须有限")
        nodes[node.node_id] = node
    for leaf in model.leaves:
        if not _is_int_not_bool(leaf.node_id):
            raise ValueError("integer tree leaf_id 类型非法")
        if leaf.node_id in leaves:
            raise ValueError("integer tree 存在重复 leaf ID")
        if not _is_int_not_bool(leaf.raw_action_id):
            raise ValueError("integer tree raw_action_id 类型非法")
        if not _is_int_not_bool(leaf.n_node_samples) or leaf.n_node_samples < 0:
            raise ValueError("integer tree leaf 样本数非法")
        if not math.isfinite(float(leaf.weighted_n_node_samples)) or leaf.weighted_n_node_samples < 0.0:
            raise ValueError("integer tree leaf 加权样本数非法")
        if not math.isfinite(float(leaf.impurity)) or leaf.impurity < 0.0:
            raise ValueError("integer tree leaf impurity 非法")
        if len(leaf.full_action_counts) != model.action_dim:
            raise ValueError("integer tree full_action_counts 长度必须等于 action_dim")
        if any((not math.isfinite(float(count))) or float(count) < 0.0 for count in leaf.full_action_counts):
            raise ValueError("integer tree full_action_counts 必须有限且非负")
        if len(leaf.action_ranking) != model.action_dim or not leaf.action_ranking:
            raise ValueError("leaf action_ranking 必须是完整排列")
        if any(not _is_int_not_bool(action_id) for action_id in leaf.action_ranking):
            raise ValueError("leaf action_ranking 元素类型非法")
        if tuple(sorted(leaf.action_ranking)) != tuple(range(model.action_dim)):
            raise ValueError("leaf action_ranking 必须覆盖完整 action_dim")
        if leaf.raw_action_id != leaf.action_ranking[0]:
            raise ValueError("leaf raw_action_id 必须等于 action_ranking 首项")
        leaves[leaf.node_id] = leaf
    if model.root_node_id not in nodes and model.root_node_id not in leaves:
        raise ValueError("integer tree node/leaf 图非法")
    reachable: set[int] = set()
    visiting: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in visiting:
            raise ValueError("integer tree graph 存在环")
        if node_id in reachable:
            return
        visiting.add(node_id)
        if node_id in leaves:
            pass
        elif node_id in nodes:
            node = nodes[node_id]
            if node.feature_index < 0 or node.feature_index >= model.state_dim:
                raise ValueError("integer tree feature_index 非法")
            if not math.isfinite(float(node.threshold_int)):
                raise ValueError("integer tree threshold_int 非法")
            if node.left_child == node.right_child:
                raise ValueError("内部节点左右子节点不能相同")
            if (node.left_child not in nodes and node.left_child not in leaves) or (
                node.right_child not in nodes and node.right_child not in leaves
            ):
                raise ValueError("integer tree child ID 非法")
            visit(node.left_child)
            visit(node.right_child)
        else:
            raise ValueError("integer tree 存在不可达引用")
        visiting.remove(node_id)
        reachable.add(node_id)
    visit(model.root_node_id)
    if reachable != set(nodes) | set(leaves):
        raise ValueError("integer tree 存在不可达 node/leaf")


def evaluate_integer_tree(model: IntegerTreeModel, state_int: Sequence[int]) -> IntegerTreeEvaluation:
    """只用 Python 整数比较执行一条树路径。"""

    if len(state_int) != model.state_dim or any(not _is_int_not_bool(v) for v in state_int):
        raise ValueError("state_int 维度或元素类型不匹配")
    nodes, leaves = _build_integer_tree_indexes(model)
    node_id = model.root_node_id
    path = [node_id]
    predicates: list[dict[str, object]] = []
    while node_id not in leaves:
        node = nodes[node_id]
        value = int(state_int[node.feature_index])
        go_left = value <= node.threshold_int
        child = node.left_child if go_left else node.right_child
        predicates.append({
            "node_id": node.node_id,
            "feature_index": node.feature_index,
            "feature_name": model.feature_names[node.feature_index],
            "operator": "<=" if go_left else ">",
            "decision": "left" if go_left else "right",
            "value_int": value,
            "threshold_int": node.threshold_int,
            "value": value,
            "threshold": node.threshold_int,
        })
        node_id = child
        path.append(node_id)
    leaf = leaves[node_id]
    return IntegerTreeEvaluation(leaf.node_id, leaf.raw_action_id, leaf.action_ranking, tuple(path), tuple(predicates))


def _model_to_dict(model: IntegerTreeModel) -> dict[str, object]:
    return {
        "schema_version": model.schema_version,
        "root_node_id": model.root_node_id,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "nodes": [asdict(node) for node in model.nodes],
        "leaves": [{**asdict(leaf), "action_ranking": list(leaf.action_ranking), "full_action_counts": list(leaf.full_action_counts)} for leaf in model.leaves],
        "feature_names": list(model.feature_names),
        "fixed_point_config_hash": model.fixed_point_config_hash,
    }


def _model_from_dict(data: dict[str, object]) -> IntegerTreeModel:
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("leaves"), list):
        raise ValueError("integer tree JSON 结构非法")
    model = IntegerTreeModel(
        schema_version=str(data["schema_version"]),
        root_node_id=_require_int(data["root_node_id"], "root_node_id"),
        state_dim=_require_int(data["state_dim"], "state_dim"),
        action_dim=_require_int(data["action_dim"], "action_dim"),
        nodes=tuple(
            IntegerTreeNode(
                node_id=_require_int(row["node_id"], "node_id"),
                feature_index=_require_int(row["feature_index"], "feature_index"),
                threshold_int=_require_int(row["threshold_int"], "threshold_int"),
                left_child=_require_int(row["left_child"], "left_child"),
                right_child=_require_int(row["right_child"], "right_child"),
            )
            for row in data["nodes"]  # type: ignore[union-attr]
        ),
        leaves=tuple(
            IntegerTreeLeaf(
                node_id=_require_int(row["node_id"], "node_id"),
                raw_action_id=_require_int(row["raw_action_id"], "raw_action_id"),
                action_ranking=tuple(_require_int(v, "action_ranking") for v in row["action_ranking"]),
                full_action_counts=tuple(_require_float(v, "full_action_counts") for v in row["full_action_counts"]),
                n_node_samples=_require_int(row["n_node_samples"], "n_node_samples"),
                weighted_n_node_samples=_require_float(row["weighted_n_node_samples"], "weighted_n_node_samples"),
                impurity=_require_float(row["impurity"], "impurity"),
            )
            for row in data["leaves"]  # type: ignore[union-attr]
        ),
        feature_names=tuple(str(v) for v in data["feature_names"]),  # type: ignore[arg-type]
        fixed_point_config_hash=str(data["fixed_point_config_hash"]),
    )
    _validate_integer_tree_model(model)
    return model


def save_integer_tree_json(model: IntegerTreeModel, path: Path) -> str:
    """保存规范 JSON，并返回内容 hash。"""

    _validate_integer_tree_model(model)
    text = json.dumps(_model_to_dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Write the exact bytes that are hashed. Text-mode writes translate LF to
    # CRLF on Windows, which made artifact_manifest.json contain a stale hash.
    payload = (text + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_integer_tree_json(path: Path) -> IntegerTreeModel:
    model = _model_from_dict(json.loads(path.read_text(encoding="utf-8")))
    return model


def integer_tree_hash(model: IntegerTreeModel) -> str:
    text = json.dumps(_model_to_dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
