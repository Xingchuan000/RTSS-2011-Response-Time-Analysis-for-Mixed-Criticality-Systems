"""整数树 artifact 到执行模型的严格解析器。

该模块只负责把磁盘上的 JSON 还原为 ``amc_py`` 已有的整数树数据结构。
它不修复节点编号、排序叶子或补齐缺失字段；任何结构不完整都必须直接
失败，这样 compiler 和 verifier 消费的是同一份明确 artifact 身份。
"""

from __future__ import annotations

from typing import Any, Mapping

from amc_py.viper.integer_tree import (
    IntegerTreeLeaf,
    IntegerTreeModel,
    IntegerTreeNode,
)


def integer_tree_from_dict(data: Mapping[str, Any]) -> IntegerTreeModel:
    """严格解析完整节点/叶子形式的 integer tree JSON。"""

    def integer(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} 必须是整数")
        return int(value)

    nodes_data = data.get("nodes")
    leaves_data = data.get("leaves")
    if not isinstance(nodes_data, list) or not isinstance(leaves_data, list):
        raise ValueError("integer_tree.json 必须包含 nodes 和 leaves 数组")
    nodes = tuple(
        IntegerTreeNode(
            node_id=integer(row["node_id"], "node_id"),
            feature_index=integer(row["feature_index"], "feature_index"),
            threshold_int=integer(row["threshold_int"], "threshold_int"),
            left_child=integer(row["left_child"], "left_child"),
            right_child=integer(row["right_child"], "right_child"),
        )
        for row in nodes_data
    )
    leaves = tuple(
        IntegerTreeLeaf(
            node_id=integer(row["node_id"], "leaf.node_id"),
            raw_action_id=integer(row["raw_action_id"], "leaf.raw_action_id"),
            action_ranking=tuple(integer(value, "leaf.action_ranking") for value in row["action_ranking"]),
            full_action_counts=tuple(float(value) for value in row.get("full_action_counts", [])),
            n_node_samples=integer(row.get("n_node_samples", 0), "leaf.n_node_samples"),
            weighted_n_node_samples=float(row.get("weighted_n_node_samples", 0)),
            impurity=float(row.get("impurity", 0)),
        )
        for row in leaves_data
    )
    if any(len(leaf.full_action_counts) != int(data["action_dim"]) for leaf in leaves):
        raise ValueError("leaf.full_action_counts 长度必须等于 action_dim")
    return IntegerTreeModel(
        schema_version=str(data["schema_version"]),
        root_node_id=integer(data["root_node_id"], "root_node_id"),
        state_dim=integer(data["state_dim"], "state_dim"),
        action_dim=integer(data["action_dim"], "action_dim"),
        nodes=nodes,
        leaves=leaves,
        feature_names=tuple(str(value) for value in data["feature_names"]),
        fixed_point_config_hash=str(data["fixed_point_config_hash"]),
    )
