"""VIPER tree artifact 的保存与加载。

新增 leaf-level audit 的离线 leaf rule table 导出：
- export_tree_leaf_table(): 导出每个叶子的规则路径，供 HOUT leaf audit 聚合时 join。
- save_tree_policy_artifact(): 在保存 rules.txt 之外新增 leaf_rules.json 和 leaf_rules.csv。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

from amc_py.viper.tree_policy import TreeBudgetPolicy


def export_tree_rules_text(classifier, feature_names: tuple[str, ...]) -> str:
    """导出带特征名的可读规则文本。"""

    return export_text(classifier, feature_names=list(feature_names))


def export_tree_leaf_table(
    classifier,
    feature_names: tuple[str, ...],
    action_definitions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """导出每个叶子的规则路径，供 HOUT leaf audit 聚合时 join。

    每个 leaf row 包含：
    - leaf_id: 叶子节点编号。
    - path_depth: 从根到叶子的深度。
    - path_node_ids: 路径上所有节点编号。
    - path_predicates: 路径上每个内部节点的分裂条件与阈值。
    - predicted_action_id: 叶子预测的动作编号（已映射到 classifier.classes_）。
    - predicted_action_definition: 叶子预测动作的语义描述。
    - leaf_n_node_samples / leaf_weighted_n_node_samples / leaf_impurity / leaf_value: 叶子统计信息。
    """

    tree = classifier.tree_
    n_nodes = tree.node_count
    leaves: list[dict[str, object]] = []

    # DFS 收集每个叶子从根到该叶子的路径
    def _dfs_collect(node_id: int, path_nodes: list[int]) -> None:
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
        if is_leaf:
            # 生成路径上的 predicate 列表
            path_predicates: list[dict[str, object]] = []
            for pid in path_nodes:
                feature_idx = int(tree.feature[pid])
                if feature_idx < 0:
                    continue
                threshold = float(tree.threshold[pid])
                feature_name = feature_names[feature_idx]
                went_left = pid in path_nodes and (
                    tree.children_left[pid] == path_nodes[path_nodes.index(pid) + 1]
                    if path_nodes.index(pid) + 1 < len(path_nodes)
                    else False
                )
                # 判断该节点走向：左子节点为 "<="，右子节点为 ">"
                child = path_nodes[path_nodes.index(pid) + 1] if path_nodes.index(pid) + 1 < len(path_nodes) else -1
                if child == tree.children_left[pid]:
                    operator = "<="
                    decision = "left"
                else:
                    operator = ">"
                    decision = "right"
                predicate: dict[str, object] = {
                    "node_id": int(pid),
                    "feature_index": feature_idx,
                    "feature_name": feature_name,
                    "threshold": threshold,
                    "operator": operator,
                    "decision": decision,
                }
                path_predicates.append(predicate)

            leaf_value_flat = tree.value[node_id].ravel().tolist()
            leaf_predicted_class_raw = int(tree.value[node_id].ravel().argmax())
            predicted_action_id = int(classifier.classes_[leaf_predicted_class_raw])
            predicted_action_def = (
                dict(action_definitions[predicted_action_id])
                if 0 <= predicted_action_id < len(action_definitions)
                else {}
            )

            # 生成扁平化的可读路径文本
            path_text_parts: list[str] = []
            for p in path_predicates:
                path_text_parts.append(
                    f"{p['feature_name']} {p['operator']} {p['threshold']:.6f}"
                )
            path_text = " AND ".join(path_text_parts) if path_text_parts else "LEAF"

            leaves.append({
                "leaf_id": int(node_id),
                "path_depth": int(len(path_predicates)),
                "path_node_ids": [int(pid) for pid in path_nodes],
                "path_predicates": path_predicates,
                "predicted_action_id": predicted_action_id,
                "predicted_action_definition": predicted_action_def,
                "predicted_action_summary": str(predicted_action_def.get("action_name", str(predicted_action_id))),
                "leaf_n_node_samples": int(tree.n_node_samples[node_id]),
                "leaf_weighted_n_node_samples": float(tree.weighted_n_node_samples[node_id]),
                "leaf_impurity": float(tree.impurity[node_id]),
                "leaf_value": leaf_value_flat,
                "path_text": path_text,
            })
            return

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        _dfs_collect(left_child, path_nodes + [left_child])
        _dfs_collect(right_child, path_nodes + [right_child])

    _dfs_collect(0, [0])
    return leaves


def save_tree_policy_artifact(
    output_dir: Path,
    *,
    classifier,
    metadata: dict[str, object],
    feature_names: tuple[str, ...],
    action_definitions: list[dict[str, object]],
) -> Path:
    """把一棵树保存成计划要求的 artifact 目录。

    在原有 rules.txt 之外，新增 leaf_rules.json 和 leaf_rules.csv 供 leaf audit 使用。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "model.joblib")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(list(feature_names), handle, ensure_ascii=False, indent=2)
    with (output_dir / "action_definitions.json").open("w", encoding="utf-8") as handle:
        json.dump(action_definitions, handle, ensure_ascii=False, indent=2)
    with (output_dir / "rules.txt").open("w", encoding="utf-8") as handle:
        handle.write(export_tree_rules_text(classifier, feature_names))

    # 导出 leaf rules table
    leaf_table = export_tree_leaf_table(classifier, feature_names, action_definitions)
    with (output_dir / "leaf_rules.json").open("w", encoding="utf-8") as handle:
        json.dump(leaf_table, handle, ensure_ascii=False, indent=2)

    # CSV 版本：扁平字段，便于人工快速浏览
    csv_path = output_dir / "leaf_rules.csv"
    if leaf_table:
        csv_fieldnames = [
            "leaf_id",
            "path_depth",
            "predicted_action_id",
            "predicted_action_summary",
            "leaf_n_node_samples",
            "leaf_impurity",
            "path_text",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in leaf_table:
                # 将 path_predicates 转为 JSON 字符串写入额外的 predicates 列
                # 这里 CSV 只保留精简字段，完整数据在 leaf_rules.json 中
                writer.writerow(row)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("leaf_id,path_depth,predicted_action_id,predicted_action_summary,leaf_n_node_samples,leaf_impurity,path_text\n")

    return output_dir


def load_tree_policy_artifact(tree_artifact_dir: Path) -> TreeBudgetPolicy:
    """从 artifact 目录恢复 tree policy。"""

    classifier = joblib.load(tree_artifact_dir / "model.joblib")
    with (tree_artifact_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (tree_artifact_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = tuple(json.load(handle))
    with (tree_artifact_dir / "action_definitions.json").open("r", encoding="utf-8") as handle:
        action_definitions = list(json.load(handle))
    return TreeBudgetPolicy(
        classifier=classifier,
        metadata=metadata,
        feature_names=feature_names,
        action_definitions=action_definitions,
    )
