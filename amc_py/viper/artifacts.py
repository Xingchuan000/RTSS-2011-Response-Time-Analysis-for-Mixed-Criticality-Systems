"""VIPER tree artifact 的保存与加载。

新增 leaf-level audit 的离线 leaf rule table 导出：
- export_tree_leaf_table(): 导出每个叶子的规则路径，供 HOUT leaf audit 聚合时 join。
- save_tree_policy_artifact(): 在保存 rules.txt 之外新增 leaf_rules.json 和 leaf_rules.csv。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

from amc_py.viper.tree_policy import TreeBudgetPolicy
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash, fixed_point_config_to_dict, fixed_point_config_from_dict
from amc_py.viper.integer_tree import compile_sklearn_tree_to_integer, integer_tree_hash, load_integer_tree_json, save_integer_tree_json
from amc_py.viper.schema import VIPER_ARTIFACT_SCHEMA_VERSION, resolve_deployment_semantics_version
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy


def required_artifact_files(metadata: dict[str, object], *, require_integer_tree: bool = False) -> tuple[str, ...]:
    """返回由 schema 决定的 artifact 必需文件集合。"""
    integer = require_integer_tree or metadata.get("artifact_schema_version") == VIPER_ARTIFACT_SCHEMA_VERSION
    common = ("model.joblib", "metadata.json", "feature_names.json", "action_definitions.json", "rules.txt")
    if integer:
        return common + ("fixed_point_config.json", "integer_tree.json", "leaf_rules_int.json", "leaf_rules_int.csv", "artifact_manifest.json")
    return common


def _action_definitions_hash(action_definitions: list[dict[str, object]]) -> str:
    return hashlib.sha256(json.dumps(action_definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_artifact_directory(path: Path, *, require_integer_tree: bool = False) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"artifact 缺少 metadata.json: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for filename in required_artifact_files(metadata, require_integer_tree=require_integer_tree):
        if not (path / filename).is_file():
            raise ValueError(f"artifact 缺少必需文件: {filename}")
    manifest_path = path / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            file_path = path / str(item["relative_path"] if isinstance(item, dict) else item)
            if not file_path.is_file():
                raise ValueError(f"artifact manifest 文件不存在: {file_path.name}")
            if isinstance(item, dict) and item.get("sha256"):
                actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
                if actual != item["sha256"]:
                    raise ValueError(f"artifact 文件 hash 不一致: {file_path.name}")


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
    fixed_point_config: FixedPointConfig | None = None,
) -> Path:
    """把一棵树保存成计划要求的 artifact 目录。

    在原有 rules.txt 之外，新增 leaf_rules.json 和 leaf_rules.csv 供 leaf audit 使用。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    use_integer_artifact = fixed_point_config is not None
    fixed_point_config = fixed_point_config or FixedPointConfig()
    integer_model = compile_sklearn_tree_to_integer(
        classifier,
        feature_names=feature_names,
        fixed_point_config_hash=fixed_point_config_hash(fixed_point_config),
        state_dim=len(feature_names),
        action_dim=len(action_definitions),
    )
    save_integer_tree_json(integer_model, output_dir / "integer_tree.json")
    (output_dir / "fixed_point_config.json").write_text(json.dumps(fixed_point_config_to_dict(fixed_point_config), ensure_ascii=False, indent=2), encoding="utf-8")
    if use_integer_artifact:
        action_validation_mode = metadata.get("action_validation_mode", "legacy")
        strict_cap = bool(metadata.get("strict_candidate_deploy_cap", False))
        carry_over = bool(metadata.get("carry_over_aware_safety", False))
        guard_units = int(metadata.get("lo_budget_overrun_guard_units", 0))
        fallback_mode = metadata.get("fallback_mode", "ranked_valid_or_none")
        tree_state_encoding = metadata.get("tree_state_encoding", "fixed_point_int")
        deployment_ver = resolve_deployment_semantics_version(
            tree_state_encoding=tree_state_encoding,
            tree_fallback_mode=fallback_mode,
            action_validation_mode=action_validation_mode,
            strict_candidate_deploy_cap=strict_cap,
            carry_over_aware_safety=carry_over,
            lo_budget_overrun_guard_units=guard_units,
        )
        metadata = {
            **metadata,
            "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION,
            "fixed_point_config_hash": fixed_point_config_hash(fixed_point_config),
            "integer_tree_hash": integer_tree_hash(integer_model),
            "action_definitions_hash": _action_definitions_hash(action_definitions),
            "deployment_uses_sklearn": False,
            "fallback_mode": fallback_mode,
            "deployment_semantics_version": deployment_ver,
            "action_validation_mode": action_validation_mode,
            "strict_candidate_deploy_cap": strict_cap,
            "carry_over_aware_safety": carry_over,
            "lo_budget_overrun_guard_units": guard_units,
            "budget_overrun_semantics": metadata.get("budget_overrun_semantics", "strictly_greater_than_release_budget"),
            "tree_state_encoding": metadata.get("tree_state_encoding", "fixed_point_int"),
            "tree_fallback_mode": fallback_mode,
        }
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

    int_nodes = {node.node_id: node for node in integer_model.nodes}
    int_leaves = {leaf.node_id: leaf for leaf in integer_model.leaves}
    int_leaf_table: list[dict[str, object]] = []
    def collect_int(node_id: int, conditions: list[dict[str, object]]) -> None:
        if node_id in int_leaves:
            leaf = int_leaves[node_id]
            path_text = " AND ".join(f"{item['feature_name']} {item['operator']} {item['threshold_int']}" for item in conditions) or "LEAF"
            int_leaf_table.append({"leaf_id": leaf.node_id, "raw_action_id": leaf.raw_action_id, "path_conditions": conditions, "path_text": path_text, "depth": len(conditions), "sample_count": leaf.sample_count})
            return
        node = int_nodes[node_id]
        for child, operator in ((node.left_child, "<="), (node.right_child, ">")):
            collect_int(child, conditions + [{"node_id": node.node_id, "feature_index": node.feature_index, "feature_name": integer_model.feature_names[node.feature_index], "threshold_int": node.threshold_int, "operator": operator, "decision": "left" if operator == "<=" else "right"}])
    collect_int(integer_model.root_node_id, [])
    (output_dir / "leaf_rules_int.json").write_text(json.dumps(int_leaf_table, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "leaf_rules_int.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["leaf_id", "raw_action_id", "predicted_action_id", "leaf_n_node_samples", "sample_count", "depth", "leaf_impurity", "path_text"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(int_leaf_table)
    manifest_files = []
    for filename in ("metadata.json", "fixed_point_config.json", "integer_tree.json", "feature_names.json", "action_definitions.json", "leaf_rules_int.json", "leaf_rules_int.csv"):
        file_path = output_dir / filename
        manifest_files.append({"relative_path": filename, "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(), "size_bytes": file_path.stat().st_size, "schema_version": VIPER_ARTIFACT_SCHEMA_VERSION})
    manifest = {"artifact_manifest_schema_version": "artifact_manifest_v1", "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION, "integer_tree_hash": integer_tree_hash(integer_model), "fixed_point_config_hash": fixed_point_config_hash(fixed_point_config), "files": manifest_files}
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    return output_dir


def load_tree_policy_artifact(tree_artifact_dir: Path, *, require_integer_tree: bool = False, allow_legacy_fallback: bool = True, fixed_point_config: FixedPointConfig | None = None) -> TreeBudgetPolicy | IntegerTreeBudgetPolicy:
    """从 artifact 目录恢复 tree policy。"""

    with (tree_artifact_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_artifact_directory(tree_artifact_dir, require_integer_tree=require_integer_tree)
    with (tree_artifact_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = tuple(json.load(handle))
    with (tree_artifact_dir / "action_definitions.json").open("r", encoding="utf-8") as handle:
        action_definitions = list(json.load(handle))
    is_new_schema = metadata.get("artifact_schema_version") == VIPER_ARTIFACT_SCHEMA_VERSION
    if is_new_schema and not (tree_artifact_dir / "integer_tree.json").exists():
        raise ValueError("integer artifact metadata 与文件不一致")
    if is_new_schema and (tree_artifact_dir / "integer_tree.json").exists():
        config = fixed_point_config_from_dict(json.loads((tree_artifact_dir / "fixed_point_config.json").read_text(encoding="utf-8")))
        if fixed_point_config is not None and fixed_point_config_hash(config) != fixed_point_config_hash(fixed_point_config):
            raise ValueError("artifact fixed-point config 不一致")
        model = load_integer_tree_json(tree_artifact_dir / "integer_tree.json")
        if model.fixed_point_config_hash != fixed_point_config_hash(config):
            raise ValueError("integer tree fixed-point hash 不一致")
        manifest = json.loads((tree_artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("integer_tree_hash") != integer_tree_hash(model):
            raise ValueError("artifact integer tree hash 不一致")
        if int(model.state_dim) != len(feature_names) or int(model.action_dim) != len(action_definitions):
            raise ValueError("integer tree feature/action dimension 不一致")
        if metadata.get("fallback_mode") != "top1_or_noop":
            raise ValueError("integer artifact fallback_mode 必须是 top1_or_noop")
        if metadata.get("action_definitions_hash") != _action_definitions_hash(action_definitions):
            raise ValueError("artifact action definitions hash 不一致")
        # 新 integer artifact 必须包含所有关键部署语义字段，缺一不可。
        required_semantic_keys = (
            "action_validation_mode",
            "strict_candidate_deploy_cap",
            "carry_over_aware_safety",
            "lo_budget_overrun_guard_units",
            "budget_overrun_semantics",
            "tree_state_encoding",
            "tree_fallback_mode",
            "deployment_semantics_version",
        )
        for key in required_semantic_keys:
            if key not in metadata:
                raise ValueError(f"integer artifact 缺少关键部署语义字段: {key}")
        return IntegerTreeBudgetPolicy(model, metadata, feature_names, action_definitions, config)
    if require_integer_tree:
        raise ValueError("要求整数 tree artifact，但 artifact 缺少 integer_tree.json")
    if not allow_legacy_fallback:
        raise ValueError("禁止 legacy tree artifact fallback")
    classifier = joblib.load(tree_artifact_dir / "model.joblib")
    return TreeBudgetPolicy(
        classifier=classifier,
        metadata=metadata,
        feature_names=feature_names,
        action_definitions=action_definitions,
    )
