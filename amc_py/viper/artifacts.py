"""VIPER tree artifact 的保存与加载。

新增 leaf-level audit 的离线 leaf rule table 导出：
- export_tree_leaf_table(): 导出每个叶子的规则路径，供 HOUT leaf audit 聚合时 join。
- save_tree_policy_artifact(): 在保存 rules.txt 之外新增 leaf_rules.json 和 leaf_rules.csv。
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
import tempfile

import joblib
import numpy as np
from sklearn.tree import export_text

from amc_py.viper.tree_policy import TreeBudgetPolicy
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_from_dict, fixed_point_config_hash
from amc_py.viper.integer_tree import (
    IntegerTreeModel,
    compile_sklearn_tree_to_integer,
    integer_tree_hash,
    load_integer_tree_json,
    save_integer_tree_json,
)
from amc_py.viper.schema import VIPER_ARTIFACT_SCHEMA_VERSION
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_hash(data: object) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _validate_int_vector(values: Sequence[object], *, field_name: str) -> tuple[int, ...]:
    """严格验证整数向量，拒绝 bool / float / 其他非 int 值。"""

    validated: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} 必须全部是 int")
        validated.append(int(value))
    return tuple(validated)


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
    verification_states: Sequence[tuple[int, ...]] | None = None,
) -> Path:
    """把一棵树保存成计划要求的 artifact 目录。

    在原有 rules.txt 之外，新增 leaf_rules.json 和 leaf_rules.csv 供 leaf audit 使用。
    """
    if output_dir.exists():
        raise ValueError(f"artifact 目录已存在，拒绝覆盖: {output_dir}")
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        metadata_to_write = dict(metadata)
        is_fixed_point = metadata_to_write.get("student_state_encoding") == "fixed_point_int"
        if is_fixed_point:
            if verification_states is None or len(verification_states) == 0:
                raise ValueError("fixed_point_int artifact 必须提供 verification_states")
            raw_config = metadata_to_write.get("fixed_point_config")
            if not isinstance(raw_config, dict):
                raise ValueError("fixed_point_int artifact 缺少 fixed_point_config")
            config = fixed_point_config_from_dict(raw_config)
            integer_verification_states = tuple(
                _validate_int_vector(state, field_name="verification_state") for state in verification_states
            )
            integer_model = compile_sklearn_tree_to_integer(
                classifier,
                state_dim=int(metadata_to_write["state_dim"]),
                action_dim=int(metadata_to_write["action_dim"]),
                feature_names=feature_names,
                fixed_point_config=config,
                verification_states=integer_verification_states,
            )
            verification_state_hash = _canonical_json_hash([list(state) for state in integer_verification_states])
            metadata_to_write.update({
                "tree_artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION,
                "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none",
                "tree_state_encoding": "fixed_point_int",
                "tree_fixed_point_scale": int(config.scale),
                "tree_fixed_point_config_hash": fixed_point_config_hash(config),
                "integer_equivalence_verified": True,
                "integer_equivalence_state_count": len(integer_verification_states),
                "integer_equivalence_verification_state_hash": verification_state_hash,
                "runtime_policy_type": "integer_tree_ranked_valid_or_none",
            })
        else:
            metadata_to_write.setdefault("tree_artifact_schema_version", "legacy_sklearn_ranked_valid_or_none")
            metadata_to_write.setdefault("tree_runtime_policy_type", "legacy_sklearn_ranked_valid_or_none")
            metadata_to_write.setdefault("tree_state_encoding", "legacy_float32")
            metadata_to_write.setdefault("tree_fixed_point_scale", None)
            metadata_to_write.setdefault("tree_fixed_point_config_hash", None)
            metadata_to_write.setdefault("integer_equivalence_verified", False)
            metadata_to_write.setdefault("integer_equivalence_state_count", 0)
            metadata_to_write.setdefault("integer_equivalence_verification_state_hash", None)
            metadata_to_write.setdefault("runtime_policy_type", "legacy_sklearn_ranked_valid_or_none")
            integer_model = None

        joblib.dump(classifier, tmp_dir / "model.joblib")
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata_to_write, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (tmp_dir / "feature_names.json").write_text(
            json.dumps(list(feature_names), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (tmp_dir / "action_definitions.json").write_text(
            json.dumps(action_definitions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (tmp_dir / "rules.txt").write_text(export_tree_rules_text(classifier, feature_names), encoding="utf-8")

        leaf_table = export_tree_leaf_table(classifier, feature_names, action_definitions)
        (tmp_dir / "leaf_rules.json").write_text(
            json.dumps(leaf_table, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        csv_path = tmp_dir / "leaf_rules.csv"
        if leaf_table:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "leaf_id",
                        "path_depth",
                        "predicted_action_id",
                        "predicted_action_summary",
                        "leaf_n_node_samples",
                        "leaf_impurity",
                        "path_text",
                    ],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(leaf_table)
        else:
            csv_path.write_text(
                "leaf_id,path_depth,predicted_action_id,predicted_action_summary,leaf_n_node_samples,leaf_impurity,path_text\n",
                encoding="utf-8",
            )

        manifest_hashes: dict[str, str] = {
            name: _sha256_file(tmp_dir / name)
            for name in (
                "metadata.json",
                "feature_names.json",
                "action_definitions.json",
                "integer_tree.json",
                "fixed_point_config.json",
            )
            if (tmp_dir / name).exists()
        }

        if is_fixed_point:
            tree_path = tmp_dir / "integer_tree.json"
            tree_hash = save_integer_tree_json(integer_model, tree_path)  # type: ignore[arg-type]
            (tmp_dir / "fixed_point_config.json").write_text(
                json.dumps(raw_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            int_leaf_table = export_integer_tree_leaf_table(integer_model)  # type: ignore[arg-type]
            (tmp_dir / "leaf_rules_int.json").write_text(
                json.dumps(int_leaf_table, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with (tmp_dir / "leaf_rules_int.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["leaf_id", "path_depth", "raw_action_id", "action_ranking", "path_text"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(int_leaf_table)
            manifest_hashes.update({
                "leaf_rules.json": _sha256_file(tmp_dir / "leaf_rules.json"),
                "leaf_rules.csv": _sha256_file(tmp_dir / "leaf_rules.csv"),
                "integer_tree.json": tree_hash,
                "fixed_point_config.json": _sha256_file(tmp_dir / "fixed_point_config.json"),
                "leaf_rules_int.json": _sha256_file(tmp_dir / "leaf_rules_int.json"),
                "leaf_rules_int.csv": _sha256_file(tmp_dir / "leaf_rules_int.csv"),
                "model.joblib": _sha256_file(tmp_dir / "model.joblib"),
            })
            manifest = {
                "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION,
                "integer_tree_schema_version": integer_model.schema_version,
                "integer_tree_hash": integer_tree_hash(integer_model),
                "integer_tree_file_hash": tree_hash,
                "fixed_point_config_hash": integer_model.fixed_point_config_hash,
                "integer_equivalence_verified": True,
                "integer_equivalence_state_count": len(integer_verification_states),
                "integer_equivalence_verification_state_hash": metadata_to_write["integer_equivalence_verification_state_hash"],
                "file_hashes": manifest_hashes,
            }
            (tmp_dir / "artifact_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            manifest = {
                "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION,
                "integer_tree_schema_version": None,
                "integer_tree_hash": None,
                "integer_tree_file_hash": None,
                "fixed_point_config_hash": None,
                "integer_equivalence_verified": False,
                "integer_equivalence_state_count": 0,
                "integer_equivalence_verification_state_hash": None,
                "file_hashes": {
                    **manifest_hashes,
                    "rules.txt": _sha256_file(tmp_dir / "rules.txt"),
                    "leaf_rules.json": _sha256_file(tmp_dir / "leaf_rules.json"),
                    "leaf_rules.csv": _sha256_file(tmp_dir / "leaf_rules.csv"),
                    "model.joblib": _sha256_file(tmp_dir / "model.joblib"),
                },
            }
            (tmp_dir / "artifact_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        tmp_dir.rename(output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def export_integer_tree_leaf_table(model: IntegerTreeModel) -> list[dict[str, object]]:
    """导出只包含整数 predicate 的 leaf 规则表。"""

    nodes = {node.node_id: node for node in model.nodes}
    rows: list[dict[str, object]] = []
    for leaf in model.leaves:
        path: list[int] = []
        predicates: list[dict[str, object]] = []
        def walk(node_id: int) -> bool:
            path.append(node_id)
            if node_id == leaf.node_id:
                return True
            node = nodes.get(node_id)
            if node is None:
                path.pop()
                return False
            for child, decision, operator in ((node.left_child, "left", "<="), (node.right_child, "right", ">")):
                before = len(predicates)
                predicates.append({"node_id": node.node_id, "feature_index": node.feature_index, "feature_name": model.feature_names[node.feature_index], "operator": operator, "decision": decision, "threshold_int": node.threshold_int})
                if walk(child):
                    return True
                del predicates[before:]
            path.pop()
            return False
        walk(model.root_node_id)
        rows.append({
            "leaf_id": leaf.node_id,
            "path_depth": len(predicates),
            "path_node_ids": path,
            "path_predicates": predicates,
            "raw_action_id": leaf.raw_action_id,
            "action_ranking": list(leaf.action_ranking),
            "full_action_counts": list(leaf.full_action_counts),
            "n_node_samples": leaf.n_node_samples,
            "weighted_n_node_samples": leaf.weighted_n_node_samples,
            "impurity": leaf.impurity,
            "path_text": " AND ".join(f"{p['feature_name']} {p['operator']} {p['threshold_int']}" for p in predicates) or "LEAF",
        })
    return rows


def load_tree_policy_artifact(tree_artifact_dir: Path, *, require_integer_tree: bool = False):
    """从 artifact 目录恢复 tree policy。"""

    with (tree_artifact_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (tree_artifact_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = tuple(json.load(handle))
    with (tree_artifact_dir / "action_definitions.json").open("r", encoding="utf-8") as handle:
        action_definitions = list(json.load(handle))
    state_dim = int(metadata.get("state_dim", len(feature_names)))
    action_dim = int(metadata.get("action_dim", len(action_definitions)))
    if state_dim != len(feature_names):
        raise ValueError("tree metadata.state_dim 与 feature_names 长度不一致")
    if action_dim != len(action_definitions):
        raise ValueError("tree metadata.action_dim 与 action_definitions 长度不一致")
    tree_state_encoding = str(metadata.get("student_state_encoding", "legacy_float32"))
    runtime_policy_type = str(metadata.get("runtime_policy_type", "legacy_sklearn_ranked_valid_or_none"))
    tree_artifact_schema_version = metadata.get("tree_artifact_schema_version")
    if tree_state_encoding == "fixed_point_int":
        integer_path = tree_artifact_dir / "integer_tree.json"
        config_path = tree_artifact_dir / "fixed_point_config.json"
        manifest_path = tree_artifact_dir / "artifact_manifest.json"
        required_paths = [
            tree_artifact_dir / "metadata.json",
            tree_artifact_dir / "feature_names.json",
            tree_artifact_dir / "action_definitions.json",
            tree_artifact_dir / "rules.txt",
            tree_artifact_dir / "leaf_rules.json",
            tree_artifact_dir / "leaf_rules.csv",
            integer_path,
            config_path,
            tree_artifact_dir / "leaf_rules_int.json",
            tree_artifact_dir / "leaf_rules_int.csv",
            manifest_path,
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise ValueError("integer artifact 文件不完整，禁止回退到 sklearn")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_schema_version") != VIPER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("未知 VIPER artifact schema")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config = fixed_point_config_from_dict(config_data)
        if metadata.get("fixed_point_config") != config_data:
            raise ValueError("metadata.fixed_point_config 与 fixed_point_config.json 不一致")
        expected_config_hash = fixed_point_config_hash(config)
        if metadata.get("fixed_point_config_hash") != expected_config_hash:
            raise ValueError("metadata.fixed_point_config_hash 与 fixed_point_config.json 不一致")
        if metadata.get("tree_fixed_point_config_hash") != expected_config_hash:
            raise ValueError("metadata.tree_fixed_point_config_hash 与 fixed_point_config.json 不一致")
        if metadata.get("tree_fixed_point_scale") != int(config.scale):
            raise ValueError("metadata.tree_fixed_point_scale 与 fixed_point_config.json 不一致")
        if metadata.get("tree_state_encoding") != "fixed_point_int":
            raise ValueError("metadata.tree_state_encoding 与 fixed_point_config.json 不一致")
        if metadata.get("tree_runtime_policy_type") != "integer_tree_ranked_valid_or_none":
            raise ValueError("metadata.tree_runtime_policy_type 与 fixed_point_config.json 不一致")
        if metadata.get("runtime_policy_type") != "integer_tree_ranked_valid_or_none":
            raise ValueError("metadata.runtime_policy_type 与 fixed_point_config.json 不一致")
        if metadata.get("tree_artifact_schema_version") != manifest.get("artifact_schema_version"):
            raise ValueError("metadata.tree_artifact_schema_version 与 manifest 不一致")
        model = load_integer_tree_json(integer_path)
        if manifest.get("integer_tree_schema_version") != model.schema_version:
            raise ValueError("integer tree schema version 校验失败")
        if manifest.get("integer_tree_file_hash") != _sha256_file(integer_path):
            raise ValueError("integer tree file hash 校验失败")
        if manifest.get("integer_tree_hash") != integer_tree_hash(model):
            raise ValueError("integer tree hash 校验失败")
        if manifest.get("fixed_point_config_hash") != expected_config_hash:
            raise ValueError("fixed-point config hash 校验失败")
        if model.fixed_point_config_hash != expected_config_hash:
            raise ValueError("fixed-point config hash 校验失败")
        if manifest.get("integer_equivalence_verified") is not True:
            raise ValueError("integer_equivalence_verified 未声明为 true")
        if int(manifest.get("integer_equivalence_state_count", 0)) <= 0:
            raise ValueError("integer_equivalence_state_count 非法")
        if metadata.get("integer_equivalence_verified") is not True:
            raise ValueError("metadata.integer_equivalence_verified 未声明为 true")
        if int(metadata.get("integer_equivalence_state_count", 0)) != int(manifest.get("integer_equivalence_state_count", 0)):
            raise ValueError("metadata.integer_equivalence_state_count 与 manifest 不一致")
        if metadata.get("integer_equivalence_verification_state_hash") != manifest.get("integer_equivalence_verification_state_hash"):
            raise ValueError("metadata.integer_equivalence_verification_state_hash 与 manifest 不一致")
        if metadata.get("integer_equivalence_verification_state_hash") is None:
            raise ValueError("metadata.integer_equivalence_verification_state_hash 缺失")
        if metadata.get("state_dim") != state_dim or metadata.get("action_dim") != action_dim:
            raise ValueError("metadata.state_dim/action_dim 与 feature/action 定义不一致")
        if model.state_dim != state_dim or model.action_dim != action_dim:
            raise ValueError("integer tree 维度与 metadata 不一致")
        required_file_hashes = manifest.get("file_hashes")
        if not isinstance(required_file_hashes, dict):
            raise ValueError("artifact_manifest 缺少 file_hashes")
        for filename in (
            "metadata.json",
            "feature_names.json",
            "action_definitions.json",
            "integer_tree.json",
            "fixed_point_config.json",
        ):
            file_path = tree_artifact_dir / filename
            if not file_path.exists():
                raise ValueError(f"integer artifact 缺少文件: {filename}")
            expected_hash = required_file_hashes.get(filename)
            if expected_hash != _sha256_file(file_path):
                raise ValueError(f"artifact 文件 hash 校验失败: {filename}")
        for filename in ("rules.txt", "leaf_rules.json", "leaf_rules.csv", "leaf_rules_int.json", "leaf_rules_int.csv", "model.joblib"):
            file_path = tree_artifact_dir / filename
            if not file_path.exists():
                continue
            expected_hash = required_file_hashes.get(filename)
            if expected_hash is not None and expected_hash != _sha256_file(file_path):
                raise ValueError(f"artifact 文件 hash 校验失败: {filename}")
        return IntegerTreeBudgetPolicy(
            model=model,
            metadata={
                **metadata,
                "tree_state_encoding": "fixed_point_int",
                "tree_fixed_point_scale": int(config.scale),
                "tree_fixed_point_config_hash": expected_config_hash,
                "tree_artifact_schema_version": manifest.get("artifact_schema_version"),
                "integer_equivalence_verified": True,
                "integer_equivalence_state_count": int(manifest["integer_equivalence_state_count"]),
                "integer_equivalence_verification_state_hash": manifest.get("integer_equivalence_verification_state_hash"),
                "runtime_policy_type": "integer_tree_ranked_valid_or_none",
                "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none",
            },
            feature_names=feature_names,
            action_definitions=action_definitions,
            fixed_point_config=config,
        )
    if require_integer_tree:
        raise ValueError("当前 artifact 不是 integer tree artifact")
    legacy_runtime_policy_type = runtime_policy_type if runtime_policy_type else "legacy_sklearn_ranked_valid_or_none"
    if legacy_runtime_policy_type != "legacy_sklearn_ranked_valid_or_none":
        raise ValueError("legacy artifact 的 runtime_policy_type 不合法")
    if tree_state_encoding not in {"legacy_float32", "float32"}:
        raise ValueError("legacy artifact 的 tree_state_encoding 不合法")
    classifier = joblib.load(tree_artifact_dir / "model.joblib")
    if hasattr(classifier, "n_features_in_") and int(classifier.n_features_in_) != state_dim:
        raise ValueError("legacy artifact 的 state_dim 与模型特征维度不一致")
    return TreeBudgetPolicy(
        classifier=classifier,
        metadata={
            **metadata,
            "tree_state_encoding": "legacy_float32",
            "tree_fixed_point_scale": None,
            "tree_fixed_point_config_hash": None,
            "tree_artifact_schema_version": tree_artifact_schema_version,
            "integer_equivalence_verified": False,
            "integer_equivalence_state_count": 0,
            "runtime_policy_type": "legacy_sklearn_ranked_valid_or_none",
            "tree_runtime_policy_type": "legacy_sklearn_ranked_valid_or_none",
        },
        feature_names=feature_names,
        action_definitions=action_definitions,
    )
