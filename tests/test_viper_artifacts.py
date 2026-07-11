"""VIPER artifacts 测试。

新增测试：
- test_save_tree_policy_artifact_writes_leaf_rules: 验证 leaf_rules.json/csv 导出。
- Patch 2: artifact fail-closed 语义校验。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.artifacts import (
    save_tree_policy_artifact,
    load_tree_policy_artifact,
    validate_integer_artifact_semantics,
)
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash, fixed_point_config_to_dict
from amc_py.viper.integer_tree import compile_sklearn_tree_to_integer, integer_tree_hash
from amc_py.viper.schema import (
    VIPER_ARTIFACT_SCHEMA_VERSION,
    VIPER_ARTIFACT_SCHEMA_VERSION_RANKED,
    INTEGER_TREE_SCHEMA_VERSION,
    INTEGER_TREE_SCHEMA_VERSION_RANKED,
)


def _simple_classifier_and_defs() -> tuple:
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    action_defs = [
        {"action_id": 0, "action_name": "noop"},
        {"action_id": 1, "action_name": "increase_0"},
        {"action_id": 2, "action_name": "decrease_0"},
    ]
    return clf, action_defs


def _make_fixed_point_config() -> FixedPointConfig:
    return FixedPointConfig(scale=100, min_int=-100, max_int=100)


def _regenerate_manifest(artifact_dir: Path) -> None:
    """修改 metadata 后重新计算 manifest 中的文件 hash，避免校验失败。"""
    manifest_path = artifact_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        file_path = artifact_dir / str(item["relative_path"] if isinstance(item, dict) else item)
        if file_path.exists() and isinstance(item, dict) and item.get("sha256"):
            item["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def test_save_tree_policy_artifact_writes_leaf_rules(tmp_path: Path) -> None:
    """save_tree_policy_artifact 应输出 leaf_rules.json 和 leaf_rules.csv。"""

    clf, action_defs = _simple_classifier_and_defs()
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata={"state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0"},
        feature_names=("f0",),
        action_definitions=action_defs,
    )

    assert (tmp_path / "leaf_rules.json").exists()
    assert (tmp_path / "leaf_rules.csv").exists()

    with (tmp_path / "leaf_rules.json").open("r", encoding="utf-8") as f:
        leaf_table = json.load(f)

    assert isinstance(leaf_table, list)
    assert len(leaf_table) >= 2  # 至少有 2 个叶子节点
    for leaf in leaf_table:
        assert "leaf_id" in leaf
        assert "path_depth" in leaf
        assert "path_predicates" in leaf
        assert "predicted_action_id" in leaf
        assert "leaf_n_node_samples" in leaf
        assert "path_text" in leaf


# ========== Patch 2 新增测试 ==========


def test_new_artifact_missing_manifest_fallback_rejected(tmp_path: Path) -> None:
    """缺少 manifest fallback_mode 字段的新（非 legacy）artifact 应被拒绝。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "ranked_valid_or_none",
        "tree_fallback_mode": "ranked_valid_or_none",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "fixed_ranked_deployment_v1",
        "runtime_policy_type": "integer_tree_ranked_valid_or_none",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    # 删除 manifest 中的 fallback_mode 字段
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["fallback_mode"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="fallback_mode"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def _save_ranked_integer_artifact(tmp_path: Path) -> None:
    """写入一个可供 integer-only 加载测试使用的最小 ranked artifact。"""
    clf, action_defs = _simple_classifier_and_defs()
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata={
            "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
            "fallback_mode": "ranked_valid_or_none",
            "tree_fallback_mode": "ranked_valid_or_none",
            "action_validation_mode": "formal_v1",
            "strict_candidate_deploy_cap": True,
            "carry_over_aware_safety": True,
            "lo_budget_overrun_guard_units": 1,
            "budget_overrun_semantics": "strictly_greater_than_release_budget",
            "tree_state_encoding": "fixed_point_int",
            "deployment_semantics_version": "fixed_ranked_deployment_v1",
            "runtime_policy_type": "integer_tree_ranked_valid_or_none",
        },
        feature_names=("f0",), action_definitions=action_defs,
        fixed_point_config=_make_fixed_point_config(),
    )


def test_fixed_ranked_hout_does_not_call_joblib_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """require_integer_tree 的部署加载不能退回读取 model.joblib。"""
    _save_ranked_integer_artifact(tmp_path)
    import amc_py.viper.artifacts as artifacts_module

    monkeypatch.setattr(
        artifacts_module.joblib, "load",
        lambda *_args, **_kwargs: pytest.fail("integer HOUT 不得调用 joblib.load"),
    )
    loaded = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    assert type(loaded).__name__ == "IntegerTreeBudgetPolicy"


def test_fixed_ranked_hout_does_not_call_sklearn_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """integer policy rollout 只解释 integer_tree，不得调用 sklearn predict。"""
    _save_ranked_integer_artifact(tmp_path)
    from sklearn.tree import DecisionTreeClassifier

    monkeypatch.setattr(DecisionTreeClassifier, "predict", lambda *_a, **_k: pytest.fail("不得调用 sklearn predict"))
    policy = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    selected, _ = policy.select_action_id((1.0,), (True, True, True))
    assert selected is not None


def test_fixed_ranked_hout_does_not_call_sklearn_predict_proba(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ranked integer policy 的动作排序不能调用 sklearn predict_proba。"""
    _save_ranked_integer_artifact(tmp_path)
    from sklearn.tree import DecisionTreeClassifier

    monkeypatch.setattr(DecisionTreeClassifier, "predict_proba", lambda *_a, **_k: pytest.fail("不得调用 sklearn predict_proba"))
    policy = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    assert policy.predict_action_ranking((1.0,))


def test_require_integer_tree_never_falls_back_when_integer_file_missing(tmp_path: Path) -> None:
    """integer_tree.json 丢失时必须 fail-closed，不能读取 legacy model.joblib。"""
    _save_ranked_integer_artifact(tmp_path)
    (tmp_path / "integer_tree.json").unlink()
    with pytest.raises(ValueError, match="integer_tree.json"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_top1_with_ranked_artifact_schema_rejected(tmp_path: Path) -> None:
    """top1 fallback_mode 搭配 ranked artifact schema 应被拒绝。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "formal_deployment_v1",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    # 修改 metadata artifact_schema_version 为 ranked
    meta_path = tmp_path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["artifact_schema_version"] = VIPER_ARTIFACT_SCHEMA_VERSION_RANKED
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _regenerate_manifest(tmp_path)
    with pytest.raises(ValueError, match="schema"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_ranked_with_v1_tree_rejected(tmp_path: Path) -> None:
    """ranked fallback 搭配 v1 integer tree 应被拒绝。
    本测试通过直接调用 validate_integer_artifact_semantics 来验证。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    # 使用 ranked=False 显式编译 v1 tree
    integer_model = compile_sklearn_tree_to_integer(
        clf, feature_names=("f0",),
        fixed_point_config_hash=fixed_point_config_hash(fpc),
        state_dim=1, action_dim=3, ranked=False)
    # 验证 compile 确实产生了 v1 schema
    assert integer_model.schema_version == INTEGER_TREE_SCHEMA_VERSION
    metadata = {
        "fallback_mode": "ranked_valid_or_none",
        "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION_RANKED,
        "runtime_policy_type": "integer_tree_ranked_valid_or_none",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "ranked_valid_or_none",
        "deployment_semantics_version": "fixed_ranked_deployment_v1",
        "integer_tree_schema_version": INTEGER_TREE_SCHEMA_VERSION,
    }
    manifest = {
        "fallback_mode": "ranked_valid_or_none",
        "artifact_schema_version": VIPER_ARTIFACT_SCHEMA_VERSION_RANKED,
        "integer_tree_schema_version": INTEGER_TREE_SCHEMA_VERSION,
        "integer_tree_hash": integer_tree_hash(integer_model),
        "fixed_point_config_hash": fixed_point_config_hash(fpc),
        "action_definitions_hash": hashlib.sha256(json.dumps(action_defs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "runtime_policy_type": "integer_tree_ranked_valid_or_none",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "fixed_ranked_deployment_v1",
    }
    with pytest.raises(ValueError, match="integer_tree_ranked_v2"):
        validate_integer_artifact_semantics(
            metadata=metadata, manifest=manifest, model=integer_model,
            fixed_point_config=fpc, feature_names=("f0",), action_definitions=action_defs,
        )


def test_manifest_runtime_policy_type_mismatch_rejected(tmp_path: Path) -> None:
    """manifest runtime_policy_type 与 fallback_mode 不一致时应被拒绝。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "top1_or_noop",
        "deployment_semantics_version": "formal_deployment_v1",
        "runtime_policy_type": "integer_tree_top1_or_noop",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    # 修改 metadata runtime_policy_type 为错误值
    meta_path = tmp_path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["runtime_policy_type"] = "integer_tree_ranked_valid_or_none"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _regenerate_manifest(tmp_path)
    with pytest.raises(ValueError, match="runtime_policy_type"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_manifest_deployment_version_mismatch_rejected(tmp_path: Path) -> None:
    """deployment_semantics_version 与其他字段不一致时应被拒绝。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "top1_or_noop",
        "deployment_semantics_version": "formal_deployment_v1",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    # 将 deployment_semantics_version 改为不匹配的值
    meta_path = tmp_path / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["deployment_semantics_version"] = "legacy_baseline_v1"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _regenerate_manifest(tmp_path)
    with pytest.raises(ValueError, match="deployment_semantics_version"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_manifest_covers_model_and_integer_files(tmp_path: Path) -> None:
    """manifest.files 应覆盖 model.joblib, metadata.json, integer_tree.json 等关键文件。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "top1_or_noop",
        "deployment_semantics_version": "formal_deployment_v1",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_names = {item["relative_path"] for item in manifest.get("files", [])}
    required = {"model.joblib", "metadata.json", "integer_tree.json", "feature_names.json",
                "action_definitions.json", "rules.txt", "leaf_rules_int.json", "leaf_rules_int.csv",
                "fixed_point_config.json"}
    assert required.issubset(file_names)


def test_best_ranked_artifact_is_validated_as_integer(tmp_path: Path) -> None:
    """best/ ranked artifact 应作为 integer artifact 加载校验。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "ranked_valid_or_none",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "ranked_valid_or_none",
        "deployment_semantics_version": "fixed_ranked_deployment_v1",
        "runtime_policy_type": "integer_tree_ranked_valid_or_none",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    # 加载为 integer tree policy
    loaded = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy
    assert isinstance(loaded, IntegerTreeBudgetPolicy)
    assert loaded.metadata["artifact_schema_version"] == VIPER_ARTIFACT_SCHEMA_VERSION_RANKED


def test_legacy_v1_top1_runtime_round_trip(tmp_path: Path) -> None:
    """历史 v1 top1 artifact 仍可加载和执行。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "tree_fallback_mode": "top1_or_noop",
        "deployment_semantics_version": "formal_deployment_v1",
        "runtime_policy_type": "integer_tree_top1_or_noop",
    }
    save_tree_policy_artifact(tmp_path, classifier=clf, metadata=metadata,
                              feature_names=("f0",), action_definitions=action_defs,
                              fixed_point_config=fpc)
    loaded = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    assert loaded.metadata["artifact_schema_version"] == VIPER_ARTIFACT_SCHEMA_VERSION
    assert loaded.metadata["fallback_mode"] == "top1_or_noop"
