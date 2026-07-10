"""VIPER integer tree 专项测试。

测试内容：
1. integer artifact 不调用 sklearn runtime；
2. best/ artifact 加载为 IntegerTreeBudgetPolicy；
3. metadata 缺失或不一致时 fail closed。
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
)
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy
from amc_py.viper.fixed_point import FixedPointConfig


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
    """构造合法的定点配置。"""
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


def test_integer_artifact_loads_as_integer_tree_policy(tmp_path: Path):
    """integer artifact 应加载为 IntegerTreeBudgetPolicy，不依赖 sklearn。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1,
        "action_dim": 3,
        "method": "bc",
        "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "formal_deployment_v1",
    }
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata=metadata,
        feature_names=("f0",),
        action_definitions=action_defs,
        fixed_point_config=fpc,
    )
    policy = load_tree_policy_artifact(tmp_path, require_integer_tree=True)
    assert isinstance(policy, IntegerTreeBudgetPolicy)
    assert policy.metadata["deployment_uses_sklearn"] is False


def test_integer_artifact_missing_semantic_field_fails(tmp_path: Path):
    """缺少关键部署语义字段的 integer artifact 应加载失败。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1,
        "action_dim": 3,
        "method": "bc",
        "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "formal_deployment_v1",
    }
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata=metadata,
        feature_names=("f0",),
        action_definitions=action_defs,
        fixed_point_config=fpc,
    )
    # 手动删除 metadata.json 中的关键字段，模拟残缺 artifact
    with (tmp_path / "metadata.json").open("r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    saved_meta = {k: v for k, v in saved_meta.items() if k != "lo_budget_overrun_guard_units"}
    with (tmp_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(saved_meta, f)
    _regenerate_manifest(tmp_path)

    with pytest.raises(ValueError, match="缺少关键部署语义字段"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_integer_artifact_fallback_mode_wrong_fails(tmp_path: Path):
    """fallback_mode 不是 top1_or_noop 时应失败。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1,
        "action_dim": 3,
        "method": "bc",
        "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
        "deployment_semantics_version": "legacy_mixed_semantics_v1",
    }
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata=metadata,
        feature_names=("f0",),
        action_definitions=action_defs,
        fixed_point_config=fpc,
    )
    # 修改 fallback_mode 并重新 hash
    with (tmp_path / "metadata.json").open("r", encoding="utf-8") as f:
        saved = json.load(f)
    saved["fallback_mode"] = "ranked_valid_or_none"
    with (tmp_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(saved, f)
    _regenerate_manifest(tmp_path)

    with pytest.raises(ValueError, match="fallback_mode"):
        load_tree_policy_artifact(tmp_path, require_integer_tree=True)


def test_deployment_semantics_version_legacy_not_formal(tmp_path: Path):
    """当不完全满足 formal_deployment_v1 条件时，应写 legacy 版本。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1,
        "action_dim": 3,
        "method": "bc",
        "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "legacy",
        "strict_candidate_deploy_cap": False,
        "carry_over_aware_safety": False,
        "lo_budget_overrun_guard_units": 0,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
    }
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata=metadata,
        feature_names=("f0",),
        action_definitions=action_defs,
        fixed_point_config=fpc,
    )
    # 加载后检查 metadata 中的 deployment_semantics_version
    with (tmp_path / "metadata.json").open("r", encoding="utf-8") as f:
        saved_metadata = json.load(f)
    assert saved_metadata["deployment_semantics_version"] != "formal_deployment_v1"
    assert "legacy" in saved_metadata["deployment_semantics_version"]


def test_deployment_semantics_version_formal_v1_correct(tmp_path: Path):
    """全部 formal 条件满足时，应写 formal_deployment_v1。"""
    clf, action_defs = _simple_classifier_and_defs()
    fpc = _make_fixed_point_config()
    metadata = {
        "state_dim": 1,
        "action_dim": 3,
        "method": "bc",
        "tree_id": "t0",
        "fallback_mode": "top1_or_noop",
        "action_validation_mode": "formal_v1",
        "strict_candidate_deploy_cap": True,
        "carry_over_aware_safety": True,
        "lo_budget_overrun_guard_units": 1,
        "budget_overrun_semantics": "strictly_greater_than_release_budget",
        "tree_state_encoding": "fixed_point_int",
    }
    save_tree_policy_artifact(
        tmp_path,
        classifier=clf,
        metadata=metadata,
        feature_names=("f0",),
        action_definitions=action_defs,
        fixed_point_config=fpc,
    )
    with (tmp_path / "metadata.json").open("r", encoding="utf-8") as f:
        saved_metadata = json.load(f)
    assert saved_metadata["deployment_semantics_version"] == "formal_deployment_v1"
    assert saved_metadata["action_validation_mode"] == "formal_v1"
    assert saved_metadata["strict_candidate_deploy_cap"] is True
    assert saved_metadata["carry_over_aware_safety"] is True
    assert saved_metadata["lo_budget_overrun_guard_units"] == 1
