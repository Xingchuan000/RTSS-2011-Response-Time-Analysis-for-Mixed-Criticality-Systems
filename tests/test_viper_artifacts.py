"""VIPER artifacts 测试。

新增测试：
- test_save_tree_policy_artifact_writes_leaf_rules: 验证 leaf_rules.json/csv 导出。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.artifacts import save_tree_policy_artifact


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
