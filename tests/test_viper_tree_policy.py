"""TreeBudgetPolicy 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.artifacts import load_tree_policy_artifact, save_tree_policy_artifact
from amc_py.viper.tree_policy import TreeBudgetPolicy


def _policy() -> TreeBudgetPolicy:
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    return TreeBudgetPolicy(
        classifier=clf,
        metadata={"state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0"},
        feature_names=("f0",),
        action_definitions=[{"action_id": 0}, {"action_id": 1}, {"action_id": 2}],
    )


def test_tree_policy_fallbacks_to_valid_action() -> None:
    policy = _policy()
    action_id, info = policy.select_action_id((2.0,), (True, False, False))
    assert action_id == 0
    assert info["tree_raw_top1_invalid"] is True


def test_tree_policy_returns_none_when_no_valid_action() -> None:
    policy = _policy()
    action_id, info = policy.select_action_id((1.0,), (False, False, False))
    assert action_id is None
    assert info["tree_no_valid_action"] is True


def test_tree_artifact_roundtrip(tmp_path: Path) -> None:
    policy = _policy()
    save_tree_policy_artifact(
        tmp_path,
        classifier=policy.classifier,
        metadata=policy.metadata,
        feature_names=policy.feature_names,
        action_definitions=policy.action_definitions,
    )
    loaded = load_tree_policy_artifact(tmp_path)
    assert loaded.predict_action_ranking((1.0,)) == policy.predict_action_ranking((1.0,))
