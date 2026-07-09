"""TreeBudgetPolicy 测试。

新增测试：
- test_tree_policy_traces_leaf_and_path: 验证 trace_decision_path 输出基本字段。
- test_select_action_id_can_include_decision_trace: 验证 include_decision_trace=True 时 info 包含 trace 字段。
- test_predict_full_proba_same_as_ranking: 验证 _predict_full_proba 与 predict_action_ranking 概率一致性。
- test_action_definition: 验证 action_definition 返回值。
"""

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


def test_tree_policy_traces_leaf_and_path() -> None:
    """trace_decision_path 应返回完整的 leaf/path/probability 诊断信息。"""

    policy = _policy()
    trace = policy.trace_decision_path((2.0,))
    assert isinstance(trace["tree_leaf_id"], int)
    assert trace["tree_path_depth"] >= 1
    assert trace["tree_path_predicates"]
    assert trace["tree_leaf_n_node_samples"] >= 1
    assert trace["tree_leaf_predicted_class_id"] in {0, 1, 2}
    assert len(trace["tree_action_proba"]) == 3
    assert len(trace["tree_action_ranking"]) == 3
    assert isinstance(trace["tree_leaf_value"], list)


def test_select_action_id_can_include_decision_trace() -> None:
    """include_decision_trace=True 时，select_action_id 返回的 info 应包含 trace 字段。"""

    policy = _policy()
    action_id, info = policy.select_action_id(
        (2.0,), (True, True, True), include_decision_trace=True
    )
    assert "tree_leaf_id" in info
    assert "tree_path_predicates" in info
    assert "tree_action_proba" in info
    assert "tree_action_ranking" in info
    assert "tree_leaf_impurity" in info


def test_predict_full_proba_same_as_ranking() -> None:
    """_predict_full_proba 生成的概率应与 predict_action_ranking 的排序一致。"""

    policy = _policy()
    state = (1.0,)
    proba = policy._predict_full_proba(state)
    ranking = policy.predict_action_ranking(state)
    for i in range(len(ranking) - 1):
        assert proba[ranking[i]] >= proba[ranking[i + 1]]


def test_action_definition() -> None:
    """action_definition 应返回正确的动作描述。"""

    policy = _policy()
    assert policy.action_definition(None) is None
    assert policy.action_definition(-1) is None
    assert policy.action_definition(100) is None
    defn = policy.action_definition(0)
    assert defn is not None
    assert defn["action_id"] == 0
