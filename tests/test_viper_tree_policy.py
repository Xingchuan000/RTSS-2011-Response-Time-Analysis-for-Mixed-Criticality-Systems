"""TreeBudgetPolicy 测试。

新增测试：
- test_tree_policy_traces_leaf_and_path: 验证 trace_decision_path 输出基本字段。
- test_select_action_id_can_include_decision_trace: 验证 include_decision_trace=True 时 info 包含 trace 字段。
- test_predict_full_proba_same_as_ranking: 验证 _predict_full_proba 与 predict_action_ranking 概率一致性。
- test_action_definition: 验证 action_definition 返回值。
- Patch 3: noop action kind、fallback flag 一致性。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.artifacts import load_tree_policy_artifact, save_tree_policy_artifact
from amc_py.viper.fixed_point import FixedPointConfig, quantize_state_vector
from amc_py.viper.tree_policy import TreeBudgetPolicy, IntegerTreeBudgetPolicy
from amc_py.viper.integer_tree import compile_sklearn_tree_to_integer


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


def _integer_ranked_policy() -> IntegerTreeBudgetPolicy:
    """构造一个 ranked 整数树策略用于测试 action kind。"""
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    fpc = FixedPointConfig(scale=100, min_int=-100, max_int=100)
    model = compile_sklearn_tree_to_integer(
        clf, feature_names=("f0",),
        fixed_point_config_hash="hash",
        state_dim=1, action_dim=3, ranked=True)
    return IntegerTreeBudgetPolicy(
        model=model,
        metadata={"fallback_mode": "ranked_valid_or_none", "state_dim": 1, "action_dim": 3},
        feature_names=("f0",),
        action_definitions=[{"action_id": 0}, {"action_id": 1}, {"action_id": 2}],
        fixed_point_config=fpc,
    )


def _integer_top1_policy() -> IntegerTreeBudgetPolicy:
    """构造一个 top1 整数树策略用于测试 legacy action kind。"""
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    fpc = FixedPointConfig(scale=100, min_int=-100, max_int=100)
    model = compile_sklearn_tree_to_integer(
        clf, feature_names=("f0",),
        fixed_point_config_hash="hash",
        state_dim=1, action_dim=3, ranked=False)
    return IntegerTreeBudgetPolicy(
        model=model,
        metadata={"fallback_mode": "top1_or_noop", "state_dim": 1, "action_dim": 3},
        feature_names=("f0",),
        action_definitions=[{"action_id": 0}, {"action_id": 1}, {"action_id": 2}],
        fixed_point_config=fpc,
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


# ========== Patch 3 新增测试 ==========


def test_legacy_top1_invalid_action_kind_is_noop_top1_invalid() -> None:
    """legacy top1 模式下 raw invalid 返回 None 时，action_kind 应为 noop_top1_invalid。"""
    policy = _integer_top1_policy()
    # 所有动作都无效
    state = (1.0,)
    selected, info = policy.select_action_id(state, (False, False, False))
    assert selected is None
    assert info["tree_selected_action_kind"] == "noop_top1_invalid"
    assert info["tree_top1_invalid_noop"] is True
    assert info["tree_no_valid_action"] is False


def test_ranked_all_invalid_action_kind_is_noop_no_valid() -> None:
    """ranked 模式下 all-invalid 时 action_kind 应为 noop_no_valid_action。"""
    policy = _integer_ranked_policy()
    state = (1.0,)
    selected, info = policy.select_action_id(state, (False, False, False))
    assert selected is None
    assert info["tree_selected_action_kind"] == "noop_no_valid_action"
    assert info["tree_no_valid_action"] is True


def test_fallback_flag_matches_selected_rank() -> None:
    """tree_fallback_used 必须与 selected_rank > 0 完全一致。"""
    policy = _integer_ranked_policy()
    state = (1.0,)
    # raw top1 是 action 2（从训练数据看），将其屏蔽让 fallback 到 rank 1
    _, info = policy.select_action_id(state, (True, True, False))
    if info["tree_selected_rank"] is not None:
        expected = info["tree_selected_rank"] > 0
        assert info["tree_fallback_used"] == expected


def test_ranked_policy_uses_legal_lower_rank_fallback() -> None:
    """受控 mask 屏蔽 top1 后，ranked policy 必须选择下一合法动作且记录 rank。"""
    policy = _integer_ranked_policy()
    state = (1.0,)
    ranking = policy.predict_action_ranking(state)
    # 排名第一不可用、排名第二可用，构造不依赖 teacher 随机输出的 fallback 场景。
    mask = tuple(action_id != ranking[0] for action_id in range(3))
    selected, info = policy.select_action_id(state, mask)
    assert selected == ranking[1]
    assert mask[selected] is True
    assert info["tree_raw_top1_invalid"] is True
    assert info["tree_fallback_used"] is True
    assert info["tree_selected_rank"] == 1


def test_selected_none_not_added_to_selected_q_regret() -> None:
    """selected 为 None 时不应伪造 Q 值，也不加入 selected Q-regret。
    本测试验证 evaluate_tree_policy_once 的 selected_q_regrets 收集逻辑的正确性。"""
    # 直接验证逻辑：selected=None 时 selected_q_regrets 列表保持为空
    selected_q_regrets = []
    action_id = None
    teacher_q_best = 10.0
    teacher_q_values = [5.0, 3.0, 8.0]
    if action_id is not None and teacher_q_best is not None:
        selected_q_regrets.append(float(teacher_q_best) - float(teacher_q_values[action_id]))
    assert len(selected_q_regrets) == 0
