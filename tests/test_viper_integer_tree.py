import hashlib
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.integer_tree import (
    compile_sklearn_tree_to_integer,
    evaluate_integer_tree,
    integer_tree_hash,
    load_integer_tree_json,
    save_integer_tree_json,
)
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy


def _model():
    x = np.asarray([[0], [100], [200], [300]], dtype=np.int32)
    y = np.asarray([0, 1, 1, 2], dtype=np.int64)
    classifier = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, y)
    return classifier, compile_sklearn_tree_to_integer(
        classifier,
        state_dim=1,
        action_dim=4,
        feature_names=("x",),
        fixed_point_config=FixedPointConfig(scale=1000, output_max=1000),
    )


def test_integer_tree_preserves_leaf_and_complete_ranking(tmp_path: Path) -> None:
    classifier, model = _model()
    for state in ((0,), (99,), (100,), (299,), (300,)):
        integer_eval = evaluate_integer_tree(model, state)
        sklearn_leaf = int(classifier.apply(np.asarray([state], dtype=np.int32))[0])
        assert integer_eval.leaf_id == sklearn_leaf
        assert tuple(sorted(integer_eval.action_ranking)) == (0, 1, 2, 3)
    before = integer_tree_hash(model)
    file_hash = save_integer_tree_json(model, tmp_path / "integer_tree.json")
    assert file_hash == hashlib.sha256((tmp_path / "integer_tree.json").read_bytes()).hexdigest()
    restored = load_integer_tree_json(tmp_path / "integer_tree.json")
    assert integer_tree_hash(restored) == before


def test_integer_tree_rejects_dimension_and_bad_state() -> None:
    _, model = _model()
    with pytest.raises(ValueError):
        evaluate_integer_tree(model, (1, 2))
    with pytest.raises(ValueError):
        evaluate_integer_tree(model, (True,))


def test_integer_tree_policy_all_invalid_defensively_returns_explicit_noop() -> None:
    _, model = _model()
    config = FixedPointConfig(scale=1000, output_max=1000)
    policy = IntegerTreeBudgetPolicy(
        model=model,
        metadata={"state_dim": 1, "action_dim": 4},
        feature_names=("x",),
        action_definitions=[
            {"action_id": 0}, {"action_id": 1}, {"action_id": 2},
            {"action_id": 3, "is_noop": True},
        ],
        fixed_point_config=config,
    )
    action_id, info = policy.select_action_id((0.1,), (False, False, False, False))
    assert action_id == 3
    assert info["tree_no_valid_action"] is True
    assert info["tree_fallback_used"] is True
    assert info["tree_selected_rank"] is None
    assert info["tree_defensive_fallback_to_explicit_noop"] is True
