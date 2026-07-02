"""VIPER trainer 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from amc_py.viper.dataset import ViperSample
from amc_py.viper.training import train_cart_tree


def _sample(action_id: int) -> ViperSample:
    return ViperSample(
        teacher_id="t0",
        taskset_seed=None,
        scenario_seed=0,
        scenario_split="train",
        horizon=10,
        decision_index=0,
        time=0,
        state_vector=(float(action_id),),
        valid_action_mask=(True, True, True, True, True, True),
        teacher_action_id=action_id,
        teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        q_best=1.0,
        q_second_best=0.0,
        q_worst=0.0,
        q_margin_second=1.0,
        viper_weight=1.0,
        behavior_policy="oracle",
        behavior_action_id=action_id,
        tree_iteration=None,
        raw_budgets_json="{}",
        raw_recent_costs_json="{}",
        mask_reject_reasons_json="{}",
    )


def test_train_cart_tree_keeps_full_action_dim_metadata() -> None:
    classifier, metadata = train_cart_tree(
        [_sample(0), _sample(0)],
        method="bc",
        max_depth=2,
        min_samples_leaf=1,
        criterion="gini",
        weight_mode="uniform",
        resample_size=None,
        random_seed=0,
        state_dim=1,
        action_dim=6,
    )
    assert classifier is not None
    assert metadata["action_dim"] == 6
