"""formal_v1 action evaluation 专项测试。

测试内容：
1. evaluator 不产生状态副作用（计数器不变）；
2. mask/step 一致性验证；
3. mask 拒绝动作不会被 step 执行。
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv, ActionCandidateEvaluation
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario
from amc_py.rl.feature_config import FeatureConfig


def _make_simple_tasks() -> list[Task]:
    return [
        Task(name="h1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task(name="h2", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task(name="l1", period=25, deadline=25, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def test_formal_v1_evaluator_is_pure_no_side_effects():
    """_evaluate_single_action 不应修改任何计数器或引擎状态。"""
    tasks = _make_simple_tasks()
    scenario = make_nominal_scenario()
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_RA),
        check_safety=True,
        action_space="single",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        action_validation_mode="formal_v1",
    )
    env.reset(seed=0)

    # 记录调用前的计数器状态
    safety_checked_before = env._safety_checked_actions
    safety_accepted_before = env._safety_accepted_actions
    safety_rejected_before = env._safety_rejected_actions
    carry_over_applied_before = env._carry_over_envelope_applied_count
    carry_over_changed_before = env._carry_over_changed_candidate_count
    carry_over_selected_before = env._carry_over_selected_action_check_count
    carry_over_changed_selected_before = env._carry_over_changed_selected_candidate_count

    # 连续调用多次 evaluator
    for _ in range(3):
        env._evaluate_single_action(0)

    # 所有计数器应保持不变
    assert env._safety_checked_actions == safety_checked_before
    assert env._safety_accepted_actions == safety_accepted_before
    assert env._safety_rejected_actions == safety_rejected_before
    assert env._carry_over_envelope_applied_count == carry_over_applied_before
    assert env._carry_over_changed_candidate_count == carry_over_changed_before
    assert env._carry_over_selected_action_check_count == carry_over_selected_before
    assert env._carry_over_changed_selected_candidate_count == carry_over_changed_selected_before


def test_formal_v1_cache_hit_works():
    """连续两次 valid_action_mask 应命中缓存且计数器不变。"""
    tasks = _make_simple_tasks()
    scenario = make_nominal_scenario()
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_RA),
        check_safety=True,
        action_space="single",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        action_validation_mode="formal_v1",
    )
    env.reset(seed=0)

    # 第一次 mask
    mask1 = env.valid_action_mask()
    cache_hits_before = env._formal_v1_cache_hit_count
    cache_misses_before = env._formal_v1_cache_miss_count
    eval_count_before = env._formal_v1_evaluation_count

    # 第二次 mask 应命中缓存（状态未变）
    mask2 = env.valid_action_mask()

    assert mask1 == mask2
    assert env._formal_v1_cache_hit_count == cache_hits_before + 1
    # cache miss 不变
    assert env._formal_v1_cache_miss_count == cache_misses_before
    # evaluation count 不变
    assert env._formal_v1_evaluation_count == eval_count_before


def test_formal_v1_mask_rejected_action_not_executed():
    """mask 中标记为 invalid 的动作无法在 step 中被执行。"""
    tasks = _make_simple_tasks()
    scenario = make_nominal_scenario()
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_RA),
        check_safety=True,
        action_space="single",
        budget_increase_ratio=0.50,
        budget_decrease_ratio=0.05,
        action_validation_mode="formal_v1",
        forbid_decreasing_hi_budgets=True,
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()

    # 找到第一个被 mask 标记为 invalid 的非 noop 动作
    invalid_action_id = None
    for aid, valid in enumerate(mask):
        if not valid:
            invalid_action_id = aid
            break

    if invalid_action_id is None:
        pytest.skip("当前设置下所有动作都合法，无法测试 invalid action 拒绝")

    # 执行该被拒绝动作
    result = env.step(invalid_action_id)
    assert not result.info["accepted"], "被 mask 拒绝的动作不应被 step 接受"


def test_formal_v1_legal_action_not_recorded_as_noop():
    """合法 formal action 不应被标记为 noop。"""
    tasks = _make_simple_tasks()
    scenario = make_nominal_scenario()
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_RA),
        check_safety=True,
        action_space="single",
        budget_increase_ratio=0.10,
        budget_decrease_ratio=0.05,
        action_validation_mode="formal_v1",
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()

    # 找到合法动作（noop 是 action_id=0）
    legal_action_id = None
    for aid, valid in enumerate(mask):
        if valid and aid > 0:  # 非 noop
            legal_action_id = aid
            break

    if legal_action_id is None:
        pytest.skip("当前设置下没有非 noop 的合法动作")

    result = env.step(legal_action_id)
    assert not result.info.get("is_noop"), "合法预算动作不应被标记为 noop"
    assert result.info["accepted"], "合法预算动作应被接受"
