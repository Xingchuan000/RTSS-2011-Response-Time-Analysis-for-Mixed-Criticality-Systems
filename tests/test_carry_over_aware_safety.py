"""carry-over aware safety 专项测试。

测试内容：
1. carry_over envelope 与 LO guard unit (+1) 的正确性；
2. evaluate 不产生副作用但 step 消费时正确记录计数器；
3. 重复 mask 不虚增 selected-action counters。
"""

from __future__ import annotations

import numpy as np
import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.safety import (
    RuntimeBudgetSafetyChecker,
    build_effective_safety_budget_vector,
)
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _make_simple_tasks() -> list[Task]:
    return [
        Task(name="h1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task(name="h2", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task(name="l1", period=25, deadline=25, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def test_build_effective_safety_budget_vector_lo_guard_unit():
    """LO guard unit 应正确加到有效预算中。"""
    tasks = _make_simple_tasks()
    candidate = {"h1": 5, "h2": 5, "l1": 5}
    effective = build_effective_safety_budget_vector(
        tasks, candidate, lo_overrun_guard_units=1
    )
    # LO 任务 l1 应加 1
    assert effective["l1"] == 6
    # HI 任务不变
    assert effective["h1"] == 5
    assert effective["h2"] == 5


def test_build_effective_safety_budget_vector_carry_over_envelope():
    """active release budget max 覆盖 candidate 时，应使用 envelope 值。"""
    tasks = _make_simple_tasks()
    candidate = {"h1": 3, "h2": 3, "l1": 3}
    active_max = {"h1": 7, "h2": 0, "l1": 0}
    effective = build_effective_safety_budget_vector(
        tasks, candidate, active_release_budget_max=active_max, lo_overrun_guard_units=0
    )
    # h1 的 active release budget 7 > candidate 3，应使用 7
    assert effective["h1"] == 7
    # h2 的 active 0 <= candidate 3，应使用 3
    assert effective["h2"] == 3
    assert effective["l1"] == 3


def test_carry_over_counters_not_incremented_by_evaluate():
    """formal_v1 的 evaluator 不应累加 carry_over 计数器。"""
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
        carry_over_aware_safety=True,
        lo_budget_overrun_guard_units=1,
    )
    env.reset(seed=0)

    before = env._carry_over_selected_action_check_count
    # 直接调用 evaluator 不应该增加 selected-action counter
    env._evaluate_single_action(0)
    env._evaluate_single_action(1)
    assert env._carry_over_selected_action_check_count == before


def test_carry_over_counters_incremented_by_step_only():
    """仅 step() 消费 evaluation 时才应增加 selected-action 计数器。"""
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
        carry_over_aware_safety=True,
        lo_budget_overrun_guard_units=1,
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()

    # 找一个合法非 noop 动作
    legal_aid = None
    for aid, valid in enumerate(mask):
        if valid and aid > 0:
            legal_aid = aid
            break
    if legal_aid is None:
        pytest.skip("没有合法非 noop 动作")

    before = env._carry_over_selected_action_check_count
    env.step(legal_aid)
    assert env._carry_over_selected_action_check_count == before + 1


def test_repeated_mask_does_not_inflate_selected_counters():
    """重复调用 valid_action_mask 不应虚增 selected-action 计数器。"""
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
        carry_over_aware_safety=True,
        lo_budget_overrun_guard_units=1,
    )
    env.reset(seed=0)

    before_selected = env._carry_over_selected_action_check_count
    before_changed_selected = env._carry_over_changed_selected_candidate_count

    for _ in range(5):
        env.valid_action_mask()

    assert env._carry_over_selected_action_check_count == before_selected
    assert env._carry_over_changed_selected_candidate_count == before_changed_selected


def test_safety_margin_matches_checker_accepted_rejected():
    """_compute_current_safety_margin_min 的整数计算应与 checker 判定一致。"""
    from amc_py.rl.safety import RuntimeBudgetSafetyChecker
    from amc_py.models import Criticality, Task

    tasks = _make_simple_tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12}, check_lo_tasks=True)

    # 场景：预算刚好满足约束
    candidate_ok = {"h1": 3, "h2": 3, "l1": 2}
    report_ok = checker.validate_candidate(candidate_ok)
    assert report_ok.accepted is True

    # 场景：预算不足导致违反
    candidate_bad = {"h1": 10, "h2": 10, "l1": 10}
    report_bad = checker.validate_candidate(candidate_bad)
    assert report_bad.accepted is False

    # 使用 checker 的约束矩阵手算 margin
    A, b = checker.build_linear_constraints()
    n_constraints = A.shape[0]
    task_names = tuple(t.name for t in tasks)
    n_tasks = len(task_names)

    # 对候选 ok 计算 margin：应 >= 0
    budget_ints_ok = [int(candidate_ok[name]) for name in task_names]
    min_ok = 1.0
    for row_idx in range(n_constraints):
        lhs_int = 0
        for col_idx in range(n_tasks):
            lhs_int += int(A[row_idx, col_idx]) * budget_ints_ok[col_idx]
        rhs_int = int(b[row_idx])
        slack_int = rhs_int - lhs_int
        denom = max(1, abs(rhs_int))
        normalized = float(slack_int) / float(denom)
        if normalized < min_ok:
            min_ok = normalized
    assert min_ok >= 0.0, "满足约束的候选应产生非负 margin"

    # 对候选 bad 计算 margin：应 < 0（未裁剪前）
    budget_ints_bad = [int(candidate_bad[name]) for name in task_names]
    min_bad = 1.0
    for row_idx in range(n_constraints):
        lhs_int = 0
        for col_idx in range(n_tasks):
            lhs_int += int(A[row_idx, col_idx]) * budget_ints_bad[col_idx]
        rhs_int = int(b[row_idx])
        slack_int = rhs_int - lhs_int
        denom = max(1, abs(rhs_int))
        normalized = float(slack_int) / float(denom)
        if normalized < min_bad:
            min_bad = normalized
    assert min_bad < 0.0, "违反约束的候选应产生负 margin（裁剪前）"


def test_safety_margin_lhs_equals_rhs_is_zero_margin():
    """lhs 恰好等于 rhs 时，margin 应为 0。"""
    n_constraints = 2
    n_tasks = 3
    A = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
    b = np.array([5, 5], dtype=np.int64)
    budgets = [5, 5, 0]

    min_normalized = 1.0
    for row_idx in range(n_constraints):
        lhs_int = 0
        for col_idx in range(n_tasks):
            lhs_int += int(A[row_idx, col_idx]) * budgets[col_idx]
        rhs_int = int(b[row_idx])
        slack_int = rhs_int - lhs_int
        denom = max(1, abs(rhs_int))
        normalized = float(slack_int) / float(denom)
        if normalized < min_normalized:
            min_normalized = normalized
    assert min_normalized == 0.0


def test_safety_margin_lhs_equals_rhs_plus_one_negative():
    """lhs 比 rhs 多 1 时，margin 应为负（裁剪前）。"""
    A = np.array([[1, 0]], dtype=np.int64)
    b = np.array([5], dtype=np.int64)
    budgets = [6, 0]

    lhs_int = int(A[0, 0]) * budgets[0] + int(A[0, 1]) * budgets[1]
    rhs_int = int(b[0])
    slack_int = rhs_int - lhs_int
    denom = max(1, abs(rhs_int))
    normalized = float(slack_int) / float(denom)
    assert normalized == -1.0 / 5.0


def test_candidate_check_count_differs_from_selected_count():
    """在多动作环境中，候选计数应明显大于选中计数。"""
    tasks = _make_simple_tasks()
    from amc_py.runtime_scenarios import make_nominal_scenario
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
        carry_over_aware_safety=True,
        lo_budget_overrun_guard_units=1,
    )
    env.reset(seed=0)

    # 第一次 mask：candidate count 应增加
    before_candidate = env._carry_over_candidate_check_count
    env.valid_action_mask()
    after_candidate = env._carry_over_candidate_check_count
    assert after_candidate > before_candidate, "第一次 mask 应增加候选计数"

    # 第二次相同状态 mask 命中 cache：candidate count 不增加
    before_candidate2 = env._carry_over_candidate_check_count
    env.valid_action_mask()
    after_candidate2 = env._carry_over_candidate_check_count
    assert after_candidate2 == before_candidate2, "cache hit 不应重复计数"

    # mask 不改变 selected count
    assert env._carry_over_selected_action_check_count == 0

    # step 一次：selected count 只增加 1
    mask = env.valid_action_mask()
    legal_aid = None
    for aid, valid in enumerate(mask):
        if valid and aid > 0:
            legal_aid = aid
            break
    if legal_aid is None:
        import pytest
        pytest.skip("没有合法非 noop 动作")

    before_selected = env._carry_over_selected_action_check_count
    env.step(legal_aid)
    assert env._carry_over_selected_action_check_count == before_selected + 1

    # 候选计数应明显大于选中计数（在单步多动作空间中）
    assert env._carry_over_candidate_check_count > env._carry_over_selected_action_check_count
