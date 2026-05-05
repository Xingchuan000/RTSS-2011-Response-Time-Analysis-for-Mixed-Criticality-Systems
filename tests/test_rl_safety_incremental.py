"""增量 safety 检查与完整检查一致性测试。"""

from __future__ import annotations

import math
import random

import numpy as np

from amc_py.amc import build_design_r_lo_map
from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate, build_budget_action_space
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, merge_budget_candidate


def _tasks() -> list[Task]:
    return [
        Task("h1", 10, 10, 2, 3, Criticality.HI),
        Task("h2", 15, 15, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 2, Criticality.LO),
        Task("l2", 20, 20, 3, 3, Criticality.LO),
        Task("l3", 25, 25, 2, 2, Criticality.LO),
    ]


def test_incremental_mask_decision_matches_full_checker_on_random_actions() -> None:
    """随机抽样动作时，增量约束判定应与完整 checker 一致。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(
        ordered_tasks=tasks,
        design_r_lo=build_design_r_lo_map(tasks),
        check_lo_tasks=True,
    )
    budget_state = BudgetState.from_tasks(tasks)
    actions = list(build_budget_action_space(tasks, action_space="triple", include_explicit_noop=True))
    rng = random.Random(0)

    matrix_a, bounds = checker.build_linear_constraints()
    task_names = [task.name for task in tasks]
    task_upper_bounds = [
        (task.c_hi if task.criticality is Criticality.HI and task.c_hi > 0 else task.deadline)
        if task.criticality is Criticality.HI
        else task.deadline
        for task in tasks
    ]
    budget_before = dict(budget_state.budgets)
    budget_array = np.array([float(budget_before[name]) for name in task_names], dtype=np.float64)
    slack = bounds - (matrix_a @ budget_array)

    sample_actions = rng.sample(actions, k=min(1000, len(actions)))
    for action in sample_actions:
        if action.is_noop:
            assert True
            continue
        updates = apply_budget_action_candidate(
            action=action,
            budget_state=budget_state,
            ordered_tasks=tasks,
        )
        merged = merge_budget_candidate(budget_state, updates)
        full_accept = checker.validate_candidate(merged).accepted

        delta_pairs: list[tuple[int, float]] = []
        if action.increase_idx is not None:
            inc_idx = action.increase_idx
            inc_name = task_names[inc_idx]
            old_inc = budget_before[inc_name]
            inc_value = math.ceil(old_inc * (1.0 + action.increase_ratio))
            inc_value = min(inc_value, task_upper_bounds[inc_idx])
            inc_value = max(1, inc_value)
            if inc_value == old_inc:
                incremental_accept = False
                assert incremental_accept == full_accept
                continue
            delta_pairs.append((inc_idx, float(inc_value - old_inc)))
        rejected_for_effective = False
        for dec_idx in action.decrease_indices:
            dec_name = task_names[dec_idx]
            old_dec = budget_before[dec_name]
            dec_value = math.floor(old_dec * (1.0 - action.decrease_ratio))
            dec_value = max(1, dec_value)
            if dec_value == old_dec:
                rejected_for_effective = True
                break
            delta_pairs.append((dec_idx, float(dec_value - old_dec)))
        if rejected_for_effective:
            incremental_accept = False
            assert incremental_accept == full_accept
            continue

        delta_lhs = np.zeros_like(slack)
        for idx, delta in delta_pairs:
            delta_lhs += matrix_a[:, idx] * delta
        incremental_accept = bool(np.all(delta_lhs <= slack))
        assert incremental_accept == full_accept
