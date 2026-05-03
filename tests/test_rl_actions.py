"""离散预算动作空间测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate, build_budget_action_space


def _task(name: str, c_lo: int, c_hi: int, crit: Criticality) -> Task:
    """便捷构造任务。"""

    return Task(name=name, period=20, deadline=20, c_lo=c_lo, c_hi=c_hi, criticality=crit)


def test_action_count_for_n3_and_n4() -> None:
    """动作数应满足 n(n-1)(n-2)/2。"""

    tasks3 = [_task("a", 2, 4, Criticality.HI), _task("b", 2, 2, Criticality.LO), _task("c", 2, 2, Criticality.LO)]
    tasks4 = tasks3 + [_task("d", 2, 2, Criticality.LO)]
    assert len(build_budget_action_space(tasks3)) == 3
    assert len(build_budget_action_space(tasks4)) == 12


def test_each_action_modifies_three_distinct_tasks() -> None:
    """每个动作必须恰好涉及三个不同任务。"""

    tasks = [_task("a", 2, 4, Criticality.HI), _task("b", 2, 2, Criticality.LO), _task("c", 2, 2, Criticality.LO), _task("d", 2, 2, Criticality.LO)]
    for action in build_budget_action_space(tasks):
        touched = {action.increase_task, *action.decrease_tasks}
        assert len(touched) == 3


def test_action_order_is_deterministic() -> None:
    """多次构建动作空间时顺序必须一致。"""

    tasks = [_task("a", 2, 4, Criticality.HI), _task("b", 2, 2, Criticality.LO), _task("c", 2, 2, Criticality.LO), _task("d", 2, 2, Criticality.LO)]
    first = build_budget_action_space(tasks)
    second = build_budget_action_space(tasks)
    assert first == second


def test_candidate_uses_ceil_for_increase_and_floor_for_decrease() -> None:
    """增加预算用 ceil，减少预算用 floor。"""

    tasks = [_task("a", 3, 10, Criticality.HI), _task("b", 3, 3, Criticality.LO), _task("c", 3, 3, Criticality.LO)]
    budget = BudgetState.from_tasks(tasks)
    action = build_budget_action_space(tasks)[0]
    candidate = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
    assert candidate[action.increase_task] == 4
    for dec_name in action.decrease_tasks:
        assert candidate[dec_name] == 2


def test_candidate_does_not_mutate_original_budget_state() -> None:
    """候选更新不得原地修改 BudgetState。"""

    tasks = [_task("a", 2, 4, Criticality.HI), _task("b", 2, 2, Criticality.LO), _task("c", 2, 2, Criticality.LO)]
    budget = BudgetState.from_tasks(tasks)
    before = dict(budget.budgets)
    action = build_budget_action_space(tasks)[0]
    _ = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
    assert budget.budgets == before


def test_candidate_budget_never_below_one_and_hi_never_exceeds_chi() -> None:
    """预算下限为 1，HI 增加后不超过 c_hi。"""

    tasks = [_task("hi", 9, 10, Criticality.HI), _task("b", 1, 1, Criticality.LO), _task("c", 1, 1, Criticality.LO)]
    budget = BudgetState.from_tasks(tasks)
    action = build_budget_action_space(tasks)[0]
    candidate = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
    assert candidate["hi"] <= 10
    for value in candidate.values():
        assert value >= 1
