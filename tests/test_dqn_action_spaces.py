"""DQN 动作空间配置测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate, build_budget_action_space
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks(n: int = 20) -> list[Task]:
    """构造测试任务集。"""

    tasks: list[Task] = []
    for idx in range(n):
        if idx == 0:
            tasks.append(Task(f"t{idx}", 20, 20, 2, 3, Criticality.HI))
        else:
            tasks.append(Task(f"t{idx}", 20 + idx, 20 + idx, 2, 2, Criticality.LO))
    return tasks


def test_action_count_for_n20_triple_and_pair() -> None:
    """n=20 时 triple/pair 动作数应符合公式。"""

    tasks = _tasks(20)
    triple_actions = build_budget_action_space(tasks, action_space="triple")
    pair_actions = build_budget_action_space(tasks, action_space="pair")
    assert len(triple_actions) == 3420
    assert len(pair_actions) == 380


def test_include_explicit_noop_exists_and_keeps_budget_unchanged() -> None:
    """显式 noop 动作应存在且不修改预算。"""

    tasks = _tasks(6)
    actions = build_budget_action_space(tasks, action_space="pair", include_explicit_noop=True)
    noop = actions[-1]
    assert noop.is_noop is True
    budget = BudgetState.from_tasks(tasks)
    updates = apply_budget_action_candidate(action=noop, budget_state=budget, ordered_tasks=tasks)
    assert updates == {}


def test_pair_action_updates_exactly_two_tasks() -> None:
    """pair 动作应只包含一个增预算任务和一个降预算任务。"""

    tasks = _tasks(6)
    budget = BudgetState.from_tasks(tasks)
    action = build_budget_action_space(tasks, action_space="pair")[0]
    updates = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
    assert len(updates) == 2


def test_safety_mask_supports_triple_and_pair_action_space() -> None:
    """安全 mask 对 triple/pair 两种动作空间都应可用。"""

    tasks = _tasks(6)
    triple_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="triple",
    )
    pair_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="pair",
    )
    triple_env.reset(seed=0)
    pair_env.reset(seed=0)
    assert len(triple_env.valid_action_mask()) == triple_env.action_space_size
    assert len(pair_env.valid_action_mask()) == pair_env.action_space_size
