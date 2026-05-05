"""阶段 6：Random agent wrapper 测试。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import RandomBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.rl.safety import RuntimeBudgetSafetyChecker
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 3, Criticality.LO),
        Task("l2", 15, 15, 2, 3, Criticality.LO),
    ]


def test_random_wrapper_generates_action_log() -> None:
    """Random agent 下应生成动作日志。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=1),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=50),
    )
    assert len(result.action_log) > 0


def test_rejected_actions_increase_counter_when_safety_rejects() -> None:
    """安全检查拒绝时 rejected_actions 应增加。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks)
    strict_checker = RuntimeBudgetSafetyChecker(
        ordered_tasks=tasks,
        design_r_lo={"h": 1},
        check_lo_tasks=True,
    )
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=0),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=50, check_safety=True),
        safety_checker=strict_checker,
    )
    assert result.rejected_actions >= 1


def test_accepted_actions_create_budget_update_events() -> None:
    """动作被接受时应产生 budget update events。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=2),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=50, check_safety=False),
    )
    assert result.accepted_actions >= 1
    assert len(result.runtime_result.budget_update_events) >= 1


def test_fixed_seed_makes_action_log_reproducible() -> None:
    """固定种子后 action log 应可复现。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks)
    cfg = RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS)
    a = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=9),
        runtime_config=cfg,
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=50, check_safety=False),
    )
    b = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=9),
        runtime_config=cfg,
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=50, check_safety=False),
    )
    assert a.action_log == b.action_log


def test_runtime_wrapper_forbid_decreasing_hi_budgets_rejects_hi_decrease() -> None:
    """wrapper 启用 HI 预算保护后，应出现 decrease_hi_forbidden 拒绝。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks, action_space="pair")
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=RandomBudgetAgent(actions=actions, seed=0),
        runtime_config=RuntimeConfig(end_time=60, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(
            agent_period=10,
            end_time=60,
            check_safety=False,
            forbid_decreasing_hi_budgets=True,
        ),
    )

    assert any(row.get("reject_reason") == "decrease_hi_forbidden" for row in result.action_log)


def test_runtime_wrapper_budget_floor_rejects_decrease_below_initial_floor() -> None:
    """wrapper 也应按与 DQN env 相同的 budget floor 语义拒绝坏动作。"""

    class _SingleActionAgent:
        """始终返回同一个动作，便于稳定命中 floor 违规路径。"""

        def __init__(self, action) -> None:
            self._action = action

        def select_action(self, observation):  # noqa: ANN001
            return self._action

    tasks = [
        Task("h", 20, 20, 10, 12, Criticality.HI),
        Task("l1", 20, 20, 10, 10, Criticality.LO),
    ]
    actions = build_budget_action_space(tasks, action_space="pair")
    # 选择 decrease l1 的动作。初始预算 10，在 floor=1.0 下，任何 decrease 都应被拒绝。
    selected_action = next(action for action in actions if "l1" in action.decrease_tasks)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=_SingleActionAgent(selected_action),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(
            agent_period=10,
            end_time=20,
            check_safety=False,
            budget_floor_ratio=1.0,
        ),
    )

    assert any(str(row.get("reject_reason", "")).startswith("budget_floor_violation") for row in result.action_log)
