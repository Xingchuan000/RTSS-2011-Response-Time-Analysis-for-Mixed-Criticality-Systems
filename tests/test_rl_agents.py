"""阶段 5：基线预算 agent 测试。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.types import AgentObservation


def _obs() -> AgentObservation:
    """构造统一测试观测。"""

    return AgentObservation(0, (0.0, 0.0), {"a": 1}, {"a": 0})


def _tasks() -> list[Task]:
    """构造最小可用任务集。"""

    return [
        Task("a", 10, 10, 2, 3, Criticality.HI),
        Task("b", 12, 12, 2, 2, Criticality.LO),
        Task("c", 15, 15, 2, 2, Criticality.LO),
    ]


def test_noop_agent_always_returns_none() -> None:
    """NoOpBudgetAgent 必须始终返回 None。"""

    agent = NoOpBudgetAgent()
    assert agent.select_action(_obs()) is None
    assert agent.select_action(_obs()) is None


def test_random_agent_returns_action_from_action_space() -> None:
    """RandomBudgetAgent 返回动作必须来自给定动作空间。"""

    actions = build_budget_action_space(_tasks())
    agent = RandomBudgetAgent(actions=actions, seed=1)
    action = agent.select_action(_obs())
    assert action in actions


def test_random_agent_same_seed_produces_same_sequence() -> None:
    """同种子下两个随机 agent 序列应一致。"""

    actions = build_budget_action_space(_tasks())
    a1 = RandomBudgetAgent(actions=actions, seed=7)
    a2 = RandomBudgetAgent(actions=actions, seed=7)
    seq1 = [a1.select_action(_obs()).action_id for _ in range(10)]
    seq2 = [a2.select_action(_obs()).action_id for _ in range(10)]
    assert seq1 == seq2


def test_random_agent_returns_none_for_empty_action_space() -> None:
    """空动作空间下应返回 None。"""

    agent = RandomBudgetAgent(actions=tuple(), seed=0)
    assert agent.select_action(_obs()) is None


def test_heuristic_agent_selects_high_pressure_increase_target() -> None:
    """启发式策略应选择压力最大的任务做 increase。"""

    tasks = _tasks()
    actions = build_budget_action_space(tasks)
    agent = HeuristicBudgetAgent(actions=actions)
    obs = AgentObservation(
        time=0,
        state_vector=(0.0,) * (2 * len(tasks)),
        raw_budgets={"a": 2, "b": 2, "c": 2},
        raw_recent_costs={"a": 5, "b": 1, "c": 1},
    )
    action = agent.select_action(obs)
    assert action is not None
    assert action.increase_task == "a"


def test_heuristic_agent_returns_none_when_action_missing() -> None:
    """动作空间不包含目标动作时应回退到 NoOp。"""

    agent = HeuristicBudgetAgent(actions=tuple())
    assert agent.select_action(_obs()) is None
