"""阶段 6：NoOp agent wrapper 测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import BudgetAgent, NoOpBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.rl.types import AgentObservation
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_single_lo_overrun_scenario


def _tasks() -> list[Task]:
    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 12, 12, 2, 3, Criticality.LO),
        Task("l2", 15, 15, 2, 3, Criticality.LO),
    ]


def test_noop_agent_has_zero_accepted_actions() -> None:
    """NoOp agent 下 accepted_actions 必须为 0。"""

    tasks = _tasks()
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_single_lo_overrun_scenario("l", actual_cost=4),
        agent=NoOpBudgetAgent(),
        runtime_config=RuntimeConfig(end_time=80, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=80),
    )
    assert result.accepted_actions == 0


def test_noop_matches_amc_plus_baseline_key_metrics() -> None:
    """NoOp 结果应与 AMC+ baseline 的关键统计一致。"""

    tasks = _tasks()
    scenario = make_single_lo_overrun_scenario("l", actual_cost=4)
    cfg = RuntimeConfig(end_time=80, semantics=RuntimeSemantics.AMC_PLUS)

    baseline = simulate_ordered_taskset_event_driven(tasks, scenario, cfg)
    wrapped = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=scenario,
        agent=NoOpBudgetAgent(),
        runtime_config=cfg,
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=80),
    ).runtime_result

    assert baseline.mode_change_count() == wrapped.mode_change_count()
    assert baseline.lo_job_cancellation_count() == wrapped.lo_job_cancellation_count()
    assert len(baseline.deadline_misses) == len(wrapped.deadline_misses)


class _ExplicitNoOpAgent(BudgetAgent):
    """测试用 agent：每一步都显式返回动作空间中的 noop 动作。"""

    def __init__(self, tasks: list[Task]):
        actions = build_budget_action_space(tasks, action_space="pair", include_explicit_noop=True)
        self._noop_action = actions[-1]

    def select_action(self, observation: AgentObservation):  # noqa: D401, ANN201, ARG002
        return self._noop_action


def test_explicit_noop_agent_is_logged_as_noop_and_keeps_budget_unchanged() -> None:
    """显式 noop 在 runtime wrapper 中应走 noop 分支且预算保持不变。"""

    tasks = _tasks()
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_single_lo_overrun_scenario("l", actual_cost=4),
        agent=_ExplicitNoOpAgent(tasks),
        runtime_config=RuntimeConfig(end_time=40, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=40),
    )

    assert result.noop_actions > 0
    assert result.accepted_actions == 0
    assert result.rejected_actions == 0
    for row in result.action_log:
        assert row["noop"] is True
        assert row["is_explicit_noop"] is True
        assert row["updates"] == {}
        assert row["budget_before"] == row["budget_after"]
        assert row["safety_checked"] is False
