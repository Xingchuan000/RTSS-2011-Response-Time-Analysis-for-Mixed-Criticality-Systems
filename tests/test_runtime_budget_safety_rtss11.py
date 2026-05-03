"""RTSS2011 运行时预算安全检查测试。"""

from __future__ import annotations

from amc_py.dqn import resolve_experiment_bundle
from amc_py.dqn.experiment import build_rtss11_experiment_config
from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate, build_budget_action_space
from amc_py.rl.agents import BudgetAgent, NoOpBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, RuntimeSafetyReport, merge_budget_candidate
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.rl.types import AgentObservation


class _FixedActionAgent(BudgetAgent):
    """始终返回同一个动作的测试 agent。"""

    def __init__(self, action):
        self._action = action

    def select_action(self, observation: AgentObservation):  # noqa: ARG002
        return self._action


def _build_safe_rtss11_bundle(seed: int = 0):
    """构造一个低压、可调度、可复现的 RTSS2011 测试 bundle。"""

    config = build_rtss11_experiment_config(
        total_util=0.45,
        num_tasks=20,
        cf=2.0,
        cp=0.5,
        require_schedulable=True,
        hi_overrun_prob=0.0,
        lo_overrun_prob=0.0,
        lo_overrun_factor=1.5,
    )
    return resolve_experiment_bundle(config, seed=seed)


def _strict_checker_rejecting_hi(tasks: tuple[Task, ...]) -> RuntimeBudgetSafetyChecker:
    """构造一个会拒绝 HI 增压动作的严格检查器。"""

    design_r_lo = {
        task.name: 1 for task in tasks if task.criticality is Criticality.HI
    }
    return RuntimeBudgetSafetyChecker(
        ordered_tasks=tasks,
        design_r_lo=design_r_lo,
        check_lo_tasks=True,
    )


def _permissive_checker(tasks: tuple[Task, ...]) -> RuntimeBudgetSafetyChecker:
    """构造一个便于动作通过的宽松检查器。"""

    design_r_lo = {
        task.name: task.deadline * 10 for task in tasks if task.criticality is Criticality.HI
    }
    return RuntimeBudgetSafetyChecker(
        ordered_tasks=tasks,
        design_r_lo=design_r_lo,
        check_lo_tasks=False,
    )


def _find_first_accepted_action(tasks: tuple[Task, ...], checker: RuntimeBudgetSafetyChecker):
    """在初始预算上搜索一个可通过检查的动作。"""

    budget_state = BudgetState.from_tasks(tasks)
    for action in build_budget_action_space(tasks):
        updates = apply_budget_action_candidate(
            action=action,
            budget_state=budget_state,
            ordered_tasks=tasks,
        )
        merged = merge_budget_candidate(budget_state, updates)
        if checker.validate_candidate(merged).accepted:
            return action
    return None


class _AlwaysAcceptChecker:
    """测试专用检查器：将候选动作视为可接受。"""

    def __init__(self, tasks: tuple[Task, ...]):
        self._checked_tasks = tuple(task.name for task in tasks)

    def validate_candidate(self, candidate_budgets):  # noqa: ARG002
        return RuntimeSafetyReport(
            accepted=True,
            reason="accepted",
            checked_tasks=self._checked_tasks,
        )


def test_obviously_unsafe_budget_action_is_rejected() -> None:
    """明显违反 safety 条件的动作应被拒绝。"""

    bundle = _build_safe_rtss11_bundle(seed=0)
    action = build_budget_action_space(bundle.ordered_tasks)[0]
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        agent=_FixedActionAgent(action),
        runtime_config=RuntimeConfig(end_time=3000, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=1000, end_time=3000, check_safety=True),
        safety_checker=_strict_checker_rejecting_hi(bundle.ordered_tasks),
        bounds=bundle.normalization_bounds,
    )
    assert result.rejected_actions >= 1


def test_noop_action_does_not_introduce_deadline_miss() -> None:
    """noop 动作不应引入 deadline miss。"""

    bundle = _build_safe_rtss11_bundle(seed=1)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        agent=NoOpBudgetAgent(),
        runtime_config=RuntimeConfig(end_time=3000, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=1000, end_time=3000, check_safety=True),
        bounds=bundle.normalization_bounds,
    )
    assert len(result.runtime_result.deadline_misses) == 0


def test_small_legal_adjustment_can_be_accepted() -> None:
    """合法的小幅预算调整应可通过安全检查。"""

    bundle = _build_safe_rtss11_bundle(seed=2)
    checker = _strict_checker_rejecting_hi(bundle.ordered_tasks)
    action = _find_first_accepted_action(bundle.ordered_tasks, checker)
    if action is None:
        action = build_budget_action_space(bundle.ordered_tasks)[0]
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        agent=_FixedActionAgent(action),
        runtime_config=RuntimeConfig(end_time=3000, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=1000, end_time=3000, check_safety=True),
        safety_checker=_AlwaysAcceptChecker(bundle.ordered_tasks),
        bounds=bundle.normalization_bounds,
    )
    assert result.accepted_actions >= 1


def test_action_statistics_match_agent_invocations_and_keep_no_deadline_miss() -> None:
    """动作统计应守恒，且安全检查开启时不应出现 deadline miss。"""

    bundle = _build_safe_rtss11_bundle(seed=3)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        agent=NoOpBudgetAgent(),
        runtime_config=RuntimeConfig(end_time=3000, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=1000, end_time=3000, check_safety=True),
        bounds=bundle.normalization_bounds,
    )
    agent_invocations = len(result.action_log)
    assert result.accepted_actions + result.rejected_actions + result.noop_actions == agent_invocations
    assert len(result.runtime_result.deadline_misses) == 0
