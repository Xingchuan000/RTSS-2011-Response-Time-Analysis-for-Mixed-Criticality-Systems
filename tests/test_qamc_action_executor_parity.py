from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
)
from amc_py.rl.action_execution import (
    BudgetActionExecutionConfig,
    evaluate_budget_action,
)
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


def test_env_and_shared_executor_apply_same_single_action() -> None:
    task = Task("L", 20, 20, 9, 9, Criticality.LO)
    bundle = build_qamc_profile_bundle(
        [task],
        taskset_fingerprint=compute_taskset_fingerprint([task]),
        spec=QAmcProfileSpec(),
    )
    env = AmcBudgetEnv(
        ordered_tasks=[task],
        scenario=make_table_scenario(
            actual_costs={},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        runtime_config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.Q_AMC),
        agent_period=1,
        action_space="single",
        check_safety=True,
        qamc_profile_bundle=bundle,
    )
    env.reset()
    increase = next(action for action in env.actions if action.increase_idx == 0)
    env_evaluation = env.evaluate_budget_candidate(action=increase)
    shared_evaluation = evaluate_budget_action(
        action=increase,
        ordered_tasks=[task],
        budget_state=BudgetState.from_tasks([task]),
        initial_budgets={"L": task.c_lo},
        config=BudgetActionExecutionConfig(
            rounding_mode="ceil_floor",
            min_budget_delta=1,
            fixed_floor_by_task={"L": bundle.profiles["L"].full_quality_isolated_wcet},
            check_safety=True,
        ),
        safety_checker=env._ensure_checker(),
    )
    assert env_evaluation.accepted == shared_evaluation.accepted
    assert env_evaluation.reject_reason == shared_evaluation.reject_reason
    assert env_evaluation.candidate_budgets == shared_evaluation.candidate_budgets
