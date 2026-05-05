"""mask_detail_mode 行为测试。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 2, Criticality.LO),
        Task("l2", 15, 15, 2, 2, Criticality.LO),
    ]


def test_mask_detail_mode_minimal_only_keeps_small_fields() -> None:
    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        mask_detail_mode="minimal",
        include_explicit_noop=True,
    )
    env.reset(seed=0)
    _ = env.valid_action_mask()
    detail = env._last_mask_details[0]
    assert set(detail.keys()).issubset({"action_id", "valid", "reject_reason", "is_noop"})


def test_mask_detail_mode_full_keeps_candidate_fields() -> None:
    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        mask_detail_mode="full",
        include_explicit_noop=True,
    )
    env.reset(seed=0)
    _ = env.valid_action_mask()
    detail = env._last_mask_details[0]
    assert "candidate_budgets" in detail
