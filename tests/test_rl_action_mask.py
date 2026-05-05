"""运行时动作掩码测试。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks_with_bound_pressure() -> list[Task]:
    """构造能稳定触发上下界无效动作的任务集。"""

    return [
        Task("h", 10, 10, 1, 2, Criticality.HI),
        Task("l1", 12, 12, 1, 1, Criticality.LO),
        Task("l2", 15, 15, 1, 1, Criticality.LO),
    ]


def test_valid_action_mask_length_matches_action_space_size() -> None:
    """动作掩码长度应与动作空间大小一致。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()
    assert len(mask) == env.action_space_size


def test_valid_action_mask_can_identify_invalid_action() -> None:
    """动作掩码至少应能识别一个无效动作。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()
    assert any(not item for item in mask)


def test_forbid_decreasing_hi_budgets_masks_hi_decrease_actions() -> None:
    """开启 HI 预算保护后，所有 decrease 含 HI 的动作都必须被 mask。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="pair",
        forbid_decreasing_hi_budgets=True,
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()

    for action in env._actions:
        if action.is_noop:
            continue
        has_hi_decrease = any(env.ordered_tasks[idx].criticality is Criticality.HI for idx in action.decrease_indices)
        if has_hi_decrease:
            assert mask[action.action_id] is False

    reject_reason_counts = env.mask_log[-1]["reject_reason_counts"]
    assert int(reject_reason_counts.get("decrease_hi_forbidden", 0)) > 0
