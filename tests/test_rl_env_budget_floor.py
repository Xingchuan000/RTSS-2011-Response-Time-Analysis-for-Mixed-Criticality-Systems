"""budget floor 约束测试。"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    """构造便于触发 budget floor 的三任务任务集。"""

    return [
        Task("h", period=20, deadline=20, c_lo=10, c_hi=12, criticality=Criticality.HI),
        Task("l1", period=20, deadline=20, c_lo=10, c_hi=10, criticality=Criticality.LO),
        Task("l2", period=20, deadline=20, c_lo=10, c_hi=10, criticality=Criticality.LO),
    ]


def _env(*, budget_floor_ratio: float, include_explicit_noop: bool = True) -> AmcBudgetEnv:
    """构造关闭 safety、仅观察 floor 行为的 pair 动作环境。"""

    return AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=40, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        check_safety=False,
        action_space="pair",
        budget_increase_ratio=0.05,
        budget_decrease_ratio=0.05,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        mask_detail_mode="full",
    )


def _find_action_id_decreasing_task(env: AmcBudgetEnv, task_name: str) -> int:
    """找到一个会降低指定任务预算的离散动作编号。"""

    for action in env._actions:  # noqa: SLF001
        if action.is_noop:
            continue
        if task_name in action.decrease_tasks:
            return int(action.action_id)
    raise AssertionError(f"未找到 decrease {task_name} 的动作")


def test_budget_floor_ratio_validation_rejects_out_of_range_values() -> None:
    """budget_floor_ratio 必须限制在 [0, 1]。"""

    with pytest.raises(ValueError):
        _env(budget_floor_ratio=-0.1)

    with pytest.raises(ValueError):
        _env(budget_floor_ratio=1.1)


def test_explicit_noop_remains_valid_when_budget_floor_is_enabled() -> None:
    """显式 noop 不应被 budget floor 影响，必须始终保持合法。"""

    env = _env(budget_floor_ratio=0.9, include_explicit_noop=True)
    env.reset(seed=0)
    mask = env.valid_action_mask()
    noop_action_id = env.action_space_size - 1

    assert mask[noop_action_id] is True
    assert env._last_mask_details[noop_action_id]["valid"] is True  # noqa: SLF001
    assert env._last_mask_details[noop_action_id]["reject_reason"] is None  # noqa: SLF001


def test_valid_action_mask_rejects_decrease_that_would_cross_budget_floor() -> None:
    """一旦 decrease 会把预算压到 floor 以下，该动作必须被 mask。"""

    env = _env(budget_floor_ratio=0.8)
    env.reset(seed=0)
    # 初始预算是 10，floor=ceil(10*0.8)=8。
    # 先把 l1 当前预算调到 8，再执行一次 5% decrease，会得到 7，从而违反 floor。
    env._engine.runtime_budgets.apply_updates({"l1": 8})  # noqa: SLF001
    action_id = _find_action_id_decreasing_task(env, "l1")

    mask = env.valid_action_mask()

    assert mask[action_id] is False
    assert str(env._last_mask_details[action_id]["reject_reason"]).startswith("budget_floor_violation")  # noqa: SLF001
    reject_reason_counts = env.mask_log[-1]["reject_reason_counts"]
    assert int(reject_reason_counts.get("budget_floor_violation", 0)) >= 1


def test_step_rejects_budget_floor_violation_even_when_mask_is_bypassed() -> None:
    """直接传入违规 action_id 时，step 也必须兜底拒绝，并保持预算不变。"""

    env = _env(budget_floor_ratio=0.8)
    env.reset(seed=0)
    env._engine.runtime_budgets.apply_updates({"l1": 8})  # noqa: SLF001
    action_id = _find_action_id_decreasing_task(env, "l1")
    budget_before = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001

    result = env.step(action_id)
    budget_after = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001

    assert result.info["accepted"] is False
    assert str(result.info["reject_reason"]).startswith("budget_floor_violation")
    assert budget_after == budget_before


def test_budget_floor_ratio_zero_preserves_old_behavior_for_same_state() -> None:
    """floor_ratio=0 时，不应因为 floor 而额外拒绝动作。"""

    env = _env(budget_floor_ratio=0.0, include_explicit_noop=False)
    env.reset(seed=0)
    env._engine.runtime_budgets.apply_updates({"l1": 8})  # noqa: SLF001
    action_id = _find_action_id_decreasing_task(env, "l1")

    mask = env.valid_action_mask()

    assert mask[action_id] is True
