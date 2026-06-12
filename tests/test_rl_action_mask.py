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


def test_residual_ranked_mask_uses_resolved_concrete_targets() -> None:
    """residual_ranked 的 full mask detail 必须写出解析后的 concrete increase/decrease 目标。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="residual_ranked",
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()
    assert len(mask) == env.action_space_size

    details = env._last_mask_details
    assert len(details) == env.action_space_size
    transfer_rows = [
        row
        for row in details
        if row.get("residual_action_type") in {"transfer_to_lo_risk_from_global_low", "transfer_to_lo_risk_from_lo_low"}
    ]
    assert transfer_rows, "预期至少存在 transfer residual 槽位"
    assert any(row.get("resolved_increase_task") is not None for row in transfer_rows)
    assert any(len(tuple(row.get("resolved_decrease_tasks", ()))) >= 1 for row in transfer_rows)


def test_residual_ranked_global_low_risk_excludes_hi_when_hi_decrease_forbidden() -> None:
    """forbid_decreasing_hi_budgets=True 时，global_low_risk 不应再把 HI 作为 decrease 目标。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="residual_ranked",
        forbid_decreasing_hi_budgets=True,
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    _ = env.valid_action_mask()

    for row in env._last_mask_details:
        if row.get("residual_action_type") not in {
            "decrease_lowest_risk",
            "transfer_to_lo_risk_from_global_low",
            "transfer_to_lo_risk_from_global_low2",
        }:
            continue
        dec_indices = tuple(row.get("decrease_indices", ()))
        for idx in dec_indices:
            assert env.ordered_tasks[idx].criticality is not Criticality.HI


def test_residual_ranked_mask_applies_residual_guard_rejection() -> None:
    """开启 residual safety fallback 时，mask 中应直接出现 residual_guard_* 拒绝原因。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="residual_ranked",
        enable_residual_safety_fallback=True,
        residual_guard_hi_pressure_abs_limit=0.0,
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    _ = env.valid_action_mask()
    assert any(
        isinstance(row.get("reject_reason"), str) and str(row.get("reject_reason")).startswith("residual_guard_")
        for row in env._last_mask_details
    )


def test_deploy_cap_mask_blocks_lo_increase_when_ratio_reaches_cap() -> None:
    """当 LO 任务预算达到 deploy cap 后，对应 increase 动作必须被 mask。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="single",
        enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0,
    )
    env.reset(seed=0)
    # 这里直接把当前运行时预算抬到初始预算的 4 倍，严格复现计划文档里的触发条件。
    env._engine.runtime_budgets.budgets["l1"] = 4  # noqa: SLF001

    mask = env.valid_action_mask()
    increase_action = next(
        action for action in env._actions if action.increase_task == "l1" and not action.decrease_indices  # noqa: SLF001
    )
    assert mask[increase_action.action_id] is False
    reject_reason_counts = env.mask_log[-1]["reject_reason_counts"]
    assert int(reject_reason_counts.get("deploy_cap_increase_mask", 0)) > 0


def test_deploy_cap_mask_does_not_block_lo_increase_below_cap() -> None:
    """当 LO 任务预算比值未达到阈值时，不应额外触发 deploy cap mask。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="single",
        enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0,
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    env._engine.runtime_budgets.budgets["l1"] = 3  # noqa: SLF001

    _ = env.valid_action_mask()
    increase_action = next(
        action for action in env._actions if action.increase_task == "l1" and not action.decrease_indices  # noqa: SLF001
    )
    detail = env._last_mask_details[increase_action.action_id]  # noqa: SLF001
    assert str(detail.get("reject_reason", "") or "") != "deploy_cap_increase_mask"
    assert not str(detail.get("reject_reason", "") or "").startswith("deploy_cap_increase_mask")


def test_deploy_cap_mask_does_not_block_decrease_and_step_fallback_rejects_increase() -> None:
    """deploy cap 只限制 increase；同时 step() 兜底也必须拒绝被 cap 的 increase 动作。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks_with_bound_pressure(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=30, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="single",
        enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0,
    )
    env.reset(seed=0)
    env._engine.runtime_budgets.budgets["l1"] = 4  # noqa: SLF001

    decrease_action = next(
        action for action in env._actions if action.decrease_tasks == ("l1",)  # noqa: SLF001
    )
    increase_action = next(
        action for action in env._actions if action.increase_task == "l1" and not action.decrease_indices  # noqa: SLF001
    )

    mask = env.valid_action_mask()
    assert mask[decrease_action.action_id] is True

    result = env.step(increase_action.action_id)
    assert bool(result.info["accepted"]) is False
    assert str(result.info["reject_reason"]).startswith("deploy_cap_increase_mask")
