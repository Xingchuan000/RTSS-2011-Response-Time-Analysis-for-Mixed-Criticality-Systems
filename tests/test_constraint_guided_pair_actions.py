"""constraint-guided pair 动作空间专项测试。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.constraint_guided_pair import select_constraint_guided_decrease_targets
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    """构造一个可触发 v11 风险特征与安全约束检查的小任务集。"""

    return [
        Task("h0", 20, 20, 3, 5, Criticality.HI),
        Task("h1", 30, 30, 3, 6, Criticality.HI),
        Task("l0", 25, 25, 2, 2, Criticality.LO),
        Task("l1", 40, 40, 2, 2, Criticality.LO),
        Task("l2", 50, 50, 2, 2, Criticality.LO),
        Task("l3", 60, 60, 2, 2, Criticality.LO),
    ]


def _build_env(*, observation_mode: str = "v11_full_10d", mask_detail_mode: str = "minimal") -> AmcBudgetEnv:
    """构造 constraint-guided pair 测试环境。"""

    return AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=120, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=20,
        action_space="constraint_guided_pair",
        include_explicit_noop=True,
        budget_increase_ratio=0.015,
        budget_decrease_ratio=0.015,
        budget_floor_ratio=0.9,
        feature_config=FeatureConfig(observation_mode=observation_mode),
        mask_detail_mode=mask_detail_mode,
        constraint_guided_pair_top_k_risk=3,
        constraint_guided_pair_top_k_decrease=5,
    )


def test_constraint_guided_pair_action_space_size() -> None:
    """动作数应满足 bundled transfer: top_k_risk + explicit noop。"""

    env = _build_env()
    assert env.action_space_size == 4


def test_constraint_guided_pair_mask_smoke() -> None:
    """reset + valid_action_mask 应可运行，且 mask 长度正确。"""

    env = _build_env()
    _ = env.reset(seed=0)
    mask = env.valid_action_mask()
    assert len(mask) == env.action_space_size
    assert any(mask)


def test_constraint_guided_pair_step_valid_action() -> None:
    """选择一个合法动作后 step 应返回下一观测。"""

    env = _build_env()
    _ = env.reset(seed=0)
    mask = env.valid_action_mask()
    action_id = next(i for i, ok in enumerate(mask) if ok)
    result = env.step(action_id)
    assert result.observation is not None


def test_legacy_action_space_sizes_unchanged() -> None:
    """single/pair/triple 动作数量保持历史公式。"""

    tasks = _tasks()
    single_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=80, semantics=RuntimeSemantics.AMC_PLUS),
        action_space="single",
    )
    pair_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=80, semantics=RuntimeSemantics.AMC_PLUS),
        action_space="pair",
    )
    triple_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=80, semantics=RuntimeSemantics.AMC_PLUS),
        action_space="triple",
    )
    assert single_env.action_space_size == 12
    assert pair_env.action_space_size == 30
    assert triple_env.action_space_size == 60


def test_constraint_guided_pair_full_mask_details() -> None:
    """full 详情下应包含动态解析出的 increase/decrease 索引。"""

    env = _build_env(mask_detail_mode="full")
    _ = env.reset(seed=1)
    _ = env.valid_action_mask()
    dynamic_rows = [item for item in env._last_mask_details if item.get("is_constraint_guided_pair")]  # noqa: SLF001
    assert dynamic_rows
    assert any(("increase_idx" in item and "decrease_indices" in item) for item in dynamic_rows)


def test_constraint_guided_pair_unsupported_observation_mode_masks_slots() -> None:
    """非 v11 观测模式时，constraint-guided 槽位应被统一拒绝。"""

    env = _build_env(observation_mode="v10_basic", mask_detail_mode="full")
    _ = env.reset(seed=2)
    mask = env.valid_action_mask()
    # 显式 noop 保持合法，其余 constraint-guided slots 全部拒绝。
    assert mask[-1] is True
    assert all(not value for value in mask[:-1])
    assert all(
        item.get("reject_reason") == "constraint_guided_unsupported_observation_mode"
        for item in env._last_mask_details[:-1]  # noqa: SLF001
    )


def test_constraint_guided_pair_reuses_diagnostic_decrease_candidates() -> None:
    """env 动态解析出的 decrease_idx 必须来自统一诊断候选函数。"""

    env = _build_env(mask_detail_mode="full")
    _ = env.reset(seed=3)
    _ = env.valid_action_mask()
    for action in env._actions:  # noqa: SLF001
        if not action.is_constraint_guided_pair:
            continue
        resolved = env._resolve_constraint_guided_pair_action(action, check_safety=True)  # noqa: SLF001
        if not resolved.valid:
            continue
        assert resolved.increase_idx is not None
        assert len(resolved.decrease_indices) >= 1
        assert len(resolved.decrease_indices) <= env.constraint_guided_pair_top_k_decrease
        assert env._engine is not None  # noqa: SLF001
        dec_names = {env.ordered_tasks[idx].name for idx in resolved.decrease_indices}
        single_updates = {name: value for name, value in resolved.updates.items() if name not in dec_names}
        if not single_updates:
            # increase-only 被接受的路径不会走到这里；防御性保持测试健壮。
            continue
        single_budgets = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001
        single_budgets.update(single_updates)
        diagnosis = env.diagnose_candidate_budget_update(new_budgets=single_budgets)
        candidates = select_constraint_guided_decrease_targets(
            ordered_tasks=env.ordered_tasks,
            row_coefficients=diagnosis.row_coefficients,
            current_budgets=dict(env._engine.runtime_budgets.budgets),  # noqa: SLF001
            initial_budgets=env._initial_budgets,  # noqa: SLF001
            increase_indices={resolved.increase_idx},
            budget_floor_ratio=env.budget_floor_ratio,
            top_k=env.constraint_guided_pair_top_k_decrease,
            prefer_lo=env.constraint_guided_pair_prefer_lo,
        )
        assert set(resolved.decrease_indices).issubset(set(candidates))
