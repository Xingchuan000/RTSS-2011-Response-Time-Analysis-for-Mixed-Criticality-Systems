"""DQN 动作空间配置测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate, build_budget_action_space
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks(n: int = 20) -> list[Task]:
    """构造测试任务集。"""

    tasks: list[Task] = []
    for idx in range(n):
        if idx == 0:
            tasks.append(Task(f"t{idx}", 20, 20, 2, 3, Criticality.HI))
        else:
            tasks.append(Task(f"t{idx}", 20 + idx, 20 + idx, 2, 2, Criticality.LO))
    return tasks


def _anchor_tasks() -> list[Task]:
    """构造包含 mc_lo_2 的最小任务集，用于 anchor 动作测试。"""

    return [
        Task("mc_hi_0", 20, 20, 2, 3, Criticality.HI),
        Task("mc_lo_1", 24, 24, 2, 2, Criticality.LO),
        Task("mc_lo_2", 28, 28, 2, 2, Criticality.LO),
        Task("mc_lo_3", 32, 32, 2, 2, Criticality.LO),
    ]


def test_action_count_for_n20_triple_and_pair() -> None:
    """n=20 时 triple/pair 动作数应符合公式。"""

    tasks = _tasks(20)
    triple_actions = build_budget_action_space(tasks, action_space="triple")
    pair_actions = build_budget_action_space(tasks, action_space="pair")
    assert len(triple_actions) == 3420
    assert len(pair_actions) == 380


def test_triple_action_with_explicit_noop_has_expected_count_and_distinct_tasks() -> None:
    """triple+显式noop 时动作数应为 3421，且三任务互不相同。"""

    tasks = _tasks(20)
    actions = build_budget_action_space(tasks, action_space="triple", include_explicit_noop=True)
    assert len(actions) == 3421
    for action in actions[:-1]:
        inc = action.increase_task
        dec1, dec2 = action.decrease_tasks
        assert inc is not None
        assert inc != dec1
        assert inc != dec2
        assert dec1 != dec2


def test_include_explicit_noop_exists_and_keeps_budget_unchanged() -> None:
    """显式 noop 动作应存在且不修改预算。"""

    tasks = _tasks(6)
    actions = build_budget_action_space(tasks, action_space="pair", include_explicit_noop=True)
    noop = actions[-1]
    assert noop.is_noop is True
    budget = BudgetState.from_tasks(tasks)
    updates = apply_budget_action_candidate(action=noop, budget_state=budget, ordered_tasks=tasks)
    assert updates == {}


def test_pair_action_updates_exactly_two_tasks() -> None:
    """pair 动作应只包含一个增预算任务和一个降预算任务。"""

    tasks = _tasks(6)
    budget = BudgetState.from_tasks(tasks)
    action = build_budget_action_space(tasks, action_space="pair")[0]
    updates = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
    assert len(updates) == 2


def test_safety_mask_supports_triple_and_pair_action_space() -> None:
    """安全 mask 对 triple/pair 两种动作空间都应可用。"""

    tasks = _tasks(6)
    triple_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="triple",
    )
    pair_env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="pair",
    )
    triple_env.reset(seed=0)
    pair_env.reset(seed=0)
    assert len(triple_env.valid_action_mask()) == triple_env.action_space_size
    assert len(pair_env.valid_action_mask()) == pair_env.action_space_size


def test_constraint_guided_pair_action_count_matches_slot_formula() -> None:
    """constraint_guided_pair 动作数应满足 dynamic_slots + 可选显式 noop。"""

    tasks = _tasks(6)
    actions = build_budget_action_space(
        tasks,
        action_space="constraint_guided_pair",
        constraint_guided_pair_top_k_risk=3,
        constraint_guided_pair_top_k_decrease=5,
        include_explicit_noop=True,
    )
    assert len(actions) == 4
    assert actions[-1].is_noop is True
    assert actions[0].is_constraint_guided_pair is True


def test_constraint_guided_pair_cannot_apply_before_runtime_resolution() -> None:
    """constraint_guided_pair 槽位必须先在 env 中解析，不能直接套用候选更新函数。"""

    tasks = _tasks(6)
    budget = BudgetState.from_tasks(tasks)
    action = build_budget_action_space(
        tasks,
        action_space="constraint_guided_pair",
        constraint_guided_pair_top_k_risk=1,
        constraint_guided_pair_top_k_decrease=1,
    )[0]
    try:
        _ = apply_budget_action_candidate(action=action, budget_state=budget, ordered_tasks=tasks)
        assert False, "预期抛出 ValueError"
    except ValueError as exc:
        assert "constraint_guided_pair action must be resolved by AmcBudgetEnv before applying" in str(exc)


def test_residual_ranked_action_count_and_slots_match_v2_design() -> None:
    """residual_ranked 动作槽位应固定为 15 个，并满足设计分布。"""

    tasks = _tasks(6)
    actions = build_budget_action_space(tasks, action_space="residual_ranked")
    assert len(actions) == 15
    assert [action.action_id for action in actions] == list(range(15))
    assert actions[0].is_noop is True
    assert actions[0].residual_action_type == "noop"
    transfer_count = sum(
        1
        for action in actions
        if action.residual_action_type in {
            "transfer_to_lo_risk_from_global_low",
            "transfer_to_lo_risk_from_lo_low",
            "transfer_to_lo_risk_from_global_low2",
        }
    )
    assert transfer_count == 6


def test_residual_anchor_mc_lo_2_action_space_contains_direct_anchor_slot() -> None:
    """residual_anchor_mc_lo_2 应固定为 5 槽位：noop + anchor + 3 个 safe increase。"""

    actions = build_budget_action_space(_anchor_tasks(), action_space="residual_anchor_mc_lo_2")
    assert len(actions) == 5
    assert [a.action_id for a in actions] == [0, 1, 2, 3, 4]
    assert actions[0].residual_action_type == "noop"
    anchor = actions[1]
    assert anchor.residual_action_type == "direct_safe_increase_anchor"
    assert anchor.increase_task == "mc_lo_2"
    assert anchor.increase_idx is not None
    assert [actions[2].residual_rank, actions[3].residual_rank, actions[4].residual_rank] == [0, 1, 2]
    assert all(actions[idx].residual_action_type == "safe_increase_lo_risk" for idx in (2, 3, 4))


def test_residual_anchor_mc_lo_2_anchor_step_only_increases_mc_lo_2() -> None:
    """anchor 槽位执行后应仅增加 mc_lo_2，且不带 decrease。"""

    env = AmcBudgetEnv(
        ordered_tasks=_anchor_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space="residual_anchor_mc_lo_2",
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()
    assert mask[1] is True

    step = env.step(1)
    assert step.done is False
    assert step.info["accepted"] is True
    assert step.info["residual_action_type"] == "direct_safe_increase_anchor"
    assert step.info["residual_resolved_increase_task"] == "mc_lo_2"
    assert tuple(step.info["residual_resolved_decrease_tasks"]) == ()
