"""显式 noop 在动作掩码与执行路径中的语义测试。"""

from __future__ import annotations

from amc_py.dqn import DqnBudgetAgent, DqnConfig
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _noop_pressure_tasks() -> list[Task]:
    """构造一个“非 noop 动作必无效”的任务集。

    设计意图：
    1. 所有任务初始预算都为 1；
    2. 所有任务的可增长上界也都为 1（LO 由 deadline 限制，HI 由 C_HI 限制）；
    3. pair 动作里的“加预算”和“减预算”都会被裁剪回 1，形成 no_effective_budget_change。
    """

    return [
        Task("h", period=2, deadline=1, c_lo=1, c_hi=1, criticality=Criticality.HI),
        Task("l1", period=2, deadline=1, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("l2", period=2, deadline=1, c_lo=1, c_hi=1, criticality=Criticality.LO),
    ]


def _build_env() -> AmcBudgetEnv:
    """构造开启显式 noop 的 pair 动作环境。"""

    return AmcBudgetEnv(
        ordered_tasks=_noop_pressure_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=5,
        check_safety=False,
        action_space="pair",
        include_explicit_noop=True,
        # 该测试需要断言 mask 明细中的 updates 字段，因此显式开启 full 详情模式。
        mask_detail_mode="full",
    )


def test_explicit_noop_is_always_valid_in_mask() -> None:
    """`include_explicit_noop=true` 时，显式 noop 在 mask 中必须恒为 True。"""

    env = _build_env()
    env.reset(seed=0)
    mask = env.valid_action_mask()
    noop_action_id = env.action_space_size - 1

    assert mask[noop_action_id] is True
    assert sum(mask) == 1
    assert env._last_mask_details[noop_action_id]["valid"] is True
    assert env._last_mask_details[noop_action_id]["reject_reason"] is None
    assert env._last_mask_details[noop_action_id]["updates"] == {}
    assert env._last_mask_details[noop_action_id]["is_noop"] is True
    reject_reason_counts = env.mask_log[-1]["reject_reason_counts"]
    assert reject_reason_counts.get("no_effective_budget_change", 0) == env.action_space_size - 1


def test_when_only_noop_is_valid_agent_selects_explicit_noop() -> None:
    """当其它动作都无效时，带 mask 的 DQN 选择器应返回显式 noop 的 action_id。"""

    env = _build_env()
    obs = env.reset(seed=0)
    mask = env.valid_action_mask()
    noop_action_id = env.action_space_size - 1
    agent = DqnBudgetAgent(
        observation_dim=len(obs.state_vector),
        action_dim=env.action_space_size,
        config=DqnConfig(
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            hidden_layers=(8, 4),
            seed=0,
            network_seed=0,
            exploration_seed=0,
            replay_seed=0,
        ),
    )

    selected_action_id = agent.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
    assert selected_action_id == noop_action_id


def test_explicit_noop_step_keeps_budget_unchanged_and_is_accepted() -> None:
    """执行显式 noop 后预算必须逐元素不变，且该动作应记为 accepted。"""

    env = _build_env()
    env.reset(seed=0)
    _ = env.valid_action_mask()
    noop_action_id = env.action_space_size - 1
    budget_before = dict(env._engine.runtime_budgets.budgets)

    result = env.step(noop_action_id)
    budget_after = dict(env._engine.runtime_budgets.budgets)

    assert budget_before == budget_after
    assert result.info["accepted"] is True
    assert result.info["is_noop"] is True
    assert result.info["updates"] == {}
