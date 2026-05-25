"""Masked DQN target 计算测试。"""

from __future__ import annotations

import math

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig, Transition


def _build_agent() -> DqnBudgetAgent:
    """构造用于 target 断言的最小 agent。"""

    config = DqnConfig(
        gamma=1.0,
        learning_rate=0.0,
        replay_capacity=10,
        min_replay_size=1,
        batch_size=1,
        target_update_freq=100,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        hidden_layers=(4,),
        seed=0,
    )
    return DqnBudgetAgent(observation_dim=2, action_dim=3, config=config)


def _reset_network_biases(
    agent: DqnBudgetAgent,
    *,
    target_bias: tuple[float, float, float],
    policy_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """将 policy/target 网络输出固定为可预测值。"""

    with torch.no_grad():
        for param in agent.policy_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
        policy_last = agent.policy_network.model[-1]
        target_last = agent.target_network.model[-1]
        policy_last.bias.copy_(torch.tensor(policy_bias, dtype=torch.float32))
        target_last.bias.copy_(torch.tensor(target_bias, dtype=torch.float32))


def test_invalid_next_actions_must_not_pollute_bootstrap_target() -> None:
    """next mask 屏蔽后，非法动作的超大 Q 值不应进入 target。"""

    agent = _build_agent()
    _reset_network_biases(agent, target_bias=(1000.0, 1.0, 2.0))
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=1.0,
            next_state=(0.0, 0.0),
            done=False,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(False, True, True),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    # Double DQN 默认启用：policy 在合法动作 1/2 中按并列 argmax 选择 1，
    # target 评估 action 1 得到 next_q=1，因此 target=reward+next_q=2。
    # 当前 Q=0，Huber 误差 |2| 对应损失为 |2|-0.5 = 1.5。
    assert math.isclose(loss, 1.5, rel_tol=1e-6)


def test_double_dqn_uses_policy_action_and_target_value() -> None:
    """Double DQN 应由 policy 选动作、target 评估该动作。"""

    agent = _build_agent()
    _reset_network_biases(
        agent,
        policy_bias=(0.0, 20.0, 10.0),
        target_bias=(0.0, 10.0, 100.0),
    )
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=0.0,
            next_state=(0.0, 0.0),
            done=False,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(True, True, True),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    # policy 选择 action 1，但 target 的最大值在 action 2。
    # 若错误使用标准 DQN target，误差会按 100 计算；Double DQN 正确误差为 10。
    assert math.isclose(loss, 9.5, rel_tol=1e-6)


def test_standard_dqn_branch_keeps_target_network_max() -> None:
    """关闭 double_dqn 后应保留标准 DQN 的 target max 对照逻辑。"""

    agent = DqnBudgetAgent(observation_dim=2, action_dim=3, config=_build_agent().config, double_dqn=False)
    _reset_network_biases(
        agent,
        policy_bias=(0.0, 20.0, 10.0),
        target_bias=(0.0, 10.0, 100.0),
    )
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=0.0,
            next_state=(0.0, 0.0),
            done=False,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(True, True, True),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    assert math.isclose(loss, 99.5, rel_tol=1e-6)


def test_next_mask_all_false_must_make_next_q_zero() -> None:
    """标准 DQN 对照分支中，next mask 全 False 时 next_q 必须为 0。"""

    agent = _build_agent()
    agent.double_dqn = False
    _reset_network_biases(agent, target_bias=(1000.0, 999.0, 998.0))
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=1.0,
            next_state=(0.0, 0.0),
            done=False,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(False, False, False),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    # Huber loss 在 |error|=1 时取 0.5。
    assert math.isclose(loss, 0.5, rel_tol=1e-6)
    assert math.isfinite(loss)


def test_done_transition_must_not_bootstrap_even_when_next_mask_has_valid_actions() -> None:
    """done=True 时必须停止 bootstrap。"""

    agent = _build_agent()
    _reset_network_biases(agent, target_bias=(1000.0, 999.0, 998.0))
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=1.0,
            next_state=(0.0, 0.0),
            done=True,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(True, True, True),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    # done=True 时 target=reward=1，当前 Q=0，误差为 1，对应 Huber=0.5。
    assert math.isclose(loss, 0.5, rel_tol=1e-6)


def test_action_aware_increase_noop_masks_decrease_in_target_bootstrap() -> None:
    """action-aware+increase_noop 时，target bootstrap 不应使用 decrease 动作。"""

    config = DqnConfig(
        gamma=1.0,
        learning_rate=0.0,
        replay_capacity=10,
        min_replay_size=1,
        batch_size=1,
        target_update_freq=100,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        hidden_layers=(4,),
        seed=0,
        q_network_type="action_aware",
        action_feature_mode="static_v1",
        action_aware_mask_mode="increase_noop",
    )
    agent = DqnBudgetAgent(
        observation_dim=2,
        action_dim=3,
        config=config,
        action_features=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        action_feature_names=("is_noop", "is_increase", "is_decrease"),
    )
    agent._network_q_values = lambda network, _states: (  # type: ignore[method-assign]
        torch.tensor([[1.0, 100.0, 2.0]], dtype=torch.float32, requires_grad=(network is agent.policy_network))
        if network is agent.policy_network
        else torch.tensor([[0.0, 50.0, 4.0]], dtype=torch.float32)
    )
    agent.remember(
        Transition(
            state=(0.0, 0.0),
            action_id=0,
            reward=0.0,
            next_state=(0.0, 0.0),
            done=False,
            valid_action_mask=(True, True, True),
            next_valid_action_mask=(True, True, True),
        )
    )
    loss = agent.optimize_one_step()
    assert loss is not None
    # mask 后 next action 在 {increase, noop} 中选择 noop(action2)，target Q=4；
    # 当前 policy Q(action0)=1，误差=3，Huber=|3|-0.5=2.5。
    assert math.isclose(loss, 2.5, rel_tol=1e-6)
