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


def _reset_network_biases(agent: DqnBudgetAgent, *, target_bias: tuple[float, float, float]) -> None:
    """将 policy/target 网络输出固定为可预测值。"""

    with torch.no_grad():
        for param in agent.policy_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
        target_last = agent.target_network.model[-1]
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
    assert math.isclose(loss, 9.0, rel_tol=1e-6)


def test_next_mask_all_false_must_make_next_q_zero() -> None:
    """当 next mask 全 False 时，next_q 必须为 0，且 loss 有限。"""

    agent = _build_agent()
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
    assert math.isclose(loss, 1.0, rel_tol=1e-6)
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
    assert math.isclose(loss, 1.0, rel_tol=1e-6)
