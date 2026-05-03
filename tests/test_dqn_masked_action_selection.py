"""DQN 掩码动作选择测试。"""

from __future__ import annotations

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig


STATE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def _config(**kwargs: object) -> DqnConfig:
    """构造测试用配置。"""

    defaults = {
        "gamma": 0.9,
        "learning_rate": 1e-3,
        "replay_capacity": 20,
        "min_replay_size": 2,
        "batch_size": 2,
        "target_update_freq": 2,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "hidden_layers": (8, 4),
        "seed": 11,
    }
    defaults.update(kwargs)
    return DqnConfig(**defaults)


def _agent(config: DqnConfig | None = None) -> DqnBudgetAgent:
    """构造测试 agent。"""

    return DqnBudgetAgent(observation_dim=len(STATE), action_dim=3, config=config or _config())


def _set_q_bias(agent: DqnBudgetAgent, q_values: tuple[float, float, float]) -> None:
    """将输出层偏置固定为便于断言的 Q 值。"""

    with torch.no_grad():
        for param in agent.policy_network.parameters():
            param.zero_()
        final_policy = agent.policy_network.model[-1]
        final_policy.bias.copy_(torch.tensor(q_values, dtype=torch.float32))


def test_random_exploration_never_selects_invalid_action() -> None:
    """epsilon=1 时也只能在合法动作中随机探索。"""

    agent = _agent(_config(epsilon_start=1.0, epsilon_end=1.0))
    chosen = [agent.select_action_id(STATE, valid_action_mask=(False, True, False), training=True) for _ in range(10)]
    assert chosen == [1] * 10


def test_greedy_selection_avoids_masked_high_q_action() -> None:
    """epsilon=0 时不能选择被屏蔽但 Q 值更高的动作。"""

    agent = _agent(_config(epsilon_start=0.0, epsilon_end=0.0))
    _set_q_bias(agent, (10.0, 1.0, 0.5))
    assert agent.select_action_id(STATE, valid_action_mask=(False, True, True), training=False) == 1


def test_all_masked_actions_return_none() -> None:
    """所有动作都被屏蔽时应返回 None。"""

    agent = _agent()
    assert agent.select_action_id(STATE, valid_action_mask=(False, False, False), training=False) is None
