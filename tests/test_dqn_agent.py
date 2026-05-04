"""DQN agent 测试。"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig, Transition


STATE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
NEXT_STATE = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _config(**kwargs: object) -> DqnConfig:
    """构造便于测试的 DQN 配置。"""

    defaults = {
        "gamma": 0.9,
        "learning_rate": 1e-3,
        "replay_capacity": 50,
        "min_replay_size": 2,
        "batch_size": 2,
        "target_update_freq": 2,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "hidden_layers": (8, 4),
        "seed": 123,
    }
    defaults.update(kwargs)
    return DqnConfig(**defaults)


def _agent(config: DqnConfig | None = None) -> DqnBudgetAgent:
    """构造测试 agent。"""

    return DqnBudgetAgent(observation_dim=len(STATE), action_dim=3, config=config or _config())


def _transition(action_id: int, reward: float = 1.0, done: bool = False) -> Transition:
    """构造测试 transition。"""

    return Transition(
        state=STATE,
        action_id=action_id,
        reward=reward,
        next_state=NEXT_STATE,
        done=done,
        valid_action_mask=(True, True, True),
        next_valid_action_mask=(True, True, True),
    )


def _set_q_bias(agent: DqnBudgetAgent, q_values: tuple[float, float, float]) -> None:
    """将网络输出固定到易于断言的 Q 值。"""

    with torch.no_grad():
        for param in agent.policy_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
        final_policy = agent.policy_network.model[-1]
        final_target = agent.target_network.model[-1]
        final_policy.bias.copy_(torch.tensor(q_values, dtype=torch.float32))
        final_target.bias.copy_(torch.tensor(q_values, dtype=torch.float32))


def test_epsilon_one_can_sample_random_actions() -> None:
    """epsilon=1 时应发生随机探索。"""

    agent = _agent(_config(epsilon_start=1.0, epsilon_end=1.0))
    chosen = [agent.select_action_id(STATE, training=True) for _ in range(20)]
    assert len(set(chosen)) > 1


def test_epsilon_zero_selects_max_q_action() -> None:
    """epsilon=0 时应选择 Q 值最大的动作。"""

    agent = _agent(_config(epsilon_start=0.0, epsilon_end=0.0))
    _set_q_bias(agent, (0.1, 2.0, 1.0))
    assert agent.select_action_id(STATE, training=False) == 1


def test_valid_action_mask_blocks_invalid_actions() -> None:
    """合法动作掩码应屏蔽非法动作。"""

    agent = _agent(_config(epsilon_start=0.0, epsilon_end=0.0))
    _set_q_bias(agent, (3.0, 2.0, 1.0))
    action_id = agent.select_action_id(STATE, valid_action_mask=(False, True, True), training=False)
    assert action_id == 1


def test_all_invalid_actions_return_none() -> None:
    """所有动作非法时应返回 None。"""

    agent = _agent()
    assert agent.select_action_id(STATE, valid_action_mask=(False, False, False), training=False) is None


def test_optimize_returns_none_when_replay_is_insufficient() -> None:
    """样本不足时 optimize_one_step 应返回 None。"""

    agent = _agent()
    agent.remember(_transition(0))
    assert agent.optimize_one_step() is None


def test_optimize_returns_finite_loss_when_replay_is_ready() -> None:
    """样本足够时 optimize_one_step 应返回有限 loss。"""

    agent = _agent()
    agent.remember(_transition(0, reward=1.0))
    agent.remember(_transition(1, reward=0.5))
    loss = agent.optimize_one_step()
    assert loss is not None
    assert math.isfinite(loss)


def test_save_and_load_keep_same_greedy_output(tmp_path: Path) -> None:
    """save/load 后同一状态的 greedy 输出应一致。"""

    agent = _agent(_config(epsilon_start=0.0, epsilon_end=0.0))
    _set_q_bias(agent, (0.5, 1.5, 1.0))
    expected = agent.select_action_id(STATE, training=False)

    model_path = tmp_path / "dqn_agent.pt"
    agent.save(model_path)
    loaded = DqnBudgetAgent.load(model_path)

    assert loaded.select_action_id(STATE, training=False) == expected
