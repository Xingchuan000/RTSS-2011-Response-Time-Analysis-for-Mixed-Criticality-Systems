"""Taskwise DQN v2 的单元测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig
from amc_py.dqn.network import TaskwiseDqnNetwork


def test_taskwise_v2_network_shape_variants() -> None:
    """v1/v2 四种组合都必须保持 `[batch, 25]` 输出。"""

    variants = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    x = torch.zeros(4, 128, dtype=torch.float32)
    for use_task_embedding, use_action_bias in variants:
        net = TaskwiseDqnNetwork(
            task_count=12,
            per_task_feature_dim=10,
            global_feature_dim=8,
            action_dim=25,
            noop_action_id=24,
            use_task_embedding=use_task_embedding,
            task_embedding_dim=8,
            use_action_bias=use_action_bias,
            action_bias_init=0.0,
        )
        y = net(x)
        assert y.shape == (4, 25)


def test_taskwise_v2_agent_save_and_load_preserves_structure(tmp_path: Path) -> None:
    """保存/恢复后应保留 task embedding 与 action bias 的网络结构。"""

    config = DqnConfig(
        network_arch="taskwise",
        task_count=12,
        per_task_feature_dim=10,
        global_feature_dim=8,
        taskwise_use_task_embedding=True,
        taskwise_task_embedding_dim=8,
        taskwise_use_action_bias=True,
        taskwise_action_bias_init=0.0,
    )
    agent = DqnBudgetAgent(
        observation_dim=128,
        action_dim=25,
        config=config,
        noop_action_id=24,
        device="cpu",
    )
    model_path = tmp_path / "taskwise_v2_agent.pt"
    agent.save(model_path)

    loaded = DqnBudgetAgent.load(model_path, device="cpu")
    assert loaded.config.taskwise_use_task_embedding is True
    assert loaded.config.taskwise_task_embedding_dim == 8
    assert loaded.config.taskwise_use_action_bias is True
    assert loaded.config.taskwise_action_bias_init == 0.0
    assert isinstance(loaded.policy_network, TaskwiseDqnNetwork)
    assert loaded.policy_network.use_task_embedding is True
    assert loaded.policy_network.task_embedding is not None
    assert loaded.policy_network.use_action_bias is True
    assert loaded.policy_network.action_bias is not None


def test_taskwise_v2_rejects_non_positive_embedding_dim() -> None:
    """开启 task embedding 时，embedding 维度必须为正整数。"""

    try:
        TaskwiseDqnNetwork(
            task_count=12,
            per_task_feature_dim=10,
            global_feature_dim=8,
            action_dim=25,
            noop_action_id=24,
            use_task_embedding=True,
            task_embedding_dim=0,
        )
    except ValueError as exc:
        assert "task_embedding_dim" in str(exc)
        return
    raise AssertionError("预期 task_embedding_dim 非法时抛出 ValueError")
