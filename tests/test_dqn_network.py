"""DQN 网络结构测试。"""

from __future__ import annotations

import torch

from amc_py.dqn.network import DqnNetwork


def test_network_can_initialize() -> None:
    """DqnNetwork 应可正常初始化。"""

    network = DqnNetwork(input_dim=6, output_dim=3, hidden_layers=(8, 4))
    assert network is not None


def test_forward_returns_expected_shape() -> None:
    """前向传播输出形状应为 [batch_size, output_dim]。"""

    network = DqnNetwork(input_dim=6, output_dim=3, hidden_layers=(8, 4))
    batch = torch.randn(5, 6)
    output = network(batch)
    assert output.shape == (5, 3)


def test_forward_output_contains_no_nan() -> None:
    """网络输出不应包含 NaN。"""

    network = DqnNetwork(input_dim=6, output_dim=3, hidden_layers=(8, 4))
    batch = torch.randn(5, 6)
    output = network(batch)
    assert not torch.isnan(output).any()


def test_backward_can_run_once() -> None:
    """网络应可执行一次反向传播。"""

    network = DqnNetwork(input_dim=6, output_dim=3, hidden_layers=(8, 4))
    batch = torch.randn(4, 6)
    target = torch.randn(4, 3)
    loss = torch.nn.functional.mse_loss(network(batch), target)
    loss.backward()
    has_grad = any(param.grad is not None for param in network.parameters())
    assert has_grad
