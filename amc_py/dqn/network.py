"""DQN 使用的前馈神经网络。"""

from __future__ import annotations

import torch
from torch import nn


class DqnNetwork(nn.Module):
    """将状态向量映射为每个离散动作的 Q 值。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: tuple[int, ...] | None):
        """构建多层感知机结构。"""

        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正整数")
        if output_dim <= 0:
            raise ValueError("output_dim 必须为正整数")

        if hidden_layers is None:
            task_count = max(1, input_dim // 2)
            hidden_layers = (max(4, task_count), max(4, task_count // 2))

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden_layers 中的维度必须为正整数")
            # 每个隐藏层后接 ReLU，与文档约束保持一致。
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层直接给出每个动作的 Q 值，不做额外激活。
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入形状为 [batch_size, input_dim]。"""

        return self.model(x)
