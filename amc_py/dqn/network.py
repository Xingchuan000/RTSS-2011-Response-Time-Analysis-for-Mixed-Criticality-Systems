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


class ActionAwareQNetwork(nn.Module):
    """共享 MLP：对每个动作描述符计算 Q(s,a)。"""

    def __init__(
        self,
        observation_dim: int,
        action_feature_dim: int,
        hidden_layers: tuple[int, ...] | None,
    ):
        """构建输入为 [state, action_feature] 的共享打分网络。"""

        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim 必须为正整数")
        if action_feature_dim <= 0:
            raise ValueError("action_feature_dim 必须为正整数")

        input_dim = observation_dim + action_feature_dim
        if hidden_layers is None:
            task_count = max(1, observation_dim // 2)
            hidden_layers = (max(4, task_count), max(4, task_count // 2))

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden_layers 中的维度必须为正整数")
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # 输出单个动作在该状态下的 Q 值。
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self.observation_dim = observation_dim
        self.action_feature_dim = action_feature_dim

    def forward(self, states: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
        - states: [B, observation_dim]
        - action_features: [A, F] 或 [B, A, F]
        返回：
        - q_values: [B, A]
        """

        if states.dim() != 2:
            raise ValueError("states 必须是二维张量 [B, observation_dim]")
        batch_size = int(states.shape[0])
        if states.shape[1] != self.observation_dim:
            raise ValueError("states 的 observation_dim 与网络定义不一致")

        if action_features.dim() == 2:
            if action_features.shape[1] != self.action_feature_dim:
                raise ValueError("action_features 的 feature_dim 与网络定义不一致")
            expanded_action_features = action_features.unsqueeze(0).expand(batch_size, -1, -1)
        elif action_features.dim() == 3:
            if action_features.shape[0] != batch_size:
                raise ValueError("action_features 为三维时，batch 维必须与 states 一致")
            if action_features.shape[2] != self.action_feature_dim:
                raise ValueError("action_features 的 feature_dim 与网络定义不一致")
            expanded_action_features = action_features
        else:
            raise ValueError("action_features 必须是二维 [A,F] 或三维 [B,A,F] 张量")

        action_count = int(expanded_action_features.shape[1])
        # 把每个状态复制到所有动作槽位，再与动作描述符拼接后统一送入共享 MLP。
        state_expanded = states.unsqueeze(1).expand(-1, action_count, -1)
        network_input = torch.cat([state_expanded, expanded_action_features], dim=-1)
        flat_q = self.model(network_input.reshape(batch_size * action_count, -1))
        return flat_q.reshape(batch_size, action_count)
