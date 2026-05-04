"""DQN 预算动作 agent 实现。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import Adam

from amc_py.dqn.config import DqnConfig
from amc_py.dqn.network import DqnNetwork
from amc_py.dqn.replay import ReplayBuffer
from amc_py.dqn.types import Transition


class DqnBudgetAgent:
    """围绕离散 `action_id` 与 `state_vector` 工作的 DQN agent。"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DqnConfig,
        hidden_layers: tuple[int, ...] | None = None,
        device: str | None = None,
    ):
        """初始化网络、优化器和经验回放池。"""

        if observation_dim <= 0:
            raise ValueError("observation_dim 必须为正整数")
        if action_dim <= 0:
            raise ValueError("action_dim 必须为正整数")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config
        self.hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers
        self.device = torch.device(device or "cpu")

        # 新字段为空时沿用旧的 seed 语义，保证历史配置不改参数也能复现实验。
        exploration_seed = config.seed if config.exploration_seed is None else config.exploration_seed
        network_seed = config.seed if config.network_seed is None else config.network_seed
        replay_seed = config.seed if config.replay_seed is None else config.replay_seed
        # 使用独立随机数生成器，保证探索行为在固定 exploration_seed 下可复现。
        self._rng = random.Random(exploration_seed)
        # 在构建网络前固定 torch 随机种子，保证参数初始化在 network_seed 下可复现。
        torch.manual_seed(network_seed)
        # 回放池采样使用单独 replay_seed，避免与探索随机流互相污染。
        self.replay_buffer = ReplayBuffer(capacity=config.replay_capacity, seed=replay_seed)
        self.policy_network = DqnNetwork(
            input_dim=observation_dim,
            output_dim=action_dim,
            hidden_layers=self.hidden_layers,
        ).to(self.device)
        self.target_network = DqnNetwork(
            input_dim=observation_dim,
            output_dim=action_dim,
            hidden_layers=self.hidden_layers,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.optimization_steps = 0
        self.epsilon_step = 0
        self.current_epsilon = float(config.epsilon_start)

    def _compute_epsilon(self) -> float:
        """按线性衰减规则计算当前 epsilon。"""

        if self.config.epsilon_decay_steps <= 0:
            return float(self.config.epsilon_end)
        progress = min(1.0, self.epsilon_step / self.config.epsilon_decay_steps)
        return float(
            self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * progress
        )

    def _valid_action_ids(self, valid_action_mask: tuple[bool, ...] | None) -> list[int]:
        """将合法动作掩码转换为动作编号列表。"""

        if valid_action_mask is None:
            return list(range(self.action_dim))
        if len(valid_action_mask) != self.action_dim:
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        return [action_id for action_id, is_valid in enumerate(valid_action_mask) if is_valid]

    def _greedy_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None,
    ) -> int:
        """在给定掩码约束下选择 Q 值最大的合法动作。"""

        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)[0]
        if valid_action_mask is not None:
            mask_tensor = torch.tensor(valid_action_mask, dtype=torch.bool, device=self.device)
            # 按文档要求，将非法动作的 Q 值视为 -inf，保证 greedy 不会选到它们。
            q_values = q_values.masked_fill(~mask_tensor, float("-inf"))
        return int(torch.argmax(q_values).item())

    def select_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None = None,
        training: bool = True,
    ) -> int | None:
        """按 epsilon-greedy 规则选择离散动作编号。"""

        if len(state_vector) != self.observation_dim:
            raise ValueError("state_vector 维度与 observation_dim 不一致")

        valid_action_ids = self._valid_action_ids(valid_action_mask)
        if not valid_action_ids:
            return None

        self.current_epsilon = self._compute_epsilon()
        if training and self._rng.random() < self.current_epsilon:
            action_id = int(self._rng.choice(valid_action_ids))
        else:
            action_id = self._greedy_action_id(state_vector, valid_action_mask)

        if training:
            self.epsilon_step += 1
            self.current_epsilon = self._compute_epsilon()
        return action_id

    def remember(self, transition: Transition) -> None:
        """将 transition 写入经验回放池。"""

        if transition.action_id < 0 or transition.action_id >= self.action_dim:
            raise ValueError("transition.action_id 超出动作空间范围")
        self.replay_buffer.push(transition)

    def optimize_one_step(self) -> float | None:
        """执行一次 DQN 单步优化；样本不足时返回 None。"""

        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action_id for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        next_valid_masks = torch.tensor(
            [item.next_valid_action_mask for item in batch],
            dtype=torch.bool,
            device=self.device,
        )

        policy_q = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            # Bellman bootstrap 必须与动作选择共享同一套合法动作语义：
            # 先屏蔽非法动作，再做 max。
            masked_next_q_values = next_q_values.masked_fill(~next_valid_masks, float("-inf"))
            next_q = masked_next_q_values.max(dim=1).values
            # 若 next_state 没有任何合法动作，则 bootstrap 值按 0 处理。
            has_any_valid_action = next_valid_masks.any(dim=1)
            next_q = torch.where(has_any_valid_action, next_q, torch.zeros_like(next_q))
            # done=True 时必须终止 bootstrap。
            next_q = torch.where(dones > 0.0, torch.zeros_like(next_q), next_q)
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = self.loss_fn(policy_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """将 policy network 参数同步到 target network。"""

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, path: Path) -> None:
        """保存训练状态，便于后续继续训练或评估。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_network_state_dict": self.policy_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_layers": self.hidden_layers,
                "optimization_steps": self.optimization_steps,
                "epsilon_step": self.epsilon_step,
                "current_epsilon": self.current_epsilon,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        device: str | None = None,
    ) -> "DqnBudgetAgent":
        """从磁盘恢复 DQN agent。"""

        checkpoint = torch.load(path, map_location=device or "cpu")
        config = DqnConfig(**checkpoint["config"])
        agent = cls(
            observation_dim=int(checkpoint["observation_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
            hidden_layers=tuple(checkpoint["hidden_layers"]) if checkpoint["hidden_layers"] is not None else None,
            device=device,
        )
        agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.optimization_steps = int(checkpoint["optimization_steps"])
        agent.epsilon_step = int(checkpoint["epsilon_step"])
        agent.current_epsilon = float(checkpoint["current_epsilon"])
        return agent
