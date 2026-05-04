"""DQN 训练配置定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DqnConfig:
    """集中管理 DQN 基础超参数。"""

    # 折扣因子，对应 Bellman target 中未来回报的权重。
    gamma: float = 0.99
    # Adam 优化器的学习率。
    learning_rate: float = 5e-5
    # 经验回放池最大容量。
    replay_capacity: int = 10000
    # 开始优化前要求的最小样本数。
    min_replay_size: int = 500
    # 单次优化采样的 batch 大小。
    batch_size: int = 64
    # target network 的同步频率，单位为优化步数。
    target_update_freq: int = 100
    # epsilon-greedy 的初始探索率。
    epsilon_start: float = 1.0
    # epsilon-greedy 的最终探索率。
    epsilon_end: float = 0.05
    # epsilon 从起点衰减到终点所经历的训练步数。
    epsilon_decay_steps: int = 5000
    # MLP 隐藏层宽度；若为空则按任务规模自动推导。
    hidden_layers: tuple[int, ...] | None = None
    # 随机种子，用于回放采样与探索行为复现。
    seed: int = 0
