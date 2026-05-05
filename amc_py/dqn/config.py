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
    # 按稳定性修改计划改为 5，更接近论文设置，减少 target 过旧导致的 TD 目标漂移。
    target_update_freq: int = 5
    # 梯度裁剪阈值（L2 norm）。
    # 训练时在 `loss.backward()` 之后、`optimizer.step()` 之前执行裁剪，
    # 用于抑制 TD 误差偶发放大时的梯度爆炸风险。
    grad_clip_norm: float = 10.0
    # epsilon-greedy 的初始探索率。
    epsilon_start: float = 1.0
    # epsilon-greedy 的最终探索率。
    epsilon_end: float = 0.05
    # epsilon 从起点衰减到终点所经历的训练步数。
    epsilon_decay_steps: int = 5000
    # 阶段 1：在 epsilon 探索分支里，若显式 noop 合法，则优先采样 noop 的概率。
    # 该概率只作用于“探索分支”，不影响 greedy 动作选择逻辑。
    noop_exploration_prob: float = 0.0
    # MLP 隐藏层宽度；若为空则按任务规模自动推导。
    hidden_layers: tuple[int, ...] | None = None
    # 随机种子，用于回放采样与探索行为复现。
    seed: int = 0
    # 网络参数初始化随机种子。阶段 0 需要把“网络初始化随机性”单独暴露。
    network_seed: int | None = None
    # epsilon-greedy 探索随机种子。阶段 0 需要把“动作探索随机性”单独暴露。
    exploration_seed: int | None = None
    # 经验回放采样随机种子。阶段 0 需要把“replay 采样随机性”单独暴露。
    replay_seed: int | None = None
