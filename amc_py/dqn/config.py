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
    # epsilon 探索动作采样模式：
    # - epsilon_greedy：保持旧逻辑，epsilon 触发后从合法动作中采样（含 noop 优先逻辑）；
    # - epsilon_safe_increase_mixture：epsilon 触发后，先按 safe_increase_explore_prob
    #   尝试从“当前合法 increase-only 动作”采样，未命中或无可选 increase 时回到旧逻辑。
    # - epsilon_increase_coverage：epsilon 触发后，先按 safe_increase_explore_prob
    #   尝试从“当前合法 increase-only 动作”中选择历史探索次数最少的动作；
    #   若并列最少，则随机打散并从并列集合中采样；未命中或无可选 increase 时回到旧逻辑。
    exploration_mode: str = "epsilon_greedy"
    # 仅当 exploration_mode 为 safe_increase_mixture / increase_coverage 时生效。
    # 含义：epsilon 探索触发时，以该概率进入 increase-only 分支。
    safe_increase_explore_prob: float = 0.0
    # MLP 隐藏层宽度；若为空则按任务规模自动推导。
    hidden_layers: tuple[int, ...] | None = None
    # Q 网络结构类型：
    # - mlp:         旧版 Q(s)->all actions；
    # - action_aware: 新版共享 Q(s,a) 打分网络。
    q_network_type: str = "mlp"
    # action_feature_mode:
    # - static_v1: 固定动作描述符，不需要在 replay transition 中保存每步特征；
    # - dynamic_v1: 状态相关动作描述符，必须把当前/下一步特征矩阵一起写入 transition。
    action_feature_mode: str = "static_v1"
    # action_aware_mask_mode:
    # - none: 使用环境原始合法动作 mask；
    # - increase_noop: 仅用于诊断，会屏蔽 decrease 动作。
    action_aware_mask_mode: str = "none"
    # 随机种子，用于回放采样与探索行为复现。
    seed: int = 0
    # 网络参数初始化随机种子。阶段 0 需要把“网络初始化随机性”单独暴露。
    network_seed: int | None = None
    # epsilon-greedy 探索随机种子。阶段 0 需要把“动作探索随机性”单独暴露。
    exploration_seed: int | None = None
    # 经验回放采样随机种子。阶段 0 需要把“replay 采样随机性”单独暴露。
    replay_seed: int | None = None
