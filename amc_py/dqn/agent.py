"""DQN 预算动作 agent 实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from amc_py.dqn.config import DqnConfig
from amc_py.dqn.network import DqnNetwork, TaskwiseDqnNetwork
from amc_py.dqn.replay import ReplayBuffer
from amc_py.dqn.types import Transition


@dataclass(frozen=True, slots=True)
class NoopQDiagnostics:
    """显式 noop 动作在 DQN Q 值排序中的诊断结果。

    这些字段只描述 policy network 对“当前可决策状态”的估值，不参与训练更新：
    - `noop_q_*`：显式 noop 动作自身的 Q 值统计；
    - `noop_q_rank_*`：显式 noop 在合法动作集合里的 Q 值排名，1 表示最高；
    - `noop_q_margin_to_best_mean`：最佳合法动作 Q 值减去 noop Q 值的平均差距；
    - `noop_q_is_best_rate`：noop 与最佳合法动作并列或独占第一的样本比例；
    - `noop_valid_rate`：noop 在采样决策状态中被 mask 标为合法的比例；
    - `sample_count`：本次诊断纳入的决策状态数量。
    """

    noop_q_mean: float | None
    noop_q_std: float | None
    noop_q_rank_mean: float | None
    noop_q_rank_median: float | None
    noop_q_rank_min: float | None
    noop_q_rank_max: float | None
    noop_q_margin_to_best_mean: float | None
    noop_q_is_best_rate: float | None
    noop_valid_rate: float | None
    sample_count: int


def _resolve_torch_device(device: str | None = None) -> torch.device:
    """解析 DQN 使用的设备，默认自动优先选择 Apple Metal(MPS)。

    选择策略：
    - 若调用方显式传入 `device`，则严格使用该值；
    - 否则若当前 PyTorch 环境支持且可用 `mps`，优先使用 `mps`；
    - 否则回退到 `cpu`。

    这里不自动尝试 `cuda`，因为当前用户运行环境是 macOS，目标是优先启用
    Apple Silicon / Metal 加速；若后续需要显式使用其他设备，可继续通过参数覆盖。
    """

    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DqnBudgetAgent:
    """围绕离散 `action_id` 与 `state_vector` 工作的 DQN agent。"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DqnConfig,
        noop_action_id: int | None = None,
        hidden_layers: tuple[int, ...] | None = None,
        device: str | None = None,
        double_dqn: bool = True,
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
        # Double DQN 开关：
        # - True：按文档要求由 policy network 在 next state 上选择 greedy action，
        #   再由 target network 评估该 action 的 Q 值；
        # - False：保留原始标准 DQN 逻辑，直接对 target network 的合法动作 Q 值取 max，
        #   用于后续 ablation 对照。
        self.double_dqn = bool(double_dqn)
        # 默认自动优先选择 `mps`，这样在支持 Metal 的 Mac 上无需额外传参即可启用硬件加速。
        self.device = _resolve_torch_device(device)
        # 显式 noop 的离散动作编号。
        # - None 表示当前动作空间没有显式 noop；
        # - 非 None 时，epsilon 探索分支可按 noop_exploration_prob 优先采样该动作。
        self.noop_action_id = noop_action_id
        # 阶段 1 参数：探索分支中“优先采样显式 noop”的概率。
        # 该值必须在 [0, 1]，否则配置不合法。
        self.noop_exploration_prob = float(config.noop_exploration_prob)
        if not 0.0 <= self.noop_exploration_prob <= 1.0:
            raise ValueError("noop_exploration_prob must be in [0, 1]")

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
        self.policy_network = self._build_network(
            observation_dim=observation_dim,
            action_dim=action_dim,
            noop_action_id=noop_action_id,
        ).to(self.device)
        self.target_network = self._build_network(
            observation_dim=observation_dim,
            action_dim=action_dim,
            noop_action_id=noop_action_id,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(self.policy_network.parameters(), lr=config.learning_rate)
        # 稳定性修改 A：
        # 将 MSELoss 替换为 Huber loss（PyTorch: SmoothL1Loss）。
        # 原因：MSE 对大 TD error 做平方放大，容易在 Q 值偏离时放大梯度并加剧发散；
        # Huber 在误差较大区间退化为 L1，梯度增长更温和，通常更稳。
        self.loss_fn = nn.SmoothL1Loss()
        self.optimization_steps = 0
        self.epsilon_step = 0
        self.current_epsilon = float(config.epsilon_start)
        # 探索行为统计：
        # - exploration_action_count：进入 epsilon 探索并成功选出动作的总次数；
        # - exploration_noop_action_count：上述探索动作中，显式 noop 被选中的次数。
        self.exploration_action_count = 0
        self.exploration_noop_action_count = 0

    def _build_network(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        noop_action_id: int | None,
    ) -> nn.Module:
        """按配置构造具体网络实例。

        这里把所有 taskwise 结构约束集中在一起校验，避免训练脚本、load 逻辑、
        以及后续评估路径各自维护一套重复判断而产生漂移。
        """

        if self.config.network_arch == "mlp":
            return DqnNetwork(
                input_dim=observation_dim,
                output_dim=action_dim,
                hidden_layers=self.hidden_layers,
            )
        if self.config.network_arch != "taskwise":
            raise ValueError(f"不支持的 network_arch: {self.config.network_arch}")

        if self.config.task_count is None:
            raise ValueError("taskwise network requires config.task_count")
        if self.config.per_task_feature_dim is None:
            raise ValueError("taskwise network requires config.per_task_feature_dim")
        if self.config.global_feature_dim is None:
            raise ValueError("taskwise network requires config.global_feature_dim")
        if self.config.taskwise_use_task_embedding and self.config.taskwise_task_embedding_dim <= 0:
            raise ValueError("taskwise network requires positive taskwise_task_embedding_dim")
        if noop_action_id is None:
            raise ValueError("taskwise network requires explicit noop_action_id")

        task_count = int(self.config.task_count)
        per_task_feature_dim = int(self.config.per_task_feature_dim)
        global_feature_dim = int(self.config.global_feature_dim)
        expected_observation_dim = task_count * per_task_feature_dim + global_feature_dim
        expected_action_dim = 2 * task_count + 1
        expected_noop_action_id = 2 * task_count

        if observation_dim != expected_observation_dim:
            raise ValueError(
                "taskwise network requires observation_dim == "
                "task_count * per_task_feature_dim + global_feature_dim, "
                f"收到 observation_dim={observation_dim}, expected={expected_observation_dim}"
            )
        if action_dim != expected_action_dim:
            raise ValueError(
                "taskwise network requires action_dim == 2 * task_count + 1, "
                f"收到 action_dim={action_dim}, expected={expected_action_dim}"
            )
        if noop_action_id != expected_noop_action_id:
            raise ValueError(
                "taskwise network requires noop_action_id == 2 * task_count, "
                f"收到 noop_action_id={noop_action_id}, expected={expected_noop_action_id}"
            )

        return TaskwiseDqnNetwork(
            task_count=task_count,
            per_task_feature_dim=per_task_feature_dim,
            global_feature_dim=global_feature_dim,
            action_dim=action_dim,
            noop_action_id=noop_action_id,
            use_task_embedding=self.config.taskwise_use_task_embedding,
            task_embedding_dim=self.config.taskwise_task_embedding_dim,
            use_action_bias=self.config.taskwise_use_action_bias,
            action_bias_init=self.config.taskwise_action_bias_init,
        )

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
            # 进入 epsilon 探索分支后，先按文档策略尝试“优先采样显式 noop”。
            # 触发条件必须同时满足：
            # 1) 动作空间存在 noop_action_id；
            # 2) 该 noop 在当前 mask 下合法；
            # 3) noop_exploration_prob > 0；
            # 4) 采样命中 noop_exploration_prob。
            self.exploration_action_count += 1
            should_pick_noop = (
                self.noop_action_id is not None
                and self.noop_action_id in valid_action_ids
                and self.noop_exploration_prob > 0.0
                and self._rng.random() < self.noop_exploration_prob
            )
            if should_pick_noop:
                action_id = int(self.noop_action_id)
                self.exploration_noop_action_count += 1
            else:
                # 若本轮未选 noop，则从“非 noop 合法动作”中均匀采样。
                # 当非 noop 集为空（例如动作空间只有显式 noop）时，回退到全部合法动作集合。
                non_noop_valid_action_ids = [
                    candidate_action_id
                    for candidate_action_id in valid_action_ids
                    if candidate_action_id != self.noop_action_id
                ]
                action_id = int(self._rng.choice(non_noop_valid_action_ids or valid_action_ids))
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
        # 这里先把 Python 层的 tuple/list 批量拼成连续的 NumPy 数组，
        # 再统一转换成 torch tensor。这样可以减少 `torch.tensor(list_of_tuples)` 在
        # Python 侧逐元素解包与 dtype 推断的开销，尤其在训练中频繁优化时更明显。
        states_np = np.asarray([item.state for item in batch], dtype=np.float32)
        next_states_np = np.asarray([item.next_state for item in batch], dtype=np.float32)
        actions_np = np.asarray([item.action_id for item in batch], dtype=np.int64)
        rewards_np = np.asarray([item.reward for item in batch], dtype=np.float32)
        dones_np = np.asarray([item.done for item in batch], dtype=np.float32)
        next_valid_masks_np = np.asarray(
            [item.next_valid_action_mask for item in batch],
            dtype=np.bool_,
        )

        # `states` / `next_states` 是网络前向传播使用的批量状态张量；
        # `actions` 是 gather 所需的列索引，因此保持 int64 并扩展成 [B, 1]。
        states = torch.from_numpy(states_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        next_valid_masks = torch.from_numpy(next_valid_masks_np).to(self.device)

        policy_q = self.policy_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN target 第一步：policy network 只负责“选动作”。
                # 这里对 next state 的 policy Q 值套用 next_valid_masks，把非法动作置为 -inf，
                # 确保 argmax 只会在当前环境允许的动作集合中产生 greedy action。
                next_policy_q_values = self.policy_network(next_states)
                next_policy_q_values = next_policy_q_values.masked_fill(
                    ~next_valid_masks,
                    float("-inf"),
                )
                next_actions = next_policy_q_values.argmax(dim=1, keepdim=True)

                # Double DQN target 第二步：target network 只负责“评估动作”。
                # gather 的列索引来自 policy network 选出的 next_actions，因此 bootstrap
                # 使用的是 Q_target(s', argmax_a Q_policy(s', a))，而不是标准 DQN 的
                # max_a Q_target(s', a)，从而降低大动作空间下的 max over actions 高估。
                next_target_q_values = self.target_network(next_states)
                next_q = next_target_q_values.gather(1, next_actions).squeeze(1)
            else:
                # 标准 DQN 对照分支：保持原实现语义，直接在 target network 输出上屏蔽非法动作，
                # 然后取合法动作中的最大 Q 值作为 bootstrap target。
                next_q_values = self.target_network(next_states)
                masked_next_q_values = next_q_values.masked_fill(~next_valid_masks, float("-inf"))
                next_q = masked_next_q_values.max(dim=1).values
                has_any_valid_action = next_valid_masks.any(dim=1)
                next_q = torch.where(has_any_valid_action, next_q, torch.zeros_like(next_q))
            # terminal transition 不 bootstrap：done=True 时只保留即时 reward。
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = self.loss_fn(policy_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # 稳定性修改 B：
        # 在反向传播后、参数更新前做梯度裁剪，限制总体梯度范数上界。
        # 这里严格按配置值执行，不引入额外“自适应/动态阈值”逻辑。
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=self.config.grad_clip_norm,
        )
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """将 policy network 参数同步到 target network。"""

        self.target_network.load_state_dict(self.policy_network.state_dict())

    @torch.no_grad()
    def compute_noop_q_diagnostics(
        self,
        states: torch.Tensor,
        valid_masks: torch.Tensor,
        noop_action_id: int | None = None,
    ) -> NoopQDiagnostics:
        """计算显式 noop 在合法动作 Q 值集合中的位置。

        参数说明：
        - `states`：形状为 `[N, observation_dim]` 的决策状态张量；
        - `valid_masks`：形状为 `[N, action_dim]` 的 bool 张量，True 表示对应动作合法；
        - `noop_action_id`：显式 noop 的动作编号；不传时使用 agent 初始化时解析到的编号。

        诊断口径严格按文档执行：rank 只在 valid actions 中计算，1 表示 noop 的
        Q 值不低于任何合法动作；如果没有显式 noop 或 noop 在所有样本中都无效，
        与 Q 值相关的字段返回 None，仅保留样本数与 noop_valid_rate。
        """

        resolved_noop_action_id = self.noop_action_id if noop_action_id is None else noop_action_id
        if resolved_noop_action_id is None:
            return NoopQDiagnostics(None, None, None, None, None, None, None, None, None, 0)
        if states.numel() == 0:
            return NoopQDiagnostics(None, None, None, None, None, None, None, None, None, 0)

        # 诊断只读取 policy network 的 Q 值，不应改变训练/评估模式之外的状态。
        q_values = self.policy_network(states)
        noop_valid = valid_masks[:, resolved_noop_action_id]
        sample_count = int(states.shape[0])
        noop_valid_count = int(noop_valid.sum().item())
        noop_valid_rate = noop_valid_count / sample_count if sample_count else 0.0
        if noop_valid_count == 0:
            return NoopQDiagnostics(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                float(noop_valid_rate),
                sample_count,
            )

        # 非法动作统一置为 -inf，这样 best/rank/margin 都只反映合法动作集合。
        valid_q_values = q_values.masked_fill(~valid_masks, float("-inf"))
        noop_q_values = q_values[:, resolved_noop_action_id]
        valid_q_rows_with_noop = valid_q_values[noop_valid]
        noop_q_valid = noop_q_values[noop_valid]

        best_q = valid_q_rows_with_noop.max(dim=1).values
        margin_to_best = best_q - noop_q_valid
        # rank=1 表示没有合法动作的 Q 值严格大于 noop；并列最高也算第一。
        ranks = (valid_q_rows_with_noop > noop_q_valid.unsqueeze(1)).sum(dim=1).float() + 1.0
        noop_is_best = ranks == 1.0

        return NoopQDiagnostics(
            noop_q_mean=float(noop_q_valid.mean().item()),
            noop_q_std=float(noop_q_valid.std(unbiased=False).item()) if noop_valid_count > 1 else 0.0,
            noop_q_rank_mean=float(ranks.mean().item()),
            noop_q_rank_median=float(ranks.median().item()),
            noop_q_rank_min=float(ranks.min().item()),
            noop_q_rank_max=float(ranks.max().item()),
            noop_q_margin_to_best_mean=float(margin_to_best.mean().item()),
            noop_q_is_best_rate=float(noop_is_best.float().mean().item()),
            noop_valid_rate=float(noop_valid_rate),
            sample_count=sample_count,
        )

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
                "noop_action_id": self.noop_action_id,
                "network_arch": self.config.network_arch,
                "task_count": self.config.task_count,
                "per_task_feature_dim": self.config.per_task_feature_dim,
                "global_feature_dim": self.config.global_feature_dim,
                "taskwise_use_task_embedding": self.config.taskwise_use_task_embedding,
                "taskwise_task_embedding_dim": self.config.taskwise_task_embedding_dim,
                "taskwise_use_action_bias": self.config.taskwise_use_action_bias,
                "taskwise_action_bias_init": self.config.taskwise_action_bias_init,
                "double_dqn": self.double_dqn,
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

        resolved_device = _resolve_torch_device(device)
        checkpoint = torch.load(path, map_location=resolved_device)
        config = DqnConfig(**checkpoint["config"])
        agent = cls(
            observation_dim=int(checkpoint["observation_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
            noop_action_id=checkpoint.get("noop_action_id"),
            hidden_layers=tuple(checkpoint["hidden_layers"]) if checkpoint["hidden_layers"] is not None else None,
            device=str(resolved_device),
            double_dqn=bool(checkpoint.get("double_dqn", True)),
        )
        agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.optimization_steps = int(checkpoint["optimization_steps"])
        agent.epsilon_step = int(checkpoint["epsilon_step"])
        agent.current_epsilon = float(checkpoint["current_epsilon"])
        return agent
