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
from amc_py.dqn.network import ActionAwareQNetwork, DqnNetwork
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
    """解析 DQN 训练使用的 torch device。

    约定如下：
    - 如果调用方显式传入 `device`，就严格使用该设备，并在硬件不可用时立即报错；
    - 如果没有显式指定设备，则保持旧默认行为：macOS 上优先 `mps`，否则使用 `cpu`；
    - 不在默认分支里自动切换到 `cuda`，避免改变既有实验的默认训练口径。

    这里故意不做静默回退：用户一旦明确请求 `cuda` 或 `mps`，就必须真的可用。
    """

    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested DQN device '{device}', but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --dqn-device cpu."
            )
        if requested.type == "mps" and not (
            torch.backends.mps.is_built() and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                f"Requested DQN device '{device}', but MPS is not available in this PyTorch build. "
                "Use --dqn-device cpu or install a compatible PyTorch build."
            )
        return requested
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
        increase_action_ids: tuple[int, ...] | None = None,
        hidden_layers: tuple[int, ...] | None = None,
        device: str | None = None,
        double_dqn: bool = True,
        action_features: tuple[tuple[float, ...], ...] | None = None,
        action_feature_names: tuple[str, ...] | None = None,
        action_feature_dim: int | None = None,
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
        # increase-only 动作编号集合（静态动作语义），用于训练期 safe increase 探索分支。
        # 这里不在 agent 内部推断动作语义，统一由上层环境解析后传入，避免口径漂移。
        self.increase_action_ids = tuple(int(action_id) for action_id in (increase_action_ids or ()))
        self.increase_action_id_set = set(self.increase_action_ids)
        # 阶段 1 参数：探索分支中“优先采样显式 noop”的概率。
        # 该值必须在 [0, 1]，否则配置不合法。
        self.noop_exploration_prob = float(config.noop_exploration_prob)
        if not 0.0 <= self.noop_exploration_prob <= 1.0:
            raise ValueError("noop_exploration_prob must be in [0, 1]")
        # 新增探索模式配置校验：
        # - epsilon_greedy：完全沿用旧实现；
        # - epsilon_safe_increase_mixture：训练期 epsilon 探索时，按概率偏向 increase-only 动作。
        # - epsilon_increase_coverage：训练期 epsilon 探索时，按概率优先选择历史最少访问的 increase-only 动作。
        self.exploration_mode = str(config.exploration_mode)
        if self.exploration_mode not in {
            "epsilon_greedy",
            "epsilon_safe_increase_mixture",
            "epsilon_increase_coverage",
            "epsilon_plateau_soft_target_balanced",
        }:
            raise ValueError(f"unsupported exploration_mode: {self.exploration_mode}")
        self.safe_increase_explore_prob = float(config.safe_increase_explore_prob)
        if not 0.0 <= self.safe_increase_explore_prob <= 1.0:
            raise ValueError("safe_increase_explore_prob must be in [0, 1]")
        self.plateau_balanced_mix_prob = float(config.plateau_balanced_mix_prob)
        if not 0.0 <= self.plateau_balanced_mix_prob <= 1.0:
            raise ValueError("plateau_balanced_mix_prob must be in [0, 1]")
        self.q_network_type = str(config.q_network_type)
        self.action_feature_mode = str(config.action_feature_mode)
        self.action_aware_mask_mode = str(config.action_aware_mask_mode)
        self.action_feature_names = tuple(action_feature_names or ())
        self.action_feature_dim = 0
        self._action_features_tensor: torch.Tensor | None = None
        self._action_aware_selectable_mask: torch.Tensor | None = None
        if self.action_aware_mask_mode not in {"none", "increase_noop"}:
            raise ValueError(f"unsupported action_aware_mask_mode: {self.action_aware_mask_mode}")

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
        if self.q_network_type == "mlp":
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
        elif self.q_network_type == "action_aware":
            # action-aware 需要知道动作描述符维度以构建网络。
            # static_v1 一般直接传入固定 action_features；
            # dynamic_v1 可只传 feature_dim，运行时每步再 set_action_features。
            feature_dim = 0
            if action_features is not None:
                feature_dim = len(action_features[0]) if action_features and action_features[0] else 0
            elif action_feature_dim is not None:
                feature_dim = int(action_feature_dim)
            if feature_dim <= 0:
                raise ValueError("action_aware requires non-empty action feature dimension")
            self.policy_network = ActionAwareQNetwork(
                observation_dim=observation_dim,
                action_feature_dim=feature_dim,
                hidden_layers=self.hidden_layers,
            ).to(self.device)
            self.target_network = ActionAwareQNetwork(
                observation_dim=observation_dim,
                action_feature_dim=feature_dim,
                hidden_layers=self.hidden_layers,
            ).to(self.device)
        else:
            raise ValueError(f"unsupported q_network_type: {self.q_network_type}")
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        if action_features is not None:
            self.set_action_features(action_features, action_feature_names=action_feature_names)

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
        # safe increase 探索统计：
        # - exploration_safe_increase_action_count：通过 safe increase 分支采样成功次数；
        # - exploration_all_valid_action_count：走默认 all-valid/non-noop 采样次数；
        # - exploration_safe_increase_fallback_count：触发 safe increase 分支但无合法 increase 动作时回退次数。
        self.exploration_safe_increase_action_count = 0
        self.exploration_all_valid_action_count = 0
        self.exploration_safe_increase_fallback_count = 0
        # coverage-based increase exploration 统计：
        # - increase_exploration_visit_counts：仅记录 coverage 分支实际选中过的 increase-only 动作；
        # - exploration_increase_coverage_action_count：coverage 分支成功命中的次数；
        # - exploration_increase_coverage_tie_count：coverage 分支中出现并列最少候选的次数。
        self.increase_exploration_visit_counts: dict[int, int] = {
            int(action_id): 0 for action_id in self.increase_action_ids
        }
        self.exploration_increase_coverage_action_count = 0
        self.exploration_increase_coverage_tie_count = 0
        # plateau-triggered soft target-balanced exploration 状态：
        # - active_episodes_remaining > 0 时，训练期 epsilon 探索分支允许进入 balanced 采样；
        # - burst_count 记录一共触发了多少次 burst；
        # - action_count / fallback_count 用于训练日志观察 burst 实际效果。
        self.plateau_balanced_active_episodes_remaining = 0
        self.plateau_balanced_burst_count = 0
        self.plateau_balanced_action_count = 0
        self.plateau_balanced_fallback_count = 0

    def set_action_features(
        self,
        action_features: tuple[tuple[float, ...], ...],
        action_feature_names: tuple[str, ...] | None = None,
    ) -> None:
        """设置 action-aware 网络使用的静态动作描述符。"""

        if len(action_features) != self.action_dim:
            raise ValueError("action_features 行数必须等于 action_dim")
        if not action_features:
            raise ValueError("action_features 不能为空")
        feature_dim = len(action_features[0])
        if feature_dim <= 0:
            raise ValueError("action_features 的每行维度必须大于 0")
        if any(len(row) != feature_dim for row in action_features):
            raise ValueError("action_features 的每行长度必须一致")
        if action_feature_names is not None and len(action_feature_names) != feature_dim:
            raise ValueError("action_feature_names 长度必须与 action feature 维度一致")
        self.action_feature_dim = feature_dim
        self.action_feature_names = tuple(action_feature_names or ())
        self._action_features_tensor = torch.tensor(action_features, dtype=torch.float32, device=self.device)
        self._action_aware_selectable_mask = None
        if self._is_action_aware_increase_noop_mode():
            if not self.action_feature_names:
                raise ValueError("increase_noop mode requires action_feature_names")
            try:
                noop_idx = self.action_feature_names.index("is_noop")
                increase_idx = self.action_feature_names.index("is_increase")
            except ValueError as exc:
                raise ValueError("increase_noop mode requires is_noop/is_increase action features") from exc
            selectable = [
                bool(float(row[noop_idx]) >= 0.5 or float(row[increase_idx]) >= 0.5)
                for row in action_features
            ]
            self._action_aware_selectable_mask = torch.tensor(selectable, dtype=torch.bool, device=self.device)

    def _is_action_aware_increase_noop_mode(self) -> bool:
        """当前是否启用 action-aware increase+noop 诊断 mask。"""

        return self.q_network_type == "action_aware" and self.action_aware_mask_mode == "increase_noop"

    def _apply_action_aware_valid_mask(
        self,
        valid_action_mask: tuple[bool, ...] | None,
    ) -> tuple[bool, ...] | None:
        """在需要时，把动作合法掩码与 action-aware 诊断掩码相交。"""

        if not self._is_action_aware_increase_noop_mode():
            return valid_action_mask
        if self._action_aware_selectable_mask is None:
            raise RuntimeError("increase_noop mode requires action feature mask")
        mode_mask = tuple(bool(value) for value in self._action_aware_selectable_mask.detach().cpu().tolist())
        if valid_action_mask is None:
            return mode_mask
        return tuple(bool(a and b) for a, b in zip(valid_action_mask, mode_mask))

    def _apply_action_aware_next_valid_masks(self, next_valid_masks: torch.Tensor) -> torch.Tensor:
        """在训练 target 计算前应用 action-aware 诊断掩码。"""

        if not self._is_action_aware_increase_noop_mode():
            return next_valid_masks
        if self._action_aware_selectable_mask is None:
            raise RuntimeError("increase_noop mode requires action feature mask")
        return next_valid_masks & self._action_aware_selectable_mask.unsqueeze(0)

    def _network_q_values(
        self,
        network: nn.Module,
        states: torch.Tensor,
        action_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """统一处理两类 Q 网络的前向调用。"""

        if self.q_network_type == "mlp":
            return network(states)
        if self.q_network_type == "action_aware":
            features = action_features if action_features is not None else self._action_features_tensor
            if features is None:
                raise RuntimeError("action_aware network requires action features")
            assert isinstance(network, ActionAwareQNetwork)
            return network(states, features)
        raise ValueError(f"unsupported q_network_type: {self.q_network_type}")

    def _batch_action_features_tensor(
        self,
        batch: list[Transition],
        *,
        next_features: bool,
    ) -> torch.Tensor | None:
        """把 replay batch 中的动态动作特征拼成 [B, A, F] 张量。

        说明：
        - 仅在 `action_aware + dynamic_v1` 下需要该张量；
        - `static_v1` 继续复用 agent 内部固定特征矩阵，不从 transition 读取。
        """

        if self.q_network_type != "action_aware":
            return None
        if self.action_feature_mode == "static_v1":
            return None
        key = "next_action_features" if next_features else "action_features"
        matrices = [getattr(item, key) for item in batch]
        if any(matrix is None for matrix in matrices):
            raise RuntimeError(f"{self.action_feature_mode} requires {key} in replay transition")
        arr = np.asarray(matrices, dtype=np.float32)
        if arr.ndim != 3:
            raise RuntimeError(f"{key} must have shape [B,A,F]")
        if arr.shape[1] != self.action_dim:
            raise RuntimeError(f"{key} action_dim mismatch")
        if arr.shape[2] != self.action_feature_dim:
            raise RuntimeError(f"{key} feature_dim mismatch")
        return torch.from_numpy(arr).to(self.device)

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

        valid_action_mask = self._apply_action_aware_valid_mask(valid_action_mask)
        if valid_action_mask is None:
            return list(range(self.action_dim))
        if len(valid_action_mask) != self.action_dim:
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        return [action_id for action_id, is_valid in enumerate(valid_action_mask) if is_valid]

    def _valid_increase_action_ids(self, valid_action_ids: list[int]) -> list[int]:
        """从当前合法动作中筛出 increase-only 候选。"""

        return [
            candidate_action_id
            for candidate_action_id in valid_action_ids
            if candidate_action_id in self.increase_action_id_set
        ]

    def _sample_uniform_increase_exploration_action(self, valid_action_ids: list[int]) -> int | None:
        """在合法 increase-only 动作中均匀采样。"""

        valid_increase_action_ids = self._valid_increase_action_ids(valid_action_ids)
        if not valid_increase_action_ids:
            return None
        self.exploration_safe_increase_action_count += 1
        return int(self._rng.choice(valid_increase_action_ids))

    def _sample_coverage_increase_exploration_action(self, valid_action_ids: list[int]) -> int | None:
        """在合法 increase-only 动作中优先选择历史访问次数最少的动作。"""

        valid_increase_action_ids = self._valid_increase_action_ids(valid_action_ids)
        if not valid_increase_action_ids:
            return None

        # coverage 只统计当前仍然合法的 increase 动作，避免旧 checkpoint 或动作集合变化时
        # 出现键缺失。这里为所有候选补齐计数项，保证后续最小值比较口径一致。
        for action_id in valid_increase_action_ids:
            self.increase_exploration_visit_counts.setdefault(int(action_id), 0)

        min_count = min(
            self.increase_exploration_visit_counts.get(int(action_id), 0)
            for action_id in valid_increase_action_ids
        )
        least_visited_action_ids = [
            int(action_id)
            for action_id in valid_increase_action_ids
            if self.increase_exploration_visit_counts.get(int(action_id), 0) == min_count
        ]

        if len(least_visited_action_ids) > 1:
            self.exploration_increase_coverage_tie_count += 1

        action_id = int(self._rng.choice(least_visited_action_ids))
        self.increase_exploration_visit_counts[action_id] = (
            self.increase_exploration_visit_counts.get(action_id, 0) + 1
        )
        self.exploration_safe_increase_action_count += 1
        self.exploration_increase_coverage_action_count += 1
        return action_id

    def start_plateau_balanced_burst(self, burst_episodes: int, *, reset_counts: bool = False) -> None:
        """启动一个 plateau-triggered soft balanced exploration burst。

        该方法只负责打开 burst 状态，不改变 validation 或 greedy 选择逻辑。
        若已有 burst 正在进行，则保留更长的剩余 episode 数，避免新触发把旧 burst
        无意缩短。
        """

        if burst_episodes <= 0:
            return
        self.plateau_balanced_active_episodes_remaining = max(
            self.plateau_balanced_active_episodes_remaining,
            int(burst_episodes),
        )
        self.plateau_balanced_burst_count += 1
        if reset_counts:
            for action_id in list(self.increase_exploration_visit_counts.keys()):
                self.increase_exploration_visit_counts[int(action_id)] = 0

    def on_episode_end(self) -> None:
        """训练循环每个 episode 结束时调用，用于推进 burst 的剩余 episode 计数。"""

        if self.plateau_balanced_active_episodes_remaining > 0:
            self.plateau_balanced_active_episodes_remaining -= 1

    @property
    def plateau_balanced_is_active(self) -> bool:
        """当前是否处于 plateau-triggered balanced burst 中。"""

        return self.plateau_balanced_active_episodes_remaining > 0

    def _greedy_action_id(
        self,
        state_vector: tuple[float, ...],
        valid_action_mask: tuple[bool, ...] | None,
    ) -> int:
        """在给定掩码约束下选择 Q 值最大的合法动作。"""

        valid_action_mask = self._apply_action_aware_valid_mask(valid_action_mask)
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self._network_q_values(self.policy_network, state_tensor)[0]
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
            self.exploration_action_count += 1

            # 新模式只作用于训练期 epsilon 分支；greedy 与 validation(training=False) 完全不变。
            if self.exploration_mode == "epsilon_plateau_soft_target_balanced":
                should_try_balanced = (
                    self.plateau_balanced_is_active
                    and self.plateau_balanced_mix_prob > 0.0
                    and self._rng.random() < self.plateau_balanced_mix_prob
                )
                if should_try_balanced:
                    action_id = self._sample_coverage_increase_exploration_action(valid_action_ids)
                    if action_id is None:
                        # coverage-balanced 分支无合法 increase 动作时，严格回退到旧版探索逻辑。
                        self.plateau_balanced_fallback_count += 1
                        self.exploration_safe_increase_fallback_count += 1
                        action_id = self._sample_default_exploration_action(valid_action_ids)
                    else:
                        self.plateau_balanced_action_count += 1
                else:
                    action_id = self._sample_default_exploration_action(valid_action_ids)
            elif self.exploration_mode in {"epsilon_safe_increase_mixture", "epsilon_increase_coverage"}:
                should_try_safe_increase = (
                    self.safe_increase_explore_prob > 0.0
                    and self._rng.random() < self.safe_increase_explore_prob
                )
                if should_try_safe_increase:
                    if self.exploration_mode == "epsilon_increase_coverage":
                        action_id = self._sample_coverage_increase_exploration_action(valid_action_ids)
                    else:
                        action_id = self._sample_uniform_increase_exploration_action(valid_action_ids)
                    if action_id is None:
                        # 计划要求：safe 分支无合法 increase 候选时，回退到默认探索逻辑。
                        self.exploration_safe_increase_fallback_count += 1
                        action_id = self._sample_default_exploration_action(valid_action_ids)
                else:
                    action_id = self._sample_default_exploration_action(valid_action_ids)
            else:
                action_id = self._sample_default_exploration_action(valid_action_ids)
        else:
            action_id = self._greedy_action_id(state_vector, valid_action_mask)

        if training:
            self.epsilon_step += 1
            self.current_epsilon = self._compute_epsilon()
        return action_id

    def _sample_default_exploration_action(self, valid_action_ids: list[int]) -> int:
        """执行旧版 epsilon 探索采样逻辑（含 noop 优先），供多种探索模式复用。"""

        # 旧逻辑：进入 epsilon 探索后，先尝试按 noop_exploration_prob 选择显式 noop。
        should_pick_noop = (
            self.noop_action_id is not None
            and self.noop_action_id in valid_action_ids
            and self.noop_exploration_prob > 0.0
            and self._rng.random() < self.noop_exploration_prob
        )
        if should_pick_noop:
            self.exploration_noop_action_count += 1
            return int(self.noop_action_id)

        # 未选 noop 时，从非 noop 合法动作均匀采样；若集合为空则回退到全部合法动作。
        non_noop_valid_action_ids = [
            candidate_action_id
            for candidate_action_id in valid_action_ids
            if candidate_action_id != self.noop_action_id
        ]
        self.exploration_all_valid_action_count += 1
        return int(self._rng.choice(non_noop_valid_action_ids or valid_action_ids))

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
        next_valid_masks = self._apply_action_aware_next_valid_masks(next_valid_masks)

        state_action_features = self._batch_action_features_tensor(batch, next_features=False)
        next_action_features = self._batch_action_features_tensor(batch, next_features=True)
        policy_q = self._network_q_values(
            self.policy_network,
            states,
            action_features=state_action_features,
        ).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN target 第一步：policy network 只负责“选动作”。
                # 这里对 next state 的 policy Q 值套用 next_valid_masks，把非法动作置为 -inf，
                # 确保 argmax 只会在当前环境允许的动作集合中产生 greedy action。
                next_policy_q_values = self._network_q_values(
                    self.policy_network,
                    next_states,
                    action_features=next_action_features,
                )
                next_policy_q_values = next_policy_q_values.masked_fill(
                    ~next_valid_masks,
                    float("-inf"),
                )
                next_actions = next_policy_q_values.argmax(dim=1, keepdim=True)

                # Double DQN target 第二步：target network 只负责“评估动作”。
                # gather 的列索引来自 policy network 选出的 next_actions，因此 bootstrap
                # 使用的是 Q_target(s', argmax_a Q_policy(s', a))，而不是标准 DQN 的
                # max_a Q_target(s', a)，从而降低大动作空间下的 max over actions 高估。
                next_target_q_values = self._network_q_values(
                    self.target_network,
                    next_states,
                    action_features=next_action_features,
                )
                next_q = next_target_q_values.gather(1, next_actions).squeeze(1)
                has_any_valid_action = next_valid_masks.any(dim=1)
                next_q = torch.where(has_any_valid_action, next_q, torch.zeros_like(next_q))
            else:
                # 标准 DQN 对照分支：保持原实现语义，直接在 target network 输出上屏蔽非法动作，
                # 然后取合法动作中的最大 Q 值作为 bootstrap target。
                next_q_values = self._network_q_values(
                    self.target_network,
                    next_states,
                    action_features=next_action_features,
                )
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
        q_values = self._network_q_values(self.policy_network, states)
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
                "increase_action_ids": self.increase_action_ids,
                "increase_exploration_visit_counts": dict(self.increase_exploration_visit_counts),
                "exploration_increase_coverage_action_count": self.exploration_increase_coverage_action_count,
                "exploration_increase_coverage_tie_count": self.exploration_increase_coverage_tie_count,
                "plateau_balanced_active_episodes_remaining": self.plateau_balanced_active_episodes_remaining,
                "plateau_balanced_burst_count": self.plateau_balanced_burst_count,
                "plateau_balanced_action_count": self.plateau_balanced_action_count,
                "plateau_balanced_fallback_count": self.plateau_balanced_fallback_count,
                "double_dqn": self.double_dqn,
                "q_network_type": self.q_network_type,
                "action_feature_mode": self.action_feature_mode,
                "action_aware_mask_mode": self.action_aware_mask_mode,
                "action_feature_names": self.action_feature_names,
                "action_feature_dim": self.action_feature_dim,
                # dynamic_v1 的动作特征是“状态相关”的，不能把 checkpoint 中的最后一帧
                # 当作固定特征复用；因此仅 static_v1 持久化固定 action_features。
                "action_features": (
                    self._action_features_tensor.detach().cpu().tolist()
                    if self._action_features_tensor is not None and self.action_feature_mode == "static_v1"
                    else None
                ),
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
        config_dict = dict(checkpoint["config"])
        config_dict.setdefault("q_network_type", checkpoint.get("q_network_type", "mlp"))
        config_dict.setdefault("action_feature_mode", checkpoint.get("action_feature_mode", "static_v1"))
        config_dict.setdefault("action_aware_mask_mode", checkpoint.get("action_aware_mask_mode", "none"))
        config = DqnConfig(**config_dict)
        checkpoint_action_feature_mode = str(checkpoint.get("action_feature_mode", config.action_feature_mode))
        checkpoint_increase_action_ids = checkpoint.get("increase_action_ids")
        increase_action_ids = (
            tuple(int(action_id) for action_id in checkpoint_increase_action_ids)
            if checkpoint_increase_action_ids is not None
            else None
        )
        checkpoint_action_features = checkpoint.get("action_features")
        if checkpoint_action_feature_mode != "static_v1":
            checkpoint_action_features = None
        checkpoint_action_features_tuple = (
            tuple(tuple(float(value) for value in row) for row in checkpoint_action_features)
            if checkpoint_action_features is not None
            else None
        )
        checkpoint_action_feature_names = checkpoint.get("action_feature_names")
        agent = cls(
            observation_dim=int(checkpoint["observation_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=config,
            noop_action_id=checkpoint.get("noop_action_id"),
            increase_action_ids=increase_action_ids,
            hidden_layers=tuple(checkpoint["hidden_layers"]) if checkpoint["hidden_layers"] is not None else None,
            device=str(resolved_device),
            double_dqn=bool(checkpoint.get("double_dqn", True)),
            action_features=checkpoint_action_features_tuple,
            action_feature_names=(
                tuple(str(value) for value in checkpoint_action_feature_names)
                if checkpoint_action_feature_names is not None
                else None
            ),
            action_feature_dim=int(checkpoint.get("action_feature_dim", 0) or 0),
        )
        agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.optimization_steps = int(checkpoint["optimization_steps"])
        agent.epsilon_step = int(checkpoint["epsilon_step"])
        agent.current_epsilon = float(checkpoint["current_epsilon"])
        raw_visit_counts = checkpoint.get("increase_exploration_visit_counts")
        if raw_visit_counts is not None:
            agent.increase_exploration_visit_counts = {
                int(action_id): int(count) for action_id, count in raw_visit_counts.items()
            }
            for action_id in agent.increase_action_ids:
                agent.increase_exploration_visit_counts.setdefault(int(action_id), 0)
        else:
            agent.increase_exploration_visit_counts = {
                int(action_id): 0 for action_id in agent.increase_action_ids
            }
        agent.exploration_increase_coverage_action_count = int(
            checkpoint.get("exploration_increase_coverage_action_count", 0)
        )
        agent.exploration_increase_coverage_tie_count = int(
            checkpoint.get("exploration_increase_coverage_tie_count", 0)
        )
        agent.plateau_balanced_active_episodes_remaining = int(
            checkpoint.get("plateau_balanced_active_episodes_remaining", 0)
        )
        agent.plateau_balanced_burst_count = int(checkpoint.get("plateau_balanced_burst_count", 0))
        agent.plateau_balanced_action_count = int(checkpoint.get("plateau_balanced_action_count", 0))
        agent.plateau_balanced_fallback_count = int(checkpoint.get("plateau_balanced_fallback_count", 0))
        return agent
