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


class TaskwiseDqnNetwork(nn.Module):
    """按“任务级特征 + 全局特征”结构化建模 single 动作空间的 Q 网络。

    第一版实现边界严格按任务文档执行：
    - 仅支持 `single` 动作空间；
    - 仅支持显式 noop；
    - 输出顺序必须是 `[increase_0..N-1, decrease_0..N-1, noop]`；
    - 输入状态必须是 `task_count * per_task_feature_dim + global_feature_dim`。
    """

    def __init__(
        self,
        *,
        task_count: int,
        per_task_feature_dim: int,
        global_feature_dim: int,
        action_dim: int,
        noop_action_id: int,
        task_hidden_dim: int = 64,
        global_hidden_dim: int = 64,
        joint_hidden_dim: int = 64,
        use_task_embedding: bool = False,
        task_embedding_dim: int = 8,
        use_action_bias: bool = False,
        action_bias_init: float = 0.0,
    ) -> None:
        """构建 taskwise DQN 网络。

        参数约束全部在构造阶段一次性校验，避免训练过程中才暴露结构不匹配问题。
        这里不提供对其他动作空间/观测模式的兼容分支，确保第一版行为边界清晰。
        """

        super().__init__()
        if task_count <= 0:
            raise ValueError("task_count 必须为正整数")
        if per_task_feature_dim <= 0:
            raise ValueError("per_task_feature_dim 必须为正整数")
        if global_feature_dim <= 0:
            raise ValueError("global_feature_dim 必须为正整数")
        if task_hidden_dim <= 0 or global_hidden_dim <= 0 or joint_hidden_dim <= 0:
            raise ValueError("taskwise hidden dim 必须为正整数")
        if use_task_embedding and task_embedding_dim <= 0:
            raise ValueError("task_embedding_dim 必须为正整数")
        expected_action_dim = 2 * task_count + 1
        expected_noop_action_id = 2 * task_count
        if action_dim != expected_action_dim:
            raise ValueError(
                "TaskwiseDqnNetwork 仅支持 single+explicit noop 动作顺序，"
                f"期望 action_dim={expected_action_dim}，实际收到 {action_dim}"
            )
        if noop_action_id != expected_noop_action_id:
            raise ValueError(
                "TaskwiseDqnNetwork 仅支持 noop 位于最后一个动作槽位，"
                f"期望 noop_action_id={expected_noop_action_id}，实际收到 {noop_action_id}"
            )

        self.task_count = task_count
        self.per_task_feature_dim = per_task_feature_dim
        self.global_feature_dim = global_feature_dim
        self.action_dim = action_dim
        self.noop_action_id = noop_action_id
        self.expected_input_dim = task_count * per_task_feature_dim + global_feature_dim
        # taskwise-v2 的两个可选开关都显式记录在网络对象上，便于 save/load 后做结构自检。
        self.use_task_embedding = bool(use_task_embedding)
        self.task_embedding_dim = int(task_embedding_dim) if self.use_task_embedding else 0
        self.use_action_bias = bool(use_action_bias)

        if self.use_task_embedding:
            # 任务编号 embedding 只按当前固定 taskset 的槽位数构建；
            # 本实现不做跨 task_count 迁移或动态扩缩容。
            self.task_embedding = nn.Embedding(task_count, self.task_embedding_dim)
        else:
            self.task_embedding = None

        task_encoder_input_dim = per_task_feature_dim + self.task_embedding_dim

        # task encoder 对每个任务共享参数，强制网络复用“任务局部特征 -> 动作价值”的表达。
        self.task_encoder = nn.Sequential(
            nn.Linear(task_encoder_input_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Linear(task_hidden_dim, task_hidden_dim),
            nn.ReLU(),
        )
        # global encoder 只编码全局上下文，供所有任务动作共享。
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, global_hidden_dim),
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        # increase / decrease 头分离，允许网络学习“同一任务在增预算与减预算下的不同价值函数”。
        self.inc_head = nn.Sequential(
            nn.Linear(task_hidden_dim + global_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )
        self.dec_head = nn.Sequential(
            nn.Linear(task_hidden_dim + global_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )
        # noop 只依赖全局状态，不绑定具体任务。
        self.noop_head = nn.Sequential(
            nn.Linear(global_hidden_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 1),
        )
        if self.use_action_bias:
            # 为每个固定动作槽位提供一个独立可学习偏置，
            # 用来补回共享 taskwise 结构中缺失的 per-action 常数项表达能力。
            self.action_bias = nn.Parameter(torch.full((action_dim,), float(action_bias_init)))
        else:
            self.register_parameter("action_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入必须是 `[batch_size, state_dim]`。"""

        if x.ndim != 2:
            raise ValueError("TaskwiseDqnNetwork expects input shape [batch_size, state_dim]")
        if x.shape[1] != self.expected_input_dim:
            raise ValueError(
                "TaskwiseDqnNetwork 输入维度不匹配，"
                f"期望 {self.expected_input_dim}，实际收到 {x.shape[1]}"
            )

        batch_size = x.shape[0]
        task_feature_total_dim = self.task_count * self.per_task_feature_dim
        # 状态向量前半段是按任务顺序平铺的任务级特征，后半段是共享全局特征。
        task_flat = x[:, :task_feature_total_dim]
        global_features = x[:, task_feature_total_dim:]
        task_features = task_flat.reshape(batch_size, self.task_count, self.per_task_feature_dim)
        if self.use_task_embedding:
            # task id 必须放在与输入相同的 device 上，否则在 CPU/MPS/GPU 混合场景下会报错。
            task_ids = torch.arange(self.task_count, device=x.device)
            task_id_embeddings = self.task_embedding(task_ids)
            task_id_embeddings = task_id_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            # 把“局部观测特征”与“固定任务身份”拼接后再送入共享 task encoder，
            # 这样模型就能在共享参数框架下记住不同任务槽位的个性化价值。
            task_features = torch.cat([task_features, task_id_embeddings], dim=-1)

        # Linear 支持对最后一维做投影，因此这里可以直接对 [B, N, D] 逐任务编码。
        task_embeddings = self.task_encoder(task_features)
        global_embedding = self.global_encoder(global_features)
        # 将同一条样本的全局表示广播到每个任务槽位，与任务表示拼接后分别打分。
        global_expanded = global_embedding.unsqueeze(1).expand(-1, self.task_count, -1)
        joint_embedding = torch.cat([task_embeddings, global_expanded], dim=-1)

        q_increase = self.inc_head(joint_embedding).squeeze(-1)
        q_decrease = self.dec_head(joint_embedding).squeeze(-1)
        q_noop = self.noop_head(global_embedding)
        # 输出顺序必须与 single 动作空间完全一致，不能做任何重排或兼容性兜底。
        q_values = torch.cat([q_increase, q_decrease, q_noop], dim=1)
        if self.use_action_bias:
            q_values = q_values + self.action_bias.view(1, -1)
        return q_values
