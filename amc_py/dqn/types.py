"""DQN 训练过程中的基础数据类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transition:
    """单条状态转移样本。"""

    # 当前决策时刻的观测向量。
    state: tuple[float, ...]
    # 执行的离散动作编号。
    action_id: int
    # 本步即时奖励。
    reward: float
    # 下一决策时刻的观测向量。
    next_state: tuple[float, ...]
    # 是否已经到达 episode 终点。
    done: bool
    # 当前状态下各离散动作是否合法（与 action_dim 等长）。
    valid_action_mask: tuple[bool, ...]
    # 下一状态下各离散动作是否合法（与 action_dim 等长）。
    next_valid_action_mask: tuple[bool, ...]
