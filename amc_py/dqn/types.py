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
