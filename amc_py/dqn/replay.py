"""DQN 经验回放池实现。"""

from __future__ import annotations

from collections import deque
import random

from amc_py.dqn.types import Transition


class ReplayBuffer:
    """使用固定容量队列保存 transition，并支持可复现采样。"""

    def __init__(self, capacity: int, seed: int | None = None):
        """初始化回放池。"""

        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        """追加一条 transition；若容量已满则自动丢弃最旧样本。"""

        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """随机采样一个 batch；样本不足时抛出明确异常。"""

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"回放池样本不足：当前仅有 {len(self._buffer)} 条，无法采样 {batch_size} 条"
            )
        return self._rng.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        """返回当前样本数量。"""

        return len(self._buffer)
