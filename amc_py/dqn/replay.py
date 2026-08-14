"""DQN 经验回放池实现。"""

from __future__ import annotations

from collections import deque
import random

from amc_py.dqn.types import Transition


class NStepTransitionAccumulator:
    """把连续 1-step transition 聚合为 n-step replay transition。

    聚合规则：
    - 保留第一个 transition 的 state/action/current mask/current action features；
    - reward 使用 ``r0 + gamma*r1 + ...``；
    - next_state / next mask / next action features 取聚合窗口最后一个 transition；
    - 非 terminal 情况达到 n 步后发射一条样本；
    - episode terminal 时把不足 n 步的尾部样本全部 flush，且保持 done=True。

    ``n_step=1`` 时严格退化为原始 transition 流，不改变既有训练口径。
    """

    def __init__(self, n_step: int, gamma: float):
        if n_step <= 0:
            raise ValueError("n_step 必须为正整数")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma 必须在 (0, 1] 内")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._pending: deque[Transition] = deque()

    def append(self, transition: Transition) -> list[Transition]:
        """加入一条 1-step transition，并返回本次可写入 replay 的聚合样本。"""

        self._pending.append(transition)
        emitted: list[Transition] = []

        if transition.done:
            # terminal 时所有 pending 前缀都有完整的 episode 结局，可全部 flush。
            while self._pending:
                emitted.append(self._aggregate_prefix(min(self.n_step, len(self._pending))))
                self._pending.popleft()
            return emitted

        if len(self._pending) >= self.n_step:
            emitted.append(self._aggregate_prefix(self.n_step))
            self._pending.popleft()
        return emitted

    def _aggregate_prefix(self, length: int) -> Transition:
        if length <= 0 or length > len(self._pending):
            raise ValueError("invalid n-step prefix length")
        items = list(self._pending)[:length]
        first = items[0]
        last = items[-1]
        discounted_reward = 0.0
        discount = 1.0
        for item in items:
            discounted_reward += discount * float(item.reward)
            discount *= self.gamma
        return Transition(
            state=first.state,
            action_id=first.action_id,
            reward=float(discounted_reward),
            next_state=last.next_state,
            done=bool(last.done),
            valid_action_mask=first.valid_action_mask,
            next_valid_action_mask=last.next_valid_action_mask,
            action_features=first.action_features,
            next_action_features=last.next_action_features,
            bootstrap_steps=int(length),
        )

    @property
    def pending_count(self) -> int:
        """返回尚未形成 replay transition 的尾部 1-step 样本数。"""

        return len(self._pending)


class ReplayBuffer:
    """使用固定容量 ring buffer 保存 transition，并支持可复现采样。"""

    def __init__(self, capacity: int, seed: int | None = None):
        """初始化回放池。

        这里严格按固定容量 ring buffer 的方式组织内部存储：
        - `self._buffer` 只保存当前仍然有效的 transition；
        - `self._position` 指向“下一次写入时应覆盖的位置”；
        - 当回放池尚未装满时，`push()` 直接追加；
        - 当回放池已经装满时，`push()` 覆盖最旧样本。

        这样做的核心目的，是让 `sample()` 可以直接在内部 `list` 上执行
        `random.sample()`，避免每次采样前都把整个缓冲区从 `deque` 复制成新 `list`。
        """

        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        # 对外暴露的容量语义保持不变：最多只能保留 `capacity` 条 transition。
        self.capacity = int(capacity)
        # `_buffer` 直接使用 Python `list` 保存有效样本。
        # 当回放池未满时，列表长度就是当前真实样本数；
        # 当回放池已满时，列表长度固定等于 `capacity`。
        self._buffer: list[Transition] = []
        # `_position` 记录下一次写入的位置。
        # 它会沿着 `[0, capacity)` 循环前进，从而形成 ring buffer。
        self._position = 0
        # 独立随机数生成器用于保证固定 seed 下采样结果可复现，
        # 同时避免与其他随机流程共用全局随机状态。
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        """写入一条 transition。

        写入规则严格遵循 ring buffer 语义：
        - 若当前样本数还未达到容量上限，则直接追加到列表尾部；
        - 若当前样本数已经达到容量上限，则覆盖 `_position` 指向的旧样本；
        - 无论追加还是覆盖，写入完成后都将 `_position` 向前推进一格。

        由于 `_position` 会按容量取模循环，因此被覆盖的总是“最早应该淘汰”的样本。
        """

        # 未满时继续扩容，保证已有样本都能被保留下来。
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            # 已满时原地覆盖旧样本，避免任何额外容器复制或重排。
            self._buffer[self._position] = transition
        # 写入指针循环前进，指向下一次应写入的位置。
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        """随机采样一个 batch。

        这里直接对内部 `self._buffer` 做无放回均匀采样，严格保持原有采样语义，
        但不再在采样前复制整个缓冲区，从而消除额外的 CPU 与内存开销。
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"回放池样本不足：当前仅有 {len(self._buffer)} 条，无法采样 {batch_size} 条"
            )
        # 直接在内部有效样本列表上采样，避免每次 `sample()` 都复制整个缓冲区。
        return self._rng.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        """返回当前有效样本数量。"""

        # 由于 `_buffer` 只保存当前仍有效的 transition，因此其长度就是外部应看到的长度。
        return len(self._buffer)
