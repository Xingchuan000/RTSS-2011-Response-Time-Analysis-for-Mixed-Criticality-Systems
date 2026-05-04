"""DQN 经验回放池测试。"""

from __future__ import annotations

from amc_py.dqn.replay import ReplayBuffer
from amc_py.dqn.types import Transition


def _transition(idx: int) -> Transition:
    """构造便于断言的测试样本。"""

    return Transition(
        state=(float(idx), float(idx + 1)),
        action_id=idx % 3,
        reward=float(idx),
        next_state=(float(idx + 2), float(idx + 3)),
        done=False,
        valid_action_mask=(True, True, True),
        next_valid_action_mask=(True, True, True),
    )


def test_push_increases_length() -> None:
    """push 后长度应增加。"""

    replay = ReplayBuffer(capacity=3, seed=0)
    replay.push(_transition(0))
    assert len(replay) == 1


def test_length_never_exceeds_capacity() -> None:
    """超过容量后长度不应继续增长。"""

    replay = ReplayBuffer(capacity=2, seed=0)
    replay.push(_transition(0))
    replay.push(_transition(1))
    replay.push(_transition(2))
    assert len(replay) == 2


def test_sample_returns_requested_batch_size() -> None:
    """sample 应返回指定数量的样本。"""

    replay = ReplayBuffer(capacity=5, seed=0)
    for idx in range(4):
        replay.push(_transition(idx))
    batch = replay.sample(3)
    assert len(batch) == 3


def test_fixed_seed_makes_sampling_reproducible() -> None:
    """固定种子时采样结果应可复现。"""

    replay_a = ReplayBuffer(capacity=5, seed=7)
    replay_b = ReplayBuffer(capacity=5, seed=7)
    for idx in range(5):
        item = _transition(idx)
        replay_a.push(item)
        replay_b.push(item)

    sample_a = replay_a.sample(3)
    sample_b = replay_b.sample(3)
    assert sample_a == sample_b
