"""DQN 经验回放池测试。"""

from __future__ import annotations

import pytest

from amc_py.dqn.replay import ReplayBuffer
from amc_py.dqn.types import Transition


def _transition(idx: int) -> Transition:
    """构造便于断言的测试样本。

    这里让 `action_id` 直接等于传入的 `idx`，这样在覆盖旧样本的测试里，
    我们可以直接通过 `action_id` 判断哪些 transition 仍然留在回放池中。
    """

    return Transition(
        state=(float(idx), float(idx + 1)),
        # 直接保留唯一编号，方便在测试中检查样本是否被覆盖淘汰。
        action_id=idx,
        reward=float(idx),
        next_state=(float(idx + 2), float(idx + 3)),
        done=False,
        # 掩码长度只需与测试里的离散动作编号范围兼容即可。
        valid_action_mask=(True, True, True, True, True, True),
        next_valid_action_mask=(True, True, True, True, True, True),
    )


def test_push_increases_length() -> None:
    """当回放池未满时，push 后长度应立即增加。"""

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
    """sample 应返回请求的 batch 大小。"""

    replay = ReplayBuffer(capacity=5, seed=0)
    for idx in range(4):
        replay.push(_transition(idx))
    batch = replay.sample(3)
    assert len(batch) == 3


def test_replay_buffer_overwrites_oldest_transition() -> None:
    """容量写满后，后续写入应覆盖最老的 transition。"""

    replay = ReplayBuffer(capacity=3, seed=123)
    for idx in range(5):
        replay.push(_transition(idx))

    sampled = replay.sample(3)
    sampled_ids = {item.action_id for item in sampled}
    # 容量为 3 时连续写入 0,1,2,3,4，理论上只应保留最新的 2,3,4。
    assert sampled_ids == {2, 3, 4}


def test_replay_buffer_samples_from_partial_buffer() -> None:
    """当回放池未满时，采样结果只能来自当前已有样本。"""

    replay = ReplayBuffer(capacity=5, seed=123)
    replay.push(_transition(0))
    replay.push(_transition(1))
    replay.push(_transition(2))

    sampled = replay.sample(2)
    sampled_ids = {item.action_id for item in sampled}
    assert len(sampled) == 2
    # 这里只允许从 0、1、2 这三条已写入样本中采样，不能出现任何未写入编号。
    assert sampled_ids.issubset({0, 1, 2})


def test_replay_buffer_rejects_invalid_batch_size() -> None:
    """非法 batch_size 应抛出 ValueError。"""

    replay = ReplayBuffer(capacity=3, seed=123)
    replay.push(_transition(0))

    # batch_size 为 0 没有实际意义，必须明确拒绝。
    with pytest.raises(ValueError):
        replay.sample(0)

    # 负数 batch_size 同样属于调用方错误，也必须显式报错。
    with pytest.raises(ValueError):
        replay.sample(-1)

    # 当前只有 1 条样本，要求采样 2 条时应报告样本不足。
    with pytest.raises(ValueError):
        replay.sample(2)


def test_fixed_seed_makes_sampling_reproducible() -> None:
    """固定种子时采样结果应可复现。"""

    replay_a = ReplayBuffer(capacity=10, seed=42)
    replay_b = ReplayBuffer(capacity=10, seed=42)
    for idx in range(10):
        item = _transition(idx)
        replay_a.push(item)
        replay_b.push(item)

    sample_a = replay_a.sample(5)
    sample_b = replay_b.sample(5)
    # 两个回放池在写入顺序和随机种子完全一致时，采样结果也应完全一致。
    assert sample_a == sample_b
