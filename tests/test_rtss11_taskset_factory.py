"""RTSS2011 任务集工厂测试。"""

from __future__ import annotations

import pytest

from amc_py.dqn import build_rtss11_taskset, build_schedulable_rtss11_taskset
from amc_py.generator import taskset_total_util
from amc_py.models import Criticality, SchedulabilityResult, Task


def _serialize_taskset(tasks: list[Task]) -> list[tuple[str, int, int, int, int, str]]:
    """把任务集序列化为稳定可比较结构。"""

    return [
        (task.name, task.period, task.deadline, task.c_lo, task.c_hi, task.criticality.value)
        for task in tasks
    ]


def test_rtss11_taskset_is_deterministic() -> None:
    """同一 seed 与参数组合应生成完全一致的任务集。"""

    ts1 = build_rtss11_taskset(seed=1, total_util=0.65)
    ts2 = build_rtss11_taskset(seed=1, total_util=0.65)
    assert _serialize_taskset(ts1) == _serialize_taskset(ts2)


def test_rtss11_taskset_has_expected_number_of_tasks() -> None:
    """任务数量应等于调用参数。"""

    tasks = build_rtss11_taskset(seed=1, total_util=0.65, num_tasks=20)
    assert len(tasks) == 20


def test_rtss11_taskset_task_parameters_are_valid() -> None:
    """任务参数应满足模型约束。"""

    tasks = build_rtss11_taskset(seed=2, total_util=0.65, num_tasks=20)
    for task in tasks:
        assert task.period > 0
        assert task.deadline == task.period
        assert task.c_lo > 0
        if task.criticality is Criticality.HI:
            assert task.c_hi >= task.c_lo
        # 现有 Task 模型要求所有任务都提供 c_hi，因此这里不对 LO 任务增加额外约束。


def test_rtss11_taskset_total_util_matches_target_approximately() -> None:
    """LO 模式总利用率应与目标利用率近似一致。"""

    target = 0.65
    tasks = build_rtss11_taskset(seed=3, total_util=target, num_tasks=20)
    actual = taskset_total_util(tasks, mode=Criticality.LO)
    assert abs(actual - target) <= 0.01


def test_build_schedulable_rtss11_taskset_returns_schedulable_set() -> None:
    """可调度工厂应返回 AMC-rtb 可调度任务集。"""

    bundle = build_schedulable_rtss11_taskset(seed=1, total_util=0.55, max_attempts=100)
    assert bundle.analysis.schedulable is True
    assert bundle.attempts >= 1
    assert len(bundle.tasks) == 20


def test_build_schedulable_rtss11_taskset_raises_clear_error_when_all_attempts_fail(monkeypatch) -> None:
    """当所有尝试都不可调度时，应抛出清晰异常。"""

    def _always_unschedulable(tasks, method, priority_policy):
        return SchedulabilityResult(
            schedulable=False,
            method="amc_rtb",
            response_times={},
            details=f"forced_failure method={method} policy={priority_policy}",
        )

    monkeypatch.setattr("amc_py.dqn.experiment.evaluate_taskset", _always_unschedulable)

    with pytest.raises(RuntimeError, match="max_attempts=3"):
        build_schedulable_rtss11_taskset(seed=7, total_util=0.55, max_attempts=3)


def test_build_schedulable_rtss11_taskset_returns_actual_used_seed() -> None:
    """返回结果中的 seed 应是实际命中的 `seed + offset`。"""

    base_seed = 9
    bundle = build_schedulable_rtss11_taskset(seed=base_seed, total_util=0.55, max_attempts=100)
    assert bundle.seed >= base_seed
    assert bundle.attempts == bundle.seed - base_seed + 1
