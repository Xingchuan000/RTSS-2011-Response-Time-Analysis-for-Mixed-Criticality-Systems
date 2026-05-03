"""RTSS2011 随机运行时场景测试。"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.runtime_scenarios import make_rtss11_random_scenario


def _reference_tasks() -> list[Task]:
    """构造包含 HI/LO 任务的参考任务集。"""

    return [
        Task("H1", period=1000, deadline=1000, c_lo=40, c_hi=80, criticality=Criticality.HI),
        Task("H2", period=1500, deadline=1500, c_lo=50, c_hi=100, criticality=Criticality.HI),
        Task("L1", period=1200, deadline=1200, c_lo=30, c_hi=30, criticality=Criticality.LO),
        Task("L2", period=1800, deadline=1800, c_lo=20, c_hi=20, criticality=Criticality.LO),
    ]


def _serialize_samples(tasks: list[Task], seed: int, horizon: int = 50) -> list[tuple[str, int, int]]:
    """把场景采样结果序列化为可直接比较的结构。"""

    scenario = make_rtss11_random_scenario(tasks=tasks, seed=seed)
    rows: list[tuple[str, int, int]] = []
    for task in tasks:
        for release_index in range(horizon):
            rows.append((task.name, release_index, scenario.actual_cost_for(task, release_index)))
    return rows


def test_rtss11_random_scenario_is_deterministic() -> None:
    """同一任务集与 seed 下应得到完全一致的执行时间样本。"""

    tasks = _reference_tasks()
    s1 = _serialize_samples(tasks, seed=7, horizon=80)
    s2 = _serialize_samples(tasks, seed=7, horizon=80)
    assert s1 == s2


def test_rtss11_random_scenario_hi_actual_cost_never_exceeds_c_hi() -> None:
    """HI 任务样本必须始终满足 actual_cost <= C_HI。"""

    tasks = _reference_tasks()
    scenario = make_rtss11_random_scenario(tasks=tasks, seed=11, hi_overrun_prob=0.2)
    hi_tasks = [task for task in tasks if task.criticality is Criticality.HI]

    for task in hi_tasks:
        for release_index in range(1000):
            actual_cost = scenario.actual_cost_for(task, release_index)
            assert actual_cost <= task.c_hi


def test_rtss11_random_scenario_contains_overrun_samples() -> None:
    """在固定 seed 与足够样本数下应观测到部分 actual_cost > C_LO。"""

    tasks = _reference_tasks()
    scenario = make_rtss11_random_scenario(
        tasks=tasks,
        seed=23,
        hi_overrun_prob=0.3,
        lo_overrun_prob=0.3,
        lo_overrun_factor=1.8,
    )

    overrun_found = False
    for task in tasks:
        for release_index in range(300):
            if scenario.actual_cost_for(task, release_index) > task.c_lo:
                overrun_found = True
                break
        if overrun_found:
            break
    assert overrun_found is True


def test_rtss11_random_scenario_rejects_invalid_parameters() -> None:
    """非法参数应在场景构造阶段明确报错。"""

    tasks = _reference_tasks()
    with pytest.raises(ValueError, match="hi_overrun_prob"):
        make_rtss11_random_scenario(tasks=tasks, seed=0, hi_overrun_prob=-0.1)
    with pytest.raises(ValueError, match="hi_overrun_prob"):
        make_rtss11_random_scenario(tasks=tasks, seed=0, hi_overrun_prob=1.1)
    with pytest.raises(ValueError, match="lo_overrun_prob"):
        make_rtss11_random_scenario(tasks=tasks, seed=0, lo_overrun_prob=-0.1)
    with pytest.raises(ValueError, match="lo_overrun_prob"):
        make_rtss11_random_scenario(tasks=tasks, seed=0, lo_overrun_prob=1.1)
    with pytest.raises(ValueError, match="lo_overrun_factor"):
        make_rtss11_random_scenario(tasks=tasks, seed=0, lo_overrun_factor=1.0)
