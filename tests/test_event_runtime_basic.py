"""事件驱动 runtime 基础测试（阶段二）。"""

from __future__ import annotations

import pytest

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, SimulationResult, SystemMode
from amc_py.runtime_scenarios import make_nominal_scenario


def _lo(name: str, period: int, c_lo: int, *, deadline: int | None = None) -> Task:
    """构造 LO 任务；若未指定 deadline，则默认 D=T。"""

    return Task(
        name=name,
        period=period,
        deadline=deadline if deadline is not None else period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_event_runtime_releases_periodic_jobs() -> None:
    """事件驱动版本应能按周期持续释放 job。"""

    tasks = [_lo("t1", period=3, c_lo=1)]
    scenario = make_nominal_scenario()
    result = simulate_ordered_taskset_event_driven(
        tasks,
        scenario,
        config=RuntimeConfig(end_time=10),
    )

    # 在 end_time=10 且 period=3 时，应释放在 t=0/3/6/9 的四个 job。
    jobs = result.jobs_of("t1")
    assert len(jobs) == 4
    assert [job.release_index for job in jobs] == [0, 1, 2, 3]
    assert [job.release_time for job in jobs] == [0, 3, 6, 9]


def test_event_runtime_completes_single_job() -> None:
    """单任务单作业场景下，job 应被正确完成并写入 completion_time。"""

    tasks = [_lo("single", period=100, c_lo=4)]
    scenario = make_nominal_scenario()
    result = simulate_ordered_taskset_event_driven(
        tasks,
        scenario,
        config=RuntimeConfig(end_time=10),
    )

    jobs = result.jobs_of("single")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.executed_time == 4
    assert job.completion_time == 4
    assert not job.dropped


def test_event_runtime_returns_simulation_result() -> None:
    """返回值类型和核心字段应符合现有 SimulationResult 约定。"""

    tasks = [_lo("t1", period=5, c_lo=1)]
    scenario = make_nominal_scenario()
    result = simulate_ordered_taskset_event_driven(
        tasks,
        scenario,
        config=RuntimeConfig(end_time=6),
    )

    assert isinstance(result, SimulationResult)
    assert result.final_mode is SystemMode.LO
    assert result.end_time <= 6


def test_event_runtime_rejects_empty_taskset() -> None:
    """空任务集输入必须直接报错。"""

    with pytest.raises(ValueError, match="不能为空"):
        simulate_ordered_taskset_event_driven([], make_nominal_scenario())


def test_event_runtime_rejects_duplicate_task_names() -> None:
    """任务名重复会破坏 job 键唯一性，必须拒绝。"""

    tasks = [
        _lo("dup", period=5, c_lo=1),
        _lo("dup", period=7, c_lo=1),
    ]
    with pytest.raises(ValueError, match="重复任务名"):
        simulate_ordered_taskset_event_driven(tasks, make_nominal_scenario())
