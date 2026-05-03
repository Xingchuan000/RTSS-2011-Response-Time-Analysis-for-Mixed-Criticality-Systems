"""事件驱动 runtime 抢占与 completion 失效测试（阶段三）。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig
from amc_py.runtime_scenarios import make_table_scenario


def _lo(name: str, period: int, c_lo: int) -> Task:
    """构造 LO 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_higher_priority_arrival_preempts_lower_priority_job() -> None:
    """高优先级 job 到达后应抢占低优先级 job。"""

    high = _lo("high", period=3, c_lo=1)
    low = _lo("low", period=20, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [high, low],
        make_table_scenario({("low", 0): 6, ("high", 1): 1}),
        config=RuntimeConfig(end_time=10),
    )
    low_job = result.jobs_of("low")[0]
    high_job_1 = [job for job in result.jobs_of("high") if job.release_index == 1][0]
    assert high_job_1.completion_time == 4
    assert low_job.completion_time == 9


def test_preempted_job_later_resumes_with_remaining_execution_time() -> None:
    """被抢占 job 应在后续继续执行剩余时间，而不是从头开始。"""

    high = _lo("high", period=4, c_lo=1)
    low = _lo("low", period=20, c_lo=5)
    result = simulate_ordered_taskset_event_driven(
        [high, low],
        make_table_scenario({("low", 0): 5}),
        config=RuntimeConfig(end_time=12),
    )
    low_job = result.jobs_of("low")[0]
    assert low_job.executed_time == 5
    assert low_job.completion_time == 7


def test_stale_completion_event_is_ignored_after_preemption() -> None:
    """抢占前留下的旧 completion 事件不应导致提前完成。"""

    high = _lo("high", period=2, c_lo=1)
    low = _lo("low", period=20, c_lo=6)
    result = simulate_ordered_taskset_event_driven(
        [high, low],
        make_table_scenario({("low", 0): 6}),
        config=RuntimeConfig(end_time=13),
    )
    low_job = result.jobs_of("low")[0]
    # 若旧 completion 事件未失效，low 会在 t=7 前被错误完成。
    assert low_job.completion_time == 12


def test_same_priority_order_follows_ordered_tasks_priority_map() -> None:
    """任务优先级应严格由 ordered_tasks 顺序决定。"""

    first = _lo("first", period=20, c_lo=2)
    second = _lo("second", period=20, c_lo=2)
    result = simulate_ordered_taskset_event_driven(
        [first, second],
        make_table_scenario({("first", 0): 2, ("second", 0): 2}),
        config=RuntimeConfig(end_time=10),
    )
    first_job = result.jobs_of("first")[0]
    second_job = result.jobs_of("second")[0]
    assert first_job.completion_time == 2
    assert second_job.completion_time == 4
