"""事件驱动 runtime 的 deadline miss 测试（阶段七）。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


def _lo(name: str, period: int, c_lo: int, *, deadline: int | None = None) -> Task:
    """构造 LO 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=deadline if deadline is not None else period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def test_job_missing_deadline_is_recorded() -> None:
    """未完成且到达截止期的 job 应记录 miss。"""

    lo = _lo("l", period=10, c_lo=6, deadline=5)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("l", 0): 6}),
        config=RuntimeConfig(end_time=8),
    )
    assert result.deadline_missed() is True
    assert len(result.deadline_misses) >= 1
    assert result.deadline_misses[0].task == "l"
    assert result.deadline_misses[0].absolute_deadline == 5


def test_job_completing_at_deadline_is_not_miss() -> None:
    """在 deadline 时刻刚好完成的 job 不应被记为 miss。"""

    lo = _lo("l", period=10, c_lo=5, deadline=5)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("l", 0): 5}),
        config=RuntimeConfig(end_time=8),
    )
    assert result.deadline_missed() is False


def test_dropped_lo_job_is_not_counted_as_deadline_miss() -> None:
    """AMC+ 局部取消的 LO job 不算 deadline miss。"""

    lo = _lo("l", period=10, c_lo=3, deadline=8)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("l", 0): 5}),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.lo_job_cancellation_count() == 1
    assert result.deadline_missed() is False


def test_stop_at_first_miss_stops_event_loop() -> None:
    """开启 stop_at_first_miss 后，事件循环应在首个 miss 处提前结束。"""

    lo = _lo("l", period=6, c_lo=5, deadline=3)
    result = simulate_ordered_taskset_event_driven(
        [lo],
        make_table_scenario({("l", 0): 5, ("l", 1): 5}),
        config=RuntimeConfig(end_time=12, stop_at_first_miss=True),
    )
    assert result.deadline_missed() is True
    assert len(result.deadline_misses) == 1
    assert result.end_time == 3
