"""事件驱动 runtime 的 HI 恢复与 LO 抑制测试（阶段五）。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import make_table_scenario


def _hi(name: str, period: int, c_lo: int, c_hi: int) -> Task:
    """构造 HI 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


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


def test_hi_mode_recovers_to_lo_when_idle() -> None:
    """HI 模式在系统空闲时应恢复 LO。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=3), _lo("l", period=20, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("h", 0): 3}),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.mode_change_count() == 1
    assert result.mode_recovery_count() == 1
    assert result.final_mode is SystemMode.LO


def test_lo_arrivals_are_suppressed_while_in_hi_mode() -> None:
    """HI 模式持续期间，不应释放新的 LO jobs。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=4), _lo("l", period=1, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("h", 0): 4}),
        config=RuntimeConfig(end_time=7, semantics=RuntimeSemantics.AMC_PLUS),
    )
    lo_release_indexes = [job.release_index for job in result.jobs_of("l")]
    # HI 模式预计覆盖 t=2~4，对应 release_index 2,3 被抑制。
    assert 2 not in lo_release_indexes
    assert 3 not in lo_release_indexes
    assert 4 in lo_release_indexes


def test_suppressed_lo_arrivals_are_not_caught_up_after_recovery() -> None:
    """恢复后不会补发 HI 期间被抑制的历史 LO releases。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=4), _lo("l", period=1, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("h", 0): 4}),
        config=RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    lo_release_indexes = [job.release_index for job in result.jobs_of("l")]
    assert 2 not in lo_release_indexes
    assert 3 not in lo_release_indexes
    assert 4 in lo_release_indexes


def test_future_lo_arrivals_after_recovery_are_released_normally() -> None:
    """恢复后未来时刻的 LO releases 应正常发生。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=4), _lo("l", period=1, c_lo=1)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("h", 0): 4}),
        config=RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    lo_release_indexes = [job.release_index for job in result.jobs_of("l")]
    assert 5 in lo_release_indexes
    assert 6 in lo_release_indexes
    assert 7 in lo_release_indexes
