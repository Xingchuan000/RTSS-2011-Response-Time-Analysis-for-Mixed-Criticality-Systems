"""HI 空闲恢复 LO 模式测试（阶段 6）。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import make_single_hi_overrun_scenario


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


def _trace_mode_at(result, time: int) -> SystemMode | None:
    """读取指定 tick 的 trace mode。"""

    for tick in result.trace:
        if tick.time == time:
            return tick.mode
    return None


def test_hi_mode_recovers_to_lo_when_idle() -> None:
    """HI overrun 后，在无活动任务时应恢复 LO。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=2), _lo("l", period=20, c_lo=1)]
    result = simulate_ordered_taskset(
        tasks,
        make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS, capture_trace=True),
    )

    assert result.mode_change_count() == 1
    assert result.mode_recovery_count() == 1
    assert result.final_mode is SystemMode.LO
    assert _trace_mode_at(result, 2) is SystemMode.LO


def test_suppressed_lo_releases_do_not_catch_up_after_recovery() -> None:
    """HI 期间抑制的 LO release 不应在恢复后补发，但后续周期应正常释放。"""

    tasks = [_hi("h", period=20, c_lo=1, c_hi=4), _lo("l", period=1, c_lo=1)]
    result = simulate_ordered_taskset(
        tasks,
        make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC_PLUS),
    )

    lo_release_indexes = [job.release_index for job in result.jobs_of("l")]
    assert 2 not in lo_release_indexes
    assert 3 not in lo_release_indexes
    assert 4 in lo_release_indexes
    assert 5 in lo_release_indexes
