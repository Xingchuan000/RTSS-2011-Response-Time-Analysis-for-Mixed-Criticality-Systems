"""多次模式切换与多次 LO 取消事件测试（阶段 7）。"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
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


def test_multiple_hi_overruns_produce_multiple_mode_changes() -> None:
    """HI 多次超预算应累计记录多次模式切换。"""

    hi = _hi("h", period=4, c_lo=1, c_hi=3)
    scenario = make_table_scenario(
        actual_costs={
            ("h", 0): 3,
            ("h", 1): 1,
            ("h", 2): 3,
        }
    )
    result = simulate_ordered_taskset(
        [hi],
        scenario,
        RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.mode_change_count() == 2
    assert result.mode_recovery_count() >= 2


def test_multiple_lo_overruns_produce_multiple_cancellations_without_switch() -> None:
    """LO 多次超预算应累计取消，不应污染模式切换统计。"""

    hi = _hi("h", period=20, c_lo=1, c_hi=1)
    lo = _lo("l", period=2, c_lo=1)
    scenario = make_table_scenario(
        actual_costs={
            ("l", 0): 2,
            ("l", 1): 1,
            ("l", 2): 2,
        }
    )
    result = simulate_ordered_taskset(
        [hi, lo],
        scenario,
        RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    assert result.lo_job_cancellation_count() == 2
    assert result.mode_change_count() == 0

