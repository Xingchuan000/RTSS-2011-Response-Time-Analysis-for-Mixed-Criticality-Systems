"""事件驱动 runtime 与 RuntimeMonitor 集成测试。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_single_hi_overrun_scenario, make_single_lo_overrun_scenario, make_table_scenario


def _task(name: str, period: int, c_lo: int, c_hi: int, criticality: Criticality) -> Task:
    """便捷构造任务。"""

    return Task(name=name, period=period, deadline=period, c_lo=c_lo, c_hi=c_hi, criticality=criticality)


def test_lo_overrun_records_penalty_and_recent_execution() -> None:
    """AMC+ 中 LO overrun 应被记录为 -1.0 并更新 recent execution。"""

    lo = _task("lo", period=10, c_lo=2, c_hi=4, criticality=Criticality.LO)
    monitor = RuntimeMonitor()
    simulate_ordered_taskset_event_driven(
        [lo],
        make_single_lo_overrun_scenario("lo", release_index=0, actual_cost=4),
        config=RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC_PLUS),
        monitor=monitor,
    )
    assert monitor.lo_overrun_count >= 1
    assert monitor.recent_execution["lo"] == 3


def test_hi_overrun_records_penalty_and_recent_execution() -> None:
    """HI overrun 触发模式切换时应计入 -2.0。"""

    hi = _task("hi", period=10, c_lo=2, c_hi=4, criticality=Criticality.HI)
    monitor = RuntimeMonitor()
    simulate_ordered_taskset_event_driven(
        [hi],
        make_single_hi_overrun_scenario("hi", release_index=0, overrun_to="c_hi"),
        config=RuntimeConfig(end_time=6, semantics=RuntimeSemantics.AMC_PLUS),
        monitor=monitor,
    )
    assert monitor.hi_overrun_count >= 1
    assert monitor.recent_execution["hi"] == 4


def test_preempt_resume_does_not_double_count_job_start_reward() -> None:
    """同一 job 被抢占并恢复执行时，start 奖励只能记一次。"""

    high = _task("high", period=4, c_lo=1, c_hi=1, criticality=Criticality.LO)
    low = _task("low", period=20, c_lo=5, c_hi=5, criticality=Criticality.LO)
    monitor = RuntimeMonitor()
    simulate_ordered_taskset_event_driven(
        [high, low],
        make_table_scenario({("low", 0): 5}),
        config=RuntimeConfig(end_time=12),
        monitor=monitor,
    )
    # high 在 t=0/4/8 共 3 个 job，low 1 个 job，总 start 次数应为 4。
    assert monitor.job_start_count == 4
