"""事件驱动 runtime trace / debug 输出测试。"""

from __future__ import annotations

from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def test_event_runtime_writes_tick_trace_and_debug_events() -> None:
    """开启 capture_trace 时，应同时产出逐 tick trace 与事件级 debug 日志。"""

    tasks = [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 15, 15, 2, 2, Criticality.LO),
    ]
    engine = EventRuntimeEngine.build(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(
            end_time=30,
            capture_trace=True,
            capture_debug_events=True,
            semantics=RuntimeSemantics.AMC_PLUS,
        ),
    )

    engine.run_until(30, include_boundary=True)
    result = engine.finish()

    assert len(result.trace) == 30
    assert any(row["event"] == "job_arrival" for row in result.debug_events)
    assert any(row["event"] == "job_start" for row in result.debug_events)


def test_event_runtime_can_disable_debug_events_even_when_trace_is_enabled() -> None:
    """正式评估可在保留 trace 的同时关闭事件级 debug 日志。"""

    tasks = [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 15, 15, 2, 2, Criticality.LO),
    ]
    engine = EventRuntimeEngine.build(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(
            end_time=30,
            capture_trace=True,
            capture_debug_events=False,
            semantics=RuntimeSemantics.AMC_PLUS,
        ),
    )

    engine.run_until(30, include_boundary=True)
    result = engine.finish()

    assert len(result.trace) == 30
    assert result.debug_events == []
