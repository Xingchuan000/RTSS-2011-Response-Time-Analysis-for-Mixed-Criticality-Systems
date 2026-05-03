"""事件边界语义测试。"""

from __future__ import annotations

from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    """构造边界测试任务集。"""

    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 15, 15, 2, 2, Criticality.LO),
    ]


def test_run_until_boundary_flag_changes_behavior() -> None:
    """include_boundary 应影响 time=target 的事件是否被处理。"""

    engine_a = EventRuntimeEngine.build(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )
    engine_a.run_until(0, include_boundary=False)
    assert len(engine_a.state.active_jobs) == 0

    engine_b = EventRuntimeEngine.build(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )
    engine_b.run_until(0, include_boundary=True)
    assert len(engine_b.state.active_jobs) > 0
