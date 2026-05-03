"""阶段 6：EventRuntimeEngine 测试。"""

from __future__ import annotations

from amc_py.event_runtime import EventRuntimeEngine, simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 15, 15, 2, 2, Criticality.LO),
    ]


def test_run_until_end_matches_original_function() -> None:
    """run_until(end) 的结果应与原函数一致。"""

    tasks = _tasks()
    cfg = RuntimeConfig(end_time=50)
    scenario = make_nominal_scenario()
    direct = simulate_ordered_taskset_event_driven(tasks, scenario, cfg)

    engine = EventRuntimeEngine.build(ordered_tasks=tasks, scenario=scenario, config=cfg)
    engine.run_until(50)
    via_engine = engine.finish()

    assert via_engine.mode_change_count() == direct.mode_change_count()
    assert via_engine.lo_job_cancellation_count() == direct.lo_job_cancellation_count()
    assert len(via_engine.deadline_misses) == len(direct.deadline_misses)


def test_segmented_run_matches_single_run() -> None:
    """分段 run_until 与一次性 run_until 应一致。"""

    tasks = _tasks()
    cfg = RuntimeConfig(end_time=60)
    scenario = make_nominal_scenario()

    e1 = EventRuntimeEngine.build(ordered_tasks=tasks, scenario=scenario, config=cfg)
    e1.run_until(60)
    r1 = e1.finish()

    e2 = EventRuntimeEngine.build(ordered_tasks=tasks, scenario=scenario, config=cfg)
    e2.run_until(20)
    e2.run_until(60)
    r2 = e2.finish()

    assert r1.mode_change_count() == r2.mode_change_count()
    assert r1.lo_job_cancellation_count() == r2.lo_job_cancellation_count()
    assert len(r1.deadline_misses) == len(r2.deadline_misses)


def test_apply_budget_updates_records_event() -> None:
    """apply_budget_updates 应追加 BudgetUpdateEvent。"""

    tasks = _tasks()
    engine = EventRuntimeEngine.build(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )
    engine.run_until(5)
    engine.apply_budget_updates({"h": 3})
    result = engine.finish()
    assert len(result.budget_update_events) == 1


def test_apply_budget_updates_forces_reschedule_path() -> None:
    """apply 后继续运行不应报错，且保持可推进。"""

    tasks = _tasks()
    engine = EventRuntimeEngine.build(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )
    engine.run_until(5)
    engine.apply_budget_updates({"h": 3, "l": 1})
    engine.run_until(30)
    result = engine.finish()
    assert result.end_time == 30
