"""观测构建测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.observation import TaskNormalizationBound, build_observation


def _task(name: str, c_lo: int, c_hi: int, crit: Criticality) -> Task:
    """便捷构造任务。"""

    return Task(name=name, period=10, deadline=10, c_lo=c_lo, c_hi=c_hi, criticality=crit)


def test_observation_length_is_two_times_task_count() -> None:
    """每个任务应输出预算和最近执行量两个分量。"""

    tasks = [_task("a", 2, 4, Criticality.HI), _task("b", 1, 1, Criticality.LO)]
    obs = build_observation(
        time=0,
        ordered_tasks=tasks,
        budget_state=BudgetState.from_tasks(tasks),
        monitor=RuntimeMonitor(),
    )
    assert len(obs.state_vector) == 2 * len(tasks)


def test_observation_order_follows_ordered_tasks() -> None:
    """输出顺序必须严格跟随 ordered_tasks。"""

    tasks = [_task("t1", 2, 4, Criticality.HI), _task("t2", 3, 3, Criticality.LO)]
    monitor = RuntimeMonitor()
    monitor.record_job_completion("t1", 1)
    monitor.record_job_completion("t2", 2)
    obs = build_observation(
        time=0,
        ordered_tasks=tasks,
        budget_state=BudgetState.from_tasks(tasks),
        monitor=monitor,
    )
    assert obs.raw_budgets == {"t1": 2, "t2": 3}
    assert obs.raw_recent_costs == {"t1": 1, "t2": 2}


def test_observation_changes_with_budget_and_recent_cost() -> None:
    """预算或 recent execution 变化后，状态向量应变化。"""

    tasks = [_task("x", 2, 5, Criticality.HI)]
    budget = BudgetState.from_tasks(tasks)
    monitor = RuntimeMonitor()
    obs_before = build_observation(
        time=0,
        ordered_tasks=tasks,
        budget_state=budget,
        monitor=monitor,
    )
    budget.set_budget("x", 4)
    monitor.record_job_completion("x", 3)
    obs_after = build_observation(
        time=1,
        ordered_tasks=tasks,
        budget_state=budget,
        monitor=monitor,
    )
    assert obs_before.state_vector != obs_after.state_vector


def test_normalized_values_are_clipped_to_zero_one() -> None:
    """归一化结果必须裁剪在 [0.0, 1.0]。"""

    tasks = [_task("x", 2, 4, Criticality.HI)]
    budget = BudgetState.from_tasks(tasks)
    budget.set_budget("x", 999)
    monitor = RuntimeMonitor()
    monitor.record_job_completion("x", -10)
    obs = build_observation(
        time=0,
        ordered_tasks=tasks,
        budget_state=budget,
        monitor=monitor,
    )
    assert 0.0 <= obs.state_vector[0] <= 1.0
    assert 0.0 <= obs.state_vector[1] <= 1.0


def test_custom_bounds_are_applied() -> None:
    """自定义 bounds 下应按外部区间归一化。"""

    tasks = [_task("x", 2, 4, Criticality.HI)]
    budget = BudgetState.from_tasks(tasks)
    monitor = RuntimeMonitor()
    monitor.record_job_completion("x", 3)
    obs = build_observation(
        time=0,
        ordered_tasks=tasks,
        budget_state=budget,
        monitor=monitor,
        bounds={"x": TaskNormalizationBound(min_cost=1, max_cost=5)},
    )
    assert obs.state_vector[0] == 0.25
    assert obs.state_vector[1] == 0.5
