"""automotive workload 生成器测试。"""

from __future__ import annotations

from amc_py.automotive_workload import (
    AUTOMOTIVE_PERIOD_SET,
    AutomotiveWorkloadConfig,
    build_automotive_execution_scenario,
    build_automotive_experiment_config,
    build_automotive_workload,
)
from amc_py.dqn import build_env_from_experiment_config
from amc_py.runtime_models import RuntimeSemantics


def test_same_seed_produces_reproducible_workload() -> None:
    """同 seed 下生成结果应可复现。"""

    config = AutomotiveWorkloadConfig(num_runnables=150, seed=7, require_schedulable=False)
    a = build_automotive_workload(config)
    b = build_automotive_workload(config)
    assert a.runnables == b.runnables
    assert a.tasks == b.tasks
    assert a.normalization_bounds == b.normalization_bounds


def test_runnable_periods_follow_paper_period_set() -> None:
    """runnable 周期应来自预定义的论文周期集合。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(num_runnables=150, seed=1, require_schedulable=False)
    )
    assert {runnable.period for runnable in workload.runnables}.issubset(set(AUTOMOTIVE_PERIOD_SET))


def test_task_count_is_bounded_by_period_and_criticality_buckets() -> None:
    """每个 period / criticality 最多一个任务，因此总任务数不超过 18。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(num_runnables=250, seed=2, require_schedulable=False)
    )
    assert len(workload.tasks) <= 18


def test_each_task_deadline_equals_period() -> None:
    """生成任务应满足 D = T。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(num_runnables=150, seed=3, require_schedulable=False)
    )
    assert all(task.deadline == task.period for task in workload.tasks)


def test_hi_tasks_have_c_hi_and_bounds_are_generated() -> None:
    """HI 任务应保留 c_hi，且应能生成 normalization bounds。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(num_runnables=150, seed=4, hi_probability=0.5, require_schedulable=False)
    )
    hi_tasks = [task for task in workload.tasks if task.criticality.value == "HI"]
    assert hi_tasks
    assert all(task.c_hi >= task.c_lo for task in hi_tasks)
    assert set(workload.normalization_bounds.keys()) == {task.name for task in workload.tasks}


def test_short_episode_can_run_inside_amc_budget_env() -> None:
    """automotive workload 应可接入 AmcBudgetEnv 并跑一个短 episode。"""

    config = build_automotive_experiment_config(
        num_runnables=150,
        require_schedulable=True,
        max_attempts=20,
    )
    env = build_env_from_experiment_config(
        config,
        seed=0,
        end_time=50,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
    )
    obs = env.reset(seed=0)
    assert len(obs.state_vector) == 2 * len(env.ordered_tasks)
    step_result = env.step(None)
    assert step_result.info["time"] > 0


def test_weibull_scenario_respects_task_execution_bounds() -> None:
    """Weibull 场景应把实际执行时间裁剪到 BCET/WCET 区间。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(num_runnables=150, seed=5, require_schedulable=False)
    )
    scenario = build_automotive_execution_scenario(workload, scenario_seed=5)
    meta_map = {meta.name: meta for meta in workload.task_meta}
    for task in workload.tasks[:5]:
        actual_cost = scenario.actual_cost_for(task, 0)
        meta = meta_map[task.name]
        assert meta.bcet <= actual_cost <= meta.wcet
