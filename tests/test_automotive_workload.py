"""automotive workload 生成器测试。"""

from __future__ import annotations

from amc_py.workloads.automotive import (
    AUTOMOTIVE_PERIOD_SHARES,
    AUTOMOTIVE_PERIOD_SET,
    ACET_TABLE_US,
    AutomotiveWorkloadProvider,
    AutomotiveWorkloadConfig,
    _sample_exact_acet_us_values,
    build_automotive_execution_scenario,
    build_automotive_workload,
    build_task_to_runnables_map,
    ms_to_ticks,
    sample_runnable_cost,
)
from amc_py.dqn import build_automotive_experiment_config, build_env_from_experiment_config
from amc_py.runtime_models import RuntimeSemantics
import random


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
    assert {runnable.period_ms for runnable in workload.runnables}.issubset(set(AUTOMOTIVE_PERIOD_SET))


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


def test_automotive_provider_builds_bundle() -> None:
    """provider 应能一次性返回 tasks、scenario、bounds 与元数据。"""

    provider = AutomotiveWorkloadProvider(
        AutomotiveWorkloadConfig(num_runnables=150, seed=0, require_schedulable=False)
    )
    bundle = provider.build(seed=0)

    assert bundle.tasks
    assert bundle.scenario is not None
    assert bundle.normalization_bounds
    assert bundle.metadata is not None
    assert bundle.metadata["num_runnables"] == 150


def test_paper_exact_period_distribution_matches_table1_for_150_runnables() -> None:
    """paper_exact 应按 Table 1 的 largest remainder 分配 period 计数。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(
            num_runnables=150,
            seed=11,
            mode="paper_exact",
            require_schedulable=False,
        )
    )
    observed_counts = {
        period_ms: sum(1 for runnable in workload.runnables if runnable.period_ms == period_ms)
        for period_ms in AUTOMOTIVE_PERIOD_SET
    }
    expected_counts = dict(
        zip(
            AUTOMOTIVE_PERIOD_SET,
            [6, 3, 3, 44, 44, 6, 36, 1, 7],
            strict=True,
        )
    )
    assert observed_counts == expected_counts
    assert sum(observed_counts.values()) == 150
    assert sum(AUTOMOTIVE_PERIOD_SHARES) == 1.0


def test_paper_exact_runnables_have_weibull_parameters_and_tick_periods() -> None:
    """paper_exact runnable 应带完整 Weibull 参数，任务周期应按 tick_ns 换算。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(
            num_runnables=150,
            seed=12,
            mode="paper_exact",
            tick_ns=10,
            require_schedulable=False,
        )
    )

    assert all(runnable.weibull_shape is not None for runnable in workload.runnables)
    assert all(runnable.weibull_scale is not None for runnable in workload.runnables)
    assert all(runnable.weibull_location is not None for runnable in workload.runnables)
    assert all(task.period == ms_to_ticks(meta.period_ms, 10) for task, meta in zip(workload.tasks, workload.task_meta, strict=True))


def test_paper_exact_scenario_uses_runnable_level_sampling_sum() -> None:
    """paper_exact 场景应按 task 内 runnables 逐个采样后求和。"""

    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(
            num_runnables=150,
            seed=13,
            mode="paper_exact",
            tick_ns=10,
            require_schedulable=False,
        )
    )
    scenario = build_automotive_execution_scenario(workload, scenario_seed=13)
    task_to_runnables = build_task_to_runnables_map(workload)
    task = workload.tasks[0]
    expected_rng = random.Random(13 * 1_000_003 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(task.name)) * 1_009)
    expected_cost = sum(sample_runnable_cost(runnable, expected_rng) for runnable in task_to_runnables[task.name])
    assert scenario.actual_cost_for(task, 0) == expected_cost


def test_paper_exact_acet_sampling_is_bounded_and_sum_preserving_for_tight_bucket() -> None:
    """1000 ms 这类紧区间 bucket 也应快速返回，并满足边界与总和约束。"""

    rng = random.Random(123)
    values = _sample_exact_acet_us_values(1000, 7, rng)
    min_acet_us, avg_acet_us, max_acet_us = ACET_TABLE_US[1000]

    assert len(values) == 7
    assert all(min_acet_us <= value <= max_acet_us for value in values)
    assert abs(sum(values) - 7 * avg_acet_us) < 1e-9


def test_paper_learnable_headroom_budget_is_within_min_max_range() -> None:
    """learnable 模式下生成的预算应严格位于 min/max 区间内。"""

    floor_ratio = 0.9
    base_workload = build_automotive_workload(
        AutomotiveWorkloadConfig(
            num_runnables=150,
            seed=21,
            mode="paper_exact",
            require_schedulable=False,
        )
    )
    workload = build_automotive_workload(
        AutomotiveWorkloadConfig(
            num_runnables=150,
            seed=21,
            mode="paper_learnable_headroom",
            budget_floor_ratio=floor_ratio,
            require_schedulable=False,
        )
    )
    base_budget_map = {task.name: task.c_lo for task in base_workload.tasks}
    for task in workload.tasks:
        min_budget = max(1, int(round(float(base_budget_map[task.name]) * floor_ratio)))
        max_budget = task.c_hi if task.criticality.value == "HI" else max(task.c_hi, min(task.deadline, task.period))
        assert min_budget <= task.c_lo <= max_budget


def test_paper_learnable_headroom_is_reproducible_for_same_seed() -> None:
    """learnable 模式在相同 seed 与参数下应可复现。"""

    config = AutomotiveWorkloadConfig(
        num_runnables=150,
        seed=22,
        mode="paper_learnable_headroom",
        require_schedulable=False,
    )
    workload_a = build_automotive_workload(config)
    workload_b = build_automotive_workload(config)
    assert workload_a.tasks == workload_b.tasks
    assert workload_a.metadata == workload_b.metadata
