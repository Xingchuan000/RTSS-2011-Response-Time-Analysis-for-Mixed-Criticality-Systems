"""MC-FairGen workload（Step1-3）测试。"""

from __future__ import annotations

import random

import pytest

from amc_py.models import Criticality
from amc_py.workloads.mc_fairgen import (
    MC_FAIRGEN_AUTOMOTIVE_PERIOD_SET,
    MCFairGenWorkloadProvider,
    MCFairGenWorkloadConfig,
    _sample_period_ms,
    build_mc_fairgen_execution_scenario,
    build_mc_fairgen_workload,
    uunifast_discard,
)


def test_mc_fairgen_config_rejects_unknown_mode() -> None:
    """mode 非 paper_learnable_headroom 时应报错。"""

    with pytest.raises(ValueError, match="mode"):
        MCFairGenWorkloadConfig(mode="paper_like")  # type: ignore[arg-type]


def test_mc_fairgen_config_rejects_invalid_ranges() -> None:
    """关键区间参数非法时应报错。"""

    with pytest.raises(ValueError, match="u_hi_lo"):
        MCFairGenWorkloadConfig(u_hi_lo_min=0.4, u_hi_lo_max=0.3)


def test_mc_fairgen_period_sampler_uses_supported_periods() -> None:
    """period 采样必须来自预定义集合。"""

    rng = random.Random(0)
    config = MCFairGenWorkloadConfig(seed=0)
    sampled = {_sample_period_ms(rng, config) for _ in range(200)}
    assert sampled.issubset(set(MC_FAIRGEN_AUTOMOTIVE_PERIOD_SET))


def test_uunifast_discard_sums_to_total() -> None:
    """UUniFastDiscard 应保持总和并产生正 utilization。"""

    rng = random.Random(42)
    total = 0.55
    utils = uunifast_discard(rng, 8, total, max_per_task=0.8)
    assert len(utils) == 8
    assert all(u > 0 for u in utils)
    assert abs(sum(utils) - total) < 1e-9


def test_mc_fairgen_raw_workload_task_count() -> None:
    """生成任务数量应与配置一致。"""

    config = MCFairGenWorkloadConfig(seed=0, num_tasks=16)
    workload = build_mc_fairgen_workload(config)
    assert len(workload.tasks) == 16


def test_mc_fairgen_hi_tasks_have_c_hi_ge_c_lo() -> None:
    """HI 任务需满足 c_hi >= c_lo。"""

    workload = build_mc_fairgen_workload(MCFairGenWorkloadConfig(seed=1))
    hi_tasks = [task for task in workload.tasks if task.criticality is Criticality.HI]
    assert hi_tasks
    assert all(task.c_hi >= task.c_lo for task in hi_tasks)


def test_mc_fairgen_lo_tasks_have_c_hi_eq_c_lo() -> None:
    """LO 任务需满足 c_hi == c_lo。"""

    workload = build_mc_fairgen_workload(MCFairGenWorkloadConfig(seed=2))
    lo_tasks = [task for task in workload.tasks if task.criticality is Criticality.LO]
    assert lo_tasks
    assert all(task.c_hi == task.c_lo for task in lo_tasks)


def test_mc_fairgen_metadata_has_util_totals() -> None:
    """metadata 中应包含预算利用率统计字段。"""

    workload = build_mc_fairgen_workload(MCFairGenWorkloadConfig(seed=3))
    assert workload.metadata is not None
    assert "budget_util_total" in workload.metadata
    assert "budget_util_hi" in workload.metadata
    assert "budget_util_lo" in workload.metadata


def test_mc_fairgen_scenario_actual_costs_are_valid() -> None:
    """scenario 的实际执行时间必须满足关键级约束。"""

    config = MCFairGenWorkloadConfig(seed=0)
    workload = build_mc_fairgen_workload(config)
    scenario = build_mc_fairgen_execution_scenario(workload, 123)

    for task in workload.tasks:
        for release in range(50):
            cost = scenario.actual_cost_for(task, release)
            assert cost >= 1
            if task.criticality is Criticality.HI:
                assert cost <= task.c_hi


def test_mc_fairgen_scenario_can_produce_lo_over_budget_costs() -> None:
    """LO overrun 概率拉满时，应能看到 cost > c_lo。"""

    config = MCFairGenWorkloadConfig(seed=0, lo_overrun_prob=1.0)
    workload = build_mc_fairgen_workload(config)
    scenario = build_mc_fairgen_execution_scenario(workload, 123)
    lo_tasks = [task for task in workload.tasks if task.criticality is Criticality.LO]
    assert any(scenario.actual_cost_for(task, release) > task.c_lo for task in lo_tasks for release in range(5))


def test_mc_fairgen_provider_builds_bundle() -> None:
    """provider 应返回完整 bundle 与 mc_fairgen 元数据。"""

    provider = MCFairGenWorkloadProvider(MCFairGenWorkloadConfig(seed=0))
    bundle = provider.build(seed=0)
    assert bundle.tasks
    assert bundle.scenario is not None
    assert bundle.normalization_bounds
    assert bundle.metadata is not None
    assert bundle.metadata["workload_family"] == "mc_fairgen"


def test_mc_fairgen_fixed_taskset_seed_reproducible() -> None:
    """固定 taskset seed 时，不同输入 seed 也应生成同一任务集。"""

    provider = MCFairGenWorkloadProvider(
        MCFairGenWorkloadConfig(seed=0),
        fixed_taskset_seed=7,
    )
    bundle_a = provider.build(seed=100)
    bundle_b = provider.build(seed=200)
    assert bundle_a.tasks == bundle_b.tasks


def test_mc_fairgen_require_schedulable_smoke() -> None:
    """require_schedulable 路径应可返回并标注可调度筛选元数据。"""

    workload = build_mc_fairgen_workload(
        MCFairGenWorkloadConfig(
            seed=0,
            require_schedulable=True,
            max_attempts=20,
        )
    )
    assert workload.attempts >= 1
    assert workload.metadata is not None
    assert bool(workload.metadata.get("schedulability_checked")) is True
