"""RTSS2011 DQN experiment config 测试。"""

from __future__ import annotations

from amc_py.dqn import build_rtss11_experiment_config, resolve_experiment_bundle
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality


def test_rtss11_experiment_config_can_be_constructed() -> None:
    """应能正常创建 RTSS2011 experiment config。"""

    config = build_rtss11_experiment_config(total_util=0.65, num_tasks=20, cf=2.0, cp=0.5)
    assert config.name.startswith("rtss11_")
    assert config.check_safety is True


def test_rtss11_experiment_config_require_schedulable_true_returns_schedulable_taskset() -> None:
    """require_schedulable=True 时应返回 AMC-rtb 可调度任务集。"""

    config = build_rtss11_experiment_config(
        total_util=0.55,
        num_tasks=20,
        cf=2.0,
        cp=0.5,
        require_schedulable=True,
    )
    bundle = resolve_experiment_bundle(config, seed=3)
    result = evaluate_taskset(list(bundle.ordered_tasks), method="amc_rtb", priority_policy="opa")
    assert result.schedulable is True


def test_rtss11_experiment_config_scenario_factory_produces_valid_scenario() -> None:
    """scenario factory 应可生成可调用且满足约束的执行场景。"""

    config = build_rtss11_experiment_config(
        total_util=0.65,
        num_tasks=20,
        cf=2.0,
        cp=0.5,
        hi_overrun_prob=0.2,
        lo_overrun_prob=0.2,
        lo_overrun_factor=1.8,
    )
    bundle = resolve_experiment_bundle(config, seed=5)
    assert "rtss11_random" in bundle.scenario.name

    has_overrun = False
    for task in bundle.ordered_tasks:
        for release_index in range(120):
            actual_cost = bundle.scenario.actual_cost_for(task, release_index)
            if task.criticality is Criticality.HI:
                assert actual_cost <= task.c_hi
            if actual_cost > task.c_lo:
                has_overrun = True
    assert has_overrun is True


def test_rtss11_experiment_config_name_contains_key_parameters() -> None:
    """config name 应包含 total_util/num_tasks/cf/cp 等关键信息。"""

    config = build_rtss11_experiment_config(total_util=0.65, num_tasks=20, cf=2.0, cp=0.5)
    assert "n20" in config.name
    assert "u650" in config.name
    assert "cf20" in config.name
    assert "cp50" in config.name
