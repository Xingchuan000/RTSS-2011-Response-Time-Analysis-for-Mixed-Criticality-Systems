"""运行时仿真第 5 轮集成测试。

本文件的目标是验证“策略入口打通”而不是重复测试底层 tick 细节。
覆盖重点：
1. `simulate_taskset_with_policy` 是否正确复用了 `resolve_ordering`；
2. `dm / crmpo / opa` 三条路径是否都能从“未排序任务集”直接跑通；
3. `compare_static_and_runtime` 是否把静态分析结果、运行时结果和最终顺序统一返回；
4. `amc_py.__init__` 是否导出了第 5 轮要求的 runtime 公共 API。
"""

from __future__ import annotations

import pytest

from amc_py import (
    ExecutionScenario,
    Job,
    RuntimeConfig,
    SimulationResult,
    SystemMode,
    compare_static_and_runtime,
    make_nominal_scenario,
    make_single_hi_overrun_scenario,
    simulate_ordered_taskset,
    simulate_taskset_with_policy,
)
from amc_py.experiments import resolve_ordering
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeComparisonResult


def _integration_taskset() -> list[Task]:
    """构造一个可用于 DM/CrMPO/OPA 三策略的小型任务集。

    设计意图：
    - 至少包含一个 HI 任务，确保 CrMPO 与 HI 切换相关路径可被触发；
    - 周期/执行时间保持温和，避免在 nominal 场景里很快超载导致断言脆弱；
    - 与 `amc_rtb` + `opa` 组合兼容，便于 OPA 路径稳定通过。
    """

    return [
        Task("tau_hi", period=10, deadline=10, c_lo=1, c_hi=2, criticality=Criticality.HI),
        Task("tau_lo_1", period=12, deadline=12, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau_lo_2", period=20, deadline=20, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def test_init_exports_runtime_public_api_symbols() -> None:
    """`amc_py.__init__` 应导出第 5 轮要求的 runtime 公共符号。"""

    # 这里做“类型存在性”断言，避免未来 __init__ 回归时悄悄丢失导出。
    assert RuntimeConfig is not None
    assert SystemMode is not None
    assert Job is not None
    assert SimulationResult is not None
    assert ExecutionScenario is not None
    assert make_nominal_scenario is not None
    assert make_single_hi_overrun_scenario is not None
    assert simulate_ordered_taskset is not None
    assert simulate_taskset_with_policy is not None
    assert compare_static_and_runtime is not None


def test_simulate_taskset_with_policy_dm_matches_manual_ordered_run() -> None:
    """DM 路径下，集成入口结果应与“手动排序 + 基础仿真”一致。"""

    tasks = _integration_taskset()
    scenario = make_nominal_scenario()
    cfg = RuntimeConfig(end_time=30)

    # 手动路径：先用 resolve_ordering 得到 DM 顺序，再调用 ordered 仿真器。
    ordered = resolve_ordering(tasks, priority_policy="dm", method="amc_rtb")
    manual = simulate_ordered_taskset(ordered_tasks=ordered, scenario=scenario, config=cfg)

    # 集成路径：直接给未排序任务 + method/policy。
    integrated = simulate_taskset_with_policy(
        tasks=tasks,
        method="amc_rtb",
        priority_policy="dm",
        scenario=scenario,
        config=cfg,
    )

    # 对比几个关键可观测结果，确保两条路径语义一致。
    assert len(integrated.jobs) == len(manual.jobs)
    assert integrated.end_time == manual.end_time
    assert integrated.final_mode is manual.final_mode
    assert integrated.mode_switched() is manual.mode_switched()
    assert integrated.deadline_missed() is manual.deadline_missed()
    # 用 (task, release_index) 序列固定历史释放轨迹。
    assert [
        (job.task.name, job.release_index) for job in integrated.jobs
    ] == [
        (job.task.name, job.release_index) for job in manual.jobs
    ]


def test_simulate_taskset_with_policy_crmpo_path_runs() -> None:
    """CrMPO 路径可用：应能返回有效仿真结果对象并生成 job 历史。"""

    result = simulate_taskset_with_policy(
        tasks=_integration_taskset(),
        method="amc_rtb",
        priority_policy="crmpo",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=25),
    )

    assert isinstance(result, SimulationResult)
    assert len(result.jobs) > 0
    # nominal 场景下不应触发 HI 切换。
    assert result.mode_switched() is False


def test_simulate_taskset_with_policy_opa_path_runs() -> None:
    """OPA + amc_rtb 路径可用：无需 runtime 里额外实现排序算法。"""

    result = simulate_taskset_with_policy(
        tasks=_integration_taskset(),
        method="amc_rtb",
        priority_policy="opa",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=25),
    )

    assert isinstance(result, SimulationResult)
    assert len(result.jobs) > 0


def test_simulate_taskset_with_policy_amc_max_path_runs() -> None:
    """amc_max 属于 AMC family，bridge API 应允许并返回有效仿真结果。"""

    result = simulate_taskset_with_policy(
        tasks=_integration_taskset(),
        method="amc_max",
        priority_policy="dm",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=25),
    )

    assert isinstance(result, SimulationResult)
    assert len(result.jobs) > 0
    # nominal 场景下通常不会触发 HI 切换，这里给出基本行为断言。
    assert result.mode_switched() is False


@pytest.mark.parametrize("method", ["smc", "smc_no"])
def test_simulate_taskset_with_policy_rejects_non_amc_method(method: str) -> None:
    """bridge API 对非 AMC 方法必须 fail-fast，避免误导运行时语义解释。"""

    with pytest.raises(ValueError, match=rf"仅支持 AMC family methods.*method='{method}'"):
        simulate_taskset_with_policy(
            tasks=_integration_taskset(),
            method=method,
            priority_policy="dm",
            scenario=make_nominal_scenario(),
            config=RuntimeConfig(end_time=20),
        )


@pytest.mark.parametrize("method", ["smc", "smc_no"])
def test_compare_static_and_runtime_rejects_non_amc_method(method: str) -> None:
    """compare bridge 对非 AMC 方法也必须 fail-fast。"""

    with pytest.raises(ValueError, match=rf"仅支持 AMC family methods.*method='{method}'"):
        compare_static_and_runtime(
            tasks=_integration_taskset(),
            method=method,
            priority_policy="dm",
            scenario=make_nominal_scenario(),
            config=RuntimeConfig(end_time=20),
        )


def test_compare_static_and_runtime_returns_both_sides_and_ordering() -> None:
    """compare 接口应返回静态结果、运行时结果、顺序信息与元数据。"""

    tasks = _integration_taskset()
    result = compare_static_and_runtime(
        tasks=tasks,
        method="amc_rtb",
        priority_policy="dm",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )

    assert isinstance(result, RuntimeComparisonResult)
    assert result.method == "amc_rtb"
    assert result.priority_policy == "dm"
    # ordered_task_names 应与 resolve_ordering 的输出一致。
    expected_order = [task.name for task in resolve_ordering(tasks, "dm", "amc_rtb")]
    assert result.ordered_task_names == expected_order
    # compare 结果中应同时可访问静态/运行时两侧判定。
    assert isinstance(result.runtime_result, SimulationResult)
    assert isinstance(result.static_result.schedulable, bool)


def test_compare_nominal_schedulable_taskset_usually_has_no_runtime_miss() -> None:
    """在 nominal 场景和温和时间窗下，静态可调度任务集运行时应无 miss。"""

    result = compare_static_and_runtime(
        tasks=_integration_taskset(),
        method="amc_rtb",
        priority_policy="dm",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )

    # 这个任务集按设计在该时间窗内应保持稳定，不出现 miss。
    assert result.static_schedulable() is True
    assert result.deadline_missed() is False


def test_compare_overrun_scenario_reports_mode_switch() -> None:
    """超限场景下，compare 结果中的运行时侧应正确暴露 mode_switched。"""

    result = compare_static_and_runtime(
        tasks=_integration_taskset(),
        method="amc_rtb",
        priority_policy="dm",
        scenario=make_single_hi_overrun_scenario(
            task_name="tau_hi",
            release_index=0,
            overrun_to="c_hi",
        ),
        config=RuntimeConfig(end_time=30),
    )

    assert result.mode_switched() is True
    assert result.runtime_result.mode_switch is not None
    assert result.runtime_result.mode_switch.triggering_task == "tau_hi"
