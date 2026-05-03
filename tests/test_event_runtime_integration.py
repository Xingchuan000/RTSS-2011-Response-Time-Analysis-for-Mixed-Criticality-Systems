"""事件驱动 runtime 的策略桥接测试（阶段八）。"""

from __future__ import annotations

import pytest

from amc_py.budget_runtime import BudgetUpdate
from amc_py.event_runtime import (
    simulate_ordered_taskset_event_driven,
    simulate_taskset_with_policy_event_driven,
)
from amc_py.experiments import resolve_ordering
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, SimulationResult
from amc_py.runtime_scenarios import make_nominal_scenario, make_table_scenario


def _integration_taskset() -> list[Task]:
    """构造一个可用于桥接测试的小型任务集。"""

    return [
        Task("tau_hi", period=10, deadline=10, c_lo=1, c_hi=2, criticality=Criticality.HI),
        Task("tau_lo_1", period=12, deadline=12, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau_lo_2", period=20, deadline=20, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def test_event_runtime_policy_bridge_accepts_amc_methods() -> None:
    """bridge API 应接受 AMC family methods。"""

    result = simulate_taskset_with_policy_event_driven(
        tasks=_integration_taskset(),
        method="amc_rtb",
        priority_policy="dm",
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=30),
    )
    assert isinstance(result, SimulationResult)
    assert len(result.jobs) > 0


@pytest.mark.parametrize("method", ["smc", "smc_no"])
def test_event_runtime_policy_bridge_rejects_non_amc_methods(method: str) -> None:
    """非 AMC 方法应被显式拒绝。"""

    with pytest.raises(ValueError, match=rf"仅支持 AMC family methods.*method='{method}'"):
        simulate_taskset_with_policy_event_driven(
            tasks=_integration_taskset(),
            method=method,
            priority_policy="dm",
            scenario=make_nominal_scenario(),
            config=RuntimeConfig(end_time=30),
        )


def test_event_runtime_policy_bridge_uses_resolved_ordering() -> None:
    """bridge 结果应与手动 resolve_ordering 后的直接调用一致。"""

    tasks = _integration_taskset()
    cfg = RuntimeConfig(end_time=30)
    scenario = make_nominal_scenario()
    ordered = resolve_ordering(tasks, priority_policy="crmpo", method="amc_rtb")
    manual = simulate_ordered_taskset_event_driven(ordered, scenario, cfg)
    bridged = simulate_taskset_with_policy_event_driven(
        tasks=tasks,
        method="amc_rtb",
        priority_policy="crmpo",
        scenario=scenario,
        config=cfg,
    )

    assert [(job.task.name, job.release_index) for job in bridged.jobs] == [
        (job.task.name, job.release_index) for job in manual.jobs
    ]
    assert bridged.final_mode is manual.final_mode
    assert bridged.end_time == manual.end_time


def test_event_runtime_with_policy_supports_budget_updates() -> None:
    """bridge API 应透传 budget_updates。"""

    hi = Task(
        "h",
        period=10,
        deadline=10,
        c_lo=3,
        c_hi=5,
        criticality=Criticality.HI,
    )
    scenario = make_table_scenario({("h", 0): 5})
    result = simulate_taskset_with_policy_event_driven(
        tasks=[hi],
        method="amc_max",
        priority_policy="dm",
        scenario=scenario,
        config=RuntimeConfig(end_time=8),
        budget_updates=[BudgetUpdate(time=0, updates={"h": 5})],
    )
    assert isinstance(result, SimulationResult)
    assert result.mode_change_count() == 0
