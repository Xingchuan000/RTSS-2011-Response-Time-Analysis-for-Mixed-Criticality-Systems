"""runtime wrapper 对 C-AMC-sem XF 传播的测试。"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.agents import NoOpBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
import amc_py.rl.runtime_wrapper as runtime_wrapper_module
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    """构造一组最小任务，足以触发 wrapper 建 runtime config。"""

    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 12, 12, 2, 2, Criticality.LO),
    ]


@pytest.mark.parametrize("xf", [0.25, 0.75])
def test_runtime_wrapper_propagates_c_amc_sem_degradation_ratio(
    monkeypatch: pytest.MonkeyPatch,
    xf: float,
) -> None:
    """wrapper 重建 RuntimeConfig 时必须保留调用方传入的 XF 和 primary-on-switch 字段。"""

    captured_configs: list[RuntimeConfig] = []
    real_build = runtime_wrapper_module.EventRuntimeEngine.build

    def _build_proxy(*, ordered_tasks, scenario, config, budget_state=None, budget_updates=None, monitor=None):
        captured_configs.append(config)
        return real_build(
            ordered_tasks=ordered_tasks,
            scenario=scenario,
            config=config,
            budget_state=budget_state,
            budget_updates=budget_updates,
            monitor=monitor,
        )

    monkeypatch.setattr(runtime_wrapper_module.EventRuntimeEngine, "build", _build_proxy)

    simulate_ordered_taskset_with_agent(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        agent=NoOpBudgetAgent(),
        runtime_config=RuntimeConfig(
            end_time=20,
            semantics=RuntimeSemantics.C_AMC_SEM,
            c_amc_sem_lo_degradation_ratio=xf,
            c_amc_sem_primary_on_switch_time=True,
        ),
        agent_config=AgentRuntimeConfig(agent_period=10, end_time=20),
    )

    assert captured_configs
    assert captured_configs[0].semantics is RuntimeSemantics.C_AMC_SEM
    assert captured_configs[0].c_amc_sem_lo_degradation_ratio == xf
    assert captured_configs[0].c_amc_sem_primary_on_switch_time is True
