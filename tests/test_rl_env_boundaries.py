"""AmcBudgetEnv 边界事件语义测试。"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks_for_boundary() -> list[Task]:
    """构造用于边界语义验证的最小任务集。"""

    return [
        Task("hi", period=10, deadline=10, c_lo=1, c_hi=1, criticality=Criticality.HI),
        Task("lo", period=20, deadline=20, c_lo=1, c_hi=1, criticality=Criticality.LO),
    ]


def _env(*, end_time: int = 30, agent_period: int = 10) -> AmcBudgetEnv:
    """构造边界测试环境。"""

    return AmcBudgetEnv(
        ordered_tasks=_tasks_for_boundary(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=agent_period,
    )


def test_reset_processes_time_zero_arrivals() -> None:
    """reset 后应已处理 t=0 arrival，不应留到第一次 step 才处理。"""

    env = _env()
    env.reset(seed=0)
    assert env._engine is not None
    assert env._engine.current_time == 0
    assert len(env._engine.state.active_jobs) >= 1


def test_step_processes_agent_period_boundary_events() -> None:
    """step 到达 agent 周期边界时，应包含该边界时刻的普通事件处理结果。"""

    env = _env(end_time=25, agent_period=10)
    env.reset(seed=0)
    step1 = env.step(action_id=None)
    assert step1.info["time"] == 10
    assert env._engine is not None
    assert any(job.release_index == 1 for job in env._engine.state.active_jobs)


def test_done_boundary_never_exceeds_end_time_and_step_after_done_raises() -> None:
    """done 边界不应越界，且 done 后再次 step 必须抛错。"""

    env = _env(end_time=20, agent_period=10)
    env.reset(seed=0)
    out1 = env.step(None)
    assert out1.done is False
    out2 = env.step(None)
    assert out2.done is True
    assert env._engine is not None
    assert env._engine.current_time <= 20
    assert out2.observation.state_vector
    with pytest.raises(RuntimeError):
        env.step(None)
