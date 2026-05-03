"""RL 类型定义测试。"""

from __future__ import annotations

import pytest

from amc_py.rl.types import AgentObservation, AgentStepResult


def test_agent_observation_can_be_constructed() -> None:
    """应能正常构造 AgentObservation。"""

    obs = AgentObservation(
        time=10,
        state_vector=(0.1, 0.2),
        raw_budgets={"t1": 3},
        raw_recent_costs={"t1": 2},
    )
    assert obs.time == 10
    assert obs.state_vector == (0.1, 0.2)


def test_state_vector_is_tuple_and_dataclass_is_frozen() -> None:
    """state_vector 必须是 tuple，且对象不可变。"""

    obs = AgentObservation(
        time=0,
        state_vector=(1.0,),
        raw_budgets={"t1": 1},
        raw_recent_costs={"t1": 0},
    )
    assert isinstance(obs.state_vector, tuple)
    with pytest.raises(AttributeError):
        obs.time = 1


def test_agent_step_result_info_can_carry_runtime_metrics() -> None:
    """info 应可携带运行时统计信息。"""

    obs = AgentObservation(0, (0.0,), {"t1": 1}, {"t1": 0})
    step = AgentStepResult(
        observation=obs,
        reward=0.1,
        done=False,
        info={"deadline_miss_count": 0, "mode": "LO"},
    )
    assert step.info["deadline_miss_count"] == 0
    assert step.info["mode"] == "LO"
