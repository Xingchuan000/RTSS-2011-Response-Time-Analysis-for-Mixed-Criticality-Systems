"""阶段 7：AmcBudgetEnv 测试。"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.observation import TaskNormalizationBound
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _tasks() -> list[Task]:
    return [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l1", 12, 12, 2, 3, Criticality.LO),
        Task("l2", 15, 15, 2, 3, Criticality.LO),
    ]


def _env(end_time: int = 50) -> AmcBudgetEnv:
    return AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
    )


def test_reset_returns_observation_with_2n_length() -> None:
    """reset 应返回长度为 2*n 的观测向量。"""

    env = _env()
    obs = env.reset(seed=0)
    assert len(obs.state_vector) == 2 * len(_tasks())


def test_action_space_size_matches_formula() -> None:
    """动作空间大小应满足 n(n-1)(n-2)/2。"""

    env = _env()
    n = len(_tasks())
    assert env.action_space_size == n * (n - 1) * (n - 2) // 2


def test_step_none_advances_time() -> None:
    """step(None) 应推进仿真时间。"""

    env = _env()
    env.reset()
    r = env.step(None)
    assert r.info["time"] > 0
    assert env.action_log[-1]["time"] == 0


def test_step_valid_action_returns_reward_and_next_state() -> None:
    """step(合法 action) 应返回 reward 和下一状态。"""

    env = _env()
    env.reset()
    r = env.step(0)
    assert isinstance(r.reward, float)
    assert len(r.observation.state_vector) == 2 * len(_tasks())
    assert "budget_before" in r.info
    assert "candidate_budgets" in r.info
    assert "budget_after" in r.info
    assert "valid_action_count" in r.info
    assert "masked_action_count" in r.info
    assert "selected_action_was_mask_valid" in r.info


def test_invalid_action_id_raises_value_error() -> None:
    """非法 action_id 必须抛 ValueError。"""

    env = _env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(999)


def test_step_until_done_finishes_without_infinite_loop() -> None:
    """反复 step 应最终到达 done=True。"""

    env = _env(end_time=35)
    env.reset()
    done = False
    guard = 0
    while not done:
        out = env.step(None)
        done = out.done
        guard += 1
        assert guard < 20


def test_reset_with_seed_has_reproducible_trajectory() -> None:
    """确定性场景下 reset(seed) 后轨迹应复现。"""

    env1 = _env(end_time=40)
    env2 = _env(end_time=40)
    env1.reset(seed=1)
    env2.reset(seed=1)
    traj1 = [env1.step(None).info["time"] for _ in range(4)]
    traj2 = [env2.step(None).info["time"] for _ in range(4)]
    assert traj1 == traj2


def test_reset_and_step_share_same_normalization_bounds() -> None:
    """env.reset 与 env.step 应使用同一套 bounds。"""

    env = AmcBudgetEnv(
        ordered_tasks=_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        normalization_bounds={
            "h": TaskNormalizationBound(min_cost=0, max_cost=10),
            "l1": TaskNormalizationBound(min_cost=0, max_cost=10),
            "l2": TaskNormalizationBound(min_cost=0, max_cost=10),
        },
    )
    obs0 = env.reset()
    obs1 = env.step(None).observation
    assert all(0.0 <= v <= 1.0 for v in obs0.state_vector)
    assert all(0.0 <= v <= 1.0 for v in obs1.state_vector)


def test_env_debug_statistics_are_populated_after_mask_and_step() -> None:
    """DQN 环境应累计 mask/safety 统计，供评估脚本写入 CSV。"""

    env = _env(end_time=30)
    obs = env.reset()
    assert obs.state_vector
    env.valid_action_mask()
    env.step(0)
    stats = env.debug_statistics()

    assert int(stats["safety_checked_actions"]) >= 0
    assert float(stats["valid_action_count_mean"]) >= 0.0
    assert float(stats["masked_action_count_mean"]) >= 0.0


def test_action_log_time_records_application_time_not_next_decision_time() -> None:
    """action_log.time 应记录动作实际应用时刻。"""

    env = _env(end_time=30)
    env.reset()
    env.step(0)

    assert env.action_log[-1]["time"] == 0
    assert env.action_log[-1]["action_time"] == 0
