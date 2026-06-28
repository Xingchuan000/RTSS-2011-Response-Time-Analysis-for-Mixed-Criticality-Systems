"""v13 RH-specific observation（Level 7）测试。"""

from __future__ import annotations

import pytest

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.rl.feature_config import (
    FeatureConfig,
    OBSERVATION_MODE_V10_BASIC,
    OBSERVATION_MODE_V11_FULL_10D,
    OBSERVATION_MODE_V12_FULL_14D,
    OBSERVATION_MODE_V13_RH_17D,
)
from amc_py.runtime_models import RuntimeSemantics


def _build_env(*, observation_mode: str):
    """构造用于 v13 RH-risk 观测测试的环境实例。"""

    return build_env_from_experiment_config(
        build_small_stress_experiment_config(),
        seed=0,
        end_time=80,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
        reward_mode="mendes",
        feature_config=FeatureConfig(observation_mode=observation_mode),
    )


def _first_valid_action_id(env) -> int:
    """获取当前时刻首个合法动作 id。"""

    mask = env.valid_action_mask()
    return next(i for i, valid in enumerate(mask) if valid)


def test_v13_rh_17d_state_dim_matches_feature_config() -> None:
    """v13_rh_17d 观测维度必须与 FeatureConfig.expected_state_dim() 一致。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    obs = env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)
    expected_dim = FeatureConfig(observation_mode=OBSERVATION_MODE_V13_RH_17D).expected_state_dim(n_tasks)
    assert len(obs.state_vector) == expected_dim
    # 17 * n_tasks + 16
    assert len(obs.state_vector) == 17 * n_tasks + 16


def test_v13_rh_17d_observation_length_is_17n_plus_16_after_step() -> None:
    """v13_rh_17d 模式 step 后观测维度仍为 17 * n_tasks + 16。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    env.reset(seed=0)
    action_id = _first_valid_action_id(env)
    result = env.step(action_id)
    n_tasks = len(env.ordered_tasks)
    assert len(result.observation.state_vector) == 17 * n_tasks + 16


def test_v13_rh_17d_features_are_bounded() -> None:
    """v13_rh_17d 所有特征值必须在 [0.0, 1.0] 范围内。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    obs = env.reset(seed=0)
    assert all(0.0 <= value <= 1.0 for value in obs.state_vector)

    # 多步运行后再次检查
    for _ in range(5):
        if env._done:  # noqa: SLF001
            break
        action_id = _first_valid_action_id(env)
        result = env.step(action_id)
        assert all(0.0 <= value <= 1.0 for value in result.observation.state_vector)


def test_v13_rh_17d_new_rh_features_in_bounds() -> None:
    """v13_rh_17d 新增的 RH-risk per-task 和 global 特征必须在 [0, 1] 内。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)

    # 按特征结构定位新增特征范围
    per_task_dim = 17
    global_dim = 16

    # 每个任务最后 3 维是 per-task RH-risk hint（索引 14, 15, 16 在每个任务块内）
    for task_idx in range(n_tasks):
        rh_start = task_idx * per_task_dim + 14
        rh_end = task_idx * per_task_dim + per_task_dim
        # 在每个任务块内，RH-risk hint 是最后 3 个值
        task_rh_features = env._last_observation.state_vector[rh_start:rh_end]  # noqa: SLF001
        assert len(task_rh_features) == 3
        assert all(0.0 <= value <= 1.0 for value in task_rh_features)

    # 全局最后 8 维是 RH-risk global 特征
    global_start = n_tasks * per_task_dim + 8
    global_rh_features = env._last_observation.state_vector[global_start:]  # noqa: SLF001
    assert len(global_rh_features) == 8
    assert all(0.0 <= value <= 1.0 for value in global_rh_features)


def test_existing_observation_modes_keep_previous_dimensions() -> None:
    """旧 observation mode 维度保持不变（向后兼容性测试）。"""

    # v10_basic
    env10 = _build_env(observation_mode=OBSERVATION_MODE_V10_BASIC)
    obs10 = env10.reset(seed=0)
    n_tasks = len(env10.ordered_tasks)
    assert len(obs10.state_vector) == 2 * n_tasks

    # v11_full_10d
    env11 = _build_env(observation_mode=OBSERVATION_MODE_V11_FULL_10D)
    obs11 = env11.reset(seed=0)
    n_tasks = len(env11.ordered_tasks)
    assert len(obs11.state_vector) == 10 * n_tasks + 8

    # v12_full_14d
    env12 = _build_env(observation_mode=OBSERVATION_MODE_V12_FULL_14D)
    obs12 = env12.reset(seed=0)
    n_tasks = len(env12.ordered_tasks)
    assert len(obs12.state_vector) == 14 * n_tasks + 8


def test_v13_rh_17d_feature_state_exists_after_reset() -> None:
    """v13_rh_17d 模式 reset 后 feature_state 必须存在。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    env.reset(seed=0)
    assert env._feature_state is not None  # noqa: SLF001


def test_v13_rh_17d_rh_risk_context_initialized() -> None:
    """v13_rh_17d 模式 reset 后 _last_rh_risk_context 必须初始化为全 0。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    env.reset(seed=0)
    context = env._last_rh_risk_context  # noqa: SLF001
    assert context is not None
    assert "hi_mode_pressure_mean" in context
    assert "hi_mode_pressure_max" in context
    assert "active_lo_job_rate" in context
    assert "active_lo_work_ratio" in context
    assert "active_lo_under_hi_pressure" in context
    assert "recent_active_drop_rate" in context
    assert "recent_budget_cancellation_rate" in context
    assert "recent_release_drop_rate" in context


def test_v13_rh_17d_rh_risk_context_updated_after_step() -> None:
    """v13_rh_17d 模式 step 后 _last_rh_risk_context 应被更新。"""

    env = _build_env(observation_mode=OBSERVATION_MODE_V13_RH_17D)
    env.reset(seed=0)
    action_id = _first_valid_action_id(env)
    env.step(action_id)
    context = env._last_rh_risk_context  # noqa: SLF001
    # 至少 hi_mode_pressure_mean 应该在 step 后被计算（可能仍为 0.0，但字段必须存在）
    assert isinstance(context.get("hi_mode_pressure_mean"), float)
    assert isinstance(context.get("active_lo_job_rate"), float)
