"""v11 observation（阶段 8）测试。"""

from __future__ import annotations

import pytest

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics


def _build_env(*, observation_mode: str):
    """构造用于 v10/v11 对照测试的环境实例。"""

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


def test_v10_basic_observation_length_keeps_2n() -> None:
    """旧模式长度必须保持 2 * n_tasks。"""

    env = _build_env(observation_mode="v10_basic")
    obs = env.reset(seed=0)
    assert len(obs.state_vector) == 2 * len(env.ordered_tasks)


def test_v11_full_10d_observation_length_is_10n_plus_8() -> None:
    """新模式长度必须为 10 * n_tasks + 8。"""

    env = _build_env(observation_mode="v11_full_10d")
    obs = env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)
    assert len(obs.state_vector) == 10 * n_tasks + 8


def test_v11_full_10d_observation_values_in_unit_interval() -> None:
    """新模式 reset 观测的所有值必须位于 [0, 1]。"""

    env = _build_env(observation_mode="v11_full_10d")
    obs = env.reset(seed=0)
    assert all(0.0 <= value <= 1.0 for value in obs.state_vector)


def test_v11_full_10d_step_observation_length_keeps_10n_plus_8() -> None:
    """新模式 step 后观测长度仍必须为 10 * n_tasks + 8。"""

    env = _build_env(observation_mode="v11_full_10d")
    env.reset(seed=0)
    action_id = _first_valid_action_id(env)
    result = env.step(action_id)
    n_tasks = len(env.ordered_tasks)
    assert len(result.observation.state_vector) == 10 * n_tasks + 8


@pytest.mark.parametrize(
    ("observation_mode", "expected_per_task_dim"),
    [
        ("v11_no_risk_9d", 9),
        ("v11_no_util_9d", 9),
        ("v11_no_max_9d", 9),
        ("v11_no_priority_9d", 9),
        ("v11_no_risk_no_util_8d", 8),
        ("v11_lite_6d", 6),
    ],
)
def test_v11_ablation_observation_lengths_match_feature_config(
    observation_mode: str,
    expected_per_task_dim: int,
) -> None:
    """三个新增 v11 消融模式都必须返回计划文档定义的理论维度。"""

    env = _build_env(observation_mode=observation_mode)
    obs = env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)
    assert len(obs.state_vector) == expected_per_task_dim * n_tasks + 8
    assert len(obs.state_vector) == FeatureConfig(observation_mode=observation_mode).expected_state_dim(n_tasks)


@pytest.mark.parametrize(
    "observation_mode",
    [
        "v11_no_risk_9d",
        "v11_no_util_9d",
        "v11_no_max_9d",
        "v11_no_priority_9d",
        "v11_no_risk_no_util_8d",
        "v11_lite_6d",
    ],
)
def test_v11_ablation_observation_values_in_unit_interval(observation_mode: str) -> None:
    """所有新增 v11 消融模式的观测值都必须稳定落在 [0, 1]。"""

    env = _build_env(observation_mode=observation_mode)
    obs = env.reset(seed=0)
    assert all(0.0 <= value <= 1.0 for value in obs.state_vector)


def test_v11_feature_state_exists_and_preserves_task_keys_after_step() -> None:
    """step 后 feature_state 必须存在且任务键集合保持一致。"""

    env = _build_env(observation_mode="v11_full_10d")
    env.reset(seed=0)
    assert env._feature_state is not None  # noqa: SLF001
    before_keys = set(env._feature_state.ema_cost.keys())  # noqa: SLF001

    action_id = _first_valid_action_id(env)
    env.step(action_id)

    assert env._feature_state is not None  # noqa: SLF001
    after_keys = set(env._feature_state.ema_cost.keys())  # noqa: SLF001
    assert before_keys == after_keys


def test_v11_event_window_length_never_exceeds_config_limit() -> None:
    """多步运行后 event window 长度不能超过 event_window 配置上限。"""

    env = _build_env(observation_mode="v11_full_10d")
    env.reset(seed=0)
    for _ in range(20):
        if env._done:  # noqa: SLF001
            break
        action_id = _first_valid_action_id(env)
        env.step(action_id)

    assert env._feature_state is not None  # noqa: SLF001
    assert len(env._feature_state.window_job_starts) <= env.feature_config.event_window  # noqa: SLF001


def test_v11_without_new_sample_does_not_reupdate_ema_or_history() -> None:
    """当任务无新完成样本时，EMA/history 不应在后续 step 被重复更新。"""

    env = _build_env(observation_mode="v11_full_10d")
    env.reset(seed=0)
    assert env._feature_state is not None  # noqa: SLF001
    feature_state = env._feature_state  # noqa: SLF001

    # 先把所有任务的“已消费样本版本号”对齐到当前 monitor 计数，
    # 并记录初始 EMA 与 history 长度，构造“无新样本重复 step”场景。
    for task in env.ordered_tasks:
        name = task.name
        feature_state.last_seen_completion_count[name] = env._monitor.completed_job_count_by_task.get(name, 0)  # noqa: SLF001
    ema_before = dict(feature_state.ema_cost)
    hist_len_before = {name: len(history) for name, history in feature_state.cost_history.items()}

    # 人工构造“本 step 没有任何新 job_start / overrun / completion”的更新调用。
    env._update_feature_state(  # noqa: SLF001
        delta_job_start=0,
        delta_mode_changes=0,
        delta_lo_cancellations=0,
        delta_hi_overrun=0,
        delta_lo_overrun=0,
    )

    ema_after = dict(feature_state.ema_cost)
    hist_len_after = {name: len(history) for name, history in feature_state.cost_history.items()}
    assert ema_before == ema_after
    assert hist_len_before == hist_len_after
