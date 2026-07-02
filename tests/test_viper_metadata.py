"""VIPER metadata helper 测试。"""

from __future__ import annotations

from amc_py.dqn import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.observation_metadata import build_action_definitions, build_observation_feature_names
from amc_py.runtime_models import RuntimeSemantics


def test_observation_feature_names_match_small_env_state_dim() -> None:
    feature_config = FeatureConfig(observation_mode="v11_full_10d")
    env = build_env_from_experiment_config(
        build_small_stress_experiment_config(),
        seed=0,
        end_time=50,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
        feature_config=feature_config,
    )
    obs = env.reset(seed=0)
    feature_names = build_observation_feature_names(env.ordered_tasks, feature_config)
    assert len(feature_names) == len(obs.state_vector)
    assert feature_names[0].startswith("T00.")
    assert any(name.startswith("global.") for name in feature_names)


def test_action_definitions_action_id_matches_index() -> None:
    env = build_env_from_experiment_config(
        build_small_stress_experiment_config(),
        seed=0,
        end_time=50,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
    )
    action_definitions = build_action_definitions(env._actions)  # noqa: SLF001
    assert len(action_definitions) == env.action_space_size
    assert all(row["action_id"] == idx for idx, row in enumerate(action_definitions))
