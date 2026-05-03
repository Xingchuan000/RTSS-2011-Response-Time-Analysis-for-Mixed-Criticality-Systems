"""DQN 实验工厂抽象测试。"""

from __future__ import annotations

from amc_py.dqn import (
    ExperimentConfig,
    build_env_from_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.dqn.experiment import build_seeded_taskset
from amc_py.rl.observation import TaskNormalizationBound
from amc_py.runtime_models import RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def test_default_small_stress_config_can_build_env() -> None:
    """默认 small stress 配置应能直接构造环境。"""

    env = build_env_from_experiment_config(
        build_small_stress_experiment_config(),
        seed=0,
        end_time=50,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
    )
    obs = env.reset(seed=0)
    assert len(obs.state_vector) == 2 * len(env.ordered_tasks)


def test_custom_normalization_bounds_can_build_env() -> None:
    """自定义 normalization bounds 应可注入到环境中。"""

    def custom_bounds(tasks):
        return {task.name: TaskNormalizationBound(min_cost=0.0, max_cost=20.0) for task in tasks}

    config = ExperimentConfig(
        name="custom_bounds",
        taskset_factory=build_seeded_taskset,
        scenario_factory=lambda seed, tasks: make_nominal_scenario(),
        normalization_bounds_factory=custom_bounds,
        check_safety=True,
    )
    env = build_env_from_experiment_config(
        config,
        seed=3,
        end_time=40,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
    )
    obs = env.reset(seed=3)
    assert all(0.0 <= value <= 1.0 for value in obs.state_vector)


def test_different_seed_changes_bundle_and_same_seed_is_reproducible() -> None:
    """不同 seed 应可生成不同 bundle，而同 seed 应保持可复现。"""

    config = ExperimentConfig(
        name="seeded",
        taskset_factory=build_seeded_taskset,
        scenario_factory=lambda seed, tasks: make_nominal_scenario(),
        normalization_bounds_factory=None,
        check_safety=True,
    )

    bundle_a = resolve_experiment_bundle(config, 7)
    bundle_b = resolve_experiment_bundle(config, 7)
    bundle_c = resolve_experiment_bundle(config, 8)

    assert bundle_a.ordered_tasks == bundle_b.ordered_tasks
    assert bundle_a.normalization_bounds == bundle_b.normalization_bounds
    assert bundle_a.ordered_tasks != bundle_c.ordered_tasks
