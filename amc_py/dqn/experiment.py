"""DQN 实验环境构造工厂。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random

from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.observation import NormalizationBounds, build_default_normalization_bounds
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import ExecutionScenario, make_nominal_scenario, make_table_scenario

TasksetFactory = Callable[[int], list[Task]]
ScenarioFactory = Callable[[int, Sequence[Task]], ExecutionScenario]
NormalizationBoundsFactory = Callable[[Sequence[Task]], NormalizationBounds]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """描述一组可复用的训练/评估环境构造参数。"""

    name: str
    taskset_factory: TasksetFactory
    scenario_factory: ScenarioFactory
    normalization_bounds_factory: NormalizationBoundsFactory | None = None
    check_safety: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentBundle:
    """保存一次实验实例化后的任务集、场景与归一化边界。"""

    ordered_tasks: tuple[Task, ...]
    scenario: ExecutionScenario
    normalization_bounds: NormalizationBounds


def build_small_taskset(seed: int = 0) -> list[Task]:  # noqa: ARG001
    """构造默认 small 配置使用的小任务集。"""

    return [
        Task("T1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task("T2", period=15, deadline=15, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("T3", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO),
    ]


def build_small_stress_scenario(seed: int, tasks: Sequence[Task]) -> ExecutionScenario:  # noqa: ARG001
    """构造默认 small stress 场景。"""

    return make_table_scenario(
        actual_costs={
            ("T2", 0): 5,
            ("T2", 1): 5,
            ("T2", 2): 4,
            ("T2", 3): 5,
            ("T3", 1): 5,
            ("T1", 0): 3,
            ("T1", 2): 3,
        },
        default_hi="c_lo",
        default_lo="c_lo",
    )


def build_small_nominal_scenario(seed: int, tasks: Sequence[Task]) -> ExecutionScenario:  # noqa: ARG001
    """构造默认 nominal 场景。"""

    return make_nominal_scenario()


def build_small_stress_experiment_config() -> ExperimentConfig:
    """返回默认的 small stress 实验配置。"""

    return ExperimentConfig(
        name="small_stress",
        taskset_factory=build_small_taskset,
        scenario_factory=build_small_stress_scenario,
        normalization_bounds_factory=build_default_normalization_bounds,
        check_safety=True,
    )


def build_small_nominal_experiment_config() -> ExperimentConfig:
    """返回默认的 small nominal 实验配置。"""

    return ExperimentConfig(
        name="small_nominal",
        taskset_factory=build_small_taskset,
        scenario_factory=build_small_nominal_scenario,
        normalization_bounds_factory=build_default_normalization_bounds,
        check_safety=True,
    )


def resolve_experiment_bundle(config: ExperimentConfig, seed: int) -> ExperimentBundle:
    """使用实验工厂解析出任务集、场景与归一化边界。"""

    ordered_tasks = tuple(config.taskset_factory(seed))
    scenario = config.scenario_factory(seed, ordered_tasks)
    normalization_bounds = (
        config.normalization_bounds_factory(ordered_tasks)
        if config.normalization_bounds_factory is not None
        else build_default_normalization_bounds(ordered_tasks)
    )
    return ExperimentBundle(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        normalization_bounds=normalization_bounds,
    )


def build_env_from_experiment_config(
    config: ExperimentConfig,
    *,
    seed: int,
    end_time: int,
    agent_period: int,
    semantics: RuntimeSemantics = RuntimeSemantics.AMC_PLUS,
) -> AmcBudgetEnv:
    """根据实验配置构造 `AmcBudgetEnv`，供训练与评估入口复用。"""

    bundle = resolve_experiment_bundle(config, seed)
    return AmcBudgetEnv(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        runtime_config=RuntimeConfig(end_time=end_time, semantics=semantics),
        agent_period=agent_period,
        check_safety=config.check_safety,
        normalization_bounds=bundle.normalization_bounds,
    )


def build_seeded_taskset(seed: int) -> list[Task]:
    """构造一个用于测试工厂抽象的种子驱动任务集。"""

    rng = random.Random(seed)
    hi_budget = 3 + rng.randint(0, 1)
    lo_budget = 2 + rng.randint(0, 1)
    return [
        Task("H", period=10, deadline=10, c_lo=2, c_hi=hi_budget, criticality=Criticality.HI),
        Task("L1", period=15, deadline=15, c_lo=lo_budget, c_hi=lo_budget, criticality=Criticality.LO),
        Task("L2", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO),
    ]
