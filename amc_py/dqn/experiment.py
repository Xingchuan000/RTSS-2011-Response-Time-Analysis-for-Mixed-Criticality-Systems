"""DQN 实验环境构造工厂。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random

from amc_py.experiments import evaluate_taskset, resolve_ordering
from amc_py.generator import generate_taskset
from amc_py.models import Criticality, SchedulabilityResult, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.observation import NormalizationBounds, build_default_normalization_bounds
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import (
    ExecutionScenario,
    make_nominal_scenario,
    make_rtss11_random_scenario,
    make_table_scenario,
)

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


@dataclass(frozen=True, slots=True)
class Rtss11TasksetBundle:
    """保存可调度 RTSS2011 任务集筛选结果。"""

    tasks: tuple[Task, ...]
    analysis: SchedulabilityResult
    seed: int
    attempts: int


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


def build_rtss11_taskset(
    seed: int,
    total_util: float = 0.65,
    num_tasks: int = 20,
    cf: float = 2.0,
    cp: float = 0.5,
) -> list[Task]:
    """构造 RTSS2011 风格任务集（仅生成，不做可调度筛选）。

    说明：
    - 该工厂只负责按照 RTSS2011 常见参数口径生成任务集；
    - 不在本函数内执行 AMC-rtb 分析或筛选，便于后续阶段单独复用；
    - 通过固定 seed 传入到底层 generator，保证同参可复现。
    """

    return generate_taskset(
        num_tasks=num_tasks,
        total_util=total_util,
        min_period=10,
        max_period=1000,
        time_scale=100,
        cf=cf,
        cp=cp,
        seed=seed,
        deadline_mode="implicit",
        criticality_assignment="bernoulli",
    )


def build_schedulable_rtss11_taskset(
    seed: int,
    total_util: float = 0.65,
    num_tasks: int = 20,
    cf: float = 2.0,
    cp: float = 0.5,
    max_attempts: int = 100,
) -> Rtss11TasksetBundle:
    """构造 AMC-rtb 可调度的 RTSS2011 任务集。

    实现约束：
    - 每次尝试使用 `seed + offset` 生成任务集；
    - 固定使用 AMC-rtb + OPA 进行筛选；
    - 找到可调度任务集后立刻返回；
    - 超过 `max_attempts` 仍失败时抛出包含末次分析信息的异常。
    """

    if max_attempts <= 0:
        raise ValueError("max_attempts 必须为正整数")

    last_analysis: SchedulabilityResult | None = None
    last_seed: int | None = None

    for offset in range(max_attempts):
        candidate_seed = seed + offset
        tasks = build_rtss11_taskset(
            seed=candidate_seed,
            total_util=total_util,
            num_tasks=num_tasks,
            cf=cf,
            cp=cp,
        )
        analysis = evaluate_taskset(tasks, method="amc_rtb", priority_policy="opa")
        if analysis.schedulable:
            return Rtss11TasksetBundle(
                tasks=tuple(tasks),
                analysis=analysis,
                seed=candidate_seed,
                attempts=offset + 1,
            )
        last_analysis = analysis
        last_seed = candidate_seed

    assert last_analysis is not None
    assert last_seed is not None
    raise RuntimeError(
        "未能在限定尝试次数内生成 AMC-rtb 可调度 RTSS2011 任务集："
        f"max_attempts={max_attempts}, last_seed={last_seed}, "
        f"last_details={last_analysis.details}"
    )


def build_rtss11_experiment_config(
    total_util: float = 0.65,
    num_tasks: int = 20,
    cf: float = 2.0,
    cp: float = 0.5,
    require_schedulable: bool = True,
    hi_overrun_prob: float = 0.05,
    lo_overrun_prob: float = 0.10,
    lo_overrun_factor: float = 1.5,
    max_attempts: int = 100,
) -> ExperimentConfig:
    """构造可直接接入训练/评估流程的 RTSS2011 实验配置。"""

    # 使用缓存保证同一外部 seed 下 taskset/scenario/bounds 三者一致。
    taskset_cache: dict[int, tuple[Task, ...]] = {}
    taskset_signature_cache: dict[tuple[tuple[str, int, int, int, int, str], ...], tuple[Task, ...]] = {}

    # 约定一个稳定的 seed 派生规则，显式区分 taskset 与 scenario 随机源。
    # 这样同一实验可复现，同时 taskset_seed 与 scenario_seed 互不干扰。
    def _derive_taskset_seed(seed: int) -> int:
        return seed * 2

    def _derive_scenario_seed(seed: int) -> int:
        return seed * 2 + 1

    def _task_signature(tasks: Sequence[Task]) -> tuple[tuple[str, int, int, int, int, str], ...]:
        """将任务集编码为稳定签名，便于跨工厂回查缓存。"""

        return tuple(
            (task.name, task.period, task.deadline, task.c_lo, task.c_hi, task.criticality.value)
            for task in tasks
        )

    def taskset_factory(seed: int) -> list[Task]:
        """根据配置生成（或复用）RTSS2011 任务集。"""

        if seed not in taskset_cache:
            taskset_seed = _derive_taskset_seed(seed)
            if require_schedulable:
                bundle = build_schedulable_rtss11_taskset(
                    seed=taskset_seed,
                    total_util=total_util,
                    num_tasks=num_tasks,
                    cf=cf,
                    cp=cp,
                    max_attempts=max_attempts,
                )
                tasks = bundle.tasks
            else:
                tasks = tuple(
                    build_rtss11_taskset(
                        seed=taskset_seed,
                        total_util=total_util,
                        num_tasks=num_tasks,
                        cf=cf,
                        cp=cp,
                    )
                )
            # 运行时环境要求“有序任务集”语义，这里统一按 AMC-rtb+OPA 解析顺序。
            ordered_tasks = tuple(resolve_ordering(list(tasks), priority_policy="opa", method="amc_rtb"))
            taskset_cache[seed] = ordered_tasks
            taskset_signature_cache[_task_signature(ordered_tasks)] = ordered_tasks
        return list(taskset_cache[seed])

    def scenario_factory(seed: int, tasks: Sequence[Task]) -> ExecutionScenario:
        """基于 RTSS2011 过载参数构造随机执行场景。"""

        signature = _task_signature(tasks)
        if signature not in taskset_signature_cache:
            # 当调用方绕过 taskset_factory 直接注入任务集时，这里仍保持可用。
            taskset_signature_cache[signature] = tuple(tasks)
        scenario_seed = _derive_scenario_seed(seed)
        return make_rtss11_random_scenario(
            tasks=taskset_signature_cache[signature],
            seed=scenario_seed,
            hi_overrun_prob=hi_overrun_prob,
            lo_overrun_prob=lo_overrun_prob,
            lo_overrun_factor=lo_overrun_factor,
        )

    util_tag = int(round(total_util * 1000))
    cf_tag = int(round(cf * 10))
    cp_tag = int(round(cp * 100))
    config_name = f"rtss11_n{num_tasks}_u{util_tag:03d}_cf{cf_tag:02d}_cp{cp_tag:02d}"

    return ExperimentConfig(
        name=config_name,
        taskset_factory=taskset_factory,
        scenario_factory=scenario_factory,
        normalization_bounds_factory=build_default_normalization_bounds,
        check_safety=True,
    )
