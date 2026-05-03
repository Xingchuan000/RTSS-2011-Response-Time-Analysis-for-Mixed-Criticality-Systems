"""接近论文设定的 automotive workload 生成器。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import random

from amc_py.dqn.experiment import ExperimentConfig
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality, Task
from amc_py.rl.observation import NormalizationBounds, TaskNormalizationBound
from amc_py.runtime_scenarios import ExecutionScenario

# Mendes 风格 automotive workload 常用的周期集合。
AUTOMOTIVE_PERIOD_SET: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 1000)

# 周期分布采用离散权重采样，让中短周期 runnable 更常见。
AUTOMOTIVE_PERIOD_WEIGHTS: tuple[float, ...] = (0.03, 0.05, 0.10, 0.17, 0.20, 0.18, 0.14, 0.08, 0.05)


@dataclass(frozen=True, slots=True)
class AutomotiveWorkloadConfig:
    """automotive workload 生成配置。"""

    num_runnables: int = 150
    seed: int = 0
    hi_probability: float = 0.35
    weibull_shape: float = 2.0
    lo_budget_quantile: float = 0.7
    period_scale: int = 100
    require_schedulable: bool = False
    max_attempts: int = 50

    def __post_init__(self) -> None:
        """统一校验配置合法性。"""

        if self.num_runnables not in {150, 250}:
            raise ValueError("num_runnables 仅支持 150 或 250")
        if not (0.0 < self.hi_probability < 1.0):
            raise ValueError("hi_probability 必须在 (0,1) 区间内")
        if self.weibull_shape <= 0.0:
            raise ValueError("weibull_shape 必须为正数")
        if not (0.0 <= self.lo_budget_quantile <= 1.0):
            raise ValueError("lo_budget_quantile 必须在 [0,1] 区间内")
        if self.period_scale <= 0:
            raise ValueError("period_scale 必须为正整数")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts 必须为正整数")


@dataclass(frozen=True, slots=True)
class AutomotiveRunnable:
    """单个 runnable 的抽样结果。"""

    name: str
    period: int
    criticality: Criticality
    bcet: int
    acet: int
    wcet: int


@dataclass(frozen=True, slots=True)
class AutomotiveTaskMeta:
    """聚合后任务对应的统计信息。"""

    name: str
    period: int
    criticality: Criticality
    bcet: int
    acet: int
    wcet: int
    runnable_count: int


@dataclass(frozen=True, slots=True)
class AutomotiveWorkload:
    """完整的 automotive workload 结果。"""

    config: AutomotiveWorkloadConfig
    runnables: tuple[AutomotiveRunnable, ...]
    tasks: tuple[Task, ...]
    task_meta: tuple[AutomotiveTaskMeta, ...]
    normalization_bounds: NormalizationBounds


def _stable_name_value(name: str) -> int:
    """把任务名转换为稳定整数，避免依赖 Python hash 随机化。"""

    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def _sample_period(rng: random.Random) -> int:
    """按离散分布采样 runnable 周期。"""

    return rng.choices(AUTOMOTIVE_PERIOD_SET, weights=AUTOMOTIVE_PERIOD_WEIGHTS, k=1)[0]


def _sample_runnable(idx: int, config: AutomotiveWorkloadConfig, rng: random.Random) -> AutomotiveRunnable:
    """采样单个 runnable。"""

    period = _sample_period(rng)
    criticality = Criticality.HI if rng.random() < config.hi_probability else Criticality.LO
    scaled_period = period * config.period_scale

    # 为了让 150/250 runnables 聚合后仍有机会通过 AMC-rtb，这里将单个 runnable 的利用率控制在较低范围。
    wcet_util = 10 ** rng.uniform(-3.3, -1.9)
    wcet = max(2, int(round(scaled_period * wcet_util)))
    bcet = max(1, min(wcet - 1, int(round(wcet * rng.uniform(0.35, 0.65)))))
    acet = max(bcet, min(wcet, int(round(wcet * rng.uniform(0.60, 0.90)))))

    return AutomotiveRunnable(
        name=f"r{idx}",
        period=period,
        criticality=criticality,
        bcet=bcet,
        acet=acet,
        wcet=wcet,
    )


def sample_automotive_runnables(config: AutomotiveWorkloadConfig) -> tuple[AutomotiveRunnable, ...]:
    """按配置采样完整 runnable 集合。"""

    rng = random.Random(config.seed)
    return tuple(_sample_runnable(idx, config, rng) for idx in range(config.num_runnables))


def _compute_c_lo(meta: AutomotiveTaskMeta, lo_budget_quantile: float) -> int:
    """根据 BCET/WCET 区间和 quantile 选择 LO 预算。"""

    return max(1, int(round(meta.bcet + lo_budget_quantile * (meta.wcet - meta.bcet))))


def aggregate_runnables_to_tasks(
    runnables: Sequence[AutomotiveRunnable],
    lo_budget_quantile: float,
    period_scale: int,
) -> tuple[tuple[Task, ...], tuple[AutomotiveTaskMeta, ...], NormalizationBounds]:
    """将 runnables 聚合为 period x criticality 层面的任务。"""

    grouped: dict[tuple[int, Criticality], list[AutomotiveRunnable]] = defaultdict(list)
    for runnable in runnables:
        grouped[(runnable.period, runnable.criticality)].append(runnable)

    tasks: list[Task] = []
    meta_list: list[AutomotiveTaskMeta] = []
    bounds: NormalizationBounds = {}

    for raw_period, criticality in sorted(grouped.keys(), key=lambda item: (item[0], item[1].value)):
        bucket = grouped[(raw_period, criticality)]
        task_period = raw_period * period_scale
        bcet_sum = sum(item.bcet for item in bucket)
        acet_sum = sum(item.acet for item in bucket)
        wcet_sum = sum(item.wcet for item in bucket)
        task_name = f"auto_{criticality.value.lower()}_p{raw_period}"
        meta = AutomotiveTaskMeta(
            name=task_name,
            period=task_period,
            criticality=criticality,
            bcet=bcet_sum,
            acet=acet_sum,
            wcet=wcet_sum,
            runnable_count=len(bucket),
        )
        meta_list.append(meta)

        c_lo = _compute_c_lo(meta, lo_budget_quantile)
        c_hi = meta.wcet if criticality is Criticality.HI else c_lo
        tasks.append(
            Task(
                name=task_name,
                period=task_period,
                deadline=task_period,
                c_lo=c_lo,
                c_hi=max(c_lo, c_hi),
                criticality=criticality,
            )
        )
        bounds[task_name] = TaskNormalizationBound(min_cost=float(meta.bcet), max_cost=float(meta.wcet))

    return tuple(tasks), tuple(meta_list), bounds


def build_automotive_execution_scenario(
    workload: AutomotiveWorkload,
    *,
    scenario_seed: int | None = None,
) -> ExecutionScenario:
    """基于 Weibull 执行时间抽样构造 runtime 场景。"""

    scenario_seed = workload.config.seed if scenario_seed is None else scenario_seed
    meta_map = {meta.name: meta for meta in workload.task_meta}

    def resolver(task: Task, release_index: int) -> int:
        meta = meta_map[task.name]
        sample_seed = (
            scenario_seed * 1_000_003
            + _stable_name_value(task.name) * 1_009
            + release_index * 9_173
        )
        rng = random.Random(sample_seed)

        # 由 ACET 反推 Weibull scale，使均值更接近 ACET，同时再裁剪到 [BCET, WCET]。
        scale = meta.acet / math.gamma(1.0 + 1.0 / workload.config.weibull_shape)
        sampled = rng.weibullvariate(max(scale, 1e-6), workload.config.weibull_shape)
        bounded = min(float(meta.wcet), max(float(meta.bcet), sampled))
        return int(round(bounded))

    scenario_name = f"automotive_weibull_{workload.config.num_runnables}_{scenario_seed}"
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def generate_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """生成一次 automotive workload。"""

    runnables = sample_automotive_runnables(config)
    tasks, task_meta, bounds = aggregate_runnables_to_tasks(
        runnables,
        config.lo_budget_quantile,
        config.period_scale,
    )
    return AutomotiveWorkload(
        config=config,
        runnables=runnables,
        tasks=tasks,
        task_meta=task_meta,
        normalization_bounds=bounds,
    )


def generate_schedulable_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """按 seed 递增搜索一个 AMC-rtb 可调度的 automotive workload。"""

    for offset in range(config.max_attempts):
        attempt_config = AutomotiveWorkloadConfig(
            num_runnables=config.num_runnables,
            seed=config.seed + offset,
            hi_probability=config.hi_probability,
            weibull_shape=config.weibull_shape,
            lo_budget_quantile=config.lo_budget_quantile,
            period_scale=config.period_scale,
            require_schedulable=config.require_schedulable,
            max_attempts=config.max_attempts,
        )
        workload = generate_automotive_workload(attempt_config)
        result = evaluate_taskset(list(workload.tasks), method="amc_rtb", priority_policy="dm")
        if result.schedulable:
            return workload
    raise RuntimeError("在 max_attempts 范围内未找到 AMC-rtb 可调度的 automotive workload")


def build_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """根据 require_schedulable 选项生成 workload。"""

    if config.require_schedulable:
        return generate_schedulable_automotive_workload(config)
    return generate_automotive_workload(config)


def build_automotive_experiment_config(
    *,
    num_runnables: int = 150,
    hi_probability: float = 0.35,
    weibull_shape: float = 2.0,
    lo_budget_quantile: float = 0.7,
    period_scale: int = 100,
    require_schedulable: bool = True,
    max_attempts: int = 50,
) -> ExperimentConfig:
    """构造可直接接入 DQN 实验工厂的 automotive 配置。"""

    workload_cache: dict[int, AutomotiveWorkload] = {}
    task_signature_cache: dict[tuple[tuple[str, int, int, int, int, str], ...], AutomotiveWorkload] = {}

    def get_workload(seed: int) -> AutomotiveWorkload:
        """按 seed 缓存 workload，保证 taskset/scenario/bounds 一致。"""

        if seed not in workload_cache:
            workload = build_automotive_workload(
                AutomotiveWorkloadConfig(
                    num_runnables=num_runnables,
                    seed=seed,
                    hi_probability=hi_probability,
                    weibull_shape=weibull_shape,
                    lo_budget_quantile=lo_budget_quantile,
                    period_scale=period_scale,
                    require_schedulable=require_schedulable,
                    max_attempts=max_attempts,
                )
            )
            workload_cache[seed] = workload
            signature = tuple(
                (task.name, task.period, task.deadline, task.c_lo, task.c_hi, task.criticality.value)
                for task in workload.tasks
            )
            task_signature_cache[signature] = workload
        return workload_cache[seed]

    def taskset_factory(seed: int) -> list[Task]:
        workload = get_workload(seed)
        return list(workload.tasks)

    def scenario_factory(seed: int, tasks: Sequence[Task]) -> ExecutionScenario:  # noqa: ARG001
        workload = get_workload(seed)
        return build_automotive_execution_scenario(workload, scenario_seed=seed)

    def bounds_factory(tasks: Sequence[Task]) -> NormalizationBounds:
        signature = tuple(
            (task.name, task.period, task.deadline, task.c_lo, task.c_hi, task.criticality.value)
            for task in tasks
        )
        try:
            workload = task_signature_cache[signature]
        except KeyError as exc:
            raise RuntimeError("automotive experiment bounds 与 taskset 不匹配") from exc
        return workload.normalization_bounds

    return ExperimentConfig(
        name=f"automotive_{num_runnables}",
        taskset_factory=taskset_factory,
        scenario_factory=scenario_factory,
        normalization_bounds_factory=bounds_factory,
        check_safety=True,
    )
