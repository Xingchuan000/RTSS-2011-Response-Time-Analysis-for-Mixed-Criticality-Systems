"""接近论文设定的 automotive workload provider。

该模块只负责 workload 层能力：
- 生成 runnable；
- 聚合为任务；
- 构造 runtime scenario；
- 生成 normalization bounds；
- 可选执行可调度筛选。

这里不允许依赖 DQN experiment、agent、env 或训练脚本。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import random
from typing import Literal

from amc_py.models import Criticality, Task
from amc_py.rl.observation import NormalizationBounds, TaskNormalizationBound
from amc_py.runtime_scenarios import ExecutionScenario
from amc_py.workloads.base import WorkloadBundle, WorkloadProvider

# Mendes 风格 automotive workload 常用的周期集合。
AUTOMOTIVE_PERIOD_SET: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 1000)

# fast / paper_like 继续保留当前近似实现使用的离散权重采样方式。
AUTOMOTIVE_PERIOD_WEIGHTS: tuple[float, ...] = (0.03, 0.05, 0.10, 0.17, 0.20, 0.18, 0.14, 0.08, 0.05)

# paper_exact 模式使用文档要求的固定 period 占比。
AUTOMOTIVE_PERIOD_SHARES: tuple[float, ...] = (0.04, 0.02, 0.02, 0.29, 0.29, 0.04, 0.24, 0.01, 0.05)

# Table 2：各 period 桶的 (min_acet_us, avg_acet_us, max_acet_us)。
ACET_TABLE_US: dict[int, tuple[float, float, float]] = {
    1: (0.34, 5.00, 30.11),
    2: (0.32, 4.20, 40.69),
    5: (0.36, 11.04, 83.36),
    10: (0.21, 10.09, 309.87),
    20: (0.25, 8.74, 291.42),
    50: (0.29, 17.56, 92.98),
    100: (0.21, 10.53, 420.43),
    200: (0.22, 2.56, 21.95),
    1000: (0.37, 0.43, 0.46),
}

# Table 3：各 period 桶的 BCET/WCET factor 区间。
FACTOR_TABLE: dict[int, tuple[float, float, float, float]] = {
    1: (0.19, 0.92, 1.30, 29.11),
    2: (0.12, 0.89, 1.54, 19.04),
    5: (0.17, 0.94, 1.13, 18.44),
    10: (0.05, 0.99, 1.06, 30.03),
    20: (0.11, 0.98, 1.06, 15.61),
    50: (0.32, 0.95, 1.13, 7.76),
    100: (0.09, 0.99, 1.02, 8.88),
    200: (0.45, 0.98, 1.03, 4.90),
    1000: (0.68, 0.80, 1.84, 4.75),
}

# Table 4：LO mode budget quantile。
L_WCET_PROB: dict[int, dict[Criticality, float]] = {
    1: {Criticality.LO: 0.75, Criticality.HI: 0.80},
    2: {Criticality.LO: 0.75, Criticality.HI: 0.80},
    5: {Criticality.LO: 0.75, Criticality.HI: 0.80},
    10: {Criticality.LO: 0.67, Criticality.HI: 0.75},
    20: {Criticality.LO: 0.67, Criticality.HI: 0.75},
    50: {Criticality.LO: 0.67, Criticality.HI: 0.75},
    100: {Criticality.LO: 0.50, Criticality.HI: 0.67},
    200: {Criticality.LO: 0.50, Criticality.HI: 0.67},
    1000: {Criticality.LO: 0.50, Criticality.HI: 0.67},
}

AutomotiveMode = Literal["fast", "paper_like", "paper_exact"]


@dataclass(frozen=True, slots=True)
class AutomotiveWorkloadConfig:
    """automotive workload 生成配置。

    设计约束：
    - 该配置仅描述 workload 生成语义，不承载训练器或 agent 参数；
    - `mode` 用于显式区分近似实现与论文精确实现；
    - `paper_exact` 使用文档给定的 Table 1/2/3/4 与 runnable-level sampling 语义；
    - `fast` / `paper_like` 保留现有近似实现，方便快速实验与历史行为兼容。
    """

    num_runnables: int = 150
    seed: int = 0
    mode: AutomotiveMode = "paper_like"
    tick_ns: int = 10
    hi_probability: float = 0.5
    require_schedulable: bool = False
    max_attempts: int = 50
    priority_policy: str = "dm"
    sched_method: str = "amc_rtb"
    period_scale: int = 100
    weibull_shape: float = 2.0
    lo_budget_quantile: float = 0.7

    def __post_init__(self) -> None:
        """严格校验文档要求的基础配置约束。"""

        if self.num_runnables not in {150, 250}:
            raise ValueError("num_runnables must be 150 or 250")
        if self.tick_ns <= 0:
            raise ValueError("tick_ns must be positive")
        if not 0.0 < self.hi_probability < 1.0:
            raise ValueError("hi_probability must be in (0, 1)")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.weibull_shape <= 0.0:
            raise ValueError("weibull_shape must be positive")
        if not 0.0 <= self.lo_budget_quantile <= 1.0:
            raise ValueError("lo_budget_quantile must be in [0, 1]")
        if self.period_scale <= 0:
            raise ValueError("period_scale must be positive")


@dataclass(frozen=True, slots=True)
class AutomotiveRunnable:
    """单个 runnable 的抽样结果。"""

    name: str
    period_ms: int
    criticality: Criticality
    bcet: int
    acet: int
    wcet: int
    weibull_shape: float | None = None
    weibull_scale: float | None = None
    weibull_location: int | None = None


@dataclass(frozen=True, slots=True)
class AutomotiveTaskMeta:
    """聚合后任务对应的统计信息。"""

    name: str
    period: int
    period_ms: int
    criticality: Criticality
    bcet: int
    acet: int
    wcet: int
    runnable_count: int
    runnable_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AutomotiveWorkload:
    """完整的 automotive workload 结果。"""

    config: AutomotiveWorkloadConfig
    runnables: tuple[AutomotiveRunnable, ...]
    tasks: tuple[Task, ...]
    task_meta: tuple[AutomotiveTaskMeta, ...]
    normalization_bounds: NormalizationBounds
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class AutomotiveWorkloadProvider:
    """将 automotive workload 暴露为统一 provider 接口。"""

    config: AutomotiveWorkloadConfig
    fixed_taskset_seed: int | None = None
    scenario_seed_offset: int = 0
    name: str = "automotive"

    def build(self, seed: int) -> WorkloadBundle:
        """根据外部 seed 构造完整 workload bundle。

        说明：
        - `fixed_taskset_seed` 用于固定 workload / taskset 的随机源，保证不同 episode
          或评估 seed 下 task 数量与动作空间维度保持不变；
        - `scenario_seed_offset` 用于把 scenario 的随机源与 taskset 随机源显式拆开，
          这样在固定 taskset 的同时仍能变化执行时间场景。
        """

        taskset_seed = self.fixed_taskset_seed if self.fixed_taskset_seed is not None else seed
        scenario_seed = seed + self.scenario_seed_offset
        effective_config = replace(self.config, seed=taskset_seed)
        workload = build_automotive_workload(effective_config)
        scenario = build_automotive_execution_scenario(workload, scenario_seed=scenario_seed)
        return WorkloadBundle(
            tasks=workload.tasks,
            scenario=scenario,
            normalization_bounds=workload.normalization_bounds,
            taskset_seed=workload.config.seed,
            scenario_seed=scenario_seed,
            attempts=workload.attempts,
            metadata={
                "num_runnables": workload.config.num_runnables,
                "mode": workload.config.mode,
                "task_meta": workload.task_meta,
            },
        )


def _stable_name_value(name: str) -> int:
    """把任务名转换为稳定整数，避免依赖 Python hash 随机化。"""

    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def _stable_seed(base_seed: int, task_name: str, release_index: int) -> int:
    """构造 task/release 粒度的稳定随机种子。"""

    return base_seed * 1_000_003 + _stable_name_value(task_name) * 1_009 + release_index * 9_173


def us_to_ticks(us: float, tick_ns: int = 10) -> int:
    """把 microsecond 转成内部 tick。"""

    return max(1, int(round(us * 1000.0 / tick_ns)))


def ms_to_ticks(ms: int, tick_ns: int = 10) -> int:
    """把 millisecond 转成内部 tick。"""

    return max(1, int(round(ms * 1_000_000.0 / tick_ns)))


def _sample_period(rng: random.Random) -> int:
    """按离散分布采样 runnable 周期。"""

    return rng.choices(AUTOMOTIVE_PERIOD_SET, weights=AUTOMOTIVE_PERIOD_WEIGHTS, k=1)[0]


def _sample_runnable(idx: int, config: AutomotiveWorkloadConfig, rng: random.Random) -> AutomotiveRunnable:
    """采样单个 runnable。

    当前 fast/paper_like 路径继续复用仓库已有近似逻辑：
    - 周期来自离散集合；
    - criticality 按伯努利分配；
    - BCET/ACET/WCET 由低利用率区间缩放得到。
    """

    period_ms = _sample_period(rng)
    criticality = Criticality.HI if rng.random() < config.hi_probability else Criticality.LO
    scaled_period = period_ms * config.period_scale

    # 为了让 150/250 runnables 聚合后仍有机会通过 AMC-rtb，
    # 这里继续将单个 runnable 利用率控制在较低范围。
    wcet_util = 10 ** rng.uniform(-3.3, -1.9)
    wcet = max(2, int(round(scaled_period * wcet_util)))
    bcet = max(1, min(wcet - 1, int(round(wcet * rng.uniform(0.35, 0.65)))))
    acet = max(bcet, min(wcet, int(round(wcet * rng.uniform(0.60, 0.90)))))

    return AutomotiveRunnable(
        name=f"r{idx}",
        period_ms=period_ms,
        criticality=criticality,
        bcet=bcet,
        acet=acet,
        wcet=wcet,
    )


def _largest_remainder_counts(total: int, shares: Sequence[float]) -> list[int]:
    """按 largest remainder 方法把占比稳定分配成整数计数。"""

    raw_counts = [total * share for share in shares]
    counts = [int(math.floor(value)) for value in raw_counts]
    missing = total - sum(counts)
    remainders = sorted(
        ((raw_counts[idx] - counts[idx], idx) for idx in range(len(shares))),
        key=lambda item: (-item[0], item[1]),
    )
    for _remainder, idx in remainders[:missing]:
        counts[idx] += 1
    return counts


def _assign_exact_periods(num_runnables: int, rng: random.Random) -> list[int]:
    """按 Table 1 的占比稳定生成 runnable period 列表。"""

    counts = _largest_remainder_counts(num_runnables, AUTOMOTIVE_PERIOD_SHARES)
    periods: list[int] = []
    for period_ms, count in zip(AUTOMOTIVE_PERIOD_SET, counts, strict=True):
        periods.extend([period_ms] * count)
    rng.shuffle(periods)
    return periods


def _assign_exact_criticalities(num_runnables: int, hi_probability: float, rng: random.Random) -> list[Criticality]:
    """为 paper_exact 稳定分配关键级。

    这里使用“先固定总数、再 shuffle”的方式，而不是逐个伯努利采样，
    目的是让 `hi_probability=0.5` 时更贴近论文的 50/50 语义。
    """

    hi_count = int(round(num_runnables * hi_probability))
    hi_count = max(0, min(num_runnables, hi_count))
    criticalities = [Criticality.HI] * hi_count + [Criticality.LO] * (num_runnables - hi_count)
    rng.shuffle(criticalities)
    return criticalities


def _sample_bounded_sum_values(
    count: int,
    total: float,
    lower_bound: float,
    upper_bound: float,
    rng: random.Random,
) -> list[float]:
    """在固定总和与统一上下界约束下，直接构造一组可行浮点值。

    设计目的：
    - 彻底替代 `UUniFast + rejection sampling` 的无界重试模式；
    - 对像 1000 ms 这类上下界极窄的 period bucket，也保证有限步内返回；
    - 始终满足：
      1) 每个值都位于 `[lower_bound, upper_bound]`；
      2) 所有值的和严格等于 `total`（浮点误差范围内）。
    """

    if count <= 0:
        raise ValueError("count must be positive")
    min_total = count * lower_bound
    max_total = count * upper_bound
    if total < min_total or total > max_total:
        raise ValueError("requested total is outside feasible bounded range")

    values: list[float] = []
    remaining_total = total
    remaining_count = count
    for _ in range(count - 1):
        remaining_count -= 1
        low = max(lower_bound, remaining_total - remaining_count * upper_bound)
        high = min(upper_bound, remaining_total - remaining_count * lower_bound)
        values.append(rng.uniform(low, high))
        remaining_total -= values[-1]
    values.append(remaining_total)
    rng.shuffle(values)
    return values


def _sample_exact_acet_us_values(period_ms: int, count: int, rng: random.Random) -> list[float]:
    """按 Table 2 为同一 period 桶采样一组 ACET（单位 microsecond）。

    原实现使用 `UUniFast + rejection sampling`，在上下界极窄的 bucket
    （例如 1000 ms）上可能极慢甚至近似卡住。这里改为直接构造满足：
    - 单个 ACET 在 `[min, max]` 内；
    - 总和等于 `count * avg`；
    的一组值，从根源上消除无界重试风险。
    """

    min_acet_us, avg_acet_us, max_acet_us = ACET_TABLE_US[period_ms]
    total_acet_us = count * avg_acet_us
    return _sample_bounded_sum_values(
        count=count,
        total=total_acet_us,
        lower_bound=min_acet_us,
        upper_bound=max_acet_us,
        rng=rng,
    )


def fit_shifted_weibull_from_acet_bcet_wcet(acet: int, bcet: int, wcet: int) -> tuple[float, float, int]:
    """按文档给出的 quantile 约束拟合 shifted Weibull。"""

    loc = bcet
    q_low = 1
    q_high = max(1, wcet - bcet)
    p_low = 0.00001
    p_high = 0.99999

    a = -math.log(1.0 - p_low)
    b = -math.log(1.0 - p_high)
    k = math.log(b / a) / math.log(q_high / q_low)

    mean_shifted = max(1e-9, float(acet - bcet))
    lam = mean_shifted / math.gamma(1.0 + 1.0 / k)
    return k, lam, loc


def sample_runnable_cost(runnable: AutomotiveRunnable, rng: random.Random) -> int:
    """采样单个 runnable 的一次实际执行时间。"""

    if (
        runnable.weibull_shape is None
        or runnable.weibull_scale is None
        or runnable.weibull_location is None
    ):
        raise ValueError("runnable Weibull parameters are missing")
    value = runnable.weibull_location + rng.weibullvariate(runnable.weibull_scale, runnable.weibull_shape)
    return max(runnable.bcet, min(runnable.wcet, int(round(value))))


def _build_exact_runnables(config: AutomotiveWorkloadConfig) -> tuple[AutomotiveRunnable, ...]:
    """按文档 Table 1/2/3/4 语义构造 paper_exact runnables。"""

    rng = random.Random(config.seed)
    periods = _assign_exact_periods(config.num_runnables, rng)
    criticalities = _assign_exact_criticalities(config.num_runnables, config.hi_probability, rng)

    # 先按 period 分桶，再整桶采样 ACET，保证每个 period 桶都严格使用自己的 Table 2 均值。
    period_indices: dict[int, list[int]] = defaultdict(list)
    for idx, period_ms in enumerate(periods):
        period_indices[period_ms].append(idx)

    acet_us_by_index: dict[int, float] = {}
    for period_ms, indices in period_indices.items():
        sampled_acets = _sample_exact_acet_us_values(period_ms, len(indices), rng)
        rng.shuffle(sampled_acets)
        for idx, acet_us in zip(indices, sampled_acets, strict=True):
            acet_us_by_index[idx] = acet_us

    runnables: list[AutomotiveRunnable] = []
    for idx in range(config.num_runnables):
        period_ms = periods[idx]
        criticality = criticalities[idx]
        acet_us = acet_us_by_index[idx]
        bcet_factor_min, bcet_factor_max, wcet_factor_min, wcet_factor_max = FACTOR_TABLE[period_ms]
        bcet_factor = rng.uniform(bcet_factor_min, bcet_factor_max)
        wcet_factor = rng.uniform(wcet_factor_min, wcet_factor_max)

        acet = us_to_ticks(acet_us, config.tick_ns)
        # 论文给出的 Table 3 已经保证 BCET factor < 1、WCET factor > 1，
        # 因此这里直接按表驱动计算，不再加入额外的裁剪或兜底逻辑。
        bcet = us_to_ticks(acet_us * bcet_factor, config.tick_ns)
        wcet = us_to_ticks(acet_us * wcet_factor, config.tick_ns)
        weibull_shape, weibull_scale, weibull_location = fit_shifted_weibull_from_acet_bcet_wcet(
            acet,
            bcet,
            wcet,
        )
        runnables.append(
            AutomotiveRunnable(
                name=f"r{idx}",
                period_ms=period_ms,
                criticality=criticality,
                bcet=bcet,
                acet=acet,
                wcet=wcet,
                weibull_shape=weibull_shape,
                weibull_scale=weibull_scale,
                weibull_location=weibull_location,
            )
        )

    return tuple(runnables)


def sample_automotive_runnables(config: AutomotiveWorkloadConfig) -> tuple[AutomotiveRunnable, ...]:
    """按配置采样完整 runnable 集合。"""

    if config.mode == "paper_exact":
        return _build_exact_runnables(config)

    rng = random.Random(config.seed)
    return tuple(_sample_runnable(idx, config, rng) for idx in range(config.num_runnables))


def _compute_c_lo(meta: AutomotiveTaskMeta, lo_budget_quantile: float) -> int:
    """根据 BCET/WCET 区间和 quantile 选择 LO 预算。"""

    return max(1, int(round(meta.bcet + lo_budget_quantile * (meta.wcet - meta.bcet))))


def estimate_c_lo_from_samples(
    runnables: Sequence[AutomotiveRunnable],
    period_ms: int,
    criticality: Criticality,
    rng: random.Random,
) -> int:
    """按 Table 4 通过 1000 次 runnable-level sampling 估计任务级 C_LO。"""

    probability = L_WCET_PROB[period_ms][criticality]
    samples = [sum(sample_runnable_cost(runnable, rng) for runnable in runnables) for _ in range(1000)]
    samples.sort()
    index = min(999, int(probability * 1000))
    return samples[index]


def aggregate_runnables_to_tasks(
    runnables: Sequence[AutomotiveRunnable],
    lo_budget_quantile: float,
    period_scale: int,
    *,
    mode: AutomotiveMode = "paper_like",
    tick_ns: int = 10,
    rng_seed: int = 0,
) -> tuple[tuple[Task, ...], tuple[AutomotiveTaskMeta, ...], NormalizationBounds]:
    """将 runnables 聚合为 period x criticality 层面的任务。

    参数说明：
    - `mode` 决定 C_LO 的估计方式；
    - `tick_ns` 只在 `paper_exact` 模式下用于 period -> tick 换算；
    - `rng_seed` 只在 `paper_exact` 模式下用于驱动 1000 次采样估计，保证结果可复现。
    """

    grouped: dict[tuple[int, Criticality], list[AutomotiveRunnable]] = defaultdict(list)
    for runnable in runnables:
        grouped[(runnable.period_ms, runnable.criticality)].append(runnable)

    tasks: list[Task] = []
    meta_list: list[AutomotiveTaskMeta] = []
    bounds: NormalizationBounds = {}

    for period_ms, criticality in sorted(grouped.keys(), key=lambda item: (item[0], item[1].value)):
        bucket = grouped[(period_ms, criticality)]
        task_period = ms_to_ticks(period_ms, tick_ns) if mode == "paper_exact" else period_ms * period_scale
        bcet_sum = sum(item.bcet for item in bucket)
        acet_sum = sum(item.acet for item in bucket)
        wcet_sum = sum(item.wcet for item in bucket)
        task_name = f"auto_{criticality.value.lower()}_p{period_ms}"
        meta = AutomotiveTaskMeta(
            name=task_name,
            period=task_period,
            period_ms=period_ms,
            criticality=criticality,
            bcet=bcet_sum,
            acet=acet_sum,
            wcet=wcet_sum,
            runnable_count=len(bucket),
            runnable_names=tuple(item.name for item in bucket),
        )
        meta_list.append(meta)

        if mode == "paper_exact":
            # 对每个 task 单独派生采样种子，保证 quantile 估计对 task 维度稳定可复现。
            c_lo_rng = random.Random(rng_seed + _stable_name_value(task_name))
            c_lo = estimate_c_lo_from_samples(bucket, period_ms, criticality, c_lo_rng)
        else:
            c_lo = _compute_c_lo(meta, lo_budget_quantile)

        c_hi = meta.wcet if criticality is Criticality.HI else c_lo
        tasks.append(
            Task(
                name=task_name,
                period=task_period,
                deadline=task_period,
                c_lo=c_lo,
                c_hi=c_hi,
                criticality=criticality,
            )
        )
        bounds[task_name] = TaskNormalizationBound(min_cost=float(meta.bcet), max_cost=float(meta.wcet))

    return tuple(tasks), tuple(meta_list), bounds


def build_task_to_runnables_map(workload: AutomotiveWorkload) -> dict[str, tuple[AutomotiveRunnable, ...]]:
    """把聚合后的 task 名回映射到组成它的 runnables。"""

    grouped: dict[tuple[int, Criticality], list[AutomotiveRunnable]] = defaultdict(list)
    for runnable in workload.runnables:
        grouped[(runnable.period_ms, runnable.criticality)].append(runnable)

    mapping: dict[str, tuple[AutomotiveRunnable, ...]] = {}
    for meta in workload.task_meta:
        mapping[meta.name] = tuple(grouped[(meta.period_ms, meta.criticality)])
    return mapping


def build_automotive_execution_scenario(
    workload: AutomotiveWorkload,
    *,
    scenario_seed: int | None = None,
) -> ExecutionScenario:
    """构造 runtime 场景。

    模式差异：
    - `paper_exact`：按文档要求，对 task 内所有 runnables 分别采样后求和；
    - `fast` / `paper_like`：保留原有 task-level Weibull 近似采样路径。
    """

    scenario_seed = workload.config.seed if scenario_seed is None else scenario_seed

    if workload.config.mode == "paper_exact":
        task_to_runnables = build_task_to_runnables_map(workload)

        def resolver(task: Task, release_index: int) -> int:
            """按 runnable-level 采样并求和，得到单个 job 的实际执行时间。"""

            rng = random.Random(_stable_seed(scenario_seed, task.name, release_index))
            return sum(sample_runnable_cost(runnable, rng) for runnable in task_to_runnables[task.name])

        return ExecutionScenario(name=f"automotive_{workload.config.mode}_{scenario_seed}", resolver=resolver)

    meta_map = {meta.name: meta for meta in workload.task_meta}

    def resolver(task: Task, release_index: int) -> int:
        """为单个 job 决定一次实际执行时间。"""

        meta = meta_map[task.name]
        rng = random.Random(_stable_seed(scenario_seed, task.name, release_index))

        # 近似模式继续用 task-level ACET 反推 Weibull scale，
        # 然后裁剪到 [BCET, WCET] 区间，保持历史行为不变。
        scale = meta.acet / math.gamma(1.0 + 1.0 / workload.config.weibull_shape)
        sampled = rng.weibullvariate(max(scale, 1e-6), workload.config.weibull_shape)
        bounded = min(float(meta.wcet), max(float(meta.bcet), sampled))
        return int(round(bounded))

    scenario_name = f"automotive_{workload.config.mode}_{workload.config.num_runnables}_{scenario_seed}"
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def generate_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """生成一次 automotive workload。"""

    runnables = sample_automotive_runnables(config)
    tasks, task_meta, bounds = aggregate_runnables_to_tasks(
        runnables,
        config.lo_budget_quantile,
        config.period_scale,
        mode=config.mode,
        tick_ns=config.tick_ns,
        rng_seed=config.seed,
    )
    return AutomotiveWorkload(
        config=config,
        runnables=runnables,
        tasks=tasks,
        task_meta=task_meta,
        normalization_bounds=bounds,
        attempts=1,
    )


def generate_schedulable_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """按 seed 递增搜索一个指定调度分析下可调度的 automotive workload。"""

    from amc_py.experiments import evaluate_taskset

    for offset in range(config.max_attempts):
        attempt_config = replace(config, seed=config.seed + offset)
        workload = generate_automotive_workload(attempt_config)
        result = evaluate_taskset(
            list(workload.tasks),
            method=config.sched_method,
            priority_policy=config.priority_policy,
        )
        if result.schedulable:
            return AutomotiveWorkload(
                config=workload.config,
                runnables=workload.runnables,
                tasks=workload.tasks,
                task_meta=workload.task_meta,
                normalization_bounds=workload.normalization_bounds,
                attempts=offset + 1,
            )
    raise RuntimeError("在 max_attempts 范围内未找到可调度的 automotive workload")


def build_automotive_workload(config: AutomotiveWorkloadConfig) -> AutomotiveWorkload:
    """根据 require_schedulable 选项生成 workload。"""

    if config.require_schedulable:
        return generate_schedulable_automotive_workload(config)
    return generate_automotive_workload(config)


__all__ = [
    "ACET_TABLE_US",
    "AUTOMOTIVE_PERIOD_SET",
    "AUTOMOTIVE_PERIOD_SHARES",
    "AUTOMOTIVE_PERIOD_WEIGHTS",
    "AutomotiveMode",
    "AutomotiveRunnable",
    "AutomotiveTaskMeta",
    "AutomotiveWorkload",
    "AutomotiveWorkloadConfig",
    "AutomotiveWorkloadProvider",
    "FACTOR_TABLE",
    "L_WCET_PROB",
    "aggregate_runnables_to_tasks",
    "build_automotive_execution_scenario",
    "build_automotive_workload",
    "build_task_to_runnables_map",
    "estimate_c_lo_from_samples",
    "fit_shifted_weibull_from_acet_bcet_wcet",
    "generate_automotive_workload",
    "generate_schedulable_automotive_workload",
    "ms_to_ticks",
    "sample_automotive_runnables",
    "sample_runnable_cost",
    "us_to_ticks",
]
