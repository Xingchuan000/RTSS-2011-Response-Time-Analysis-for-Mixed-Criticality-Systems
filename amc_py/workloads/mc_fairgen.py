"""独立的 MC-FairGen workload provider（Step1-3）。

本模块严格聚焦 workload 层：
- 生成任务集（taskset）；
- 构造 runtime scenario；
- 构造 observation normalization bounds。

不引入训练器、agent、扫描脚本等上层依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
import math
import random

from amc_py.models import Criticality, Task
from amc_py.rl.observation import NormalizationBounds, TaskNormalizationBound
from amc_py.runtime_scenarios import ExecutionScenario
from amc_py.workloads.base import WorkloadBundle, WorkloadProvider

MCFairGenMode = Literal["paper_learnable_headroom"]
MCFairGenPeriodSource = Literal[
    "automotive",
    "controlled_sparse",
    "controlled_medium",
    "controlled_dense",
]

MC_FAIRGEN_AUTOMOTIVE_PERIOD_SET: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 1000)
MC_FAIRGEN_AUTOMOTIVE_PERIOD_WEIGHTS: tuple[float, ...] = (0.03, 0.05, 0.10, 0.17, 0.20, 0.18, 0.14, 0.08, 0.05)
MC_FAIRGEN_CONTROLLED_PERIOD_SETS: dict[str, tuple[int, ...]] = {
    "controlled_sparse": (20, 50, 100, 200, 500),
    "controlled_medium": (10, 20, 50, 100, 200),
    "controlled_dense": (5, 10, 20, 50, 100),
}
MC_FAIRGEN_CONTROLLED_PERIOD_WEIGHTS: dict[str, tuple[float, ...]] = {
    "controlled_sparse": (0.08, 0.18, 0.32, 0.27, 0.15),
    "controlled_medium": (0.10, 0.22, 0.36, 0.22, 0.10),
    "controlled_dense": (0.10, 0.22, 0.36, 0.22, 0.10),
}


@dataclass(frozen=True, slots=True)
class MCFairGenWorkloadConfig:
    """MC-FairGen 任务集与场景生成配置。"""

    seed: int = 0
    mode: MCFairGenMode = "paper_learnable_headroom"

    num_tasks: int = 16
    hi_ratio: float = 0.5
    period_source: MCFairGenPeriodSource = "automotive"
    period_scale: int = 100
    tick_ns: int = 10

    # Multi-utilization-bound controls.
    u_hi_lo_min: float = 0.20
    u_hi_lo_max: float = 0.35
    u_hi_hi_min: float = 0.45
    u_hi_hi_max: float = 0.70
    u_lo_lo_min: float = 0.35
    u_lo_lo_max: float = 0.60

    # Runtime budget placement.
    hi_budget_rho_min: float = 0.55
    hi_budget_rho_max: float = 0.75
    lo_budget_rho_min: float = 0.05
    lo_budget_rho_max: float = 0.25

    # Runtime scenario controls.
    hi_overrun_prob: float = 0.08
    lo_overrun_prob: float = 0.40
    hi_overrun_factor_min: float = 1.02
    hi_overrun_factor_max: float = 1.25
    lo_overrun_factor_min: float = 1.05
    lo_overrun_factor_max: float = 1.80
    nominal_hi_cost_min_ratio: float = 0.65
    nominal_lo_cost_min_ratio: float = 0.60

    # Schedulability search.
    require_schedulable: bool = False
    max_attempts: int = 100
    sched_method: str = "amc_rtb"
    priority_policy: str = "dm"

    def __post_init__(self) -> None:
        """严格校验配置，失败时明确指出字段名。"""

        if self.mode != "paper_learnable_headroom":
            raise ValueError("mode must be 'paper_learnable_headroom'")
        if self.num_tasks < 3:
            raise ValueError("num_tasks must be >= 3")
        if not 0.0 < self.hi_ratio < 1.0:
            raise ValueError("hi_ratio must be in (0, 1)")
        valid_period_sources = {"automotive", *MC_FAIRGEN_CONTROLLED_PERIOD_SETS.keys()}
        if self.period_source not in valid_period_sources:
            raise ValueError(f"period_source must be one of {sorted(valid_period_sources)}")
        if self.period_scale <= 0:
            raise ValueError("period_scale must be > 0")
        if self.tick_ns <= 0:
            raise ValueError("tick_ns must be > 0")

        self._validate_positive_range("u_hi_lo", self.u_hi_lo_min, self.u_hi_lo_max)
        self._validate_positive_range("u_hi_hi", self.u_hi_hi_min, self.u_hi_hi_max)
        self._validate_positive_range("u_lo_lo", self.u_lo_lo_min, self.u_lo_lo_max)

        if self.u_hi_hi_max < self.u_hi_lo_min:
            raise ValueError("u_hi_hi_max must be >= u_hi_lo_min")

        self._validate_zero_one_range("hi_budget_rho", self.hi_budget_rho_min, self.hi_budget_rho_max)
        self._validate_zero_one_range("lo_budget_rho", self.lo_budget_rho_min, self.lo_budget_rho_max)

        self._validate_probability("hi_overrun_prob", self.hi_overrun_prob)
        self._validate_probability("lo_overrun_prob", self.lo_overrun_prob)

        if self.hi_overrun_factor_min < 1.0:
            raise ValueError("hi_overrun_factor_min must be >= 1")
        if self.hi_overrun_factor_max < self.hi_overrun_factor_min:
            raise ValueError("hi_overrun_factor_max must be >= hi_overrun_factor_min")
        if self.lo_overrun_factor_min < 1.0:
            raise ValueError("lo_overrun_factor_min must be >= 1")
        if self.lo_overrun_factor_max < self.lo_overrun_factor_min:
            raise ValueError("lo_overrun_factor_max must be >= lo_overrun_factor_min")

        self._validate_ratio("nominal_hi_cost_min_ratio", self.nominal_hi_cost_min_ratio)
        self._validate_ratio("nominal_lo_cost_min_ratio", self.nominal_lo_cost_min_ratio)

        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

    @staticmethod
    def _validate_positive_range(name: str, lo: float, hi: float) -> None:
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(f"{name}_min/{name}_max must be > 0")
        if lo > hi:
            raise ValueError(f"{name}_min must be <= {name}_max")

    @staticmethod
    def _validate_zero_one_range(name: str, lo: float, hi: float) -> None:
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"{name}_min/{name}_max must be in [0, 1]")
        if lo > hi:
            raise ValueError(f"{name}_min must be <= {name}_max")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class MCFairGenTaskMeta:
    """记录每个任务的生成时中间量，便于调试与场景构造。"""

    name: str
    period: int
    period_ms: int
    criticality: Criticality
    generated_u_lo: float
    generated_u_hi: float
    base_c_lo: int
    base_c_hi: int
    initial_budget: int


@dataclass(frozen=True, slots=True)
class MCFairGenWorkload:
    """MC-FairGen workload 完整结果。"""

    config: MCFairGenWorkloadConfig
    tasks: tuple[Task, ...]
    task_meta: tuple[MCFairGenTaskMeta, ...]
    normalization_bounds: NormalizationBounds
    attempts: int = 1
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MCFairGenWorkloadProvider:
    """把 MC-FairGen 暴露为统一 WorkloadProvider。"""

    config: MCFairGenWorkloadConfig
    fixed_taskset_seed: int | None = None
    scenario_seed_offset: int = 100000
    name: str = "mc_fairgen"

    def build(self, seed: int) -> WorkloadBundle:
        """按统一协议返回任务、场景和归一化边界。"""

        taskset_seed = self.fixed_taskset_seed if self.fixed_taskset_seed is not None else seed
        scenario_seed = seed + self.scenario_seed_offset
        effective_config = replace(self.config, seed=taskset_seed)
        workload = build_mc_fairgen_workload(effective_config)
        scenario = build_mc_fairgen_execution_scenario(workload, scenario_seed=scenario_seed)
        return WorkloadBundle(
            tasks=workload.tasks,
            scenario=scenario,
            normalization_bounds=workload.normalization_bounds,
            taskset_seed=workload.config.seed,
            scenario_seed=scenario_seed,
            attempts=workload.attempts,
            metadata={
                "workload_family": "mc_fairgen",
                "mode": workload.config.mode,
                "task_meta": workload.task_meta,
                "workload_metadata": workload.metadata,
                "num_tasks": workload.config.num_tasks,
                "hi_ratio": workload.config.hi_ratio,
            },
        )


def _stable_name_value(name: str) -> int:
    """把任务名映射到稳定整数，避免 Python hash 随机化影响复现。"""

    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def _stable_seed(base_seed: int, task_name: str, release_index: int) -> int:
    """构造 task/release 粒度稳定 seed，保证 scenario 可复现。"""

    return base_seed * 1_000_003 + _stable_name_value(task_name) * 1_009 + release_index * 9_173


def _sample_period_ms(rng: random.Random, config: MCFairGenWorkloadConfig) -> int:
    """按配置的 period source 采样 period(ms)。

    说明：
    - `automotive` 使用论文风格离散集合；
    - `controlled_*` 使用可控的稀疏/中等/稠密集合，便于可重复对比实验。
    """

    if config.period_source == "automotive":
        periods = MC_FAIRGEN_AUTOMOTIVE_PERIOD_SET
        weights = MC_FAIRGEN_AUTOMOTIVE_PERIOD_WEIGHTS
    elif config.period_source in MC_FAIRGEN_CONTROLLED_PERIOD_SETS:
        periods = MC_FAIRGEN_CONTROLLED_PERIOD_SETS[config.period_source]
        weights = MC_FAIRGEN_CONTROLLED_PERIOD_WEIGHTS[config.period_source]
    else:
        raise ValueError(f"unsupported period_source: {config.period_source}")
    return rng.choices(periods, weights=weights, k=1)[0]


def uunifast_discard(
    rng: random.Random,
    n: int,
    total_util: float,
    *,
    max_per_task: float | None = None,
    max_attempts: int = 10_000,
) -> list[float]:
    """UUniFastDiscard：生成 n 个正利用率，总和接近 total_util。

    - 每个分量 > 0；
    - 若设置 max_per_task，则每个分量 <= max_per_task；
    - 失败时抛 RuntimeError 并携带关键信息。
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if total_util <= 0.0:
        raise ValueError("total_util must be > 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if max_per_task is not None and max_per_task <= 0.0:
        raise ValueError("max_per_task must be > 0 when provided")

    for _ in range(max_attempts):
        utils: list[float] = []
        sum_u = total_util
        for i in range(1, n):
            next_sum_u = sum_u * (rng.random() ** (1.0 / (n - i)))
            utils.append(sum_u - next_sum_u)
            sum_u = next_sum_u
        utils.append(sum_u)

        if any(u <= 0.0 for u in utils):
            continue
        if max_per_task is not None and max(utils) > max_per_task:
            continue

        if abs(sum(utils) - total_util) > 1e-9:
            continue
        return utils

    raise RuntimeError(
        "uunifast_discard failed: "
        f"n={n}, total_util={total_util}, max_per_task={max_per_task}, max_attempts={max_attempts}"
    )


def _generate_raw_mc_fairgen_workload(config: MCFairGenWorkloadConfig) -> MCFairGenWorkload:
    """按 MC-FairGen 规则生成一次原始 workload（不含可调度筛选）。"""

    rng = random.Random(config.seed)

    num_hi = int(round(config.num_tasks * config.hi_ratio))
    num_hi = max(1, min(config.num_tasks - 1, num_hi))
    num_lo = config.num_tasks - num_hi

    for _ in range(1000):
        u_hi_lo_total = rng.uniform(config.u_hi_lo_min, config.u_hi_lo_max)
        u_hi_hi_total = rng.uniform(config.u_hi_hi_min, config.u_hi_hi_max)
        u_lo_lo_total = rng.uniform(config.u_lo_lo_min, config.u_lo_lo_max)
        if u_hi_hi_total >= u_hi_lo_total:
            break
    else:
        raise RuntimeError("failed to sample u_hi_hi_total >= u_hi_lo_total")

    hi_lo_utils = uunifast_discard(rng, num_hi, u_hi_lo_total, max_per_task=0.8)
    extra_total = u_hi_hi_total - u_hi_lo_total
    extra_utils = (
        uunifast_discard(rng, num_hi, extra_total, max_per_task=0.8) if extra_total > 1e-12 else [0.0] * num_hi
    )
    hi_hi_utils = [lo + extra for lo, extra in zip(hi_lo_utils, extra_utils, strict=True)]
    lo_lo_utils = uunifast_discard(rng, num_lo, u_lo_lo_total, max_per_task=0.8)

    tasks: list[Task] = []
    task_meta: list[MCFairGenTaskMeta] = []

    for i, (hi_lo_util_i, hi_hi_util_i) in enumerate(zip(hi_lo_utils, hi_hi_utils, strict=True)):
        period_ms = _sample_period_ms(rng, config)
        period = period_ms * config.period_scale

        base_c_lo = max(1, int(round(hi_lo_util_i * period)))
        base_c_hi = max(base_c_lo + 1, int(round(hi_hi_util_i * period)))

        # HI 任务预算放在 [C_LO, C_HI] 之间，制造可学习 headroom。
        rho = rng.uniform(config.hi_budget_rho_min, config.hi_budget_rho_max)
        initial_budget = int(round(base_c_lo + rho * (base_c_hi - base_c_lo)))
        initial_budget = max(base_c_lo, min(base_c_hi, initial_budget))

        name = f"mc_hi_{i}"
        tasks.append(
            Task(
                name=name,
                period=period,
                deadline=period,
                c_lo=initial_budget,
                c_hi=base_c_hi,
                criticality=Criticality.HI,
            )
        )
        task_meta.append(
            MCFairGenTaskMeta(
                name=name,
                period=period,
                period_ms=period_ms,
                criticality=Criticality.HI,
                generated_u_lo=hi_lo_util_i,
                generated_u_hi=hi_hi_util_i,
                base_c_lo=base_c_lo,
                base_c_hi=base_c_hi,
                initial_budget=initial_budget,
            )
        )

    for j, lo_lo_util_i in enumerate(lo_lo_utils):
        period_ms = _sample_period_ms(rng, config)
        period = period_ms * config.period_scale

        base_c_lo = max(1, int(round(lo_lo_util_i * period)))

        # LO 任务预算下压到 base_c_lo 的 60%~80% 区间，制造 LO service pressure。
        rho = rng.uniform(config.lo_budget_rho_min, config.lo_budget_rho_max)
        budget_scale = 0.60 + 0.80 * rho
        initial_budget = max(1, int(round(base_c_lo * budget_scale)))

        name = f"mc_lo_{j}"
        tasks.append(
            Task(
                name=name,
                period=period,
                deadline=period,
                c_lo=initial_budget,
                c_hi=initial_budget,
                criticality=Criticality.LO,
            )
        )
        task_meta.append(
            MCFairGenTaskMeta(
                name=name,
                period=period,
                period_ms=period_ms,
                criticality=Criticality.LO,
                generated_u_lo=lo_lo_util_i,
                generated_u_hi=lo_lo_util_i,
                base_c_lo=base_c_lo,
                base_c_hi=base_c_lo,
                initial_budget=initial_budget,
            )
        )

    tasks_tuple = tuple(tasks)
    task_meta_tuple = tuple(task_meta)
    normalization_bounds = build_mc_fairgen_normalization_bounds(tasks_tuple, task_meta_tuple, config)

    metadata: dict[str, Any] = {
        "workload_family": "mc_fairgen",
        "mode": config.mode,
        "num_tasks": config.num_tasks,
        "num_hi": num_hi,
        "num_lo": num_lo,
        "hi_ratio": config.hi_ratio,
        "period_source": config.period_source,
        "u_hi_lo_total": u_hi_lo_total,
        "u_hi_hi_total": u_hi_hi_total,
        "u_lo_lo_total": u_lo_lo_total,
        "budget_util_total": sum(t.c_lo / t.period for t in tasks_tuple),
        "budget_util_hi": sum(t.c_lo / t.period for t in tasks_tuple if t.criticality is Criticality.HI),
        "budget_util_lo": sum(t.c_lo / t.period for t in tasks_tuple if t.criticality is Criticality.LO),
    }

    return MCFairGenWorkload(
        config=config,
        tasks=tasks_tuple,
        task_meta=task_meta_tuple,
        normalization_bounds=normalization_bounds,
        metadata=metadata,
    )


def generate_mc_fairgen_workload(config: MCFairGenWorkloadConfig) -> MCFairGenWorkload:
    """生成一次 MC-FairGen workload（Step2/3 基础路径）。"""

    return _generate_raw_mc_fairgen_workload(config)


def generate_schedulable_mc_fairgen_workload(config: MCFairGenWorkloadConfig) -> MCFairGenWorkload:
    """按固定次数尝试生成可调度 workload。

    说明：该逻辑仅在 require_schedulable=True 时使用，且局部 import
    evaluate_taskset，保持 workload 层依赖方向不变。
    """

    from amc_py.experiments import evaluate_taskset

    for offset in range(config.max_attempts):
        candidate_config = replace(config, seed=config.seed + offset, require_schedulable=False)
        workload = _generate_raw_mc_fairgen_workload(candidate_config)
        result = evaluate_taskset(
            list(workload.tasks),
            method=config.sched_method,
            priority_policy=config.priority_policy,
        )
        if result.schedulable:
            metadata = dict(workload.metadata or {})
            metadata.update(
                {
                    "schedulability_checked": True,
                    "sched_method": config.sched_method,
                    "priority_policy": config.priority_policy,
                    "attempts": offset + 1,
                    "effective_taskset_seed": candidate_config.seed,
                }
            )
            return replace(workload, attempts=offset + 1, metadata=metadata)

    raise RuntimeError("在 max_attempts 范围内未找到可调度的 mc_fairgen workload")


def build_mc_fairgen_workload(config: MCFairGenWorkloadConfig) -> MCFairGenWorkload:
    """对外统一入口。"""

    if config.require_schedulable:
        return generate_schedulable_mc_fairgen_workload(config)
    return generate_mc_fairgen_workload(config)


def build_mc_fairgen_normalization_bounds(
    tasks: tuple[Task, ...],
    task_meta: tuple[MCFairGenTaskMeta, ...],
    config: MCFairGenWorkloadConfig,
) -> NormalizationBounds:
    """构建任务归一化边界。

    设计原则：
    - HI 任务上界至少覆盖 base_c_hi；
    - LO 任务上界覆盖 scenario 可能出现的 overrun 最大值；
    - 同时确保覆盖初始预算值，避免 runtime 下界/上界失配。
    """

    _ = tasks
    bounds: NormalizationBounds = {}
    for meta in task_meta:
        if meta.criticality is Criticality.LO:
            scenario_max = int(math.ceil(meta.base_c_lo * config.lo_overrun_factor_max))
        else:
            scenario_max = meta.base_c_hi
        max_cost = max(meta.initial_budget, scenario_max)
        bounds[meta.name] = TaskNormalizationBound(
            min_cost=float(1),
            max_cost=float(max_cost),
        )
    return bounds


def build_mc_fairgen_execution_scenario(
    workload: MCFairGenWorkload,
    scenario_seed: int,
) -> ExecutionScenario:
    """按文档规则构建 MC-FairGen runtime scenario。"""

    meta_by_name = {meta.name: meta for meta in workload.task_meta}
    config = workload.config

    def resolver(task: Task, release_index: int) -> int:
        meta = meta_by_name[task.name]
        rng = random.Random(_stable_seed(scenario_seed, task.name, release_index))

        if task.criticality is Criticality.HI:
            # HI 任务：小概率 overrun（不超过 C_HI），用于触发 mode change 信号。
            if task.c_hi > task.c_lo and rng.random() < config.hi_overrun_prob:
                low = max(task.c_lo + 1, int(math.ceil(task.c_lo * config.hi_overrun_factor_min)))
                high = min(task.c_hi, int(math.ceil(task.c_lo * config.hi_overrun_factor_max)))
                if low <= high:
                    return rng.randint(low, high)
                return min(task.c_hi, task.c_lo + 1)

            # 非 overrun 时在 [ratio*C_LO, C_LO] 内采样，保留波动。
            low = max(1, int(round(task.c_lo * config.nominal_hi_cost_min_ratio)))
            high = max(low, task.c_lo)
            return rng.randint(low, high)

        # LO 任务：高概率允许超过 C_LO，制造 cancellation pressure。
        if rng.random() < config.lo_overrun_prob:
            low = max(task.c_lo + 1, int(math.ceil(task.c_lo * config.lo_overrun_factor_min)))
            high = max(low, int(math.ceil(meta.base_c_lo * config.lo_overrun_factor_max)))
            return rng.randint(low, high)

        low = max(1, int(round(task.c_lo * config.nominal_lo_cost_min_ratio)))
        high = max(low, task.c_lo)
        return rng.randint(low, high)

    return ExecutionScenario(
        name=f"mc_fairgen_{workload.config.mode}_{scenario_seed}",
        resolver=resolver,
    )


__all__ = [
    "MCFairGenMode",
    "MCFairGenTaskMeta",
    "MCFairGenWorkload",
    "MCFairGenWorkloadConfig",
    "MCFairGenWorkloadProvider",
    "build_mc_fairgen_execution_scenario",
    "build_mc_fairgen_workload",
    "generate_mc_fairgen_workload",
    "generate_schedulable_mc_fairgen_workload",
    "uunifast_discard",
]
