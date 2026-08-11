"""Independent MC-Stratified-Dynamic workload family.

This module owns taskset construction, execution-realization sampling, and
normalization bounds for the ``mc_stratified_dynamic`` family.  It deliberately
does not depend on any experiment, selector, controller, or legacy workload
module.  In particular, the execution scenario is a persistent Markov process
whose random streams are derived only from the taskset/scenario seeds, task
name, release index, and Markov state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import random
from typing import Any, Literal

from amc_py.generator import uunifast
from amc_py.models import Criticality, Task
from amc_py.rl.observation import NormalizationBounds, TaskNormalizationBound
from amc_py.runtime_scenarios import ExecutionScenario
from amc_py.workloads.base import WorkloadBundle, WorkloadProvider


MCStratifiedDynamicPeriodFamily = Literal[
    "semi_harmonic",
    "log_uniform",
    "seed_paired",
]


MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_PERIODS_MS: tuple[int, ...] = (
    10,
    20,
    25,
    50,
    100,
    200,
)
MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_WEIGHTS: tuple[float, ...] = (
    0.10,
    0.20,
    0.20,
    0.25,
    0.15,
    0.10,
)
# Short private aliases retain the plan's suggested names while keeping the
# family prefix available for callers that prefer explicit constants.
_SEMI_HARMONIC_PERIODS_MS = MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_PERIODS_MS
_SEMI_HARMONIC_WEIGHTS = MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_WEIGHTS


@dataclass(frozen=True, slots=True)
class MCStratifiedDynamicWorkloadConfig:
    """Configuration for one independent workload realization."""

    seed: int = 0
    num_tasks: int = 12
    hi_ratio: float = 0.5
    period_family: MCStratifiedDynamicPeriodFamily = "seed_paired"
    period_scale: int = 500
    tick_ns: int = 10

    total_util_min: float = 0.58
    total_util_max: float = 0.84
    criticality_factor_min: float = 1.40
    criticality_factor_max: float = 2.20
    max_task_util: float = 0.25

    lo_budget_quantile_min: float = 0.65
    lo_budget_quantile_max: float = 0.80
    hi_budget_quantile_min: float = 0.75
    hi_budget_quantile_max: float = 0.90

    normal_cost_ratio_min: float = 0.55
    normal_cost_ratio_max: float = 0.95
    lo_stress_cost_ratio_min: float = 0.95
    lo_stress_cost_ratio_max: float = 1.35
    hi_stress_cost_ratio_min: float = 0.95
    hi_stress_cost_ratio_max: float = 1.45

    lo_stress_duty_min: float = 0.10
    lo_stress_duty_max: float = 0.35
    lo_stress_dwell_min: float = 4.0
    lo_stress_dwell_max: float = 20.0
    hi_stress_duty_min: float = 0.03
    hi_stress_duty_max: float = 0.15
    hi_stress_dwell_min: float = 2.0
    hi_stress_dwell_max: float = 10.0

    log_uniform_period_min_ms: int = 10
    log_uniform_period_max_ms: int = 200

    require_schedulable: bool = False
    max_attempts: int = 100
    sched_method: str = "amc_rtb"
    priority_policy: str = "dm"

    def __post_init__(self) -> None:
        if self.num_tasks < 2:
            raise ValueError("num_tasks must be >= 2")
        if not 0.0 < self.hi_ratio < 1.0:
            raise ValueError("hi_ratio must be in (0, 1)")
        if self.period_family not in {"semi_harmonic", "log_uniform", "seed_paired"}:
            raise ValueError("period_family must be semi_harmonic, log_uniform, or seed_paired")
        if self.period_scale <= 0:
            raise ValueError("period_scale must be > 0")
        if self.tick_ns <= 0:
            raise ValueError("tick_ns must be > 0")

        _validate_range("total_util", self.total_util_min, self.total_util_max, positive=True)
        _validate_range(
            "criticality_factor",
            self.criticality_factor_min,
            self.criticality_factor_max,
            lower=1.0,
        )
        if self.max_task_util <= 0.0 or self.max_task_util > 1.0:
            raise ValueError("max_task_util must be in (0, 1]")
        if self.total_util_max > self.num_tasks * self.max_task_util + 1e-12:
            raise ValueError("total_util_max cannot be covered by num_tasks * max_task_util")

        _validate_quantile_range(
            "lo_budget_quantile", self.lo_budget_quantile_min, self.lo_budget_quantile_max
        )
        _validate_quantile_range(
            "hi_budget_quantile", self.hi_budget_quantile_min, self.hi_budget_quantile_max
        )
        _validate_range("normal_cost_ratio", self.normal_cost_ratio_min, self.normal_cost_ratio_max, positive=True)
        _validate_range(
            "lo_stress_cost_ratio",
            self.lo_stress_cost_ratio_min,
            self.lo_stress_cost_ratio_max,
            positive=True,
        )
        _validate_range(
            "hi_stress_cost_ratio",
            self.hi_stress_cost_ratio_min,
            self.hi_stress_cost_ratio_max,
            positive=True,
        )

        _validate_duty_range("lo_stress_duty", self.lo_stress_duty_min, self.lo_stress_duty_max)
        _validate_duty_range("hi_stress_duty", self.hi_stress_duty_min, self.hi_stress_duty_max)
        _validate_range("lo_stress_dwell", self.lo_stress_dwell_min, self.lo_stress_dwell_max, lower=1.0)
        _validate_range("hi_stress_dwell", self.hi_stress_dwell_min, self.hi_stress_dwell_max, lower=1.0)

        if self.log_uniform_period_min_ms <= 0 or self.log_uniform_period_max_ms <= 0:
            raise ValueError("log_uniform_period_min_ms/max_ms must be > 0")
        if self.log_uniform_period_min_ms > self.log_uniform_period_max_ms:
            raise ValueError("log_uniform_period_min_ms must be <= log_uniform_period_max_ms")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        # This catches configurations for which the Markov parameter ranges
        # could never be legal.  Individual samples are checked again below.
        for duty, dwell in (
            (self.lo_stress_duty_max, self.lo_stress_dwell_min),
            (self.hi_stress_duty_max, self.hi_stress_dwell_min),
        ):
            if _markov_p_enter(duty, dwell) > 1.0 + 1e-12:
                raise ValueError("stress duty/dwell range implies p_enter > 1")


@dataclass(frozen=True, slots=True)
class MCStratifiedDynamicTaskMeta:
    """Generation and execution-distribution metadata for one task."""

    name: str
    criticality: Criticality
    period: int
    period_ms: int
    generated_u_lo: float
    generated_u_hi: float
    base_demand_lo: int
    base_demand_hi: int
    initial_budget: int
    budget_quantile: float
    normal_cost_min: int
    normal_cost_max: int
    stress_cost_min: int
    stress_cost_max: int
    stress_stationary_prob: float
    stress_expected_dwell: float
    stress_p_enter: float
    stress_p_exit: float


@dataclass(frozen=True, slots=True)
class MCStratifiedDynamicWorkload:
    """Complete taskset-side result before a scenario is attached."""

    config: MCStratifiedDynamicWorkloadConfig
    tasks: tuple[Task, ...]
    task_meta: tuple[MCStratifiedDynamicTaskMeta, ...]
    normalization_bounds: NormalizationBounds
    attempts: int = 1
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MCStratifiedDynamicWorkloadProvider:
    """Uniform provider adapter for the independent workload family."""

    config: MCStratifiedDynamicWorkloadConfig
    fixed_taskset_seed: int | None = None
    scenario_seed_offset: int = 100000
    name: str = "mc_stratified_dynamic"

    def build(self, seed: int) -> WorkloadBundle:
        taskset_seed = self.fixed_taskset_seed if self.fixed_taskset_seed is not None else seed
        scenario_seed = seed + self.scenario_seed_offset
        effective_config = replace(self.config, seed=taskset_seed)
        workload = build_mc_stratified_dynamic_workload(effective_config)
        scenario = build_mc_stratified_dynamic_execution_scenario(
            workload,
            scenario_seed=scenario_seed,
        )
        metadata = dict(workload.metadata or {})
        metadata.update(
            {
                "workload_family": "mc_stratified_dynamic",
                "schema_version": "mc_stratified_dynamic_workload_v1",
                "taskset_seed": taskset_seed,
                "scenario_seed": scenario_seed,
                "task_meta": workload.task_meta,
                "workload_metadata": workload.metadata,
            }
        )
        return WorkloadBundle(
            tasks=workload.tasks,
            scenario=scenario,
            normalization_bounds=workload.normalization_bounds,
            taskset_seed=taskset_seed,
            scenario_seed=scenario_seed,
            attempts=workload.attempts,
            metadata=metadata,
        )


class _CandidateRejected(RuntimeError):
    """Internal signal used by the schedulable retry loop."""


def _validate_range(
    name: str,
    low: float,
    high: float,
    *,
    positive: bool = False,
    lower: float | None = None,
) -> None:
    if positive and (low <= 0.0 or high <= 0.0):
        raise ValueError(f"{name}_min/max must be > 0")
    if lower is not None and low < lower:
        raise ValueError(f"{name}_min must be >= {lower}")
    if low > high:
        raise ValueError(f"{name}_min must be <= {name}_max")


def _validate_quantile_range(name: str, low: float, high: float) -> None:
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"{name}_min/max must satisfy 0 < min <= max <= 1")


def _validate_duty_range(name: str, low: float, high: float) -> None:
    if not 0.0 < low <= high < 1.0:
        raise ValueError(f"{name}_min/max must satisfy 0 < min <= max < 1")


def _stable_seed(*parts: object) -> int:
    """Return a process-independent integer seed for random-access streams."""

    encoded = "|".join(f"{type(part).__name__}:{part}" for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _effective_period_family(
    period_family: MCStratifiedDynamicPeriodFamily,
    taskset_seed: int,
) -> Literal["semi_harmonic", "log_uniform"]:
    if period_family == "seed_paired":
        return "semi_harmonic" if taskset_seed % 2 == 0 else "log_uniform"
    return period_family


def _sample_period_ms(
    rng: random.Random,
    config: MCStratifiedDynamicWorkloadConfig,
) -> int:
    """Sample one period in milliseconds using the resolved family."""

    family = _effective_period_family(config.period_family, config.seed)
    if family == "semi_harmonic":
        return rng.choices(
            MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_PERIODS_MS,
            weights=MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_WEIGHTS,
            k=1,
        )[0]
    sampled = math.exp(
        rng.uniform(
            math.log(config.log_uniform_period_min_ms),
            math.log(config.log_uniform_period_max_ms),
        )
    )
    return int(round(sampled))


def _sample_bounded_util_vector(
    rng: random.Random,
    num_tasks: int,
    total_util: float,
    *,
    max_per_task: float,
    max_attempts: int = 10_000,
) -> list[float]:
    """Sample a bounded UUniFast vector by rejection.

    The public ``amc_py.generator.uunifast`` is intentionally the only
    allocation primitive used here; the bound is enforced in this family-local
    helper so no legacy generator semantics leak into the new workload.
    """

    if num_tasks <= 0 or total_util <= 0.0 or max_per_task <= 0.0:
        raise ValueError("num_tasks, total_util, and max_per_task must be positive")
    if total_util > num_tasks * max_per_task + 1e-12:
        raise ValueError("total_util exceeds the bounded utilization capacity")
    for _ in range(max_attempts):
        values = uunifast(num_tasks, total_util, rng=rng)
        if all(0.0 < value <= max_per_task for value in values):
            return values
    raise RuntimeError(
        "bounded UUniFast rejection failed: "
        f"num_tasks={num_tasks}, total_util={total_util}, max_per_task={max_per_task}"
    )


def _uniform_integer_cdf(value: int, low: int, high: int) -> float:
    if value < low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low + 1) / (high - low + 1)


def _mixture_quantile_budget(
    normal_min: int,
    normal_max: int,
    stress_min: int,
    stress_max: int,
    stress_stationary_prob: float,
    quantile: float,
) -> int:
    """Return the smallest integer budget covering a discrete cost mixture.

    Each component is an inclusive discrete-uniform distribution.  The
    mixture CDF is ``(1-pi) F_normal(c) + pi F_stress(c)`` and the returned
    budget is the first integer whose CDF reaches ``quantile``.
    """

    if min(normal_min, normal_max, stress_min, stress_max) < 1:
        raise ValueError("cost range endpoints must be >= 1")
    if normal_min > normal_max or stress_min > stress_max:
        raise ValueError("cost range min must be <= max")
    if not 0.0 <= stress_stationary_prob <= 1.0:
        raise ValueError("stress_stationary_prob must be in [0, 1]")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")

    upper = max(normal_max, stress_max)
    for candidate in range(1, upper + 1):
        cdf = (
            (1.0 - stress_stationary_prob)
            * _uniform_integer_cdf(candidate, normal_min, normal_max)
            + stress_stationary_prob
            * _uniform_integer_cdf(candidate, stress_min, stress_max)
        )
        if cdf + 1e-15 >= quantile:
            return candidate
    return upper


def _markov_p_exit(expected_stress_dwell: float) -> float:
    return 1.0 / expected_stress_dwell


def _markov_p_enter(stress_stationary_prob: float, expected_stress_dwell: float) -> float:
    p_exit = _markov_p_exit(expected_stress_dwell)
    return stress_stationary_prob * p_exit / (1.0 - stress_stationary_prob)


def _sample_markov_parameters(
    rng: random.Random,
    config: MCStratifiedDynamicWorkloadConfig,
    criticality: Criticality,
) -> tuple[float, float, float, float]:
    if criticality is Criticality.HI:
        pi = rng.uniform(config.hi_stress_duty_min, config.hi_stress_duty_max)
        dwell = rng.uniform(config.hi_stress_dwell_min, config.hi_stress_dwell_max)
    else:
        pi = rng.uniform(config.lo_stress_duty_min, config.lo_stress_duty_max)
        dwell = rng.uniform(config.lo_stress_dwell_min, config.lo_stress_dwell_max)
    p_exit = _markov_p_exit(dwell)
    p_enter = _markov_p_enter(pi, dwell)
    if not 0.0 <= p_enter <= 1.0 or not 0.0 < p_exit <= 1.0:
        raise _CandidateRejected("sampled Markov parameters have an invalid transition probability")
    return pi, dwell, p_enter, p_exit


def _cost_range(base_demand: int, ratio_min: float, ratio_max: float) -> tuple[int, int]:
    low = max(1, int(round(base_demand * ratio_min)))
    high = max(low, int(round(base_demand * ratio_max)))
    return low, high


def _generate_raw_mc_stratified_dynamic_workload(
    config: MCStratifiedDynamicWorkloadConfig,
) -> MCStratifiedDynamicWorkload:
    rng = random.Random(config.seed)
    total_util_target = rng.uniform(config.total_util_min, config.total_util_max)
    generated_u_lo = _sample_bounded_util_vector(
        rng,
        config.num_tasks,
        total_util_target,
        max_per_task=config.max_task_util,
    )

    num_hi = int(round(config.num_tasks * config.hi_ratio))
    num_hi = max(1, min(config.num_tasks - 1, num_hi))
    criticalities = [Criticality.HI] * num_hi + [Criticality.LO] * (config.num_tasks - num_hi)
    rng.shuffle(criticalities)

    tasks: list[Task] = []
    task_meta: list[MCStratifiedDynamicTaskMeta] = []
    hi_index = 0
    lo_index = 0
    sampled_factors: list[float] = []

    for u_lo, criticality in zip(generated_u_lo, criticalities, strict=True):
        period_ms = _sample_period_ms(rng, config)
        period = period_ms * config.period_scale
        base_demand_lo = max(1, int(round(u_lo * period)))

        if criticality is Criticality.HI:
            factor = rng.uniform(config.criticality_factor_min, config.criticality_factor_max)
            u_hi = u_lo * factor
            sampled_factors.append(factor)
            base_demand_hi = max(base_demand_lo, int(round(u_hi * period)))
            normal_min, normal_max = _cost_range(
                base_demand_lo,
                config.normal_cost_ratio_min,
                config.normal_cost_ratio_max,
            )
            stress_min, stress_max = _cost_range(
                base_demand_hi,
                config.hi_stress_cost_ratio_min,
                config.hi_stress_cost_ratio_max,
            )
            name = f"mc_sd_hi_{hi_index}"
            hi_index += 1
            quantile = rng.uniform(config.hi_budget_quantile_min, config.hi_budget_quantile_max)
        else:
            u_hi = u_lo
            base_demand_hi = base_demand_lo
            normal_min, normal_max = _cost_range(
                base_demand_lo,
                config.normal_cost_ratio_min,
                config.normal_cost_ratio_max,
            )
            stress_min, stress_max = _cost_range(
                base_demand_lo,
                config.lo_stress_cost_ratio_min,
                config.lo_stress_cost_ratio_max,
            )
            name = f"mc_sd_lo_{lo_index}"
            lo_index += 1
            quantile = rng.uniform(config.lo_budget_quantile_min, config.lo_budget_quantile_max)

        pi, dwell, p_enter, p_exit = _sample_markov_parameters(rng, config, criticality)
        initial_budget = _mixture_quantile_budget(
            normal_min=normal_min,
            normal_max=normal_max,
            stress_min=stress_min,
            stress_max=stress_max,
            stress_stationary_prob=pi,
            quantile=quantile,
        )

        c_hi = (
            max(initial_budget, base_demand_hi, normal_max, stress_max)
            if criticality is Criticality.HI
            else initial_budget
        )
        if initial_budget > period or c_hi > period:
            raise _CandidateRejected(
                f"{name} demand/budget exceeds deadline: budget={initial_budget}, c_hi={c_hi}, period={period}"
            )

        task = Task(
            name=name,
            period=period,
            deadline=period,
            c_lo=initial_budget,
            c_hi=c_hi,
            criticality=criticality,
        )
        tasks.append(task)
        task_meta.append(
            MCStratifiedDynamicTaskMeta(
                name=name,
                criticality=criticality,
                period=period,
                period_ms=period_ms,
                generated_u_lo=u_lo,
                generated_u_hi=u_hi,
                base_demand_lo=base_demand_lo,
                base_demand_hi=base_demand_hi,
                initial_budget=initial_budget,
                budget_quantile=quantile,
                normal_cost_min=normal_min,
                normal_cost_max=normal_max,
                stress_cost_min=stress_min,
                stress_cost_max=stress_max,
                stress_stationary_prob=pi,
                stress_expected_dwell=dwell,
                stress_p_enter=p_enter,
                stress_p_exit=p_exit,
            )
        )

    tasks_tuple = tuple(tasks)
    task_meta_tuple = tuple(task_meta)
    bounds = build_mc_stratified_dynamic_normalization_bounds(
        tasks_tuple,
        task_meta_tuple,
        config,
    )

    hi_meta = [meta for meta in task_meta_tuple if meta.criticality is Criticality.HI]
    actual_factors = [meta.base_demand_hi / meta.base_demand_lo for meta in hi_meta]
    total_util_actual = sum(meta.base_demand_lo / meta.period for meta in task_meta_tuple)
    initial_budget_util_hi = sum(
        task.c_lo / task.period
        for task in tasks_tuple
        if task.criticality is Criticality.HI
    )
    initial_budget_util_lo = sum(
        task.c_lo / task.period
        for task in tasks_tuple
        if task.criticality is Criticality.LO
    )
    effective_family = _effective_period_family(config.period_family, config.seed)
    metadata: dict[str, Any] = {
        "workload_family": "mc_stratified_dynamic",
        "schema_version": "mc_stratified_dynamic_workload_v1",
        "taskset_seed": config.seed,
        "period_family": effective_family,
        "period_family_requested": config.period_family,
        "num_tasks": config.num_tasks,
        "num_hi": num_hi,
        "num_lo": config.num_tasks - num_hi,
        "total_util_target": total_util_target,
        "total_util_actual": total_util_actual,
        "generated_u_lo_total": sum(generated_u_lo),
        "criticality_factor_target": sum(sampled_factors) / len(sampled_factors),
        "criticality_factor_actual": sum(actual_factors) / len(actual_factors),
        "criticality_factor_mean": sum(actual_factors) / len(actual_factors),
        "criticality_factor_min": min(actual_factors),
        "criticality_factor_max": max(actual_factors),
        "initial_budget_util_total": initial_budget_util_hi + initial_budget_util_lo,
        "initial_budget_util_hi": initial_budget_util_hi,
        "initial_budget_util_lo": initial_budget_util_lo,
        "task_meta": task_meta_tuple,
    }
    return MCStratifiedDynamicWorkload(
        config=config,
        tasks=tasks_tuple,
        task_meta=task_meta_tuple,
        normalization_bounds=bounds,
        metadata=metadata,
    )


def generate_mc_stratified_dynamic_workload(
    config: MCStratifiedDynamicWorkloadConfig,
) -> MCStratifiedDynamicWorkload:
    """Generate one raw taskset from the taskset seed."""

    return _generate_raw_mc_stratified_dynamic_workload(config)


def generate_schedulable_mc_stratified_dynamic_workload(
    config: MCStratifiedDynamicWorkloadConfig,
) -> MCStratifiedDynamicWorkload:
    """Retry independently seeded candidates until AMC-rtb accepts one."""

    from amc_py.experiments import evaluate_taskset

    for offset in range(config.max_attempts):
        candidate_config = replace(
            config,
            seed=config.seed + offset,
            require_schedulable=False,
        )
        try:
            workload = _generate_raw_mc_stratified_dynamic_workload(candidate_config)
        except _CandidateRejected:
            continue
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

    raise RuntimeError(
        "在 max_attempts 范围内未找到可调度的 mc_stratified_dynamic workload: "
        f"max_attempts={config.max_attempts}"
    )


def build_mc_stratified_dynamic_workload(
    config: MCStratifiedDynamicWorkloadConfig,
) -> MCStratifiedDynamicWorkload:
    """Unified workload factory."""

    if config.require_schedulable:
        return generate_schedulable_mc_stratified_dynamic_workload(config)
    return generate_mc_stratified_dynamic_workload(config)


def build_mc_stratified_dynamic_normalization_bounds(
    tasks: tuple[Task, ...],
    task_meta: tuple[MCStratifiedDynamicTaskMeta, ...],
    config: MCStratifiedDynamicWorkloadConfig,
) -> NormalizationBounds:
    """Build bounds covering normal/stressed costs and design budgets."""

    _ = config
    task_by_name = {task.name: task for task in tasks}
    bounds: NormalizationBounds = {}
    for meta in task_meta:
        task = task_by_name[meta.name]
        max_cost = max(
            task.c_hi,
            meta.initial_budget,
            meta.normal_cost_max,
            meta.stress_cost_max,
        )
        bounds[meta.name] = TaskNormalizationBound(
            min_cost=1.0,
            max_cost=float(max_cost),
        )
    return bounds


def build_mc_stratified_dynamic_execution_scenario(
    workload: MCStratifiedDynamicWorkload,
    *,
    scenario_seed: int,
) -> ExecutionScenario:
    """Build a persistent, random-access Markov execution scenario.

    State is extended from release zero and cached per task.  Actual cost for
    each release uses an independent stable random-access stream, so querying
    releases in a different order cannot change either the state or the cost.
    No controller action or current runtime budget is read here.
    """

    meta_by_name = {meta.name: meta for meta in workload.task_meta}
    state_cache: dict[str, list[bool]] = {name: [] for name in meta_by_name}
    state_rngs: dict[str, random.Random] = {
        name: random.Random(
            _stable_seed(workload.config.seed, scenario_seed, name, "markov_state")
        )
        for name in meta_by_name
    }

    def ensure_state(meta: MCStratifiedDynamicTaskMeta, release_index: int) -> bool:
        cache = state_cache[meta.name]
        rng = state_rngs[meta.name]
        if not cache:
            cache.append(rng.random() < meta.stress_stationary_prob)
        while len(cache) <= release_index:
            previous = cache[-1]
            if previous:
                current = not (rng.random() < meta.stress_p_exit)
            else:
                current = rng.random() < meta.stress_p_enter
            cache.append(current)
        return cache[release_index]

    def resolver(task: Task, release_index: int) -> int:
        meta = meta_by_name[task.name]
        stressed = ensure_state(meta, release_index)
        if stressed:
            low, high = meta.stress_cost_min, meta.stress_cost_max
        else:
            low, high = meta.normal_cost_min, meta.normal_cost_max
        cost_rng = random.Random(
            _stable_seed(
                workload.config.seed,
                scenario_seed,
                task.name,
                release_index,
                "stressed" if stressed else "normal",
            )
        )
        actual_cost = cost_rng.randint(low, high)
        if meta.criticality is Criticality.HI:
            # Generation normally makes this a no-op.  The clamp is a final
            # safety guard required by ExecutionScenario's HI contract.
            actual_cost = min(actual_cost, task.c_hi)
        return actual_cost

    return ExecutionScenario(
        name=f"mc_stratified_dynamic_{workload.config.seed}_{scenario_seed}",
        resolver=resolver,
    )


__all__ = [
    "MCStratifiedDynamicPeriodFamily",
    "MCStratifiedDynamicTaskMeta",
    "MCStratifiedDynamicWorkload",
    "MCStratifiedDynamicWorkloadConfig",
    "MCStratifiedDynamicWorkloadProvider",
    "MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_PERIODS_MS",
    "MC_STRATIFIED_DYNAMIC_SEMI_HARMONIC_WEIGHTS",
    "build_mc_stratified_dynamic_execution_scenario",
    "build_mc_stratified_dynamic_normalization_bounds",
    "build_mc_stratified_dynamic_workload",
    "generate_mc_stratified_dynamic_workload",
    "generate_schedulable_mc_stratified_dynamic_workload",
    "_mixture_quantile_budget",
    "_sample_bounded_util_vector",
]
