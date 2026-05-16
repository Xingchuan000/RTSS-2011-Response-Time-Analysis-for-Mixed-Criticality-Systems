"""构建 RL 观测向量。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.feature_config import (
    OBSERVATION_MODE_V10_BASIC,
    OBSERVATION_MODE_V11_FULL_10D,
    OBSERVATION_MODE_V12_FULL_14D,
    FeatureConfig,
)
from amc_py.rl.feature_state import RuntimeFeatureState
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.types import AgentObservation


@dataclass(frozen=True, slots=True)
class TaskNormalizationBound:
    """单个任务的归一化区间定义。"""

    min_cost: float
    max_cost: float


NormalizationBounds = dict[str, TaskNormalizationBound]


def build_default_normalization_bounds(tasks: Sequence[Task]) -> NormalizationBounds:
    """按默认策略构建归一化边界。"""

    bounds: NormalizationBounds = {}
    for task in tasks:
        if task.criticality is Criticality.HI:
            bounds[task.name] = TaskNormalizationBound(min_cost=0.0, max_cost=float(task.c_hi))
        else:
            bounds[task.name] = TaskNormalizationBound(
                min_cost=0.0,
                max_cost=float(max(task.c_lo, task.deadline)),
            )
    return bounds


def _normalize(value: float, lo: float, hi: float) -> float:
    """将数值归一化并裁剪到 [0, 1]。"""

    if hi <= lo:
        raise ValueError("normalization bound 非法：max_cost 必须大于 min_cost")
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _clip01(value: float) -> float:
    """将任意浮点值裁剪到 [0, 1]，用于 v11 组合特征的最终边界保护。"""

    return max(0.0, min(1.0, float(value)))


def build_basic_observation(
    *,
    time: int,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    monitor: RuntimeMonitor,
    bounds: NormalizationBounds | None = None,
) -> AgentObservation:
    """按 v10_basic 语义构建观测。

    状态拼接格式为：
    [(budget_norm_1, recent_cost_norm_1), ..., (budget_norm_n, recent_cost_norm_n)]

    该函数保留旧版本行为，不引入任何 v11 特征。
    """

    active_bounds = bounds or build_default_normalization_bounds(ordered_tasks)
    state_values: list[float] = []
    raw_budgets: dict[str, int] = {}
    raw_recent_costs: dict[str, int] = {}

    for task in ordered_tasks:
        task_name = task.name
        budget = budget_state.budgets[task_name]
        recent_cost = monitor.recent_execution.get(task_name, 0)

        if task_name not in active_bounds:
            raise ValueError(f"normalization bounds 缺少任务 {task_name}")
        task_bound = active_bounds[task_name]
        lo = float(task_bound.min_cost)
        hi = float(task_bound.max_cost)

        state_values.append(_normalize(budget, lo, hi))
        state_values.append(_normalize(recent_cost, lo, hi))

        raw_budgets[task_name] = budget
        raw_recent_costs[task_name] = recent_cost

    return AgentObservation(
        time=time,
        state_vector=tuple(state_values),
        raw_budgets=raw_budgets,
        raw_recent_costs=raw_recent_costs,
    )


def build_v11_full_10d_observation(
    *,
    time: int,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    monitor: RuntimeMonitor,
    bounds: NormalizationBounds | None = None,
    feature_state: RuntimeFeatureState,
    feature_config: FeatureConfig,
    safety_margin_min: float = 1.0,
) -> AgentObservation:
    """按 v11_full_10d 语义构建观测（每任务 10 维 + 全局 8 维）。

    重要语义约束：
    - 任务顺序严格使用 ordered_tasks，不做动态重排；
    - 只读取当前已有统计量，不使用未来信息；
    - 所有输出特征都裁剪到 [0, 1]。
    """

    active_bounds = bounds or build_default_normalization_bounds(ordered_tasks)
    state_values: list[float] = []
    raw_budgets: dict[str, int] = {}
    raw_recent_costs: dict[str, int] = {}

    n_tasks = len(ordered_tasks)
    total_budget_util = 0.0
    hi_budget_util = 0.0
    lo_budget_util = 0.0

    for rank, task in enumerate(ordered_tasks):
        task_name = task.name
        budget = float(budget_state.budgets[task_name])
        recent_cost = float(monitor.recent_execution.get(task_name, 0))

        if task_name not in active_bounds:
            raise ValueError(f"normalization bounds 缺少任务 {task_name}")
        task_bound = active_bounds[task_name]
        lo = float(task_bound.min_cost)
        hi = float(task_bound.max_cost)

        # 每任务缓存按需初始化：EMA 初值优先使用 task.c_lo，符合实现计划要求。
        feature_state.init_task(task_name, init_cost=float(task.c_lo))

        # 读取历史特征缓存；当历史为空时 max-k 回退 recent_cost。
        ema_cost = float(feature_state.ema_cost.get(task_name, float(task.c_lo)))
        history = feature_state.cost_history.get(task_name)
        max_cost_k = float(max(history)) if history and len(history) > 0 else recent_cost
        overrun_ema = float(feature_state.overrun_ema.get(task_name, 0.0))

        is_hi = 1.0 if task.criticality is Criticality.HI else 0.0
        priority_norm = 1.0 - float(rank) / float(max(1, n_tasks - 1))
        util_budget = budget / max(1.0, float(task.period))

        total_budget_util += util_budget
        if is_hi > 0.5:
            hi_budget_util += util_budget
        else:
            lo_budget_util += util_budget

        # pred_cost 是 risk/surplus 的共同核心输入。
        pred_cost = max(
            recent_cost,
            ema_cost,
            float(feature_config.max_cost_weight) * max_cost_k,
        )
        raw_risk = (
            pred_cost / max(1.0, budget)
            + 0.5 * overrun_ema
            + 0.2 * is_hi
            + 0.1 * priority_norm
        )
        risk = _clip01(raw_risk / float(feature_config.risk_max_scale))

        raw_surplus = (budget - pred_cost) / max(1.0, budget)
        surplus = _clip01((raw_surplus + 1.0) / 2.0)

        budget_norm = _normalize(budget, lo, hi)
        recent_cost_norm = _normalize(recent_cost, lo, hi)
        ema_cost_norm = _normalize(ema_cost, lo, hi)
        max_cost_k_norm = _normalize(max_cost_k, lo, hi)

        # 每任务特征顺序必须严格固定为文档定义顺序。
        state_values.extend(
            [
                budget_norm,
                recent_cost_norm,
                ema_cost_norm,
                max_cost_k_norm,
                _clip01(overrun_ema),
                risk,
                surplus,
                is_hi,
                _clip01(priority_norm),
                _clip01(util_budget),
            ]
        )

        raw_budgets[task_name] = int(budget)
        raw_recent_costs[task_name] = int(recent_cost)

    recent_mode_change_rate = feature_state.rate(feature_state.window_mode_changes)
    recent_lo_cancel_rate = feature_state.rate(feature_state.window_lo_cancellations)
    recent_hi_overrun_rate = feature_state.rate(feature_state.window_hi_overruns)
    recent_lo_overrun_rate = feature_state.rate(feature_state.window_lo_overruns)

    state_values.extend(
        [
            _clip01(total_budget_util),
            _clip01(hi_budget_util),
            _clip01(lo_budget_util),
            _clip01(recent_mode_change_rate),
            _clip01(recent_lo_cancel_rate),
            _clip01(recent_hi_overrun_rate),
            _clip01(recent_lo_overrun_rate),
            _clip01(safety_margin_min),
        ]
    )

    return AgentObservation(
        time=time,
        state_vector=tuple(float(v) for v in state_values),
        raw_budgets=raw_budgets,
        raw_recent_costs=raw_recent_costs,
    )


def build_v12_full_14d_observation(
    *,
    time: int,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    monitor: RuntimeMonitor,
    bounds: NormalizationBounds | None = None,
    feature_state: RuntimeFeatureState,
    feature_config: FeatureConfig,
    safety_margin_min: float = 1.0,
    initial_budgets: dict[str, int] | None = None,
    safe_inc_possible_by_task: dict[str, bool] | None = None,
) -> AgentObservation:
    """按 v12_full_14d 语义构建观测（每任务 14 维 + 全局 8 维）。"""

    active_bounds = bounds or build_default_normalization_bounds(ordered_tasks)
    state_values: list[float] = []
    raw_budgets: dict[str, int] = {}
    raw_recent_costs: dict[str, int] = {}

    n_tasks = len(ordered_tasks)
    total_budget_util = 0.0
    hi_budget_util = 0.0
    lo_budget_util = 0.0

    for rank, task in enumerate(ordered_tasks):
        task_name = task.name
        budget = float(budget_state.budgets[task_name])
        recent_cost = float(monitor.recent_execution.get(task_name, 0))

        if task_name not in active_bounds:
            raise ValueError(f"normalization bounds 缺少任务 {task_name}")
        task_bound = active_bounds[task_name]
        lo = float(task_bound.min_cost)
        hi = float(task_bound.max_cost)

        feature_state.init_task(task_name, init_cost=float(task.c_lo))

        ema_cost = float(feature_state.ema_cost.get(task_name, float(task.c_lo)))
        history = feature_state.cost_history.get(task_name)
        max_cost_k = float(max(history)) if history and len(history) > 0 else recent_cost
        overrun_ema = float(feature_state.overrun_ema.get(task_name, 0.0))

        is_hi = 1.0 if task.criticality is Criticality.HI else 0.0
        priority_norm = 1.0 - float(rank) / float(max(1, n_tasks - 1))
        util_budget = budget / max(1.0, float(task.period))

        total_budget_util += util_budget
        if is_hi > 0.5:
            hi_budget_util += util_budget
        else:
            lo_budget_util += util_budget

        pred_cost = max(
            recent_cost,
            ema_cost,
            float(feature_config.max_cost_weight) * max_cost_k,
        )
        raw_risk = (
            pred_cost / max(1.0, budget)
            + 0.5 * overrun_ema
            + 0.2 * is_hi
            + 0.1 * priority_norm
        )
        risk = _clip01(raw_risk / float(feature_config.risk_max_scale))

        raw_surplus = (budget - pred_cost) / max(1.0, budget)
        surplus = _clip01((raw_surplus + 1.0) / 2.0)

        budget_norm = _normalize(budget, lo, hi)
        recent_cost_norm = _normalize(recent_cost, lo, hi)
        ema_cost_norm = _normalize(ema_cost, lo, hi)
        max_cost_k_norm = _normalize(max_cost_k, lo, hi)

        # 使用 episode 初始预算计算预算漂移；若未传入则退化到 task.c_lo 以支持单元测试。
        initial_budget = float((initial_budgets or {}).get(task_name, task.c_lo))
        budget_ratio = budget / max(1.0, initial_budget)
        positive_budget_drift = _clip01(max(0.0, budget_ratio - 1.0))
        negative_budget_drift = _clip01(max(0.0, 1.0 - budget_ratio))
        # 第一版使用 task-level overrun event 作为 cancellation pressure 的代理信号。
        task_cancel_ema = _clip01(feature_state.task_cancel_ema.get(task_name, 0.0))
        # 该特征仅作为观测 hint，不参与动作 mask。
        safe_inc_possible = 1.0 if (safe_inc_possible_by_task or {}).get(task_name, False) else 0.0

        state_values.extend(
            [
                budget_norm,
                recent_cost_norm,
                ema_cost_norm,
                max_cost_k_norm,
                _clip01(overrun_ema),
                risk,
                surplus,
                is_hi,
                _clip01(priority_norm),
                _clip01(util_budget),
                positive_budget_drift,
                negative_budget_drift,
                task_cancel_ema,
                safe_inc_possible,
            ]
        )

        raw_budgets[task_name] = int(budget)
        raw_recent_costs[task_name] = int(recent_cost)

    recent_mode_change_rate = feature_state.rate(feature_state.window_mode_changes)
    recent_lo_cancel_rate = feature_state.rate(feature_state.window_lo_cancellations)
    recent_hi_overrun_rate = feature_state.rate(feature_state.window_hi_overruns)
    recent_lo_overrun_rate = feature_state.rate(feature_state.window_lo_overruns)

    state_values.extend(
        [
            _clip01(total_budget_util),
            _clip01(hi_budget_util),
            _clip01(lo_budget_util),
            _clip01(recent_mode_change_rate),
            _clip01(recent_lo_cancel_rate),
            _clip01(recent_hi_overrun_rate),
            _clip01(recent_lo_overrun_rate),
            _clip01(safety_margin_min),
        ]
    )

    return AgentObservation(
        time=time,
        state_vector=tuple(float(v) for v in state_values),
        raw_budgets=raw_budgets,
        raw_recent_costs=raw_recent_costs,
    )


def build_observation(
    *,
    time: int,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    monitor: RuntimeMonitor,
    bounds: NormalizationBounds | None = None,
    feature_state: RuntimeFeatureState | None = None,
    feature_config: FeatureConfig | None = None,
    safety_margin_min: float = 1.0,
    initial_budgets: dict[str, int] | None = None,
    safe_inc_possible_by_task: dict[str, bool] | None = None,
) -> AgentObservation:
    """统一观测构造入口：按 observation_mode 分发到 v10 或 v11 实现。

    兼容性要求：
    - 旧调用方不传 feature_state/feature_config 时，默认走 v10_basic；
    - v11_full_10d 模式必须显式提供 feature_state，否则直接报错。
    """

    if feature_config is None or feature_config.observation_mode == OBSERVATION_MODE_V10_BASIC:
        return build_basic_observation(
            time=time,
            ordered_tasks=ordered_tasks,
            budget_state=budget_state,
            monitor=monitor,
            bounds=bounds,
        )
    if feature_config.observation_mode == OBSERVATION_MODE_V11_FULL_10D:
        if feature_state is None:
            raise ValueError("v11_full_10d 模式要求传入 feature_state")
        return build_v11_full_10d_observation(
            time=time,
            ordered_tasks=ordered_tasks,
            budget_state=budget_state,
            monitor=monitor,
            bounds=bounds,
            feature_state=feature_state,
            feature_config=feature_config,
            safety_margin_min=safety_margin_min,
        )
    if feature_config.observation_mode == OBSERVATION_MODE_V12_FULL_14D:
        if feature_state is None:
            raise ValueError("v12_full_14d 模式要求传入 feature_state")
        return build_v12_full_14d_observation(
            time=time,
            ordered_tasks=ordered_tasks,
            budget_state=budget_state,
            monitor=monitor,
            bounds=bounds,
            feature_state=feature_state,
            feature_config=feature_config,
            safety_margin_min=safety_margin_min,
            initial_budgets=initial_budgets,
            safe_inc_possible_by_task=safe_inc_possible_by_task,
        )
    raise ValueError(f"不支持的 observation_mode: {feature_config.observation_mode}")
