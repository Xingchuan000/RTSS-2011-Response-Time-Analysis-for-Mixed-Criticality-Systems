"""阶段 A：扫描不同 fixed_taskset_seed 与 budget_scale 的 baseline 与可调余量指标。"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from amc_py.dqn.experiment import build_automotive_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


def _parse_int_list_or_range(raw_value: str) -> list[int]:
    """解析整数列表或半开区间。

    支持两种形式：
    1. 逗号分隔：`0,1,2`
    2. 半开区间：`0:20`，等价于 `range(0, 20)`
    """

    text = raw_value.strip()
    if not text:
        return []

    if ":" in text:
        begin_text, end_text = (token.strip() for token in text.split(":", maxsplit=1))
        begin = int(begin_text)
        end = int(end_text)
        if end <= begin:
            raise ValueError(f"Invalid range {raw_value!r}: end must be greater than start")
        return list(range(begin, end))

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(raw_value: str) -> list[float]:
    """解析浮点数列表，例如 `0.85,0.90,1.00`。"""

    text = raw_value.strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _percentile(values: list[float], q: float) -> float:
    """线性插值分位数。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - float(lo)
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _safe_float(value: float) -> float:
    """把 NaN/Inf 清洗为 0，保证输出 CSV 不含非法数值。"""

    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _is_hi_task(task: Task) -> bool:
    """统一判断任务是否为 HI 关键级。"""

    criticality = getattr(task, "criticality", None)
    if criticality is Criticality.HI:
        return True
    name = getattr(criticality, "name", str(criticality)).upper()
    return name.endswith("HI") or name == "HI"


def _event_group(baseline_total_events_mean: float, low_event_threshold: float, high_event_threshold: float) -> str:
    """按阈值划分 baseline 事件强度。"""

    if baseline_total_events_mean < low_event_threshold:
        return "low_event"
    if baseline_total_events_mean <= high_event_threshold:
        return "medium_event"
    return "high_event"


def _slack_group(safety_margin_min_p05: float) -> str:
    """按阈值划分 slack 分组。"""

    if safety_margin_min_p05 < 0.001:
        return "tight"
    if safety_margin_min_p05 < 0.02:
        return "medium_slack"
    return "loose"


def _headroom_group(valid_action_count_mean: float) -> str:
    """legacy headroom：仅按总合法动作数分组。"""

    if valid_action_count_mean < 5.0:
        return "low_headroom"
    if valid_action_count_mean < 15.0:
        return "medium_headroom"
    return "high_headroom"


def _increase_headroom_group(valid_increase_count_mean: float) -> str:
    """按规则划分 increase headroom 分组。"""

    if valid_increase_count_mean <= 2.0:
        return "low_increase_headroom"
    if valid_increase_count_mean <= 5.0:
        return "medium_increase_headroom"
    return "high_increase_headroom"


def _decrease_headroom_group(valid_decrease_count_mean: float) -> str:
    """按规则划分 decrease headroom 分组。"""

    if valid_decrease_count_mean <= 2.0:
        return "low_decrease_headroom"
    if valid_decrease_count_mean <= 8.0:
        return "medium_decrease_headroom"
    return "high_decrease_headroom"


def _balanced_headroom_group(valid_increase_count_mean: float, valid_decrease_count_mean: float) -> str:
    """按规则判断 increase/decrease 是否平衡。"""

    ratio = valid_increase_count_mean / max(1.0, valid_decrease_count_mean)
    if ratio < 0.2:
        return "increase_limited"
    if ratio > 2.0:
        return "decrease_limited"
    return "balanced"


def _build_not_recommended_reason(
    *,
    baseline_deadline_misses_sum: float,
    event_group: str,
    valid_increase_count_mean: float,
    valid_decrease_count_mean: float,
    increase_decrease_balance: float,
) -> str:
    """构造不可推荐原因。

    约束：
    - 若 `recommended_for_dqn=True`，该字段必须为空字符串；
    - 若 `recommended_for_dqn=False`，该字段必须非空。
    """

    reasons: list[str] = []
    if baseline_deadline_misses_sum > 0.0:
        reasons.append("deadline_misses")

    if event_group == "low_event":
        reasons.append("too_few_baseline_events")
    elif event_group == "high_event":
        reasons.append("too_many_baseline_events")

    if valid_increase_count_mean < 3.0:
        reasons.append("low_valid_increase_headroom")
    if valid_decrease_count_mean < 3.0:
        reasons.append("low_valid_decrease_headroom")
    if increase_decrease_balance < 0.2:
        reasons.append("increase_decrease_imbalanced")

    return ";".join(reasons)


def _summarize_action_mask_by_type(env: AmcBudgetEnv, mask: tuple[bool, ...]) -> dict[str, float]:
    """按合法动作类型拆分 action mask 统计。"""

    valid_noop = 0
    valid_increase = 0
    valid_decrease = 0
    valid_increase_hi = 0
    valid_increase_lo = 0
    valid_decrease_hi = 0
    valid_decrease_lo = 0

    for action, is_valid in zip(env._actions, mask, strict=True):  # noqa: SLF001
        if not is_valid:
            continue

        if bool(getattr(action, "is_noop", False)):
            valid_noop += 1
            continue

        increase_idx = getattr(action, "increase_idx", None)
        if increase_idx is not None:
            valid_increase += 1
            task = env.ordered_tasks[increase_idx]
            if _is_hi_task(task):
                valid_increase_hi += 1
            else:
                valid_increase_lo += 1

        decrease_indices = getattr(action, "decrease_indices", ())
        for idx in decrease_indices:
            valid_decrease += 1
            task = env.ordered_tasks[idx]
            if _is_hi_task(task):
                valid_decrease_hi += 1
            else:
                valid_decrease_lo += 1

    return {
        "valid_noop_count": float(valid_noop),
        "valid_increase_count": float(valid_increase),
        "valid_decrease_count": float(valid_decrease),
        "valid_increase_hi_count": float(valid_increase_hi),
        "valid_increase_lo_count": float(valid_increase_lo),
        "valid_decrease_hi_count": float(valid_decrease_hi),
        "valid_decrease_lo_count": float(valid_decrease_lo),
    }


def _compute_budget_utils(ordered_tasks: list[Task], budgets: dict[str, int]) -> dict[str, float]:
    """计算预算利用率的 sum 与 per-task mean 两种口径。"""

    total_sum = 0.0
    hi_sum = 0.0
    lo_sum = 0.0
    hi_count = 0
    lo_count = 0

    for task in ordered_tasks:
        util = float(budgets[task.name]) / max(1.0, float(task.period))
        total_sum += util
        if _is_hi_task(task):
            hi_sum += util
            hi_count += 1
        else:
            lo_sum += util
            lo_count += 1

    return {
        "total_budget_util_sum": total_sum,
        "hi_budget_util_sum": hi_sum,
        "lo_budget_util_sum": lo_sum,
        "total_budget_util_mean_per_task": total_sum / max(1, len(ordered_tasks)),
        "hi_budget_util_mean_per_task": hi_sum / max(1, hi_count),
        "lo_budget_util_mean_per_task": lo_sum / max(1, lo_count),
    }


def _apply_budget_scale_to_tasks(
    *,
    ordered_tasks: list[Task],
    budget_scale: float,
    budget_floor_ratio: float,
) -> tuple[list[Task], dict[str, float]]:
    """把 budget_scale 显式应用到任务预算（`c_lo`）并返回缩放诊断。

    实现约束：
    1. 先按 `round(old_budget * budget_scale)` 计算候选预算；
    2. 再应用下界 `round(old_budget * budget_floor_ratio)`；
    3. 最后应用上界：HI 任务上界 `c_hi`，LO 任务上界 `deadline`；
    4. baseline 与 diagnostic 都使用这一份 scaled tasks，保证口径一致。
    """

    scaled_tasks: list[Task] = []
    effective_scales: list[float] = []
    changed_count = 0

    for task in ordered_tasks:
        old_budget = int(task.c_lo)
        min_budget = max(1, int(round(old_budget * budget_floor_ratio)))
        max_budget = int(task.c_hi) if _is_hi_task(task) else int(task.deadline)

        new_budget = int(round(old_budget * budget_scale))
        new_budget = max(min_budget, min(new_budget, max_budget))

        if new_budget != old_budget:
            changed_count += 1

        effective_scales.append(float(new_budget) / max(1.0, float(old_budget)))

        if _is_hi_task(task):
            scaled_tasks.append(replace(task, c_lo=new_budget))
        else:
            scaled_tasks.append(replace(task, c_lo=new_budget, c_hi=new_budget))

    stats = {
        "budget_scaled_task_count": float(changed_count),
        "budget_scale_effective_mean": mean(effective_scales) if effective_scales else 1.0,
        "budget_scale_effective_min": min(effective_scales) if effective_scales else 1.0,
        "budget_scale_effective_max": max(effective_scales) if effective_scales else 1.0,
    }
    return scaled_tasks, stats


def _run_diagnostic_on_seed(
    *,
    ordered_tasks: list[Task],
    scenario,
    feature_config: FeatureConfig,
    eval_seed: int,
    end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
) -> dict[str, float]:
    """在单个 eval_seed 上运行 noop 诊断并采集 headroom 指标。"""

    env = AmcBudgetEnv(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=agent_period,
        check_safety=True,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode="minimal",
        feature_config=feature_config,
    )

    explicit_noop_action_id: int | None = None
    for action in env._actions:  # noqa: SLF001
        if bool(getattr(action, "is_noop", False)):
            explicit_noop_action_id = int(action.action_id)
            break

    obs = env.reset(seed=eval_seed)
    done = False

    valid_action_count_values: list[float] = []
    valid_increase_count_values: list[float] = []
    valid_decrease_count_values: list[float] = []
    valid_noop_count_values: list[float] = []
    valid_increase_hi_count_values: list[float] = []
    valid_increase_lo_count_values: list[float] = []
    valid_decrease_hi_count_values: list[float] = []
    valid_decrease_lo_count_values: list[float] = []

    safety_margin_values: list[float] = []
    risk_values: list[float] = []
    surplus_values: list[float] = []
    hi_risk_values: list[float] = []
    lo_risk_values: list[float] = []
    hi_surplus_values: list[float] = []
    lo_surplus_values: list[float] = []

    total_budget_util_sum_values: list[float] = []
    hi_budget_util_sum_values: list[float] = []
    lo_budget_util_sum_values: list[float] = []
    total_budget_util_mean_per_task_values: list[float] = []
    hi_budget_util_mean_per_task_values: list[float] = []
    lo_budget_util_mean_per_task_values: list[float] = []

    diagnostic_mode_changes = 0.0
    diagnostic_lo_cancellations = 0.0
    diagnostic_deadline_misses = 0.0

    while not done:
        mask = env.valid_action_mask()
        mask_summary = _summarize_action_mask_by_type(env, mask)

        valid_action_count_values.append(float(sum(mask)))
        valid_increase_count_values.append(mask_summary["valid_increase_count"])
        valid_decrease_count_values.append(mask_summary["valid_decrease_count"])
        valid_noop_count_values.append(mask_summary["valid_noop_count"])
        valid_increase_hi_count_values.append(mask_summary["valid_increase_hi_count"])
        valid_increase_lo_count_values.append(mask_summary["valid_increase_lo_count"])
        valid_decrease_hi_count_values.append(mask_summary["valid_decrease_hi_count"])
        valid_decrease_lo_count_values.append(mask_summary["valid_decrease_lo_count"])

        action_id = explicit_noop_action_id if explicit_noop_action_id is not None else None
        step_result = env.step(action_id)
        info = step_result.info

        safety_margin_values.append(float(info.get("feature_safety_margin_min", 1.0)))
        diagnostic_mode_changes = float(info.get("mode_changes", diagnostic_mode_changes))
        diagnostic_lo_cancellations = float(info.get("lo_cancellations", diagnostic_lo_cancellations))
        diagnostic_deadline_misses = float(info.get("deadline_misses", diagnostic_deadline_misses))

        budget_utils = _compute_budget_utils(list(env.ordered_tasks), env._engine.runtime_budgets.budgets)  # noqa: SLF001
        total_budget_util_sum_values.append(budget_utils["total_budget_util_sum"])
        hi_budget_util_sum_values.append(budget_utils["hi_budget_util_sum"])
        lo_budget_util_sum_values.append(budget_utils["lo_budget_util_sum"])
        total_budget_util_mean_per_task_values.append(budget_utils["total_budget_util_mean_per_task"])
        hi_budget_util_mean_per_task_values.append(budget_utils["hi_budget_util_mean_per_task"])
        lo_budget_util_mean_per_task_values.append(budget_utils["lo_budget_util_mean_per_task"])

        vector = obs.state_vector
        task_count = len(env.ordered_tasks)
        if feature_config.observation_mode == "v11_full_10d":
            for i in range(task_count):
                risk = float(vector[i * 10 + 5])
                surplus = float(vector[i * 10 + 6])
                risk_values.append(risk)
                surplus_values.append(surplus)
                task = env.ordered_tasks[i]
                if _is_hi_task(task):
                    hi_risk_values.append(risk)
                    hi_surplus_values.append(surplus)
                else:
                    lo_risk_values.append(risk)
                    lo_surplus_values.append(surplus)

        obs = step_result.observation
        done = step_result.done

    return {
        "valid_action_count_mean": mean(valid_action_count_values) if valid_action_count_values else 0.0,
        "valid_action_count_p05": _percentile(valid_action_count_values, 0.05),
        "valid_action_count_median": median(valid_action_count_values) if valid_action_count_values else 0.0,
        "valid_increase_count_mean": mean(valid_increase_count_values) if valid_increase_count_values else 0.0,
        "valid_increase_count_p05": _percentile(valid_increase_count_values, 0.05),
        "valid_increase_count_median": median(valid_increase_count_values) if valid_increase_count_values else 0.0,
        "valid_decrease_count_mean": mean(valid_decrease_count_values) if valid_decrease_count_values else 0.0,
        "valid_decrease_count_p05": _percentile(valid_decrease_count_values, 0.05),
        "valid_decrease_count_median": median(valid_decrease_count_values) if valid_decrease_count_values else 0.0,
        "valid_noop_count_mean": mean(valid_noop_count_values) if valid_noop_count_values else 0.0,
        "valid_increase_hi_count_mean": mean(valid_increase_hi_count_values) if valid_increase_hi_count_values else 0.0,
        "valid_increase_lo_count_mean": mean(valid_increase_lo_count_values) if valid_increase_lo_count_values else 0.0,
        "valid_decrease_hi_count_mean": mean(valid_decrease_hi_count_values) if valid_decrease_hi_count_values else 0.0,
        "valid_decrease_lo_count_mean": mean(valid_decrease_lo_count_values) if valid_decrease_lo_count_values else 0.0,
        "safety_margin_min_mean": mean(safety_margin_values) if safety_margin_values else 1.0,
        "safety_margin_min_p05": _percentile(safety_margin_values, 0.05),
        "safety_margin_min_median": median(safety_margin_values) if safety_margin_values else 1.0,
        "safety_margin_min_fraction_zero": (
            sum(1 for value in safety_margin_values if value <= 1e-12) / float(len(safety_margin_values))
            if safety_margin_values
            else 0.0
        ),
        "total_budget_util_sum": mean(total_budget_util_sum_values) if total_budget_util_sum_values else 0.0,
        "hi_budget_util_sum": mean(hi_budget_util_sum_values) if hi_budget_util_sum_values else 0.0,
        "lo_budget_util_sum": mean(lo_budget_util_sum_values) if lo_budget_util_sum_values else 0.0,
        "total_budget_util_mean_per_task": (
            mean(total_budget_util_mean_per_task_values) if total_budget_util_mean_per_task_values else 0.0
        ),
        "hi_budget_util_mean_per_task": mean(hi_budget_util_mean_per_task_values) if hi_budget_util_mean_per_task_values else 0.0,
        "lo_budget_util_mean_per_task": mean(lo_budget_util_mean_per_task_values) if lo_budget_util_mean_per_task_values else 0.0,
        "risk_mean": mean(risk_values) if risk_values else 0.0,
        "risk_std": pstdev(risk_values) if len(risk_values) > 1 else 0.0,
        "surplus_mean": mean(surplus_values) if surplus_values else 0.0,
        "surplus_std": pstdev(surplus_values) if len(surplus_values) > 1 else 0.0,
        "hi_risk_mean": mean(hi_risk_values) if hi_risk_values else 0.0,
        "lo_risk_mean": mean(lo_risk_values) if lo_risk_values else 0.0,
        "hi_surplus_mean": mean(hi_surplus_values) if hi_surplus_values else 0.0,
        "lo_surplus_mean": mean(lo_surplus_values) if lo_surplus_values else 0.0,
        "diagnostic_mode_changes": diagnostic_mode_changes,
        "diagnostic_lo_cancellations": diagnostic_lo_cancellations,
        "diagnostic_deadline_misses": diagnostic_deadline_misses,
    }


@dataclass(frozen=True)
class ScanConfig:
    """扫描配置对象。

    该对象只保存可序列化的基础参数，便于多进程并行分发。
    """

    workload: str
    automotive_mode: str
    automotive_num_runnables: int
    require_schedulable: bool
    eval_seeds: tuple[int, ...]
    budget_scales: tuple[float, ...]
    end_time: int
    agent_period: int
    reward_mode: str
    action_space: str
    budget_increase_ratio: float
    budget_decrease_ratio: float
    include_explicit_noop: bool
    budget_floor_ratio: float
    forbid_decreasing_hi_budgets: bool
    observation_mode: str
    ema_alpha: float
    overrun_ema_alpha: float
    history_k: int
    event_window: int
    max_cost_weight: float
    risk_max_scale: float
    include_safety_margin: bool
    low_event_threshold: float
    high_event_threshold: float
    learnable_target_budget_util_min: float
    learnable_target_budget_util_max: float
    learnable_hi_budget_rho_min: float
    learnable_hi_budget_rho_max: float
    learnable_lo_budget_rho_min: float
    learnable_lo_budget_rho_max: float

    @classmethod
    def from_args(cls, args: argparse.Namespace, eval_seeds: list[int], budget_scales: list[float]) -> "ScanConfig":
        """从命令行参数构造不可变配置对象。"""

        return cls(
            workload=args.workload,
            automotive_mode=args.automotive_mode,
            automotive_num_runnables=args.automotive_num_runnables,
            require_schedulable=args.require_schedulable,
            eval_seeds=tuple(eval_seeds),
            budget_scales=tuple(budget_scales),
            end_time=args.end_time,
            agent_period=args.agent_period,
            reward_mode=args.reward_mode,
            action_space=args.action_space,
            budget_increase_ratio=args.budget_increase_ratio,
            budget_decrease_ratio=args.budget_decrease_ratio,
            include_explicit_noop=args.include_explicit_noop,
            budget_floor_ratio=args.budget_floor_ratio,
            forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
            observation_mode=args.observation_mode,
            ema_alpha=args.ema_alpha,
            overrun_ema_alpha=args.overrun_ema_alpha,
            history_k=args.history_k,
            event_window=args.event_window,
            max_cost_weight=args.max_cost_weight,
            risk_max_scale=args.risk_max_scale,
            include_safety_margin=args.include_safety_margin,
            low_event_threshold=args.low_event_threshold,
            high_event_threshold=args.high_event_threshold,
            learnable_target_budget_util_min=args.learnable_target_budget_util_min,
            learnable_target_budget_util_max=args.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=args.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=args.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=args.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=args.learnable_lo_budget_rho_max,
        )


def scan_one_taskset_seed_budget_scale(
    taskset_seed: int,
    budget_scale: float,
    config: ScanConfig,
) -> dict[str, int | float | str | bool]:
    """扫描单个 `(fixed_taskset_seed, budget_scale)` 组合，返回一行 CSV。"""

    if config.workload != "automotive":
        raise ValueError(f"unsupported workload: {config.workload}")

    experiment_config = build_automotive_experiment_config(
        num_runnables=config.automotive_num_runnables,
        mode=config.automotive_mode,
        require_schedulable=config.require_schedulable,
        fixed_taskset_seed=taskset_seed,
        learnable_target_budget_util_min=config.learnable_target_budget_util_min,
        learnable_target_budget_util_max=config.learnable_target_budget_util_max,
        learnable_hi_budget_rho_min=config.learnable_hi_budget_rho_min,
        learnable_hi_budget_rho_max=config.learnable_hi_budget_rho_max,
        learnable_lo_budget_rho_min=config.learnable_lo_budget_rho_min,
        learnable_lo_budget_rho_max=config.learnable_lo_budget_rho_max,
        budget_floor_ratio=config.budget_floor_ratio,
    )

    feature_config = FeatureConfig(
        observation_mode=config.observation_mode,
        ema_alpha=config.ema_alpha,
        overrun_ema_alpha=config.overrun_ema_alpha,
        history_k=config.history_k,
        event_window=config.event_window,
        max_cost_weight=config.max_cost_weight,
        risk_max_scale=config.risk_max_scale,
        include_safety_margin=config.include_safety_margin,
    )

    initial_bundle = resolve_experiment_bundle(experiment_config, config.eval_seeds[0])
    scaled_tasks_for_sched, budget_scale_stats = _apply_budget_scale_to_tasks(
        ordered_tasks=list(initial_bundle.ordered_tasks),
        budget_scale=budget_scale,
        budget_floor_ratio=config.budget_floor_ratio,
    )
    schedulable = evaluate_taskset(scaled_tasks_for_sched, method="amc_rtb", priority_policy="opa").schedulable
    num_hi_tasks = sum(1 for task in scaled_tasks_for_sched if _is_hi_task(task))
    num_lo_tasks = len(scaled_tasks_for_sched) - num_hi_tasks

    baseline_mode_changes_values: list[float] = []
    baseline_lo_cancellations_values: list[float] = []
    baseline_deadline_misses_values: list[float] = []
    diagnostic_rows: list[dict[str, float]] = []

    if not (config.require_schedulable and not schedulable):
        for eval_seed in config.eval_seeds:
            bundle_for_seed = resolve_experiment_bundle(experiment_config, eval_seed)
            scaled_tasks, _ = _apply_budget_scale_to_tasks(
                ordered_tasks=list(bundle_for_seed.ordered_tasks),
                budget_scale=budget_scale,
                budget_floor_ratio=config.budget_floor_ratio,
            )

            baseline_result = simulate_ordered_taskset_event_driven(
                ordered_tasks=scaled_tasks,
                scenario=bundle_for_seed.scenario,
                config=RuntimeConfig(end_time=config.end_time, semantics=RuntimeSemantics.AMC_PLUS),
            )
            baseline_mode_changes_values.append(float(baseline_result.mode_change_count()))
            baseline_lo_cancellations_values.append(float(baseline_result.lo_job_cancellation_count()))
            baseline_deadline_misses_values.append(float(len(baseline_result.deadline_misses)))

            diagnostic_rows.append(
                _run_diagnostic_on_seed(
                    ordered_tasks=scaled_tasks,
                    scenario=bundle_for_seed.scenario,
                    feature_config=feature_config,
                    eval_seed=eval_seed,
                    end_time=config.end_time,
                    agent_period=config.agent_period,
                    reward_mode=config.reward_mode,
                    action_space=config.action_space,
                    budget_increase_ratio=config.budget_increase_ratio,
                    budget_decrease_ratio=config.budget_decrease_ratio,
                    include_explicit_noop=config.include_explicit_noop,
                    budget_floor_ratio=config.budget_floor_ratio,
                    forbid_decreasing_hi_budgets=config.forbid_decreasing_hi_budgets,
                )
            )

    baseline_mode_changes_mean = mean(baseline_mode_changes_values) if baseline_mode_changes_values else 0.0
    baseline_lo_cancellations_mean = mean(baseline_lo_cancellations_values) if baseline_lo_cancellations_values else 0.0
    baseline_deadline_misses_sum = sum(baseline_deadline_misses_values) if baseline_deadline_misses_values else 0.0
    baseline_total_events_values = [
        mode_changes + lo_cancellations
        for mode_changes, lo_cancellations in zip(baseline_mode_changes_values, baseline_lo_cancellations_values, strict=True)
    ]
    baseline_total_events_mean = mean(baseline_total_events_values) if baseline_total_events_values else 0.0

    valid_increase_count_mean = mean(item["valid_increase_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    valid_decrease_count_mean = mean(item["valid_decrease_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    valid_action_count_mean = mean(item["valid_action_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    safety_margin_min_p05 = mean(item["safety_margin_min_p05"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    increase_decrease_balance = valid_increase_count_mean / max(1.0, valid_decrease_count_mean)

    event_group = _event_group(
        baseline_total_events_mean,
        low_event_threshold=config.low_event_threshold,
        high_event_threshold=config.high_event_threshold,
    )
    slack_group = _slack_group(safety_margin_min_p05)

    increase_headroom_group = _increase_headroom_group(valid_increase_count_mean)
    decrease_headroom_group = _decrease_headroom_group(valid_decrease_count_mean)
    balanced_headroom_group = _balanced_headroom_group(valid_increase_count_mean, valid_decrease_count_mean)

    recommended_for_dqn = (
        baseline_deadline_misses_sum == 0.0
        and event_group == "medium_event"
        and valid_increase_count_mean >= 3.0
        and valid_decrease_count_mean >= 3.0
        and increase_decrease_balance >= 0.2
    )

    not_recommended_reason = ""
    if not recommended_for_dqn:
        not_recommended_reason = _build_not_recommended_reason(
            baseline_deadline_misses_sum=baseline_deadline_misses_sum,
            event_group=event_group,
            valid_increase_count_mean=valid_increase_count_mean,
            valid_decrease_count_mean=valid_decrease_count_mean,
            increase_decrease_balance=increase_decrease_balance,
        )

    diagnostic_mode_changes_mean = mean(item["diagnostic_mode_changes"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    diagnostic_lo_cancellations_mean = mean(item["diagnostic_lo_cancellations"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    diagnostic_deadline_misses_sum = sum(item["diagnostic_deadline_misses"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    diagnostic_matches_baseline = (
        abs(diagnostic_mode_changes_mean - baseline_mode_changes_mean) < 1e-9
        and abs(diagnostic_lo_cancellations_mean - baseline_lo_cancellations_mean) < 1e-9
        and abs(diagnostic_deadline_misses_sum - baseline_deadline_misses_sum) < 1e-9
    )

    row: dict[str, Any] = {
        "fixed_taskset_seed": taskset_seed,
        "budget_scale": budget_scale,
        "budget_scaled_task_count": budget_scale_stats["budget_scaled_task_count"],
        "budget_scale_effective_mean": budget_scale_stats["budget_scale_effective_mean"],
        "budget_scale_effective_min": budget_scale_stats["budget_scale_effective_min"],
        "budget_scale_effective_max": budget_scale_stats["budget_scale_effective_max"],
        "schedulable": bool(schedulable),
        "num_eval_seeds": len(config.eval_seeds),
        "num_tasks": len(scaled_tasks_for_sched),
        "num_hi_tasks": num_hi_tasks,
        "num_lo_tasks": num_lo_tasks,
        "baseline_mode_changes_mean": baseline_mode_changes_mean,
        "baseline_mode_changes_std": pstdev(baseline_mode_changes_values) if len(baseline_mode_changes_values) > 1 else 0.0,
        "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
        "baseline_lo_cancellations_std": pstdev(baseline_lo_cancellations_values) if len(baseline_lo_cancellations_values) > 1 else 0.0,
        "baseline_total_events_mean": baseline_total_events_mean,
        "baseline_total_events_std": pstdev(baseline_total_events_values) if len(baseline_total_events_values) > 1 else 0.0,
        "baseline_deadline_misses_sum": baseline_deadline_misses_sum,
        "event_group": event_group,
        "slack_group": slack_group,
        "headroom_group": _headroom_group(valid_action_count_mean),
        "total_headroom_group": _headroom_group(valid_action_count_mean),
        "increase_headroom_group": increase_headroom_group,
        "decrease_headroom_group": decrease_headroom_group,
        "balanced_headroom_group": balanced_headroom_group,
        "recommended_for_dqn": bool(recommended_for_dqn),
        "not_recommended_reason": not_recommended_reason,
        "valid_action_count_mean": valid_action_count_mean,
        "valid_action_count_p05": mean(item["valid_action_count_p05"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_action_count_median": mean(item["valid_action_count_median"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_increase_count_mean": valid_increase_count_mean,
        "valid_increase_count_p05": mean(item["valid_increase_count_p05"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_increase_count_median": mean(item["valid_increase_count_median"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_decrease_count_mean": valid_decrease_count_mean,
        "valid_decrease_count_p05": mean(item["valid_decrease_count_p05"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_decrease_count_median": mean(item["valid_decrease_count_median"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_increase_fraction_mean": valid_increase_count_mean / max(1.0, float(len(scaled_tasks_for_sched))),
        "valid_decrease_fraction_mean": valid_decrease_count_mean / max(1.0, float(len(scaled_tasks_for_sched))),
        "increase_decrease_balance": increase_decrease_balance,
        "valid_increase_hi_count_mean": mean(item["valid_increase_hi_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_increase_lo_count_mean": mean(item["valid_increase_lo_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_decrease_hi_count_mean": mean(item["valid_decrease_hi_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_decrease_lo_count_mean": mean(item["valid_decrease_lo_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "valid_noop_count_mean": mean(item["valid_noop_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "safety_margin_min_mean": mean(item["safety_margin_min_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "safety_margin_min_p05": safety_margin_min_p05,
        "safety_margin_min_median": mean(item["safety_margin_min_median"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "safety_margin_min_fraction_zero": mean(item["safety_margin_min_fraction_zero"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "total_budget_util_sum": mean(item["total_budget_util_sum"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "hi_budget_util_sum": mean(item["hi_budget_util_sum"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "lo_budget_util_sum": mean(item["lo_budget_util_sum"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "total_budget_util_mean_per_task": (
            mean(item["total_budget_util_mean_per_task"] for item in diagnostic_rows) if diagnostic_rows else 0.0
        ),
        "hi_budget_util_mean_per_task": (
            mean(item["hi_budget_util_mean_per_task"] for item in diagnostic_rows) if diagnostic_rows else 0.0
        ),
        "lo_budget_util_mean_per_task": (
            mean(item["lo_budget_util_mean_per_task"] for item in diagnostic_rows) if diagnostic_rows else 0.0
        ),
        "risk_mean": mean(item["risk_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "risk_std": mean(item["risk_std"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "surplus_mean": mean(item["surplus_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "surplus_std": mean(item["surplus_std"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "hi_risk_mean": mean(item["hi_risk_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "lo_risk_mean": mean(item["lo_risk_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "hi_surplus_mean": mean(item["hi_surplus_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "lo_surplus_mean": mean(item["lo_surplus_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0,
        "diagnostic_mode_changes_mean": diagnostic_mode_changes_mean,
        "diagnostic_lo_cancellations_mean": diagnostic_lo_cancellations_mean,
        "diagnostic_deadline_misses_sum": diagnostic_deadline_misses_sum,
        "diagnostic_matches_baseline": bool(diagnostic_matches_baseline),
        "diagnostic_mode_delta_vs_baseline": diagnostic_mode_changes_mean - baseline_mode_changes_mean,
        "diagnostic_lo_cancel_delta_vs_baseline": diagnostic_lo_cancellations_mean - baseline_lo_cancellations_mean,
    }

    cleaned_row: dict[str, int | float | str | bool] = {}
    for key, value in row.items():
        if isinstance(value, float):
            cleaned_row[key] = _safe_float(value)
        else:
            cleaned_row[key] = value
    return cleaned_row


def _run_parallel(
    scan_pairs: list[tuple[int, float]],
    config: ScanConfig,
    workers: int,
) -> list[dict[str, int | float | str | bool]]:
    """按 `(fixed_taskset_seed, budget_scale)` 组合扫描并返回行结果。"""

    if workers <= 1:
        rows = [scan_one_taskset_seed_budget_scale(seed, scale, config) for seed, scale in scan_pairs]
        rows.sort(key=lambda row: (int(row["fixed_taskset_seed"]), float(row["budget_scale"])))
        return rows

    rows: list[dict[str, int | float | str | bool]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_taskset_seed_budget_scale, seed, scale, config): (seed, scale)
            for seed, scale in scan_pairs
        }
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            seed, scale = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Taskset scan failed for fixed_taskset_seed={seed}, budget_scale={scale}"
                ) from exc
            rows.append(row)
            completed += 1
            print(
                f"[scan] completed {completed}/{total} fixed_taskset_seed={seed} budget_scale={scale}",
                flush=True,
            )

    rows.sort(key=lambda row: (int(row["fixed_taskset_seed"]), float(row["budget_scale"])))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["automotive"], default="automotive")
    parser.add_argument("--automotive-mode", type=str, default="paper_exact")
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)
    parser.add_argument("--automotive-num-runnables", type=int, default=150)
    parser.add_argument("--require-schedulable", action="store_true")
    parser.add_argument("--fixed-taskset-seeds", type=str, required=True)
    parser.add_argument("--budget-scales", type=str, default="1.0")
    parser.add_argument("--eval-seeds", type=str, default="")
    parser.add_argument("--seeds", dest="seeds", type=str, default="")
    parser.add_argument("--end-time", type=int, default=10_000_000)
    parser.add_argument("--agent-period", type=int, default=100_000)
    parser.add_argument("--reward-mode", type=str, default="interval_v1")
    parser.add_argument("--action-space", choices=["single", "pair", "triple"], default="single")
    parser.add_argument("--budget-increase-ratio", type=float, default=0.025)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.0125)
    parser.add_argument("--include-explicit-noop", action="store_true")
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    parser.add_argument("--forbid-decreasing-hi-budgets", action="store_true")
    parser.add_argument("--observation-mode", choices=["v10_basic", "v11_full_10d"], default="v11_full_10d")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-event-threshold", type=float, default=40.0)
    parser.add_argument("--high-event-threshold", type=float, default=120.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for scanning combinations. Use 1 for serial execution.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """执行二维扫描并输出聚合 CSV。"""

    args = build_parser().parse_args()

    fixed_taskset_seeds = _parse_int_list_or_range(args.fixed_taskset_seeds)
    budget_scales = _parse_float_list(args.budget_scales)
    raw_eval_seeds = args.eval_seeds if args.eval_seeds.strip() else args.seeds
    eval_seeds = _parse_int_list_or_range(raw_eval_seeds)

    if not fixed_taskset_seeds:
        raise ValueError("--fixed-taskset-seeds 不能为空")
    if not budget_scales:
        raise ValueError("--budget-scales 不能为空")
    if not eval_seeds:
        raise ValueError("--eval-seeds/--seeds 不能为空")

    config = ScanConfig.from_args(args, eval_seeds, budget_scales)
    scan_pairs = [(seed, scale) for seed in fixed_taskset_seeds for scale in budget_scales]
    rows = _run_parallel(scan_pairs, config, args.workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fixed_taskset_seed",
        "budget_scale",
        "budget_scaled_task_count",
        "budget_scale_effective_mean",
        "budget_scale_effective_min",
        "budget_scale_effective_max",
        "schedulable",
        "num_eval_seeds",
        "num_tasks",
        "num_hi_tasks",
        "num_lo_tasks",
        "baseline_mode_changes_mean",
        "baseline_mode_changes_std",
        "baseline_lo_cancellations_mean",
        "baseline_lo_cancellations_std",
        "baseline_total_events_mean",
        "baseline_total_events_std",
        "baseline_deadline_misses_sum",
        "event_group",
        "slack_group",
        "headroom_group",
        "total_headroom_group",
        "increase_headroom_group",
        "decrease_headroom_group",
        "balanced_headroom_group",
        "recommended_for_dqn",
        "not_recommended_reason",
        "valid_action_count_mean",
        "valid_action_count_p05",
        "valid_action_count_median",
        "valid_increase_count_mean",
        "valid_increase_count_p05",
        "valid_increase_count_median",
        "valid_decrease_count_mean",
        "valid_decrease_count_p05",
        "valid_decrease_count_median",
        "valid_increase_fraction_mean",
        "valid_decrease_fraction_mean",
        "increase_decrease_balance",
        "valid_increase_hi_count_mean",
        "valid_increase_lo_count_mean",
        "valid_decrease_hi_count_mean",
        "valid_decrease_lo_count_mean",
        "valid_noop_count_mean",
        "safety_margin_min_mean",
        "safety_margin_min_p05",
        "safety_margin_min_median",
        "safety_margin_min_fraction_zero",
        "total_budget_util_sum",
        "hi_budget_util_sum",
        "lo_budget_util_sum",
        "total_budget_util_mean_per_task",
        "hi_budget_util_mean_per_task",
        "lo_budget_util_mean_per_task",
        "risk_mean",
        "risk_std",
        "surplus_mean",
        "surplus_std",
        "hi_risk_mean",
        "lo_risk_mean",
        "hi_surplus_mean",
        "lo_surplus_mean",
        "diagnostic_mode_changes_mean",
        "diagnostic_lo_cancellations_mean",
        "diagnostic_deadline_misses_sum",
        "diagnostic_matches_baseline",
        "diagnostic_mode_delta_vs_baseline",
        "diagnostic_lo_cancel_delta_vs_baseline",
    ]

    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_output.replace(args.output)

    schedulable_count = sum(1 for row in rows if bool(row["schedulable"]))
    recommended_count = sum(1 for row in rows if bool(row["recommended_for_dqn"]))
    print(f"Scanned {len(rows)} rows (seed x budget_scale)")
    print(f"Schedulable rows: {schedulable_count}")
    print(f"Recommended for DQN rows: {recommended_count}")


if __name__ == "__main__":
    main()
