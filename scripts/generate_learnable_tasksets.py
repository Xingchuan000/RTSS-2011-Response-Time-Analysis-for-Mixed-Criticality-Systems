"""生成 `paper_learnable_headroom` taskset 清单。

脚本职责：
1. 基于 automotive `paper_learnable_headroom` 模式按 candidate seed 生成候选 taskset；
2. 先做静态 reserve 检查，再做 fast 诊断；
3. 输出 accepted manifest 与 rejected 明细，便于后续复现与分析。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean

from amc_py.dqn.experiment import build_automotive_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


@dataclass(frozen=True)
class LearnableGenerateConfig:
    """learnable taskset 生成配置。"""

    automotive_num_runnables: int
    num_tasksets: int
    candidate_seed_start: int
    learnable_max_attempts: int
    learnable_min_static_increase_reserve: int
    learnable_min_static_hi_increase_reserve: int
    learnable_min_static_decrease_reserve: int
    learnable_fast_end_time: int
    learnable_fast_eval_seeds: int
    learnable_fast_event_min: float
    learnable_fast_event_max: float
    learnable_fast_min_valid_increase: float
    learnable_fast_min_valid_decrease: float
    learnable_fast_min_balance: float
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
    learnable_target_budget_util_min: float
    learnable_target_budget_util_max: float
    learnable_hi_budget_rho_min: float
    learnable_hi_budget_rho_max: float
    learnable_lo_budget_rho_min: float
    learnable_lo_budget_rho_max: float
    require_schedulable: bool
    learnable_generation_strategy: str
    learnable_enable_safety_relaxation: bool
    learnable_relax_target_valid_increase: int
    learnable_relax_max_rounds: int
    learnable_relax_step_ratio: float
    learnable_relax_min_budget_floor_ratio: float | None


def _rewrite_budgets_for_learnable_headroom(
    *,
    ordered_tasks: list[Task],
    candidate_seed: int,
    cfg: LearnableGenerateConfig,
) -> tuple[list[Task], dict[str, float]]:
    """在给定任务结构上执行 learnable headroom 预算重写。

    该函数对应 two-stage 策略中的第 2.c 步：
    - 先基于 min/max 区间采样预算；
    - 再按 target util 比例缩放并回夹；
    - 最后生成重写后的 Task 列表和预算元数据。
    """

    rng = random.Random(candidate_seed * 2_000_003 + 97_409)
    min_budget: dict[str, int] = {}
    max_budget: dict[str, int] = {}
    initial_budget: dict[str, int] = {}

    for task in ordered_tasks:
        original_budget = int(task.c_lo)
        min_b = max(1, int(round(float(original_budget) * cfg.budget_floor_ratio)))
        max_b = int(task.c_hi) if _is_hi_task(task) else int(max(task.c_hi, min(task.deadline, task.period)))
        if min_b >= max_b:
            min_budget[task.name] = original_budget
            max_budget[task.name] = original_budget
            initial_budget[task.name] = original_budget
            continue
        rho = (
            rng.uniform(cfg.learnable_hi_budget_rho_min, cfg.learnable_hi_budget_rho_max)
            if _is_hi_task(task)
            else rng.uniform(cfg.learnable_lo_budget_rho_min, cfg.learnable_lo_budget_rho_max)
        )
        sampled_budget = int(round(min_b + rho * float(max_b - min_b)))
        min_budget[task.name] = min_b
        max_budget[task.name] = max_b
        initial_budget[task.name] = max(min_b, min(max_b, sampled_budget))

    current_total_util = sum(float(initial_budget[t.name]) / float(t.period) for t in ordered_tasks)
    target_total_util = rng.uniform(cfg.learnable_target_budget_util_min, cfg.learnable_target_budget_util_max)
    scale = target_total_util / current_total_util

    rewritten_budget: dict[str, int] = {}
    for task in ordered_tasks:
        scaled = int(round(float(initial_budget[task.name]) * scale))
        rewritten_budget[task.name] = max(min_budget[task.name], min(max_budget[task.name], scaled))

    rewritten_tasks: list[Task] = []
    for task in ordered_tasks:
        c_lo = rewritten_budget[task.name]
        c_hi = task.c_hi if _is_hi_task(task) else c_lo
        rewritten_tasks.append(
            Task(
                name=task.name,
                period=task.period,
                deadline=task.deadline,
                c_lo=c_lo,
                c_hi=c_hi,
                criticality=task.criticality,
            )
        )

    actual_total_util = sum(float(rewritten_budget[t.name]) / float(t.period) for t in ordered_tasks)
    actual_hi_util = sum(float(rewritten_budget[t.name]) / float(t.period) for t in ordered_tasks if _is_hi_task(t))
    actual_lo_util = sum(float(rewritten_budget[t.name]) / float(t.period) for t in ordered_tasks if not _is_hi_task(t))
    metadata = {
        "target_budget_util": target_total_util,
        "actual_budget_util_total": actual_total_util,
        "actual_budget_util_hi": actual_hi_util,
        "actual_budget_util_lo": actual_lo_util,
        "budget_util_error": abs(actual_total_util - target_total_util),
    }
    return rewritten_tasks, metadata


def _is_hi_task(task: Task) -> bool:
    """统一判断 HI 任务。"""

    return task.criticality is Criticality.HI


def _compute_budget_range(task: Task, budget_floor_ratio: float) -> tuple[int, int]:
    """按计划定义每个任务的 min/max budget。"""

    min_budget = max(1, int(round(float(task.c_lo) * budget_floor_ratio)))
    max_budget = int(task.c_hi) if _is_hi_task(task) else int(max(task.c_hi, min(task.deadline, task.period)))
    return min_budget, max_budget


def _compute_static_reserve_counts(
    ordered_tasks: list[Task],
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    budget_floor_ratio: float,
) -> dict[str, int]:
    """计算静态 increase/decrease reserve 统计。"""

    inc_count = 0
    hi_inc_count = 0
    dec_count = 0

    for task in ordered_tasks:
        budget = int(task.c_lo)
        min_budget, max_budget = _compute_budget_range(task, budget_floor_ratio)
        inc_step = max(1, int(round(float(budget) * budget_increase_ratio)))
        dec_step = max(1, int(round(float(budget) * budget_decrease_ratio)))

        if max_budget - budget >= inc_step:
            inc_count += 1
            if _is_hi_task(task):
                hi_inc_count += 1
        if budget - min_budget >= dec_step:
            dec_count += 1

    return {
        "static_increase_reserve_count": inc_count,
        "static_hi_increase_reserve_count": hi_inc_count,
        "static_decrease_reserve_count": dec_count,
    }


def _get_valid_action_mask(env: AmcBudgetEnv) -> tuple[bool, ...]:
    """返回环境当前可用动作掩码，兼容不同 API 命名。"""

    if hasattr(env, "valid_action_mask"):
        return tuple(bool(value) for value in env.valid_action_mask())

    if hasattr(env, "action_mask"):
        candidate = env.action_mask
        if callable(candidate):
            return tuple(bool(value) for value in candidate())
        return tuple(bool(value) for value in candidate)

    raise AttributeError(f"{type(env).__name__} has neither valid_action_mask() nor action_mask")


def _get_env_actions(env: AmcBudgetEnv):
    """返回环境动作描述列表，优先公共属性，回退私有属性。"""

    if hasattr(env, "actions"):
        return env.actions
    if hasattr(env, "_actions"):
        return env._actions  # noqa: SLF001
    raise AttributeError(f"{type(env).__name__} has neither actions nor _actions")


def _count_valid_action_types(env: AmcBudgetEnv, mask: tuple[bool, ...]) -> tuple[int, int, int]:
    """统计有效 increase/decrease/noop 数量。"""

    actions = _get_env_actions(env)
    if len(actions) != len(mask):
        raise RuntimeError(f"action/mask length mismatch: actions={len(actions)}, mask={len(mask)}")

    valid_increase_count = 0
    valid_decrease_count = 0
    valid_noop_count = 0
    for action, valid in zip(actions, mask, strict=True):
        if not valid:
            continue
        if getattr(action, "is_noop", False):
            valid_noop_count += 1
            continue

        kind = getattr(action, "kind", None)
        if kind == "noop":
            valid_noop_count += 1
            continue
        if kind == "increase":
            valid_increase_count += 1
            continue
        if kind == "decrease":
            valid_decrease_count += 1
            continue

        increase_idx = getattr(action, "increase_idx", None)
        if increase_idx is not None:
            valid_increase_count += 1
        decrease_indices = getattr(action, "decrease_indices", ())
        if decrease_indices:
            valid_decrease_count += len(tuple(decrease_indices))

    return valid_increase_count, valid_decrease_count, valid_noop_count


def _diagnose_one_seed(
    *,
    ordered_tasks: list[Task],
    scenario,
    eval_seed: int,
    runtime_end_time: int,
    cfg: LearnableGenerateConfig,
    feature_config: FeatureConfig,
) -> dict[str, float]:
    """在一个 scenario seed 上执行 baseline + action mask 快速诊断。"""

    runtime_result = simulate_ordered_taskset_event_driven(
        ordered_tasks=list(ordered_tasks),
        scenario=scenario,
        config=RuntimeConfig(end_time=runtime_end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )

    env = AmcBudgetEnv(
        ordered_tasks=list(ordered_tasks),
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=runtime_end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=100_000,
        check_safety=True,
        reward_mode=cfg.reward_mode,
        action_space=cfg.action_space,
        budget_increase_ratio=cfg.budget_increase_ratio,
        budget_decrease_ratio=cfg.budget_decrease_ratio,
        include_explicit_noop=cfg.include_explicit_noop,
        budget_floor_ratio=cfg.budget_floor_ratio,
        forbid_decreasing_hi_budgets=cfg.forbid_decreasing_hi_budgets,
        mask_detail_mode="minimal",
        feature_config=feature_config,
    )
    env.reset(seed=eval_seed)
    mask = _get_valid_action_mask(env)
    valid_increase_count, valid_decrease_count, valid_noop_count = _count_valid_action_types(env, mask)
    # 读取 safety-on 掩码拒绝原因统计，用于确认 mask 的主导瓶颈。
    reject_reason_counts = {}
    if getattr(env, "_mask_log", None):  # noqa: SLF001
        reject_reason_counts = dict(env._mask_log[-1].get("reject_reason_counts", {}))  # noqa: SLF001

    # 额外跑一轮 check_safety=False 的 no-safety 诊断，
    # 用于区分“动作空间本身不足”与“安全约束导致被 mask”。
    no_safety_env = AmcBudgetEnv(
        ordered_tasks=list(ordered_tasks),
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=runtime_end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=100_000,
        check_safety=False,
        reward_mode=cfg.reward_mode,
        action_space=cfg.action_space,
        budget_increase_ratio=cfg.budget_increase_ratio,
        budget_decrease_ratio=cfg.budget_decrease_ratio,
        include_explicit_noop=cfg.include_explicit_noop,
        budget_floor_ratio=cfg.budget_floor_ratio,
        forbid_decreasing_hi_budgets=cfg.forbid_decreasing_hi_budgets,
        mask_detail_mode="minimal",
        feature_config=feature_config,
    )
    no_safety_env.reset(seed=eval_seed)
    no_safety_mask = _get_valid_action_mask(no_safety_env)
    valid_increase_count_no_safety, valid_decrease_count_no_safety, _valid_noop_count_no_safety = _count_valid_action_types(
        no_safety_env, no_safety_mask
    )

    return {
        "mode_changes": float(runtime_result.mode_change_count()),
        "lo_cancellations": float(runtime_result.lo_job_cancellation_count()),
        "total_events": float(runtime_result.mode_change_count() + runtime_result.lo_job_cancellation_count()),
        "deadline_misses": float(len(runtime_result.deadline_misses)),
        "valid_increase_count": float(valid_increase_count),
        "valid_decrease_count": float(valid_decrease_count),
        "valid_noop_count": float(valid_noop_count),
        "valid_increase_count_no_safety": float(valid_increase_count_no_safety),
        "valid_decrease_count_no_safety": float(valid_decrease_count_no_safety),
        "mask_reject_incremental_constraint_violation": float(
            reject_reason_counts.get("incremental_constraint_violation", 0)
        ),
        "mask_reject_no_effective_budget_change": float(reject_reason_counts.get("no_effective_budget_change", 0)),
        "mask_reject_budget_floor_violation": float(reject_reason_counts.get("budget_floor_violation", 0)),
        # 当前 env 中没有独立的 budget_upper_bound_violation reject reason，
        # 这里先显式输出 0，保持诊断字段稳定，便于后续对齐。
        "mask_reject_budget_upper_bound_violation": 0.0,
        "mask_reject_decrease_hi_forbidden": float(reject_reason_counts.get("decrease_hi_forbidden", 0)),
    }


def compute_initial_valid_action_counts(
    *,
    ordered_tasks: list[Task],
    scenario,
    cfg: LearnableGenerateConfig,
    feature_config: FeatureConfig,
    check_safety: bool = True,
) -> dict[str, float]:
    """只在 reset 时统计一次 valid-action 与 mask reject 原因。"""

    env = AmcBudgetEnv(
        ordered_tasks=list(ordered_tasks),
        scenario=scenario,
        runtime_config=RuntimeConfig(end_time=cfg.learnable_fast_end_time, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=100_000,
        check_safety=check_safety,
        reward_mode=cfg.reward_mode,
        action_space=cfg.action_space,
        budget_increase_ratio=cfg.budget_increase_ratio,
        budget_decrease_ratio=cfg.budget_decrease_ratio,
        include_explicit_noop=cfg.include_explicit_noop,
        budget_floor_ratio=cfg.budget_floor_ratio,
        forbid_decreasing_hi_budgets=cfg.forbid_decreasing_hi_budgets,
        mask_detail_mode="minimal",
        feature_config=feature_config,
    )
    env.reset(seed=0)
    mask = _get_valid_action_mask(env)
    valid_increase_count, valid_decrease_count, valid_noop_count = _count_valid_action_types(env, mask)
    mask_log_row = env.mask_log[-1] if env.mask_log else {}
    reject_reason_counts = mask_log_row.get("reject_reason_counts", {})
    return {
        "valid_increase_count": float(valid_increase_count),
        "valid_decrease_count": float(valid_decrease_count),
        "valid_noop_count": float(valid_noop_count),
        "increase_decrease_balance": float(valid_increase_count / max(1.0, valid_decrease_count)),
        "mask_reject_incremental_constraint_violation": float(
            reject_reason_counts.get("incremental_constraint_violation", 0)
        ),
        "mask_reject_no_effective_budget_change": float(reject_reason_counts.get("no_effective_budget_change", 0)),
        "mask_reject_budget_floor_violation": float(reject_reason_counts.get("budget_floor_violation", 0)),
        "mask_reject_budget_upper_bound_violation": float(reject_reason_counts.get("budget_upper_bound_violation", 0)),
        "mask_reject_decrease_hi_forbidden": float(reject_reason_counts.get("decrease_hi_forbidden", 0)),
    }


def choose_budget_relaxation_task(
    *,
    ordered_tasks: list[Task],
    current_budgets: dict[str, int],
    initial_budgets: dict[str, int],
    budget_floor_ratio: float,
) -> Task | None:
    """选择一个可下调预算的任务，优先 LO、长周期、高利用率。"""

    candidates: list[tuple[int, float, float, Task]] = []
    for task in ordered_tasks:
        current = int(current_budgets[task.name])
        floor = max(1, int(round(float(initial_budgets[task.name]) * budget_floor_ratio)))
        if current <= floor:
            continue
        is_hi = _is_hi_task(task)
        util = float(current) / max(1.0, float(task.period))
        candidates.append((0 if not is_hi else 1, -float(task.period), -util, task))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[:3])[0][3]


def relax_budgets_for_safety_valid_increase(
    *,
    ordered_tasks: list[Task],
    scenario,
    cfg: LearnableGenerateConfig,
    feature_config: FeatureConfig,
    initial_budgets: dict[str, int],
    target_valid_increase: int,
    max_rounds: int,
    step_ratio: float,
    budget_floor_ratio: float,
) -> tuple[list[Task], dict[str, float | bool]]:
    """按 safety-valid increase 目标迭代放松预算。"""

    current_tasks = [Task(t.name, t.period, t.deadline, t.c_lo, t.c_hi, t.criticality) for t in ordered_tasks]
    current_budget_map = {task.name: int(task.c_lo) for task in current_tasks}
    before = compute_initial_valid_action_counts(
        ordered_tasks=current_tasks,
        scenario=scenario,
        cfg=cfg,
        feature_config=feature_config,
        check_safety=True,
    )
    rounds = 0
    relaxed_task_names: set[str] = set()
    relaxed_hi_count = 0
    relaxed_lo_count = 0
    success = before["valid_increase_count"] >= float(target_valid_increase)

    while not success and rounds < max_rounds:
        task = choose_budget_relaxation_task(
            ordered_tasks=current_tasks,
            current_budgets=current_budget_map,
            initial_budgets=initial_budgets,
            budget_floor_ratio=budget_floor_ratio,
        )
        if task is None:
            break
        old_budget = current_budget_map[task.name]
        floor = max(1, int(round(float(initial_budgets[task.name]) * budget_floor_ratio)))
        new_budget = max(floor, int(round(float(old_budget) * (1.0 - step_ratio))))
        if new_budget >= old_budget:
            new_budget = max(floor, old_budget - 1)
        if new_budget >= old_budget:
            break

        updated: list[Task] = []
        for item in current_tasks:
            if item.name != task.name:
                updated.append(item)
                continue
            c_lo = int(new_budget)
            c_hi = item.c_hi if _is_hi_task(item) else c_lo
            updated.append(Task(item.name, item.period, item.deadline, c_lo, c_hi, item.criticality))
        current_tasks = updated
        current_budget_map[task.name] = int(new_budget)
        relaxed_task_names.add(task.name)
        if _is_hi_task(task):
            relaxed_hi_count += 1
        else:
            relaxed_lo_count += 1
        rounds += 1

        stats = compute_initial_valid_action_counts(
            ordered_tasks=current_tasks,
            scenario=scenario,
            cfg=cfg,
            feature_config=feature_config,
            check_safety=True,
        )
        success = stats["valid_increase_count"] >= float(target_valid_increase)

    after = compute_initial_valid_action_counts(
        ordered_tasks=current_tasks,
        scenario=scenario,
        cfg=cfg,
        feature_config=feature_config,
        check_safety=True,
    )
    budget_util_before = sum(float(initial_budgets[t.name]) / float(t.period) for t in ordered_tasks)
    budget_util_after = sum(float(current_budget_map[t.name]) / float(t.period) for t in current_tasks)
    metadata: dict[str, float | bool] = {
        "relaxation_enabled": True,
        "relaxation_success": bool(success),
        "relaxation_rounds": float(rounds),
        "valid_increase_before_relax": before["valid_increase_count"],
        "valid_increase_after_relax": after["valid_increase_count"],
        "valid_decrease_before_relax": before["valid_decrease_count"],
        "valid_decrease_after_relax": after["valid_decrease_count"],
        "balance_before_relax": before["increase_decrease_balance"],
        "balance_after_relax": after["increase_decrease_balance"],
        "relaxed_task_count": float(len(relaxed_task_names)),
        "relaxed_hi_task_count": float(relaxed_hi_count),
        "relaxed_lo_task_count": float(relaxed_lo_count),
        "budget_util_before_relax": float(budget_util_before),
        "budget_util_after_relax": float(budget_util_after),
    }
    return current_tasks, metadata


def _build_candidate_base_row(
    *,
    candidate_seed: int,
    ordered_tasks: list[Task],
    static_counts: dict[str, int],
    metadata: dict[str, object] | None,
) -> dict[str, int | float | str | bool]:
    """构造候选行的公共字段，保证 accepted/rejected 字段口径一致。"""

    workload_metadata = (metadata or {}).get("workload_metadata", {})
    if not isinstance(workload_metadata, dict):
        workload_metadata = {}
    return {
        "candidate_seed": candidate_seed,
        "num_tasks": len(ordered_tasks),
        "num_hi_tasks": sum(1 for t in ordered_tasks if _is_hi_task(t)),
        "num_lo_tasks": sum(1 for t in ordered_tasks if not _is_hi_task(t)),
        "target_budget_util": float(workload_metadata.get("target_budget_util", 0.0)),
        "actual_budget_util_total": float(workload_metadata.get("actual_budget_util_total", 0.0)),
        "actual_budget_util_hi": float(workload_metadata.get("actual_budget_util_hi", 0.0)),
        "actual_budget_util_lo": float(workload_metadata.get("actual_budget_util_lo", 0.0)),
        "budget_util_error": float(workload_metadata.get("budget_util_error", 0.0)),
        **static_counts,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--automotive-num-runnables", type=int, choices=[150, 250], default=150)
    parser.add_argument(
        "--learnable-generation-strategy",
        choices=["two_stage_from_paper_exact", "direct"],
        default="two_stage_from_paper_exact",
    )
    parser.add_argument("--num-tasksets", type=int, default=3)
    parser.add_argument("--candidate-seed-start", type=int, default=0)
    parser.add_argument("--learnable-max-attempts", type=int, default=200)
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.35)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.55)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.55)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.25)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.50)
    parser.add_argument("--learnable-min-static-increase-reserve", type=int, default=4)
    parser.add_argument("--learnable-min-static-hi-increase-reserve", type=int, default=2)
    parser.add_argument("--learnable-min-static-decrease-reserve", type=int, default=4)
    parser.add_argument("--learnable-fast-end-time", type=int, default=1_000_000)
    parser.add_argument("--learnable-fast-eval-seeds", type=int, default=5)
    parser.add_argument("--learnable-fast-event-min", type=float, default=3.0)
    parser.add_argument("--learnable-fast-event-max", type=float, default=80.0)
    parser.add_argument("--learnable-fast-min-valid-increase", type=float, default=2.0)
    parser.add_argument("--learnable-fast-min-valid-decrease", type=float, default=3.0)
    parser.add_argument("--learnable-fast-min-balance", type=float, default=0.12)
    parser.add_argument(
        "--learnable-enable-safety-relaxation",
        action="store_true",
        help="Enable safety-mask-aware budget relaxation before fast diagnostic.",
    )
    parser.add_argument("--learnable-relax-target-valid-increase", type=int, default=2)
    parser.add_argument("--learnable-relax-max-rounds", type=int, default=20)
    parser.add_argument("--learnable-relax-step-ratio", type=float, default=0.025)
    parser.add_argument("--learnable-relax-min-budget-floor-ratio", type=float, default=None)
    parser.add_argument(
        "--require-schedulable",
        action="store_true",
        help="Require generated automotive workload to be schedulable before diagnostics.",
    )
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
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-rejections", type=Path, required=True)
    return parser


def main() -> None:
    """执行 learnable taskset 生成与筛选。"""

    args = _build_parser().parse_args()
    cfg = LearnableGenerateConfig(
        automotive_num_runnables=args.automotive_num_runnables,
        num_tasksets=args.num_tasksets,
        candidate_seed_start=args.candidate_seed_start,
        learnable_max_attempts=args.learnable_max_attempts,
        learnable_min_static_increase_reserve=args.learnable_min_static_increase_reserve,
        learnable_min_static_hi_increase_reserve=args.learnable_min_static_hi_increase_reserve,
        learnable_min_static_decrease_reserve=args.learnable_min_static_decrease_reserve,
        learnable_fast_end_time=args.learnable_fast_end_time,
        learnable_fast_eval_seeds=args.learnable_fast_eval_seeds,
        learnable_fast_event_min=args.learnable_fast_event_min,
        learnable_fast_event_max=args.learnable_fast_event_max,
        learnable_fast_min_valid_increase=args.learnable_fast_min_valid_increase,
        learnable_fast_min_valid_decrease=args.learnable_fast_min_valid_decrease,
        learnable_fast_min_balance=args.learnable_fast_min_balance,
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
        learnable_target_budget_util_min=args.learnable_target_budget_util_min,
        learnable_target_budget_util_max=args.learnable_target_budget_util_max,
        learnable_hi_budget_rho_min=args.learnable_hi_budget_rho_min,
        learnable_hi_budget_rho_max=args.learnable_hi_budget_rho_max,
        learnable_lo_budget_rho_min=args.learnable_lo_budget_rho_min,
        learnable_lo_budget_rho_max=args.learnable_lo_budget_rho_max,
        require_schedulable=args.require_schedulable,
        learnable_generation_strategy=args.learnable_generation_strategy,
        learnable_enable_safety_relaxation=args.learnable_enable_safety_relaxation,
        learnable_relax_target_valid_increase=args.learnable_relax_target_valid_increase,
        learnable_relax_max_rounds=args.learnable_relax_max_rounds,
        learnable_relax_step_ratio=args.learnable_relax_step_ratio,
        learnable_relax_min_budget_floor_ratio=args.learnable_relax_min_budget_floor_ratio,
    )

    feature_config = FeatureConfig(
        observation_mode=cfg.observation_mode,
        ema_alpha=cfg.ema_alpha,
        overrun_ema_alpha=cfg.overrun_ema_alpha,
        history_k=cfg.history_k,
        event_window=cfg.event_window,
        max_cost_weight=cfg.max_cost_weight,
        risk_max_scale=cfg.risk_max_scale,
        include_safety_margin=cfg.include_safety_margin,
    )

    accepted_rows: list[dict[str, int | float | str | bool]] = []
    rejected_rows: list[dict[str, int | float | str | bool]] = []

    for attempt in range(cfg.learnable_max_attempts):
        candidate_seed = cfg.candidate_seed_start + attempt
        try:
            if cfg.learnable_generation_strategy == "two_stage_from_paper_exact":
                # two-stage: 先用 paper_exact+可调度筛选得到 base bundle，再做预算重写。
                base_config = build_automotive_experiment_config(
                    num_runnables=cfg.automotive_num_runnables,
                    mode="paper_exact",
                    require_schedulable=True,
                    fixed_taskset_seed=candidate_seed,
                    budget_floor_ratio=cfg.budget_floor_ratio,
                )
                base_bundle = resolve_experiment_bundle(base_config, seed=10_000 + candidate_seed)
                ordered_tasks, rewritten_meta = _rewrite_budgets_for_learnable_headroom(
                    ordered_tasks=list(base_bundle.ordered_tasks),
                    candidate_seed=candidate_seed,
                    cfg=cfg,
                )
                candidate_metadata = {"workload_metadata": rewritten_meta}
                candidate_scenario = base_bundle.scenario
            else:
                # 保留 direct 策略可选：直接使用 paper_learnable_headroom workload。
                experiment_config = build_automotive_experiment_config(
                    num_runnables=cfg.automotive_num_runnables,
                    mode="paper_learnable_headroom",
                    require_schedulable=cfg.require_schedulable,
                    fixed_taskset_seed=candidate_seed,
                    learnable_target_budget_util_min=cfg.learnable_target_budget_util_min,
                    learnable_target_budget_util_max=cfg.learnable_target_budget_util_max,
                    learnable_hi_budget_rho_min=cfg.learnable_hi_budget_rho_min,
                    learnable_hi_budget_rho_max=cfg.learnable_hi_budget_rho_max,
                    learnable_lo_budget_rho_min=cfg.learnable_lo_budget_rho_min,
                    learnable_lo_budget_rho_max=cfg.learnable_lo_budget_rho_max,
                    budget_floor_ratio=cfg.budget_floor_ratio,
                )
                bundle = resolve_experiment_bundle(experiment_config, seed=10_000 + candidate_seed)
                ordered_tasks = list(bundle.ordered_tasks)
                candidate_metadata = bundle.metadata
                candidate_scenario = bundle.scenario
        except RuntimeError as exc:
            rejected_rows.append(
                {
                    "taskset_id": -1,
                    "candidate_seed": candidate_seed,
                    "accepted": False,
                    "reject_reason": "reject_unschedulable_generation",
                    "error_message": str(exc),
                    "num_tasks": 0,
                    "num_hi_tasks": 0,
                    "num_lo_tasks": 0,
                    "target_budget_util": 0.0,
                    "actual_budget_util_total": 0.0,
                    "actual_budget_util_hi": 0.0,
                    "actual_budget_util_lo": 0.0,
                    "budget_util_error": 0.0,
                    "static_increase_reserve_count": 0,
                    "static_hi_increase_reserve_count": 0,
                    "static_decrease_reserve_count": 0,
                }
            )
            continue
        static_counts = _compute_static_reserve_counts(
            ordered_tasks,
            budget_increase_ratio=cfg.budget_increase_ratio,
            budget_decrease_ratio=cfg.budget_decrease_ratio,
            budget_floor_ratio=cfg.budget_floor_ratio,
        )
        base_row = _build_candidate_base_row(
            candidate_seed=candidate_seed,
            ordered_tasks=ordered_tasks,
            static_counts=static_counts,
            metadata=candidate_metadata,
        )
        initial_budget_map = {task.name: int(task.c_lo) for task in ordered_tasks}
        relaxation_metadata: dict[str, float | bool] = {
            "relaxation_enabled": False,
            "relaxation_success": False,
            "relaxation_rounds": 0.0,
            "valid_increase_before_relax": 0.0,
            "valid_increase_after_relax": 0.0,
            "valid_decrease_before_relax": 0.0,
            "valid_decrease_after_relax": 0.0,
            "balance_before_relax": 0.0,
            "balance_after_relax": 0.0,
            "relaxed_task_count": 0.0,
            "relaxed_hi_task_count": 0.0,
            "relaxed_lo_task_count": 0.0,
            "budget_util_before_relax": 0.0,
            "budget_util_after_relax": 0.0,
        }

        if (
            static_counts["static_increase_reserve_count"] < cfg.learnable_min_static_increase_reserve
            or static_counts["static_hi_increase_reserve_count"] < cfg.learnable_min_static_hi_increase_reserve
            or static_counts["static_decrease_reserve_count"] < cfg.learnable_min_static_decrease_reserve
        ):
            rejected_rows.append(
                {
                    "accepted": False,
                    "reject_reason": "reject_static_increase_reserve",
                    "error_message": "",
                    "taskset_id": -1,
                    **base_row,
                    **relaxation_metadata,
                }
            )
            continue

        # 在 fast diagnostic 前执行重写后任务集的设计时可调度性预检查。
        sched_result = evaluate_taskset(
            list(ordered_tasks),
            method="amc_rtb",
            priority_policy="opa",
        )
        if not sched_result.schedulable:
            rejected_rows.append(
                {
                    "accepted": False,
                    "reject_reason": "reject_unschedulable_amc_rtb",
                    "error_message": sched_result.details,
                    "schedulability_method": "amc_rtb",
                    "priority_policy": "opa",
                    "taskset_id": -1,
                    **base_row,
                    **relaxation_metadata,
                }
            )
            continue

        if cfg.learnable_enable_safety_relaxation:
            relax_floor_ratio = (
                cfg.learnable_relax_min_budget_floor_ratio
                if cfg.learnable_relax_min_budget_floor_ratio is not None
                else cfg.budget_floor_ratio
            )
            ordered_tasks, relaxation_metadata = relax_budgets_for_safety_valid_increase(
                ordered_tasks=ordered_tasks,
                scenario=candidate_scenario,
                cfg=cfg,
                feature_config=feature_config,
                initial_budgets=initial_budget_map,
                target_valid_increase=cfg.learnable_relax_target_valid_increase,
                max_rounds=cfg.learnable_relax_max_rounds,
                step_ratio=cfg.learnable_relax_step_ratio,
                budget_floor_ratio=relax_floor_ratio,
            )
            sched_result_after_relax = evaluate_taskset(
                list(ordered_tasks),
                method="amc_rtb",
                priority_policy="opa",
            )
            if not sched_result_after_relax.schedulable:
                rejected_rows.append(
                    {
                        "accepted": False,
                        "reject_reason": "reject_unschedulable_after_safety_relaxation",
                        "error_message": sched_result_after_relax.details,
                        "taskset_id": -1,
                        **base_row,
                        **relaxation_metadata,
                    }
                )
                continue

        try:
            fast_rows: list[dict[str, float]] = []
            for idx in range(cfg.learnable_fast_eval_seeds):
                eval_seed = candidate_seed * 1000 + idx
                if cfg.learnable_generation_strategy == "two_stage_from_paper_exact":
                    # two-stage 下场景来自 paper_exact base bundle（taskset 固定、scenario 随 eval_seed 变化）。
                    scenario_bundle = resolve_experiment_bundle(
                        build_automotive_experiment_config(
                            num_runnables=cfg.automotive_num_runnables,
                            mode="paper_exact",
                            require_schedulable=True,
                            fixed_taskset_seed=candidate_seed,
                            budget_floor_ratio=cfg.budget_floor_ratio,
                        ),
                        eval_seed,
                    )
                    scenario = scenario_bundle.scenario
                else:
                    scenario = candidate_scenario
                fast_rows.append(
                    _diagnose_one_seed(
                        ordered_tasks=ordered_tasks,
                        scenario=scenario,
                        eval_seed=eval_seed,
                        runtime_end_time=cfg.learnable_fast_end_time,
                        cfg=cfg,
                        feature_config=feature_config,
                    )
                )
        except ValueError as exc:
            message = str(exc)
            if "R_LO 不可解" in message or "无法构造运行时安全检查器" in message:
                reason = "reject_design_r_lo_unsolved"
            elif "schedul" in message.lower():
                reason = "reject_schedulability_error"
            else:
                reason = "reject_fast_runtime_value_error"
            rejected_rows.append(
                {
                    "accepted": False,
                    "reject_reason": reason,
                    "error_message": f"{type(exc).__name__}: {message}",
                    "taskset_id": -1,
                    **base_row,
                    **relaxation_metadata,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            rejected_rows.append(
                {
                    "accepted": False,
                    "reject_reason": "reject_fast_runtime_error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "taskset_id": -1,
                    **base_row,
                    **relaxation_metadata,
                }
            )
            continue

        fast_mode_changes_mean = mean(row["mode_changes"] for row in fast_rows)
        fast_lo_cancellations_mean = mean(row["lo_cancellations"] for row in fast_rows)
        fast_total_events_mean = mean(row["total_events"] for row in fast_rows)
        fast_deadline_misses_sum = sum(row["deadline_misses"] for row in fast_rows)
        fast_valid_increase_count_mean = mean(row["valid_increase_count"] for row in fast_rows)
        fast_valid_decrease_count_mean = mean(row["valid_decrease_count"] for row in fast_rows)
        fast_valid_noop_count_mean = mean(row["valid_noop_count"] for row in fast_rows)
        fast_valid_increase_count_no_safety_mean = mean(row["valid_increase_count_no_safety"] for row in fast_rows)
        fast_valid_decrease_count_no_safety_mean = mean(row["valid_decrease_count_no_safety"] for row in fast_rows)
        fast_mask_reject_incremental_constraint_violation_mean = mean(
            row["mask_reject_incremental_constraint_violation"] for row in fast_rows
        )
        fast_mask_reject_no_effective_budget_change_mean = mean(
            row["mask_reject_no_effective_budget_change"] for row in fast_rows
        )
        fast_mask_reject_budget_floor_violation_mean = mean(
            row["mask_reject_budget_floor_violation"] for row in fast_rows
        )
        fast_mask_reject_budget_upper_bound_violation_mean = mean(
            row["mask_reject_budget_upper_bound_violation"] for row in fast_rows
        )
        fast_mask_reject_decrease_hi_forbidden_mean = mean(
            row["mask_reject_decrease_hi_forbidden"] for row in fast_rows
        )
        fast_balance = fast_valid_increase_count_mean / max(1.0, fast_valid_decrease_count_mean)

        fast_recommended_for_dqn = (
            fast_deadline_misses_sum == 0.0
            and cfg.learnable_fast_event_min <= fast_total_events_mean <= cfg.learnable_fast_event_max
            and fast_valid_increase_count_mean >= cfg.learnable_fast_min_valid_increase
            and fast_valid_decrease_count_mean >= cfg.learnable_fast_min_valid_decrease
            and fast_balance >= cfg.learnable_fast_min_balance
        )
        fast_not_recommended_reason = ""
        if not fast_recommended_for_dqn:
            reasons: list[str] = []
            if fast_deadline_misses_sum > 0.0:
                reasons.append("reject_deadline_miss_count")
            if fast_total_events_mean < cfg.learnable_fast_event_min or fast_total_events_mean > cfg.learnable_fast_event_max:
                reasons.append("reject_fast_event_count")
            if (
                fast_valid_increase_count_mean < cfg.learnable_fast_min_valid_increase
                or fast_valid_decrease_count_mean < cfg.learnable_fast_min_valid_decrease
                or fast_balance < cfg.learnable_fast_min_balance
            ):
                reasons.append("reject_fast_headroom")
            fast_not_recommended_reason = ";".join(reasons)

        base_metadata = {
            **base_row,
            "automotive_mode": "paper_learnable_headroom",
            "generation_strategy": cfg.learnable_generation_strategy,
            "fast_baseline_mode_changes_mean": fast_mode_changes_mean,
            "fast_baseline_lo_cancellations_mean": fast_lo_cancellations_mean,
            "fast_baseline_total_events_mean": fast_total_events_mean,
            "fast_deadline_misses_sum": fast_deadline_misses_sum,
            "fast_valid_increase_count_mean": fast_valid_increase_count_mean,
            "fast_valid_decrease_count_mean": fast_valid_decrease_count_mean,
            "fast_valid_noop_count_mean": fast_valid_noop_count_mean,
            "fast_valid_increase_count_no_safety_mean": fast_valid_increase_count_no_safety_mean,
            "fast_valid_decrease_count_no_safety_mean": fast_valid_decrease_count_no_safety_mean,
            "fast_mask_reject_incremental_constraint_violation_mean": (
                fast_mask_reject_incremental_constraint_violation_mean
            ),
            "fast_mask_reject_no_effective_budget_change_mean": fast_mask_reject_no_effective_budget_change_mean,
            "fast_mask_reject_budget_floor_violation_mean": fast_mask_reject_budget_floor_violation_mean,
            "fast_mask_reject_budget_upper_bound_violation_mean": (
                fast_mask_reject_budget_upper_bound_violation_mean
            ),
            "fast_mask_reject_decrease_hi_forbidden_mean": fast_mask_reject_decrease_hi_forbidden_mean,
            "fast_increase_decrease_balance": fast_balance,
            "recommended_fast": fast_recommended_for_dqn,
            "fast_not_recommended_reason": fast_not_recommended_reason,
            "hi_budget_rho_min": cfg.learnable_hi_budget_rho_min,
            "hi_budget_rho_max": cfg.learnable_hi_budget_rho_max,
            "lo_budget_rho_min": cfg.learnable_lo_budget_rho_min,
            "lo_budget_rho_max": cfg.learnable_lo_budget_rho_max,
            "budget_floor_ratio": cfg.budget_floor_ratio,
            "reward_mode": cfg.reward_mode,
            "action_space": cfg.action_space,
            "budget_increase_ratio": cfg.budget_increase_ratio,
            "budget_decrease_ratio": cfg.budget_decrease_ratio,
            "include_explicit_noop": cfg.include_explicit_noop,
            "require_schedulable": cfg.require_schedulable,
            **relaxation_metadata,
        }

        if fast_recommended_for_dqn:
            accepted_rows.append({"taskset_id": len(accepted_rows), "accepted": True, "reject_reason": "", "error_message": "", **base_metadata})
            if len(accepted_rows) >= cfg.num_tasksets:
                break
        else:
            rejected_rows.append(
                {
                    "taskset_id": -1,
                    "accepted": False,
                    "reject_reason": fast_not_recommended_reason if fast_not_recommended_reason else "reject_fast_headroom",
                    "error_message": "",
                    **base_metadata,
                }
            )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_rejections.parent.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "taskset_id",
        "candidate_seed",
        "accepted",
        "reject_reason",
        "error_message",
        "automotive_mode",
        "num_tasks",
        "num_hi_tasks",
        "num_lo_tasks",
        "target_budget_util",
        "actual_budget_util_total",
        "actual_budget_util_hi",
        "actual_budget_util_lo",
        "budget_util_error",
        "hi_budget_rho_min",
        "hi_budget_rho_max",
        "lo_budget_rho_min",
        "lo_budget_rho_max",
        "static_increase_reserve_count",
        "static_hi_increase_reserve_count",
        "static_decrease_reserve_count",
        "fast_baseline_mode_changes_mean",
        "fast_baseline_lo_cancellations_mean",
        "fast_baseline_total_events_mean",
        "fast_deadline_misses_sum",
        "fast_valid_increase_count_mean",
        "fast_valid_decrease_count_mean",
        "fast_valid_noop_count_mean",
        "fast_valid_increase_count_no_safety_mean",
        "fast_valid_decrease_count_no_safety_mean",
        "fast_mask_reject_incremental_constraint_violation_mean",
        "fast_mask_reject_no_effective_budget_change_mean",
        "fast_mask_reject_budget_floor_violation_mean",
        "fast_mask_reject_budget_upper_bound_violation_mean",
        "fast_mask_reject_decrease_hi_forbidden_mean",
        "fast_increase_decrease_balance",
        "recommended_fast",
        "fast_not_recommended_reason",
        "budget_floor_ratio",
        "reward_mode",
        "action_space",
        "budget_increase_ratio",
        "budget_decrease_ratio",
        "include_explicit_noop",
        "require_schedulable",
        "relaxation_enabled",
        "relaxation_success",
        "relaxation_rounds",
        "valid_increase_before_relax",
        "valid_increase_after_relax",
        "valid_decrease_before_relax",
        "valid_decrease_after_relax",
        "balance_before_relax",
        "balance_after_relax",
        "relaxed_task_count",
        "relaxed_hi_task_count",
        "relaxed_lo_task_count",
        "budget_util_before_relax",
        "budget_util_after_relax",
    ]
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted_rows)

    reject_fields = [
        "taskset_id",
        "candidate_seed",
        "accepted",
        "reject_reason",
        "error_message",
        "num_tasks",
        "num_hi_tasks",
        "num_lo_tasks",
        "target_budget_util",
        "actual_budget_util_total",
        "actual_budget_util_hi",
        "actual_budget_util_lo",
        "budget_util_error",
        "static_increase_reserve_count",
        "static_hi_increase_reserve_count",
        "static_decrease_reserve_count",
        "fast_baseline_total_events_mean",
        "fast_valid_increase_count_mean",
        "fast_valid_decrease_count_mean",
        "fast_valid_noop_count_mean",
        "fast_valid_increase_count_no_safety_mean",
        "fast_valid_decrease_count_no_safety_mean",
        "fast_mask_reject_incremental_constraint_violation_mean",
        "fast_mask_reject_no_effective_budget_change_mean",
        "fast_mask_reject_budget_floor_violation_mean",
        "fast_mask_reject_budget_upper_bound_violation_mean",
        "fast_mask_reject_decrease_hi_forbidden_mean",
        "fast_increase_decrease_balance",
        "recommended_fast",
        "fast_not_recommended_reason",
        "relaxation_enabled",
        "relaxation_success",
        "relaxation_rounds",
        "valid_increase_before_relax",
        "valid_increase_after_relax",
        "valid_decrease_before_relax",
        "valid_decrease_after_relax",
        "balance_before_relax",
        "balance_after_relax",
        "relaxed_task_count",
        "relaxed_hi_task_count",
        "relaxed_lo_task_count",
        "budget_util_before_relax",
        "budget_util_after_relax",
    ]
    with args.output_rejections.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reject_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejected_rows)

    print(f"accepted tasksets: {len(accepted_rows)}")
    print(f"rejected candidates: {len(rejected_rows)}")


if __name__ == "__main__":
    main()
