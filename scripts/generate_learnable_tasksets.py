"""生成 `paper_learnable_headroom` taskset 清单。

脚本职责：
1. 基于 automotive `paper_learnable_headroom` 模式按 candidate seed 生成候选 taskset；
2. 先做静态 reserve 检查，再做 fast 诊断；
3. 输出 accepted manifest 与 rejected 明细，便于后续复现与分析。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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
    # fast baseline 的 LO cancellations 最小阈值：
    # 若均值低于该阈值，说明运行过程中几乎没有 LO cancellation 信号，
    # 这种样本对后续“预算调整影响 runtime 行为”的学习价值较低，应直接拒绝。
    learnable_fast_min_lo_cancellations: float
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
    enable_ranked_pair_diagnostic: bool
    ranked_pair_min_valid_count: float
    ranked_pair_top_k_risk: int
    ranked_pair_top_k_surplus: int
    ranked_pair_decrease_mode: str
    ranked_pair_include_single_controls: bool
    enable_constraint_guided_pair_diagnostic: bool
    constraint_guided_pair_min_valid_count: float
    constraint_guided_pair_top_k_risk: int
    constraint_guided_pair_top_k_decrease: int
    constraint_guided_pair_prefer_lo: bool
    learnable_selection_target: str


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


def _extract_task_rank_features(env: AmcBudgetEnv, observation) -> list[dict[str, int | float | str | bool]]:
    """提取 ranked-pair 诊断所需的逐任务特征。"""

    if env.feature_config.observation_mode != "v11_full_10d":
        raise ValueError("ranked_pair diagnostic 仅支持 v11_full_10d observation_mode")

    task_features: list[dict[str, int | float | str | bool]] = []
    vector = observation.state_vector
    for idx, task in enumerate(env.ordered_tasks):
        base = idx * 10
        task_features.append(
            {
                "index": idx,
                "name": task.name,
                "is_hi": task.criticality is Criticality.HI,
                "is_lo": task.criticality is not Criticality.HI,
                "risk": float(vector[base + 5]),
                "surplus": float(vector[base + 6]),
                "period": float(task.period),
                "priority_rank": idx,
            }
        )
    return task_features


def _sanitize_ranked_pair_candidate(candidate: dict[str, object]) -> dict[str, object] | None:
    """按计划规则清洗 ranked-pair 候选。"""

    inc = list(dict.fromkeys(int(idx) for idx in candidate["increase_indices"]))  # type: ignore[index]
    dec = list(dict.fromkeys(int(idx) for idx in candidate["decrease_indices"]))  # type: ignore[index]
    dec = [idx for idx in dec if idx not in inc]
    if not inc or not dec:
        return None
    return {"name": str(candidate["name"]), "increase_indices": inc, "decrease_indices": dec}


def _build_ranked_pair_candidates(
    *,
    env: AmcBudgetEnv,
    observation,
    top_k_risk: int,
    top_k_surplus: int,
    decrease_mode: str,
    include_single_controls: bool,
) -> list[dict[str, object]]:
    """构建 ranked-pair 候选动作集合。"""

    features = _extract_task_rank_features(env, observation)
    risk_order = sorted(
        features,
        key=lambda f: (float(f["risk"]), 1 if bool(f["is_hi"]) else 0, -int(f["priority_rank"])),
        reverse=True,
    )
    surplus_order = sorted(
        features,
        key=lambda f: (float(f["surplus"]), 1 if bool(f["is_lo"]) else 0, float(f["period"]), -float(f["risk"])),
        reverse=True,
    )
    hi_risk_order = [f for f in risk_order if bool(f["is_hi"])]
    lo_surplus_order = [f for f in surplus_order if bool(f["is_lo"])]

    risk_top = risk_order[: max(1, top_k_risk)]
    surplus_top = surplus_order[: max(1, top_k_surplus)]
    hi_risk_top = hi_risk_order[: max(1, top_k_risk)]
    lo_surplus_top = lo_surplus_order[: max(1, top_k_surplus)]

    def _top_surplus_indices(pool: list[dict[str, int | float | str | bool]], count: int) -> list[int]:
        return [int(item["index"]) for item in pool[:count]]

    dec_count = 1 if decrease_mode == "top1_surplus" else 2
    if decrease_mode == "topk_surplus":
        dec_count = max(1, top_k_surplus)

    candidates_raw: list[dict[str, object]] = []
    if risk_top and surplus_top:
        candidates_raw.append(
            {
                "name": "inc_toprisk_dec_topsurplus",
                "increase_indices": [int(risk_top[0]["index"])],
                "decrease_indices": _top_surplus_indices(surplus_top, 1),
            }
        )
        candidates_raw.append(
            {
                "name": "inc_toprisk_dec_top2surplus",
                "increase_indices": [int(risk_top[0]["index"])],
                "decrease_indices": _top_surplus_indices(surplus_top, dec_count),
            }
        )
    if hi_risk_top and lo_surplus_top:
        candidates_raw.append(
            {
                "name": "inc_tophirisk_dec_toplosurplus",
                "increase_indices": [int(hi_risk_top[0]["index"])],
                "decrease_indices": _top_surplus_indices(lo_surplus_top, 1),
            }
        )
        candidates_raw.append(
            {
                "name": "inc_tophirisk_dec_top2losurplus",
                "increase_indices": [int(hi_risk_top[0]["index"])],
                "decrease_indices": _top_surplus_indices(lo_surplus_top, dec_count),
            }
        )
    if len(risk_top) >= 2 and surplus_top:
        candidates_raw.append(
            {
                "name": "inc_top2risk_dec_top2surplus",
                "increase_indices": [int(risk_top[0]["index"]), int(risk_top[1]["index"])],
                "decrease_indices": _top_surplus_indices(surplus_top, dec_count),
            }
        )

    if include_single_controls and risk_top and surplus_top:
        candidates_raw.append(
            {"name": "control_inc_toprisk", "increase_indices": [int(risk_top[0]["index"])], "decrease_indices": []}
        )
        candidates_raw.append(
            {"name": "control_dec_topsurplus", "increase_indices": [], "decrease_indices": [int(surplus_top[0]["index"])]}
        )

    dedup: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[str, object]] = {}
    for raw in candidates_raw:
        cleaned = _sanitize_ranked_pair_candidate(raw)
        if cleaned is None:
            continue
        key = (
            str(cleaned["name"]),
            tuple(cleaned["increase_indices"]),  # type: ignore[arg-type]
            tuple(cleaned["decrease_indices"]),  # type: ignore[arg-type]
        )
        dedup[key] = cleaned
    return list(dedup.values())


def _build_candidate_budget_update(
    *,
    env: AmcBudgetEnv,
    candidate: dict[str, object],
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
) -> dict[str, int]:
    """把 ranked-pair 候选转换为候选预算向量。"""

    if env._engine is None:  # noqa: SLF001
        raise RuntimeError("环境尚未 reset")
    current = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001
    new_budgets = dict(current)
    for idx in candidate["increase_indices"]:  # type: ignore[index]
        task = env.ordered_tasks[int(idx)]
        name = task.name
        old = int(new_budgets[name])
        value = int(round(old * (1.0 + budget_increase_ratio)))
        upper = int(task.c_hi) if task.criticality is Criticality.HI else int(task.deadline)
        new_budgets[name] = max(1, min(value, upper))
    for idx in candidate["decrease_indices"]:  # type: ignore[index]
        task = env.ordered_tasks[int(idx)]
        name = task.name
        old = int(new_budgets[name])
        value = int(round(old * (1.0 - budget_decrease_ratio)))
        new_budgets[name] = max(1, value)
    return new_budgets


def _check_candidate_budget_update_no_safety(env: AmcBudgetEnv, new_budgets: dict[str, int]) -> tuple[bool, str]:
    """仅做 no-safety 约束检查（不检查 incremental safety）。"""

    original = env.check_safety
    env.check_safety = False
    try:
        return env.check_candidate_budget_update(new_budgets=new_budgets)
    finally:
        env.check_safety = original


def _diagnose_ranked_pair_candidates(env: AmcBudgetEnv, observation, cfg: LearnableGenerateConfig) -> dict[str, float]:
    """执行 ranked-pair 候选诊断并返回统计字段。"""

    candidates = _build_ranked_pair_candidates(
        env=env,
        observation=observation,
        top_k_risk=cfg.ranked_pair_top_k_risk,
        top_k_surplus=cfg.ranked_pair_top_k_surplus,
        decrease_mode=cfg.ranked_pair_decrease_mode,
        include_single_controls=cfg.ranked_pair_include_single_controls,
    )
    valid_count = 0
    valid_no_safety_count = 0
    reject_counts: Counter[str] = Counter()
    reject_counts_no_safety: Counter[str] = Counter()
    for candidate in candidates:
        new_budgets = _build_candidate_budget_update(
            env=env,
            candidate=candidate,
            budget_increase_ratio=cfg.budget_increase_ratio,
            budget_decrease_ratio=cfg.budget_decrease_ratio,
        )
        ok_no_safety, reason_no_safety = _check_candidate_budget_update_no_safety(env, new_budgets)
        if ok_no_safety:
            valid_no_safety_count += 1
        else:
            reject_counts_no_safety[reason_no_safety] += 1
        ok, reason = env.check_candidate_budget_update(new_budgets=new_budgets)
        if ok:
            valid_count += 1
        else:
            reject_counts[reason] += 1
            # 当统一聚合原因为 incremental_constraint_violation 时，
            # 额外读取 safety checker 的原始 reason，做更细粒度统计（HI/LO 约束分解）。
            if reason == "incremental_constraint_violation":
                fine_reason = env._ensure_checker().validate_candidate(new_budgets).reason  # noqa: SLF001
                if isinstance(fine_reason, str):
                    if fine_reason.startswith("hi_lo_mode_violation"):
                        reject_counts["hi_lo_mode_violation"] += 1
                    elif fine_reason.startswith("hi_mode_switch_violation"):
                        reject_counts["hi_mode_switch_violation"] += 1
                    elif fine_reason.startswith("lo_mode_violation"):
                        reject_counts["lo_mode_violation"] += 1
                    else:
                        reject_counts["unknown"] += 1
    return {
        "ranked_pair_candidate_count": float(len(candidates)),
        "valid_ranked_pair_count": float(valid_count),
        "valid_ranked_pair_count_no_safety": float(valid_no_safety_count),
        "ranked_pair_reject_incremental_constraint_violation": float(
            reject_counts.get("incremental_constraint_violation", 0)
        ),
        "ranked_pair_reject_budget_floor_violation": float(reject_counts.get("budget_floor_violation", 0)),
        "ranked_pair_reject_budget_upper_bound_violation": float(
            reject_counts.get("budget_upper_bound_violation", 0)
        ),
        "ranked_pair_reject_no_effective_budget_change": float(reject_counts.get("no_effective_budget_change", 0)),
        "ranked_pair_reject_decrease_hi_forbidden": float(reject_counts.get("decrease_hi_forbidden", 0)),
        "ranked_pair_reject_unknown": float(reject_counts.get("unknown", 0)),
        "ranked_pair_reject_hi_lo_mode_violation": float(reject_counts.get("hi_lo_mode_violation", 0)),
        "ranked_pair_reject_hi_mode_switch_violation": float(reject_counts.get("hi_mode_switch_violation", 0)),
        "ranked_pair_reject_lo_mode_violation": float(reject_counts.get("lo_mode_violation", 0)),
        "ranked_pair_reject_incremental_constraint_violation_no_safety": float(
            reject_counts_no_safety.get("incremental_constraint_violation", 0)
        ),
        "ranked_pair_reject_unknown_no_safety": float(reject_counts_no_safety.get("unknown", 0)),
    }


def _build_constraint_guided_increase_candidates(
    *,
    env: AmcBudgetEnv,
    observation,
    top_k_risk: int,
) -> list[int]:
    """按风险排序构造 increase 目标候选（包含 HI 优先子集）。"""

    features = _extract_task_rank_features(env, observation)
    risk_order = sorted(
        features,
        key=lambda f: (float(f["risk"]), 1 if bool(f["is_hi"]) else 0, -int(f["priority_rank"])),
        reverse=True,
    )
    hi_risk_order = [f for f in risk_order if bool(f["is_hi"])]
    merged = [int(item["index"]) for item in risk_order[: max(1, top_k_risk)]]
    merged.extend(int(item["index"]) for item in hi_risk_order[: max(1, top_k_risk)])
    # 去重且保持先后顺序。
    return list(dict.fromkeys(merged))


def _select_constraint_guided_decrease_targets(
    *,
    env: AmcBudgetEnv,
    diagnosis,
    increase_indices: set[int],
    budgets: dict[str, int],
    budget_floor_ratio: float,
    top_k: int,
    prefer_lo: bool,
) -> list[int]:
    """根据 violated row 系数挑选最能缓解约束违反的 decrease 目标。"""

    if diagnosis.violated_row_index is None:
        return []
    candidates: list[tuple[float, int]] = []
    for idx, task in enumerate(env.ordered_tasks):
        if idx in increase_indices:
            continue
        current = int(budgets[task.name])
        floor = max(1, int(round(float(task.c_lo) * budget_floor_ratio)))
        possible_decrease = current - floor
        if possible_decrease <= 0:
            continue
        coeff = max(0.0, float(diagnosis.row_coefficients[idx]))
        if coeff <= 0.0:
            continue
        score = coeff * float(possible_decrease)
        if prefer_lo and task.criticality is Criticality.LO:
            score *= 1.25
        candidates.append((score, idx))
    candidates.sort(reverse=True)
    return [idx for _, idx in candidates[: max(1, top_k)]]


def _diagnose_constraint_guided_pair_candidates(env: AmcBudgetEnv, observation, cfg: LearnableGenerateConfig) -> dict[str, float]:
    """执行 constraint-guided pair 诊断并返回统计字段。"""

    if env._engine is None:  # noqa: SLF001
        raise RuntimeError("环境尚未 reset")
    current_budgets = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001
    increase_candidates = _build_constraint_guided_increase_candidates(
        env=env,
        observation=observation,
        top_k_risk=cfg.constraint_guided_pair_top_k_risk,
    )
    valid_count = 0
    valid_no_safety_count = 0
    reject_counts: Counter[str] = Counter()
    candidate_count = 0

    for inc_idx in increase_candidates:
        task = env.ordered_tasks[int(inc_idx)]
        inc_name = task.name
        single_budgets = dict(current_budgets)
        old_inc = int(single_budgets[inc_name])
        inc_value = int(round(float(old_inc) * (1.0 + cfg.budget_increase_ratio)))
        upper = int(task.c_hi) if task.criticality is Criticality.HI else int(task.deadline)
        single_budgets[inc_name] = max(1, min(inc_value, upper))

        diagnosis = env.diagnose_candidate_budget_update(new_budgets=single_budgets)
        if diagnosis.accepted:
            pair_budgets = single_budgets
        else:
            dec_indices = _select_constraint_guided_decrease_targets(
                env=env,
                diagnosis=diagnosis,
                increase_indices={int(inc_idx)},
                budgets=current_budgets,
                budget_floor_ratio=cfg.budget_floor_ratio,
                top_k=cfg.constraint_guided_pair_top_k_decrease,
                prefer_lo=cfg.constraint_guided_pair_prefer_lo,
            )
            pair_budgets = dict(single_budgets)
            for dec_idx in dec_indices:
                dec_task = env.ordered_tasks[int(dec_idx)]
                dec_name = dec_task.name
                old_dec = int(pair_budgets[dec_name])
                dec_value = int(round(float(old_dec) * (1.0 - cfg.budget_decrease_ratio)))
                pair_budgets[dec_name] = max(1, dec_value)
        candidate_count += 1
        ok_no_safety, _reason_no_safety = _check_candidate_budget_update_no_safety(env, pair_budgets)
        if ok_no_safety:
            valid_no_safety_count += 1
        ok, _reason = env.check_candidate_budget_update(new_budgets=pair_budgets)
        pair_diagnosis = env.diagnose_candidate_budget_update(new_budgets=pair_budgets)
        if ok:
            valid_count += 1
        else:
            normalized = pair_diagnosis.normalized_reason
            reject_counts[normalized] += 1
            if normalized == "incremental_constraint_violation":
                fine_reason = env._ensure_checker().validate_candidate(pair_budgets).reason  # noqa: SLF001
                if isinstance(fine_reason, str):
                    if fine_reason.startswith("hi_lo_mode_violation"):
                        reject_counts["hi_lo_mode_violation"] += 1
                    elif fine_reason.startswith("hi_mode_switch_violation"):
                        reject_counts["hi_mode_switch_violation"] += 1
                    elif fine_reason.startswith("lo_mode_violation"):
                        reject_counts["lo_mode_violation"] += 1
                    else:
                        reject_counts["unknown"] += 1

    known = {
        "incremental_constraint_violation",
        "hi_lo_mode_violation",
        "hi_mode_switch_violation",
        "lo_mode_violation",
        "budget_floor_violation",
        "budget_upper_bound_violation",
        "no_effective_budget_change",
    }
    unknown_reject_count = float(
        sum(count for reason, count in reject_counts.items() if reason not in known and not reason.startswith("hi_"))
    )
    return {
        "constraint_guided_pair_candidate_count": float(candidate_count),
        "valid_constraint_guided_pair_count": float(valid_count),
        "valid_constraint_guided_pair_count_no_safety": float(valid_no_safety_count),
        "constraint_guided_pair_reject_incremental_constraint_violation": float(
            reject_counts.get("incremental_constraint_violation", 0)
        ),
        "constraint_guided_pair_reject_hi_lo_mode_violation": float(
            reject_counts.get("hi_lo_mode_violation", 0)
        ),
        "constraint_guided_pair_reject_hi_mode_switch_violation": float(
            reject_counts.get("hi_mode_switch_violation", 0)
        ),
        "constraint_guided_pair_reject_lo_mode_violation": float(
            reject_counts.get("lo_mode_violation", 0)
        ),
        "constraint_guided_pair_reject_budget_floor_violation": float(
            reject_counts.get("budget_floor_violation", 0)
        ),
        "constraint_guided_pair_reject_budget_upper_bound_violation": float(
            reject_counts.get("budget_upper_bound_violation", 0)
        ),
        "constraint_guided_pair_reject_no_effective_budget_change": float(
            reject_counts.get("no_effective_budget_change", 0)
        ),
        "constraint_guided_pair_reject_unknown": unknown_reject_count,
    }


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
    obs = env.reset(seed=eval_seed)
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

    ranked_pair_stats: dict[str, float] = {}
    if cfg.enable_ranked_pair_diagnostic:
        ranked_pair_stats = _diagnose_ranked_pair_candidates(env, obs, cfg)
    constraint_guided_pair_stats: dict[str, float] = {}
    if cfg.enable_constraint_guided_pair_diagnostic:
        constraint_guided_pair_stats = _diagnose_constraint_guided_pair_candidates(env, obs, cfg)

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
        **ranked_pair_stats,
        **constraint_guided_pair_stats,
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
    parser.add_argument("--learnable-fast-min-lo-cancellations", type=float, default=0.3)
    parser.add_argument("--enable-ranked-pair-diagnostic", action="store_true")
    parser.add_argument("--ranked-pair-min-valid-count", type=float, default=2.0)
    parser.add_argument("--ranked-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--ranked-pair-top-k-surplus", type=int, default=3)
    parser.add_argument(
        "--ranked-pair-decrease-mode",
        type=str,
        default="top2_surplus",
        choices=["top1_surplus", "top2_surplus", "topk_surplus"],
    )
    parser.add_argument("--ranked-pair-include-single-controls", action="store_true")
    parser.add_argument("--enable-constraint-guided-pair-diagnostic", action="store_true")
    parser.add_argument("--constraint-guided-pair-min-valid-count", type=float, default=1.0)
    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=4)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action="store_true")
    parser.add_argument(
        "--learnable-selection-target",
        choices=["single", "ranked_pair", "constraint_guided_pair"],
        default="single",
    )
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
        learnable_fast_min_lo_cancellations=args.learnable_fast_min_lo_cancellations,
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
        enable_ranked_pair_diagnostic=args.enable_ranked_pair_diagnostic,
        ranked_pair_min_valid_count=args.ranked_pair_min_valid_count,
        ranked_pair_top_k_risk=args.ranked_pair_top_k_risk,
        ranked_pair_top_k_surplus=args.ranked_pair_top_k_surplus,
        ranked_pair_decrease_mode=args.ranked_pair_decrease_mode,
        ranked_pair_include_single_controls=args.ranked_pair_include_single_controls,
        enable_constraint_guided_pair_diagnostic=args.enable_constraint_guided_pair_diagnostic,
        constraint_guided_pair_min_valid_count=args.constraint_guided_pair_min_valid_count,
        constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        learnable_selection_target=args.learnable_selection_target,
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
        fast_ranked_pair_candidate_count_mean = mean(
            row.get("ranked_pair_candidate_count", 0.0) for row in fast_rows
        )
        fast_valid_ranked_pair_count_mean = mean(
            row.get("valid_ranked_pair_count", 0.0) for row in fast_rows
        )
        fast_valid_ranked_pair_count_no_safety_mean = mean(
            row.get("valid_ranked_pair_count_no_safety", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_incremental_constraint_violation_mean = mean(
            row.get("ranked_pair_reject_incremental_constraint_violation", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_budget_floor_violation_mean = mean(
            row.get("ranked_pair_reject_budget_floor_violation", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_budget_upper_bound_violation_mean = mean(
            row.get("ranked_pair_reject_budget_upper_bound_violation", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_no_effective_budget_change_mean = mean(
            row.get("ranked_pair_reject_no_effective_budget_change", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_decrease_hi_forbidden_mean = mean(
            row.get("ranked_pair_reject_decrease_hi_forbidden", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_unknown_mean = mean(
            row.get("ranked_pair_reject_unknown", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_unknown_no_safety_mean = mean(
            row.get("ranked_pair_reject_unknown_no_safety", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_hi_lo_mode_violation_mean = mean(
            row.get("ranked_pair_reject_hi_lo_mode_violation", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_hi_mode_switch_violation_mean = mean(
            row.get("ranked_pair_reject_hi_mode_switch_violation", 0.0) for row in fast_rows
        )
        fast_ranked_pair_reject_lo_mode_violation_mean = mean(
            row.get("ranked_pair_reject_lo_mode_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_candidate_count_mean = mean(
            row.get("constraint_guided_pair_candidate_count", 0.0) for row in fast_rows
        )
        fast_valid_constraint_guided_pair_count_mean = mean(
            row.get("valid_constraint_guided_pair_count", 0.0) for row in fast_rows
        )
        fast_valid_constraint_guided_pair_count_no_safety_mean = mean(
            row.get("valid_constraint_guided_pair_count_no_safety", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_incremental_constraint_violation_mean = mean(
            row.get("constraint_guided_pair_reject_incremental_constraint_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean = mean(
            row.get("constraint_guided_pair_reject_hi_lo_mode_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_hi_mode_switch_violation_mean = mean(
            row.get("constraint_guided_pair_reject_hi_mode_switch_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_lo_mode_violation_mean = mean(
            row.get("constraint_guided_pair_reject_lo_mode_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_budget_floor_violation_mean = mean(
            row.get("constraint_guided_pair_reject_budget_floor_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_budget_upper_bound_violation_mean = mean(
            row.get("constraint_guided_pair_reject_budget_upper_bound_violation", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_no_effective_budget_change_mean = mean(
            row.get("constraint_guided_pair_reject_no_effective_budget_change", 0.0) for row in fast_rows
        )
        fast_constraint_guided_pair_reject_unknown_mean = mean(
            row.get("constraint_guided_pair_reject_unknown", 0.0) for row in fast_rows
        )
        fast_balance = fast_valid_increase_count_mean / max(1.0, fast_valid_decrease_count_mean)
        fast_valid_ranked_pair_to_single_increase_ratio = (
            fast_valid_ranked_pair_count_mean / max(1.0, fast_valid_increase_count_mean)
        )

        single_headroom_ok = (
            fast_valid_increase_count_mean >= cfg.learnable_fast_min_valid_increase
            and fast_valid_decrease_count_mean >= cfg.learnable_fast_min_valid_decrease
            and fast_balance >= cfg.learnable_fast_min_balance
        )
        ranked_pair_headroom_ok = (
            cfg.enable_ranked_pair_diagnostic
            and fast_valid_ranked_pair_count_mean >= cfg.ranked_pair_min_valid_count
            and fast_valid_decrease_count_mean >= cfg.learnable_fast_min_valid_decrease
        )
        constraint_guided_pair_headroom_ok = (
            cfg.enable_constraint_guided_pair_diagnostic
            and fast_valid_constraint_guided_pair_count_mean >= cfg.constraint_guided_pair_min_valid_count
            and fast_valid_decrease_count_mean >= cfg.learnable_fast_min_valid_decrease
        )
        if cfg.learnable_selection_target == "ranked_pair":
            headroom_ok = ranked_pair_headroom_ok
        elif cfg.learnable_selection_target == "constraint_guided_pair":
            headroom_ok = constraint_guided_pair_headroom_ok
        else:
            headroom_ok = single_headroom_ok
        fast_recommended_for_dqn = (
            fast_deadline_misses_sum == 0.0
            and cfg.learnable_fast_event_min <= fast_total_events_mean <= cfg.learnable_fast_event_max
            # 新增筛选规则：LO cancellations 太低则不推荐，避免接受几乎无 cancellation 信号的任务集。
            and fast_lo_cancellations_mean >= cfg.learnable_fast_min_lo_cancellations
            and headroom_ok
        )
        recommended_for_ranked_pair_dqn = (
            cfg.enable_ranked_pair_diagnostic
            and fast_deadline_misses_sum == 0.0
            and fast_valid_ranked_pair_count_mean >= cfg.ranked_pair_min_valid_count
            and cfg.learnable_fast_event_min <= fast_total_events_mean <= cfg.learnable_fast_event_max
            and fast_lo_cancellations_mean >= cfg.learnable_fast_min_lo_cancellations
        )
        recommended_for_constraint_guided_pair_dqn = (
            cfg.enable_constraint_guided_pair_diagnostic
            and fast_deadline_misses_sum == 0.0
            and fast_valid_constraint_guided_pair_count_mean >= cfg.constraint_guided_pair_min_valid_count
            and cfg.learnable_fast_event_min <= fast_total_events_mean <= cfg.learnable_fast_event_max
            and fast_lo_cancellations_mean >= cfg.learnable_fast_min_lo_cancellations
        )
        ranked_pair_not_recommended_reason = ""
        if cfg.enable_ranked_pair_diagnostic and not recommended_for_ranked_pair_dqn:
            rp_reasons: list[str] = []
            if fast_deadline_misses_sum > 0.0:
                rp_reasons.append("reject_deadline_miss_count")
            if fast_total_events_mean < cfg.learnable_fast_event_min or fast_total_events_mean > cfg.learnable_fast_event_max:
                rp_reasons.append("reject_fast_event_count")
            if fast_lo_cancellations_mean < cfg.learnable_fast_min_lo_cancellations:
                rp_reasons.append("reject_fast_low_lo_cancellations")
            if fast_valid_ranked_pair_count_mean < cfg.ranked_pair_min_valid_count:
                rp_reasons.append("reject_fast_ranked_pair_headroom")
            ranked_pair_not_recommended_reason = ";".join(rp_reasons)
        constraint_guided_pair_not_recommended_reason = ""
        if cfg.enable_constraint_guided_pair_diagnostic and not recommended_for_constraint_guided_pair_dqn:
            cgp_reasons: list[str] = []
            if fast_valid_constraint_guided_pair_count_mean < cfg.constraint_guided_pair_min_valid_count:
                cgp_reasons.append("low_valid_constraint_guided_pair_headroom")
            if fast_total_events_mean < cfg.learnable_fast_event_min:
                cgp_reasons.append("too_few_fast_events")
            if fast_total_events_mean > cfg.learnable_fast_event_max:
                cgp_reasons.append("too_many_fast_events")
            if fast_lo_cancellations_mean < cfg.learnable_fast_min_lo_cancellations:
                cgp_reasons.append("reject_fast_low_lo_cancellations")
            constraint_guided_pair_not_recommended_reason = ";".join(cgp_reasons)
        fast_not_recommended_reason = ""
        if not fast_recommended_for_dqn:
            reasons: list[str] = []
            if fast_deadline_misses_sum > 0.0:
                reasons.append("reject_deadline_miss_count")
            if fast_total_events_mean < cfg.learnable_fast_event_min or fast_total_events_mean > cfg.learnable_fast_event_max:
                reasons.append("reject_fast_event_count")
            if fast_lo_cancellations_mean < cfg.learnable_fast_min_lo_cancellations:
                reasons.append("reject_fast_low_lo_cancellations")
            if cfg.learnable_selection_target == "ranked_pair":
                if fast_valid_ranked_pair_count_mean < cfg.ranked_pair_min_valid_count:
                    reasons.append("reject_fast_ranked_pair_headroom")
                if fast_valid_decrease_count_mean < cfg.learnable_fast_min_valid_decrease:
                    reasons.append("reject_fast_headroom")
            elif cfg.learnable_selection_target == "constraint_guided_pair":
                if fast_valid_constraint_guided_pair_count_mean < cfg.constraint_guided_pair_min_valid_count:
                    reasons.append("reject_fast_constraint_guided_pair_headroom")
                if fast_valid_decrease_count_mean < cfg.learnable_fast_min_valid_decrease:
                    reasons.append("reject_fast_headroom")
            else:
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
            "fast_ranked_pair_candidate_count_mean": fast_ranked_pair_candidate_count_mean,
            "fast_valid_ranked_pair_count_mean": fast_valid_ranked_pair_count_mean,
            "fast_valid_ranked_pair_count_no_safety_mean": fast_valid_ranked_pair_count_no_safety_mean,
            "fast_valid_ranked_pair_to_single_increase_ratio": fast_valid_ranked_pair_to_single_increase_ratio,
            "fast_ranked_pair_reject_incremental_constraint_violation_mean": (
                fast_ranked_pair_reject_incremental_constraint_violation_mean
            ),
            "fast_ranked_pair_reject_budget_floor_violation_mean": fast_ranked_pair_reject_budget_floor_violation_mean,
            "fast_ranked_pair_reject_budget_upper_bound_violation_mean": (
                fast_ranked_pair_reject_budget_upper_bound_violation_mean
            ),
            "fast_ranked_pair_reject_no_effective_budget_change_mean": (
                fast_ranked_pair_reject_no_effective_budget_change_mean
            ),
            "fast_ranked_pair_reject_decrease_hi_forbidden_mean": fast_ranked_pair_reject_decrease_hi_forbidden_mean,
            "fast_ranked_pair_reject_unknown_mean": fast_ranked_pair_reject_unknown_mean,
            "fast_ranked_pair_reject_unknown_no_safety_mean": fast_ranked_pair_reject_unknown_no_safety_mean,
            "fast_ranked_pair_reject_hi_lo_mode_violation_mean": fast_ranked_pair_reject_hi_lo_mode_violation_mean,
            "fast_ranked_pair_reject_hi_mode_switch_violation_mean": (
                fast_ranked_pair_reject_hi_mode_switch_violation_mean
            ),
            "fast_ranked_pair_reject_lo_mode_violation_mean": fast_ranked_pair_reject_lo_mode_violation_mean,
            "fast_constraint_guided_pair_candidate_count_mean": fast_constraint_guided_pair_candidate_count_mean,
            "fast_valid_constraint_guided_pair_count_mean": fast_valid_constraint_guided_pair_count_mean,
            "fast_valid_constraint_guided_pair_count_no_safety_mean": (
                fast_valid_constraint_guided_pair_count_no_safety_mean
            ),
            "fast_constraint_guided_pair_reject_incremental_constraint_violation_mean": (
                fast_constraint_guided_pair_reject_incremental_constraint_violation_mean
            ),
            "fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean": (
                fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean
            ),
            "fast_constraint_guided_pair_reject_hi_mode_switch_violation_mean": (
                fast_constraint_guided_pair_reject_hi_mode_switch_violation_mean
            ),
            "fast_constraint_guided_pair_reject_lo_mode_violation_mean": (
                fast_constraint_guided_pair_reject_lo_mode_violation_mean
            ),
            "fast_constraint_guided_pair_reject_budget_floor_violation_mean": (
                fast_constraint_guided_pair_reject_budget_floor_violation_mean
            ),
            "fast_constraint_guided_pair_reject_budget_upper_bound_violation_mean": (
                fast_constraint_guided_pair_reject_budget_upper_bound_violation_mean
            ),
            "fast_constraint_guided_pair_reject_no_effective_budget_change_mean": (
                fast_constraint_guided_pair_reject_no_effective_budget_change_mean
            ),
            "fast_constraint_guided_pair_reject_unknown_mean": fast_constraint_guided_pair_reject_unknown_mean,
            "fast_increase_decrease_balance": fast_balance,
            "recommended_fast": fast_recommended_for_dqn,
            "fast_not_recommended_reason": fast_not_recommended_reason,
            "recommended_for_ranked_pair_dqn": recommended_for_ranked_pair_dqn,
            "ranked_pair_not_recommended_reason": ranked_pair_not_recommended_reason,
            "recommended_for_constraint_guided_pair_dqn": recommended_for_constraint_guided_pair_dqn,
            "constraint_guided_pair_not_recommended_reason": constraint_guided_pair_not_recommended_reason,
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
            "learnable_fast_min_lo_cancellations": cfg.learnable_fast_min_lo_cancellations,
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
                    "reject_reason": (
                        fast_not_recommended_reason
                        if fast_not_recommended_reason
                        else (
                            "reject_fast_ranked_pair_headroom"
                            if cfg.learnable_selection_target == "ranked_pair"
                            else (
                                "reject_fast_constraint_guided_pair_headroom"
                                if cfg.learnable_selection_target == "constraint_guided_pair"
                                else "reject_fast_headroom"
                            )
                        )
                    ),
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
        "fast_ranked_pair_candidate_count_mean",
        "fast_valid_ranked_pair_count_mean",
        "fast_valid_ranked_pair_count_no_safety_mean",
        "fast_valid_ranked_pair_to_single_increase_ratio",
        "fast_ranked_pair_reject_incremental_constraint_violation_mean",
        "fast_ranked_pair_reject_budget_floor_violation_mean",
        "fast_ranked_pair_reject_budget_upper_bound_violation_mean",
        "fast_ranked_pair_reject_no_effective_budget_change_mean",
        "fast_ranked_pair_reject_decrease_hi_forbidden_mean",
        "fast_ranked_pair_reject_unknown_mean",
        "fast_ranked_pair_reject_unknown_no_safety_mean",
        "fast_ranked_pair_reject_hi_lo_mode_violation_mean",
        "fast_ranked_pair_reject_hi_mode_switch_violation_mean",
        "fast_ranked_pair_reject_lo_mode_violation_mean",
        "fast_constraint_guided_pair_candidate_count_mean",
        "fast_valid_constraint_guided_pair_count_mean",
        "fast_valid_constraint_guided_pair_count_no_safety_mean",
        "fast_constraint_guided_pair_reject_incremental_constraint_violation_mean",
        "fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean",
        "fast_constraint_guided_pair_reject_hi_mode_switch_violation_mean",
        "fast_constraint_guided_pair_reject_lo_mode_violation_mean",
        "fast_constraint_guided_pair_reject_budget_floor_violation_mean",
        "fast_constraint_guided_pair_reject_budget_upper_bound_violation_mean",
        "fast_constraint_guided_pair_reject_no_effective_budget_change_mean",
        "fast_constraint_guided_pair_reject_unknown_mean",
        "fast_increase_decrease_balance",
        "recommended_fast",
        "fast_not_recommended_reason",
        "recommended_for_ranked_pair_dqn",
        "ranked_pair_not_recommended_reason",
        "recommended_for_constraint_guided_pair_dqn",
        "constraint_guided_pair_not_recommended_reason",
        "budget_floor_ratio",
        "reward_mode",
        "action_space",
        "budget_increase_ratio",
        "budget_decrease_ratio",
        "include_explicit_noop",
        "learnable_fast_min_lo_cancellations",
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
        "fast_ranked_pair_candidate_count_mean",
        "fast_valid_ranked_pair_count_mean",
        "fast_valid_ranked_pair_count_no_safety_mean",
        "fast_valid_ranked_pair_to_single_increase_ratio",
        "fast_ranked_pair_reject_incremental_constraint_violation_mean",
        "fast_ranked_pair_reject_budget_floor_violation_mean",
        "fast_ranked_pair_reject_budget_upper_bound_violation_mean",
        "fast_ranked_pair_reject_no_effective_budget_change_mean",
        "fast_ranked_pair_reject_decrease_hi_forbidden_mean",
        "fast_ranked_pair_reject_unknown_mean",
        "fast_ranked_pair_reject_unknown_no_safety_mean",
        "fast_ranked_pair_reject_hi_lo_mode_violation_mean",
        "fast_ranked_pair_reject_hi_mode_switch_violation_mean",
        "fast_ranked_pair_reject_lo_mode_violation_mean",
        "fast_constraint_guided_pair_candidate_count_mean",
        "fast_valid_constraint_guided_pair_count_mean",
        "fast_valid_constraint_guided_pair_count_no_safety_mean",
        "fast_constraint_guided_pair_reject_incremental_constraint_violation_mean",
        "fast_constraint_guided_pair_reject_hi_lo_mode_violation_mean",
        "fast_constraint_guided_pair_reject_hi_mode_switch_violation_mean",
        "fast_constraint_guided_pair_reject_lo_mode_violation_mean",
        "fast_constraint_guided_pair_reject_budget_floor_violation_mean",
        "fast_constraint_guided_pair_reject_budget_upper_bound_violation_mean",
        "fast_constraint_guided_pair_reject_no_effective_budget_change_mean",
        "fast_constraint_guided_pair_reject_unknown_mean",
        "fast_increase_decrease_balance",
        "recommended_fast",
        "fast_not_recommended_reason",
        "recommended_for_ranked_pair_dqn",
        "ranked_pair_not_recommended_reason",
        "recommended_for_constraint_guided_pair_dqn",
        "constraint_guided_pair_not_recommended_reason",
        "learnable_fast_min_lo_cancellations",
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
