"""constraint-guided transfer / pair 动作空间共享逻辑。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.safety import merge_budget_candidate


@dataclass(frozen=True, slots=True)
class ConstraintGuidedPairConfig:
    """constraint-guided 配置。"""

    top_k_risk: int = 3
    top_k_decrease: int = 5
    prefer_lo: bool = True
    budget_floor_ratio: float = 0.9
    increase_ratio: float = 0.015
    decrease_ratio: float = 0.015
    include_hi_risk_boost: bool = False
    allow_increase_only_when_safe: bool = False


@dataclass(frozen=True, slots=True)
class ConstraintGuidedResolvedAction:
    """constraint-guided 槽位在当前状态下的解析结果。"""

    valid: bool
    reject_reason: str | None
    slot_id: int
    increase_rank: int | None
    increase_idx: int | None
    decrease_indices: tuple[int, ...]
    updates: dict[str, int]
    candidate_budgets: dict[str, int]
    single_increase_was_safe: bool
    safety_checked: bool
    diagnosis_reason: str | None
    violated_row_index: int | None


@dataclass(frozen=True, slots=True)
class ConstraintGuidedTransferCandidate:
    """constraint-guided transfer 候选。

    该结构用于统一 env / scan / generator 的候选语义：
    1 个 increase + 一组 guided decrease（bundled）。
    """

    slot_index: int
    increase_rank: int
    increase_task_idx: int
    decrease_task_indices: tuple[int, ...]
    candidate_budgets: tuple[int, ...] | dict[str, int]
    accepted: bool
    reject_reason: str | None
    single_increase_accepted: bool
    violated_row_index: int | None
    violation_amount: float | None


def extract_task_rank_features_from_v11(
    *,
    ordered_tasks: Sequence[Task],
    observation_state_vector: Sequence[float],
) -> list[dict[str, int | float | bool]]:
    """从 v11_full_10d 观测向量中提取逐任务排序特征。"""

    min_expected = len(ordered_tasks) * 10
    if len(observation_state_vector) < min_expected:
        raise ValueError("extract_task_rank_features_from_v11 仅支持 v11_full_10d 观测向量")

    features: list[dict[str, int | float | bool]] = []
    for idx, task in enumerate(ordered_tasks):
        base = idx * 10
        features.append(
            {
                "index": idx,
                "is_hi": task.criticality is Criticality.HI,
                "is_lo": task.criticality is not Criticality.HI,
                "risk": float(observation_state_vector[base + 5]),
                "surplus": float(observation_state_vector[base + 6]),
                "period": float(task.period),
                "priority_rank": idx,
            }
        )
    return features


def extract_task_rank_features_from_observation(
    *,
    observation_mode: str,
    state_vector: Sequence[float],
    task_count: int,
) -> list[dict[str, int | float | bool]]:
    """从 v11/v12 观测向量中提取逐任务排序特征。

    说明：
    - v11: 每任务 10 维；
    - v12: 每任务 14 维（前 10 维语义与 v11 完全一致，额外读取新增 4 维）。
    """

    if observation_mode == "v11_full_10d":
        per_task_dim = 10
    elif observation_mode == "v12_full_14d":
        per_task_dim = 14
    else:
        raise ValueError(f"不支持的 observation_mode: {observation_mode}")

    min_expected = int(task_count) * per_task_dim
    if len(state_vector) < min_expected:
        raise ValueError("观测向量长度不足，无法解析逐任务特征")

    features: list[dict[str, int | float | bool]] = []
    for idx in range(task_count):
        base = idx * per_task_dim
        item: dict[str, int | float | bool] = {
            "index": idx,
            # 对于解析函数来说，is_hi/is_lo 不从 state 推断，交由调用方结合 task 元信息补充。
            "is_hi": False,
            "is_lo": False,
            "risk": float(state_vector[base + 5]),
            "surplus": float(state_vector[base + 6]),
            "priority_rank": idx,
        }
        if per_task_dim == 14:
            item["positive_budget_drift"] = float(state_vector[base + 10])
            item["negative_budget_drift"] = float(state_vector[base + 11])
            item["task_cancel_ema"] = float(state_vector[base + 12])
            item["safe_inc_possible"] = bool(float(state_vector[base + 13]) >= 0.5)
        features.append(item)
    return features


def build_constraint_guided_increase_candidates(
    *,
    ordered_tasks: Sequence[Task],
    observation_state_vector: Sequence[float],
    top_k_risk: int,
    include_hi_risk_boost: bool = False,
) -> list[int]:
    """按 risk 排序构造 increase 候选任务索引。"""

    task_count = len(ordered_tasks)
    per_task_10_expected = task_count * 10
    per_task_14_expected = task_count * 14
    if len(observation_state_vector) >= per_task_14_expected:
        observation_mode = "v12_full_14d"
    elif len(observation_state_vector) >= per_task_10_expected:
        observation_mode = "v11_full_10d"
    else:
        raise ValueError("观测向量长度不足，无法提取 constraint-guided 风险排序特征")
    features = extract_task_rank_features_from_observation(
        observation_mode=observation_mode,
        state_vector=observation_state_vector,
        task_count=task_count,
    )
    for item in features:
        idx = int(item["index"])
        item["is_hi"] = ordered_tasks[idx].criticality is Criticality.HI
        item["is_lo"] = not bool(item["is_hi"])
    risk_order = sorted(
        features,
        key=lambda f: (float(f["risk"]), 1 if bool(f["is_hi"]) else 0, -int(f["priority_rank"])),
        reverse=True,
    )
    count = max(1, int(top_k_risk))
    if not include_hi_risk_boost:
        return [int(item["index"]) for item in risk_order[:count]]

    hi_risk_order = [f for f in risk_order if bool(f["is_hi"])]
    merged = [int(item["index"]) for item in risk_order[:count]]
    merged.extend(int(item["index"]) for item in hi_risk_order[:count])
    return list(dict.fromkeys(merged))


def select_constraint_guided_decrease_targets(
    *,
    ordered_tasks: Sequence[Task],
    row_coefficients: Sequence[float],
    current_budgets: dict[str, int],
    initial_budgets: dict[str, int],
    increase_indices: set[int],
    budget_floor_ratio: float,
    top_k: int,
    prefer_lo: bool,
) -> list[int]:
    """根据 violated row 约束贡献分数挑选 decrease 目标。"""

    scored: list[tuple[float, int]] = []
    for idx, task in enumerate(ordered_tasks):
        if idx in increase_indices:
            continue
        current = int(current_budgets[task.name])
        floor_budget = max(1, math.ceil(float(initial_budgets[task.name]) * float(budget_floor_ratio)))
        possible_decrease = current - floor_budget
        if possible_decrease <= 0:
            continue
        coeff = max(0.0, float(row_coefficients[idx]))
        if coeff <= 0.0:
            continue
        score = coeff * float(possible_decrease)
        if prefer_lo and task.criticality is Criticality.LO:
            score *= 1.25
        scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: max(1, int(top_k))]]


def apply_single_increase_candidate(
    *,
    task_idx: int,
    budget_state: BudgetState,
    ordered_tasks: Sequence[Task],
    increase_ratio: float,
) -> dict[str, int]:
    """构造单 increase 候选更新。"""

    task = ordered_tasks[task_idx]
    name = task.name
    old_value = int(budget_state.budgets[name])
    new_value = math.ceil(old_value * (1.0 + increase_ratio))
    if task.criticality is Criticality.HI:
        upper_bound = task.c_hi if task.c_hi > 0 else task.deadline
        new_value = min(new_value, upper_bound)
    else:
        new_value = min(new_value, task.deadline)
    return {name: max(1, int(new_value))}


def apply_pair_candidate(
    *,
    increase_idx: int,
    decrease_idx: int,
    budget_state: BudgetState,
    ordered_tasks: Sequence[Task],
    increase_ratio: float,
    decrease_ratio: float,
) -> dict[str, int]:
    """兼容旧接口：构造 increase + 单 decrease 更新。"""

    updates = apply_single_increase_candidate(
        task_idx=increase_idx,
        budget_state=budget_state,
        ordered_tasks=ordered_tasks,
        increase_ratio=increase_ratio,
    )
    dec_task = ordered_tasks[decrease_idx]
    dec_name = dec_task.name
    old_dec = int(budget_state.budgets[dec_name])
    dec_value = math.floor(old_dec * (1.0 - decrease_ratio))
    updates[dec_name] = max(1, int(dec_value))
    return updates


def enumerate_constraint_guided_transfer_candidates(
    *,
    tasks,
    current_budgets,
    safety_checker,
    runtime_features,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    budget_floor_ratio: float,
    top_k_risk: int,
    top_k_decrease: int,
    prefer_lo: bool,
    include_hi_risk_boost: bool = False,
    validate_safety: bool = True,
) -> tuple[ConstraintGuidedTransferCandidate, ...]:
    """统一枚举 constraint-guided transfer 候选。

    说明：
    - 该函数只实现计划内语义：single increase -> 诊断 violated row -> bundled decrease；
    - 不引入 ranked-pair / surplus-pair / strict pair 的替代逻辑。
    """

    tasks_seq: Sequence[Task] = tuple(tasks)
    current: dict[str, int] = {task.name: int(current_budgets[task.name]) for task in tasks_seq}

    state_vector = getattr(runtime_features, "state_vector", None)
    if state_vector is None and hasattr(runtime_features, "_last_observation"):
        last_obs = getattr(runtime_features, "_last_observation")
        state_vector = None if last_obs is None else getattr(last_obs, "state_vector", None)
    if state_vector is None:
        return tuple()

    increase_candidates = build_constraint_guided_increase_candidates(
        ordered_tasks=tasks_seq,
        observation_state_vector=state_vector,
        top_k_risk=top_k_risk,
        include_hi_risk_boost=include_hi_risk_boost,
    )

    initial_budgets = dict(current)
    if hasattr(runtime_features, "_initial_budgets"):
        initial_budgets = dict(getattr(runtime_features, "_initial_budgets"))

    candidates: list[ConstraintGuidedTransferCandidate] = []
    for increase_rank, increase_idx in enumerate(increase_candidates):
        slot_index = increase_rank
        task = tasks_seq[increase_idx]
        inc_name = task.name

        old_inc = int(current[inc_name])
        inc_value = math.ceil(old_inc * (1.0 + budget_increase_ratio))
        upper = int(task.c_hi) if task.criticality is Criticality.HI and task.c_hi > 0 else int(task.deadline)
        inc_value = max(1, min(inc_value, upper))

        single_budgets = dict(current)
        single_budgets[inc_name] = inc_value

        single_diag = None
        if hasattr(runtime_features, "diagnose_candidate_budget_update"):
            single_diag = runtime_features.diagnose_candidate_budget_update(new_budgets=single_budgets)
        single_ok = bool(getattr(single_diag, "accepted", False))
        violated_row_index = getattr(single_diag, "violated_row_index", None)
        violation_amount = getattr(single_diag, "violation_amount", None)
        row_coefficients = getattr(single_diag, "row_coefficients", tuple(0.0 for _ in tasks_seq))

        dec_indices = select_constraint_guided_decrease_targets(
            ordered_tasks=tasks_seq,
            row_coefficients=row_coefficients,
            current_budgets=current,
            initial_budgets=initial_budgets,
            increase_indices={increase_idx},
            budget_floor_ratio=budget_floor_ratio,
            top_k=top_k_decrease,
            prefer_lo=prefer_lo,
        )
        dec_tuple = tuple(int(i) for i in dec_indices)
        if not dec_tuple:
            candidates.append(
                ConstraintGuidedTransferCandidate(
                    slot_index=slot_index,
                    increase_rank=increase_rank,
                    increase_task_idx=increase_idx,
                    decrease_task_indices=tuple(),
                    candidate_budgets=dict(current),
                    accepted=False,
                    reject_reason="constraint_guided_no_decrease_candidate",
                    single_increase_accepted=single_ok,
                    violated_row_index=violated_row_index,
                    violation_amount=float(violation_amount) if violation_amount is not None else None,
                )
            )
            continue

        updates = {inc_name: inc_value}
        for dec_idx in dec_tuple:
            dec_task = tasks_seq[dec_idx]
            dec_name = dec_task.name
            old_dec = int(current[dec_name])
            floor_budget = max(1, math.ceil(float(initial_budgets[dec_name]) * float(budget_floor_ratio)))
            dec_value = max(floor_budget, math.floor(old_dec * (1.0 - budget_decrease_ratio)))
            updates[dec_name] = max(1, int(dec_value))

        merged = merge_budget_candidate(BudgetState(budgets=current), updates)

        accepted = True
        reject_reason: str | None = None
        if validate_safety and safety_checker is not None:
            report = safety_checker.validate_candidate(merged)
            accepted = bool(report.accepted)
            reject_reason = None if accepted else str(report.reason)

        candidates.append(
            ConstraintGuidedTransferCandidate(
                slot_index=slot_index,
                increase_rank=increase_rank,
                increase_task_idx=increase_idx,
                decrease_task_indices=dec_tuple,
                candidate_budgets=merged,
                accepted=accepted,
                reject_reason=reject_reason,
                single_increase_accepted=single_ok,
                violated_row_index=violated_row_index,
                violation_amount=float(violation_amount) if violation_amount is not None else None,
            )
        )

    return tuple(candidates)
