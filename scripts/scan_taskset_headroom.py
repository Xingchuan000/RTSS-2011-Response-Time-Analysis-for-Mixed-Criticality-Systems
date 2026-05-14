"""阶段 A：扫描不同 fixed_taskset_seed 与 budget_scale 的 baseline 与可调余量指标。"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from amc_py.dqn.experiment import (
    build_automotive_experiment_config,
    build_mc_fairgen_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.constraint_guided_pair import (
    build_constraint_guided_increase_candidates,
    enumerate_constraint_guided_transfer_candidates,
    extract_task_rank_features_from_v11,
    select_constraint_guided_decrease_targets,
)
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from scripts.generate_learnable_tasksets import _rewrite_budgets_for_learnable_headroom


@dataclass(frozen=True)
class _TwoStageRewriteConfig:
    """two-stage 预算重写所需的最小配置子集。

    说明：
    - generate 脚本中的 `_rewrite_budgets_for_learnable_headroom` 只读取这几个字段；
    - 这里用最小配置对象复用同一重写函数，保证 scan 与生成脚本口径一致。
    """

    budget_floor_ratio: float
    learnable_target_budget_util_min: float
    learnable_target_budget_util_max: float
    learnable_hi_budget_rho_min: float
    learnable_hi_budget_rho_max: float
    learnable_lo_budget_rho_min: float
    learnable_lo_budget_rho_max: float


def _parse_bool_like(value: object) -> bool:
    """把 manifest 中的布尔样式文本解析为 bool。"""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _load_manifest_rows(
    manifest_path: str,
    *,
    seed_column: str,
    seed_limit: int | None,
    filter_recommended: bool,
) -> list[dict[str, str]]:
    """从 manifest 读取候选行并按种子去重。

    约束：
    - 按 manifest 原始顺序保留；
    - 仅保留 seed 可解析的行；
    - 使用首出现去重，避免重复扫描同一个 candidate_seed。
    """

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    if seed_column not in rows[0]:
        raise ValueError(
            f"Manifest seed column '{seed_column}' not found. Available columns: {sorted(rows[0].keys())}"
        )

    selected = rows
    if filter_recommended:
        recommendation_columns = [
            "recommended_for_constraint_guided_pair_dqn",
            "recommended_for_ranked_pair_dqn",
            "recommended_fast",
        ]
        existing = [column for column in recommendation_columns if column in rows[0]]
        if not existing:
            raise ValueError(
                "--manifest-filter-recommended was set, but no known recommendation column exists in manifest."
            )
        selected = [row for row in selected if any(_parse_bool_like(row.get(column, "")) for column in existing)]

    deduped: list[dict[str, str]] = []
    seen: set[int] = set()
    for row in selected:
        seed_text = str(row.get(seed_column, "")).strip()
        if not seed_text:
            continue
        seed_value = int(float(seed_text))
        if seed_value in seen:
            continue
        seen.add(seed_value)
        copied = dict(row)
        copied[seed_column] = str(seed_value)
        deduped.append(copied)

    if seed_limit is not None:
        deduped = deduped[:seed_limit]
    if not deduped:
        raise ValueError(f"No manifest rows selected from {manifest_path}")
    return deduped


def _write_selected_manifest_rows(path: str, rows: list[dict[str, str]]) -> None:
    """把本次真正用于扫描的 manifest 子集写盘，便于追溯。"""

    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _warn_or_raise_manifest_parameter_mismatch(
    *,
    args: argparse.Namespace,
    manifest_rows: list[dict[str, str]],
) -> None:
    """校验 CLI 参数与 manifest 内参数的一致性。

    策略：
    - 当 `--manifest-strict-parameter-check` 为真：不一致直接报错；
    - 否则打印 WARNING，继续执行。
    """

    first = manifest_rows[0]
    checks: list[tuple[str, str, float]] = [
        ("learnable_target_budget_util_min", "learnable_target_budget_util_min", float(args.learnable_target_budget_util_min)),
        ("learnable_target_budget_util_max", "learnable_target_budget_util_max", float(args.learnable_target_budget_util_max)),
        ("learnable_hi_budget_rho_min", "learnable_hi_budget_rho_min", float(args.learnable_hi_budget_rho_min)),
        ("learnable_hi_budget_rho_max", "learnable_hi_budget_rho_max", float(args.learnable_hi_budget_rho_max)),
        ("learnable_lo_budget_rho_min", "learnable_lo_budget_rho_min", float(args.learnable_lo_budget_rho_min)),
        ("learnable_lo_budget_rho_max", "learnable_lo_budget_rho_max", float(args.learnable_lo_budget_rho_max)),
        ("budget_increase_ratio", "budget_increase_ratio", float(args.budget_increase_ratio)),
        ("budget_decrease_ratio", "budget_decrease_ratio", float(args.budget_decrease_ratio)),
        ("budget_floor_ratio", "budget_floor_ratio", float(args.budget_floor_ratio)),
        ("constraint_guided_pair_top_k_risk", "constraint_guided_pair_top_k_risk", float(args.constraint_guided_pair_top_k_risk)),
        ("constraint_guided_pair_top_k_decrease", "constraint_guided_pair_top_k_decrease", float(args.constraint_guided_pair_top_k_decrease)),
    ]
    mismatches: list[str] = []
    for manifest_column, display_name, cli_value in checks:
        if manifest_column not in first:
            continue
        manifest_text = str(first.get(manifest_column, "")).strip()
        if not manifest_text:
            continue
        manifest_value = float(manifest_text)
        if abs(manifest_value - cli_value) > 1e-12:
            mismatches.append(
                f"{display_name}: CLI={cli_value} vs manifest={manifest_value}"
            )
    if not mismatches:
        return
    message = "Manifest parameter mismatch detected:\n" + "\n".join(f"- {item}" for item in mismatches)
    if args.manifest_strict_parameter_check:
        raise ValueError(message)
    print(f"WARNING: {message}", flush=True)


def _extract_task_rank_features(env: AmcBudgetEnv, observation) -> list[dict[str, int | float | bool]]:
    """提取 ranked-pair 诊断所需特征。"""

    if env.feature_config.observation_mode != "v11_full_10d":
        raise ValueError("ranked_pair diagnostic 仅支持 v11_full_10d observation_mode")
    return extract_task_rank_features_from_v11(
        ordered_tasks=env.ordered_tasks,
        observation_state_vector=observation.state_vector,
    )


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


def _sanitize_ranked_pair_candidate(candidate: dict[str, object]) -> dict[str, object] | None:
    """按计划规则清洗候选。"""

    inc = list(dict.fromkeys(int(i) for i in candidate["increase_indices"]))  # type: ignore[index]
    dec = list(dict.fromkeys(int(i) for i in candidate["decrease_indices"]))  # type: ignore[index]
    dec = [idx for idx in dec if idx not in inc]
    if not inc or not dec:
        return None
    return {"name": str(candidate["name"]), "increase_indices": inc, "decrease_indices": dec}


def _build_ranked_pair_candidates(env: AmcBudgetEnv, observation, top_k_risk: int, top_k_surplus: int, decrease_mode: str) -> list[dict[str, object]]:
    """构建 ranked-pair 候选集合。"""

    feats = _extract_task_rank_features(env, observation)
    risk_order = sorted(feats, key=lambda f: (float(f["risk"]), 1 if bool(f["is_hi"]) else 0, -int(f["priority_rank"])), reverse=True)
    surplus_order = sorted(
        feats,
        key=lambda f: (float(f["surplus"]), 1 if bool(f["is_lo"]) else 0, float(f["period"]), -float(f["risk"])),
        reverse=True,
    )
    hi_risk_order = [f for f in risk_order if bool(f["is_hi"])]
    lo_surplus_order = [f for f in surplus_order if bool(f["is_lo"])]
    risk_top = risk_order[: max(1, top_k_risk)]
    surplus_top = surplus_order[: max(1, top_k_surplus)]
    hi_top = hi_risk_order[: max(1, top_k_risk)]
    lo_top = lo_surplus_order[: max(1, top_k_surplus)]
    dec_count = 1 if decrease_mode == "top1_surplus" else 2
    if decrease_mode == "topk_surplus":
        dec_count = max(1, top_k_surplus)
    raw: list[dict[str, object]] = []
    if risk_top and surplus_top:
        raw.append({"name": "inc_toprisk_dec_topsurplus", "increase_indices": [int(risk_top[0]["index"])], "decrease_indices": [int(surplus_top[0]["index"])]})
        raw.append({"name": "inc_toprisk_dec_top2surplus", "increase_indices": [int(risk_top[0]["index"])], "decrease_indices": [int(x["index"]) for x in surplus_top[:dec_count]]})
    if hi_top and lo_top:
        raw.append({"name": "inc_tophirisk_dec_toplosurplus", "increase_indices": [int(hi_top[0]["index"])], "decrease_indices": [int(lo_top[0]["index"])]})
        raw.append({"name": "inc_tophirisk_dec_top2losurplus", "increase_indices": [int(hi_top[0]["index"])], "decrease_indices": [int(x["index"]) for x in lo_top[:dec_count]]})
    if len(risk_top) >= 2 and surplus_top:
        raw.append({"name": "inc_top2risk_dec_top2surplus", "increase_indices": [int(risk_top[0]["index"]), int(risk_top[1]["index"])], "decrease_indices": [int(x["index"]) for x in surplus_top[:dec_count]]})
    seen: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[str, object]] = {}
    for item in raw:
        cleaned = _sanitize_ranked_pair_candidate(item)
        if cleaned is None:
            continue
        key = (str(cleaned["name"]), tuple(cleaned["increase_indices"]), tuple(cleaned["decrease_indices"]))  # type: ignore[arg-type]
        seen[key] = cleaned
    return list(seen.values())


def _build_candidate_budget_update(env: AmcBudgetEnv, candidate: dict[str, object], increase_ratio: float, decrease_ratio: float) -> dict[str, int]:
    """构建候选预算向量。"""

    if env._engine is None:  # noqa: SLF001
        raise RuntimeError("环境尚未 reset")
    base = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001
    new_budgets = dict(base)
    for idx in candidate["increase_indices"]:  # type: ignore[index]
        task = env.ordered_tasks[int(idx)]
        name = task.name
        value = int(round(int(new_budgets[name]) * (1.0 + increase_ratio)))
        upper = int(task.c_hi) if task.criticality is Criticality.HI else int(task.deadline)
        new_budgets[name] = max(1, min(value, upper))
    for idx in candidate["decrease_indices"]:  # type: ignore[index]
        task = env.ordered_tasks[int(idx)]
        name = task.name
        value = int(round(int(new_budgets[name]) * (1.0 - decrease_ratio)))
        new_budgets[name] = max(1, value)
    return new_budgets


def _build_constraint_guided_increase_candidates(
    env: AmcBudgetEnv,
    observation,
    top_k_risk: int,
) -> list[int]:
    """构造 constraint-guided 的 increase 候选任务索引。

    规则与生成脚本保持一致：
    1. 先取全任务 top-k risk；
    2. 再拼接 HI 任务 top-k risk；
    3. 最后去重并保持顺序。
    """

    return build_constraint_guided_increase_candidates(
        ordered_tasks=env.ordered_tasks,
        observation_state_vector=observation.state_vector,
        top_k_risk=top_k_risk,
        include_hi_risk_boost=True,
    )


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
    """按 violated row 的约束贡献分数选择 decrease 目标。

    分数定义：
    score = max(0, coeff_j) * possible_decrease_j
    其中 possible_decrease_j = current_budget_j - floor_budget_j。
    """

    if diagnosis.violated_row_index is None:
        return []
    # 与生成脚本共享同一 decrease 选择实现，确保诊断口径一致。
    return select_constraint_guided_decrease_targets(
        ordered_tasks=env.ordered_tasks,
        row_coefficients=diagnosis.row_coefficients,
        current_budgets=budgets,
        initial_budgets=env._initial_budgets,  # noqa: SLF001
        increase_indices=increase_indices,
        budget_floor_ratio=budget_floor_ratio,
        top_k=top_k,
        prefer_lo=prefer_lo,
    )


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
    enable_ranked_pair_diagnostic: bool,
    ranked_pair_top_k_risk: int,
    ranked_pair_top_k_surplus: int,
    ranked_pair_decrease_mode: str,
    enable_constraint_guided_pair_diagnostic: bool,
    constraint_guided_pair_top_k_risk: int,
    constraint_guided_pair_top_k_decrease: int,
    constraint_guided_pair_prefer_lo: bool,
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
    ranked_pair_candidate_values: list[float] = []
    ranked_pair_valid_values: list[float] = []
    ranked_pair_valid_no_safety_values: list[float] = []
    ranked_pair_reject_incremental_values: list[float] = []
    constraint_guided_pair_valid_values: list[float] = []
    constraint_guided_pair_valid_no_safety_values: list[float] = []
    constraint_guided_pair_reject_incremental_values: list[float] = []
    constraint_guided_pair_reject_hi_lo_values: list[float] = []
    constraint_guided_pair_reject_hi_mode_switch_values: list[float] = []
    constraint_guided_pair_reject_lo_mode_values: list[float] = []
    constraint_guided_pair_reject_budget_floor_values: list[float] = []
    constraint_guided_pair_reject_unknown_values: list[float] = []

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

        if enable_ranked_pair_diagnostic:
            candidates = _build_ranked_pair_candidates(
                env,
                obs,
                top_k_risk=ranked_pair_top_k_risk,
                top_k_surplus=ranked_pair_top_k_surplus,
                decrease_mode=ranked_pair_decrease_mode,
            )
            valid_count = 0
            valid_no_safety_count = 0
            reject_counts: Counter[str] = Counter()
            for cand in candidates:
                new_budgets = _build_candidate_budget_update(
                    env,
                    cand,
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                )
                old_flag = env.check_safety
                env.check_safety = False
                ok_no_safety, _ = env.check_candidate_budget_update(new_budgets=new_budgets)
                env.check_safety = old_flag
                ok, reason = env.check_candidate_budget_update(new_budgets=new_budgets)
                if ok:
                    valid_count += 1
                else:
                    reject_counts[reason] += 1
                if ok_no_safety:
                    valid_no_safety_count += 1
            ranked_pair_candidate_values.append(float(len(candidates)))
            ranked_pair_valid_values.append(float(valid_count))
            ranked_pair_valid_no_safety_values.append(float(valid_no_safety_count))
            ranked_pair_reject_incremental_values.append(
                float(reject_counts.get("incremental_constraint_violation", 0))
            )
        if enable_constraint_guided_pair_diagnostic:
            if env._engine is None:  # noqa: SLF001
                raise RuntimeError("环境尚未 reset")
            current_budgets = dict(env._engine.runtime_budgets.budgets)  # noqa: SLF001
            candidates = enumerate_constraint_guided_transfer_candidates(
                tasks=env.ordered_tasks,
                current_budgets=current_budgets,
                safety_checker=env._ensure_checker(),  # noqa: SLF001
                runtime_features=env,
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
                top_k_risk=constraint_guided_pair_top_k_risk,
                top_k_decrease=constraint_guided_pair_top_k_decrease,
                prefer_lo=constraint_guided_pair_prefer_lo,
                validate_safety=True,
            )
            candidates_no_safety = enumerate_constraint_guided_transfer_candidates(
                tasks=env.ordered_tasks,
                current_budgets=current_budgets,
                safety_checker=env._ensure_checker(),  # noqa: SLF001
                runtime_features=env,
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
                top_k_risk=constraint_guided_pair_top_k_risk,
                top_k_decrease=constraint_guided_pair_top_k_decrease,
                prefer_lo=constraint_guided_pair_prefer_lo,
                validate_safety=False,
            )
            valid_count = int(sum(1 for item in candidates if item.accepted))
            valid_no_safety_count = int(sum(1 for item in candidates_no_safety if item.accepted))
            reject_counts: Counter[str] = Counter(item.reject_reason or "unknown" for item in candidates if not item.accepted)

            constraint_guided_pair_valid_values.append(float(valid_count))
            constraint_guided_pair_valid_no_safety_values.append(float(valid_no_safety_count))
            constraint_guided_pair_reject_incremental_values.append(
                float(reject_counts.get("incremental_constraint_violation", 0))
            )
            constraint_guided_pair_reject_hi_lo_values.append(float(reject_counts.get("hi_lo_mode_violation", 0)))
            constraint_guided_pair_reject_hi_mode_switch_values.append(
                float(reject_counts.get("hi_mode_switch_violation", 0))
            )
            constraint_guided_pair_reject_lo_mode_values.append(float(reject_counts.get("lo_mode_violation", 0)))
            constraint_guided_pair_reject_budget_floor_values.append(
                float(reject_counts.get("budget_floor_violation", 0))
            )
            known = {
                "incremental_constraint_violation",
                "hi_lo_mode_violation",
                "hi_mode_switch_violation",
                "lo_mode_violation",
                "budget_floor_violation",
                "budget_upper_bound_violation",
                "no_effective_budget_change",
            }
            unknown_count = float(sum(v for k, v in reject_counts.items() if k not in known and not k.startswith("hi_")))
            constraint_guided_pair_reject_unknown_values.append(unknown_count)

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
        "ranked_pair_candidate_count_mean": mean(ranked_pair_candidate_values) if ranked_pair_candidate_values else 0.0,
        "valid_ranked_pair_count_mean": mean(ranked_pair_valid_values) if ranked_pair_valid_values else 0.0,
        "valid_ranked_pair_count_no_safety_mean": (
            mean(ranked_pair_valid_no_safety_values) if ranked_pair_valid_no_safety_values else 0.0
        ),
        "ranked_pair_reject_incremental_constraint_violation_mean": (
            mean(ranked_pair_reject_incremental_values) if ranked_pair_reject_incremental_values else 0.0
        ),
        "valid_constraint_guided_pair_count_mean": (
            mean(constraint_guided_pair_valid_values) if constraint_guided_pair_valid_values else 0.0
        ),
        "valid_constraint_guided_pair_count_no_safety_mean": (
            mean(constraint_guided_pair_valid_no_safety_values)
            if constraint_guided_pair_valid_no_safety_values
            else 0.0
        ),
        "constraint_guided_pair_reject_incremental_constraint_violation_mean": (
            mean(constraint_guided_pair_reject_incremental_values)
            if constraint_guided_pair_reject_incremental_values
            else 0.0
        ),
        "constraint_guided_pair_reject_hi_lo_mode_violation_mean": (
            mean(constraint_guided_pair_reject_hi_lo_values) if constraint_guided_pair_reject_hi_lo_values else 0.0
        ),
        "constraint_guided_pair_reject_hi_mode_switch_violation_mean": (
            mean(constraint_guided_pair_reject_hi_mode_switch_values)
            if constraint_guided_pair_reject_hi_mode_switch_values
            else 0.0
        ),
        "constraint_guided_pair_reject_lo_mode_violation_mean": (
            mean(constraint_guided_pair_reject_lo_mode_values) if constraint_guided_pair_reject_lo_mode_values else 0.0
        ),
        "constraint_guided_pair_reject_budget_floor_violation_mean": (
            mean(constraint_guided_pair_reject_budget_floor_values)
            if constraint_guided_pair_reject_budget_floor_values
            else 0.0
        ),
        "constraint_guided_pair_reject_unknown_mean": (
            mean(constraint_guided_pair_reject_unknown_values) if constraint_guided_pair_reject_unknown_values else 0.0
        ),
    }


@dataclass(frozen=True)
class ScanConfig:
    """扫描配置对象。

    该对象只保存可序列化的基础参数，便于多进程并行分发。
    """

    workload: str
    automotive_mode: str
    automotive_num_runnables: int
    mc_fairgen_mode: str
    mc_fairgen_num_tasks: int
    mc_fairgen_hi_ratio: float
    mc_fairgen_period_source: str
    mc_fairgen_period_scale: int
    mc_fairgen_u_hi_lo_min: float
    mc_fairgen_u_hi_lo_max: float
    mc_fairgen_u_hi_hi_min: float
    mc_fairgen_u_hi_hi_max: float
    mc_fairgen_u_lo_lo_min: float
    mc_fairgen_u_lo_lo_max: float
    mc_fairgen_hi_budget_rho_min: float
    mc_fairgen_hi_budget_rho_max: float
    mc_fairgen_lo_budget_rho_min: float
    mc_fairgen_lo_budget_rho_max: float
    mc_fairgen_hi_overrun_prob: float
    mc_fairgen_lo_overrun_prob: float
    mc_fairgen_hi_overrun_factor_min: float
    mc_fairgen_hi_overrun_factor_max: float
    mc_fairgen_lo_overrun_factor_min: float
    mc_fairgen_lo_overrun_factor_max: float
    require_schedulable: bool
    eval_seeds: tuple[int, ...]
    budget_scales: tuple[float, ...]
    taskset_manifest: str | None
    manifest_seed_column: str
    manifest_rows_by_seed: dict[int, dict[str, str]]
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
    enable_ranked_pair_diagnostic: bool
    ranked_pair_min_valid_count: float
    ranked_pair_top_k_risk: int
    ranked_pair_top_k_surplus: int
    ranked_pair_decrease_mode: str
    enable_constraint_guided_pair_diagnostic: bool
    constraint_guided_pair_min_valid_count: float
    constraint_guided_pair_top_k_risk: int
    constraint_guided_pair_top_k_decrease: int
    constraint_guided_pair_prefer_lo: bool

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        eval_seeds: list[int],
        budget_scales: list[float],
        manifest_rows_by_seed: dict[int, dict[str, str]],
    ) -> "ScanConfig":
        """从命令行参数构造不可变配置对象。"""

        return cls(
            workload=args.workload,
            automotive_mode=args.automotive_mode,
            automotive_num_runnables=args.automotive_num_runnables,
            mc_fairgen_mode=args.mc_fairgen_mode,
            mc_fairgen_num_tasks=args.mc_fairgen_num_tasks,
            mc_fairgen_hi_ratio=args.mc_fairgen_hi_ratio,
            mc_fairgen_period_source=args.mc_fairgen_period_source,
            mc_fairgen_period_scale=args.mc_fairgen_period_scale,
            mc_fairgen_u_hi_lo_min=args.mc_fairgen_u_hi_lo_min,
            mc_fairgen_u_hi_lo_max=args.mc_fairgen_u_hi_lo_max,
            mc_fairgen_u_hi_hi_min=args.mc_fairgen_u_hi_hi_min,
            mc_fairgen_u_hi_hi_max=args.mc_fairgen_u_hi_hi_max,
            mc_fairgen_u_lo_lo_min=args.mc_fairgen_u_lo_lo_min,
            mc_fairgen_u_lo_lo_max=args.mc_fairgen_u_lo_lo_max,
            mc_fairgen_hi_budget_rho_min=args.mc_fairgen_hi_budget_rho_min,
            mc_fairgen_hi_budget_rho_max=args.mc_fairgen_hi_budget_rho_max,
            mc_fairgen_lo_budget_rho_min=args.mc_fairgen_lo_budget_rho_min,
            mc_fairgen_lo_budget_rho_max=args.mc_fairgen_lo_budget_rho_max,
            mc_fairgen_hi_overrun_prob=args.mc_fairgen_hi_overrun_prob,
            mc_fairgen_lo_overrun_prob=args.mc_fairgen_lo_overrun_prob,
            mc_fairgen_hi_overrun_factor_min=args.mc_fairgen_hi_overrun_factor_min,
            mc_fairgen_hi_overrun_factor_max=args.mc_fairgen_hi_overrun_factor_max,
            mc_fairgen_lo_overrun_factor_min=args.mc_fairgen_lo_overrun_factor_min,
            mc_fairgen_lo_overrun_factor_max=args.mc_fairgen_lo_overrun_factor_max,
            require_schedulable=args.require_schedulable,
            eval_seeds=tuple(eval_seeds),
            budget_scales=tuple(budget_scales),
            taskset_manifest=args.taskset_manifest,
            manifest_seed_column=args.manifest_seed_column,
            manifest_rows_by_seed=manifest_rows_by_seed,
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
            enable_ranked_pair_diagnostic=args.enable_ranked_pair_diagnostic,
            ranked_pair_min_valid_count=args.ranked_pair_min_valid_count,
            ranked_pair_top_k_risk=args.ranked_pair_top_k_risk,
            ranked_pair_top_k_surplus=args.ranked_pair_top_k_surplus,
            ranked_pair_decrease_mode=args.ranked_pair_decrease_mode,
            enable_constraint_guided_pair_diagnostic=args.enable_constraint_guided_pair_diagnostic,
            constraint_guided_pair_min_valid_count=args.constraint_guided_pair_min_valid_count,
            constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
            constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
            constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        )


def _build_scan_experiment_config(
    *,
    config: ScanConfig,
    fixed_taskset_seed: int,
    manifest_row: dict[str, str] | None = None,
):
    """按 workload 构建扫描用 experiment config。"""

    if config.workload == "automotive":
        mode = config.automotive_mode
        num_runnables = config.automotive_num_runnables
        if manifest_row:
            mode = str(manifest_row.get("automotive_mode", mode) or mode)
            num_runnables = int(float(manifest_row.get("automotive_num_runnables", num_runnables) or num_runnables))
        return build_automotive_experiment_config(
            num_runnables=num_runnables,
            mode=mode,
            require_schedulable=config.require_schedulable,
            fixed_taskset_seed=fixed_taskset_seed,
            learnable_target_budget_util_min=config.learnable_target_budget_util_min,
            learnable_target_budget_util_max=config.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=config.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=config.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=config.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=config.learnable_lo_budget_rho_max,
            budget_floor_ratio=config.budget_floor_ratio,
        )
    if config.workload == "mc_fairgen":
        mode = config.mc_fairgen_mode
        num_tasks = config.mc_fairgen_num_tasks
        hi_ratio = config.mc_fairgen_hi_ratio
        period_source = config.mc_fairgen_period_source
        period_scale = config.mc_fairgen_period_scale
        u_hi_lo_min = config.mc_fairgen_u_hi_lo_min
        u_hi_lo_max = config.mc_fairgen_u_hi_lo_max
        u_hi_hi_min = config.mc_fairgen_u_hi_hi_min
        u_hi_hi_max = config.mc_fairgen_u_hi_hi_max
        u_lo_lo_min = config.mc_fairgen_u_lo_lo_min
        u_lo_lo_max = config.mc_fairgen_u_lo_lo_max
        hi_budget_rho_min = config.mc_fairgen_hi_budget_rho_min
        hi_budget_rho_max = config.mc_fairgen_hi_budget_rho_max
        lo_budget_rho_min = config.mc_fairgen_lo_budget_rho_min
        lo_budget_rho_max = config.mc_fairgen_lo_budget_rho_max
        hi_overrun_prob = config.mc_fairgen_hi_overrun_prob
        lo_overrun_prob = config.mc_fairgen_lo_overrun_prob
        hi_overrun_factor_min = config.mc_fairgen_hi_overrun_factor_min
        hi_overrun_factor_max = config.mc_fairgen_hi_overrun_factor_max
        lo_overrun_factor_min = config.mc_fairgen_lo_overrun_factor_min
        lo_overrun_factor_max = config.mc_fairgen_lo_overrun_factor_max
        if manifest_row:
            mode = str(manifest_row.get("mc_fairgen_mode", mode) or mode)
            num_tasks = int(float(manifest_row.get("mc_fairgen_num_tasks", num_tasks) or num_tasks))
            hi_ratio = float(manifest_row.get("mc_fairgen_hi_ratio", hi_ratio) or hi_ratio)
            period_source = str(manifest_row.get("mc_fairgen_period_source", period_source) or period_source)
            period_scale = int(float(manifest_row.get("mc_fairgen_period_scale", period_scale) or period_scale))
            u_hi_lo_min = float(manifest_row.get("mc_fairgen_u_hi_lo_min", u_hi_lo_min) or u_hi_lo_min)
            u_hi_lo_max = float(manifest_row.get("mc_fairgen_u_hi_lo_max", u_hi_lo_max) or u_hi_lo_max)
            u_hi_hi_min = float(manifest_row.get("mc_fairgen_u_hi_hi_min", u_hi_hi_min) or u_hi_hi_min)
            u_hi_hi_max = float(manifest_row.get("mc_fairgen_u_hi_hi_max", u_hi_hi_max) or u_hi_hi_max)
            u_lo_lo_min = float(manifest_row.get("mc_fairgen_u_lo_lo_min", u_lo_lo_min) or u_lo_lo_min)
            u_lo_lo_max = float(manifest_row.get("mc_fairgen_u_lo_lo_max", u_lo_lo_max) or u_lo_lo_max)
            hi_budget_rho_min = float(manifest_row.get("mc_fairgen_hi_budget_rho_min", hi_budget_rho_min) or hi_budget_rho_min)
            hi_budget_rho_max = float(manifest_row.get("mc_fairgen_hi_budget_rho_max", hi_budget_rho_max) or hi_budget_rho_max)
            lo_budget_rho_min = float(manifest_row.get("mc_fairgen_lo_budget_rho_min", lo_budget_rho_min) or lo_budget_rho_min)
            lo_budget_rho_max = float(manifest_row.get("mc_fairgen_lo_budget_rho_max", lo_budget_rho_max) or lo_budget_rho_max)
            hi_overrun_prob = float(manifest_row.get("mc_fairgen_hi_overrun_prob", hi_overrun_prob) or hi_overrun_prob)
            lo_overrun_prob = float(manifest_row.get("mc_fairgen_lo_overrun_prob", lo_overrun_prob) or lo_overrun_prob)
            hi_overrun_factor_min = float(manifest_row.get("mc_fairgen_hi_overrun_factor_min", hi_overrun_factor_min) or hi_overrun_factor_min)
            hi_overrun_factor_max = float(manifest_row.get("mc_fairgen_hi_overrun_factor_max", hi_overrun_factor_max) or hi_overrun_factor_max)
            lo_overrun_factor_min = float(manifest_row.get("mc_fairgen_lo_overrun_factor_min", lo_overrun_factor_min) or lo_overrun_factor_min)
            lo_overrun_factor_max = float(manifest_row.get("mc_fairgen_lo_overrun_factor_max", lo_overrun_factor_max) or lo_overrun_factor_max)
        return build_mc_fairgen_experiment_config(
            mode=mode,
            num_tasks=num_tasks,
            hi_ratio=hi_ratio,
            period_source=period_source,
            period_scale=period_scale,
            require_schedulable=config.require_schedulable,
            fixed_taskset_seed=fixed_taskset_seed,
            u_hi_lo_min=u_hi_lo_min,
            u_hi_lo_max=u_hi_lo_max,
            u_hi_hi_min=u_hi_hi_min,
            u_hi_hi_max=u_hi_hi_max,
            u_lo_lo_min=u_lo_lo_min,
            u_lo_lo_max=u_lo_lo_max,
            hi_budget_rho_min=hi_budget_rho_min,
            hi_budget_rho_max=hi_budget_rho_max,
            lo_budget_rho_min=lo_budget_rho_min,
            lo_budget_rho_max=lo_budget_rho_max,
            hi_overrun_prob=hi_overrun_prob,
            lo_overrun_prob=lo_overrun_prob,
            hi_overrun_factor_min=hi_overrun_factor_min,
            hi_overrun_factor_max=hi_overrun_factor_max,
            lo_overrun_factor_min=lo_overrun_factor_min,
            lo_overrun_factor_max=lo_overrun_factor_max,
        )
    raise ValueError(f"unsupported workload: {config.workload}")


def scan_one_taskset_seed_budget_scale(
    taskset_seed: int,
    budget_scale: float,
    config: ScanConfig,
) -> dict[str, int | float | str | bool]:
    """扫描单个 `(fixed_taskset_seed, budget_scale)` 组合，返回一行 CSV。"""

    manifest_row = config.manifest_rows_by_seed.get(taskset_seed, {})
    experiment_config = _build_scan_experiment_config(
        config=config,
        fixed_taskset_seed=taskset_seed,
        manifest_row=manifest_row if manifest_row else None,
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

    manifest_row = config.manifest_rows_by_seed.get(taskset_seed, {})
    # 当使用 manifest 扫描时，必须复现 generate 脚本的 two-stage 路径：
    # paper_exact(require_schedulable=True) -> learnable budget rewrite。
    if config.taskset_manifest is not None and config.workload == "automotive":
        base_config = build_automotive_experiment_config(
            num_runnables=config.automotive_num_runnables,
            mode="paper_exact",
            require_schedulable=True,
            fixed_taskset_seed=taskset_seed,
            budget_floor_ratio=config.budget_floor_ratio,
        )
        base_bundle = resolve_experiment_bundle(base_config, seed=10_000 + taskset_seed)
        rewrite_cfg = _TwoStageRewriteConfig(
            budget_floor_ratio=config.budget_floor_ratio,
            learnable_target_budget_util_min=config.learnable_target_budget_util_min,
            learnable_target_budget_util_max=config.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=config.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=config.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=config.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=config.learnable_lo_budget_rho_max,
        )
        rewritten_tasks, rewritten_meta = _rewrite_budgets_for_learnable_headroom(
            ordered_tasks=list(base_bundle.ordered_tasks),
            candidate_seed=taskset_seed,
            cfg=rewrite_cfg,  # type: ignore[arg-type]
        )
        scaled_tasks_for_sched, budget_scale_stats = _apply_budget_scale_to_tasks(
            ordered_tasks=list(rewritten_tasks),
            budget_scale=budget_scale,
            budget_floor_ratio=config.budget_floor_ratio,
        )
    elif config.taskset_manifest is not None and config.workload == "mc_fairgen":
        initial_bundle = resolve_experiment_bundle(experiment_config, config.eval_seeds[0])
        rewritten_meta = {}
        scaled_tasks_for_sched, budget_scale_stats = _apply_budget_scale_to_tasks(
            ordered_tasks=list(initial_bundle.ordered_tasks),
            budget_scale=budget_scale,
            budget_floor_ratio=config.budget_floor_ratio,
        )
    else:
        initial_bundle = resolve_experiment_bundle(experiment_config, config.eval_seeds[0])
        rewritten_meta = {}
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
            if config.taskset_manifest is not None and config.workload == "automotive":
                # manifest 模式下固定同一份 two-stage 任务集，仅替换 scenario seed。
                scenario_bundle = resolve_experiment_bundle(base_config, eval_seed)
                scaled_tasks = list(scaled_tasks_for_sched)
                scenario = scenario_bundle.scenario
            else:
                bundle_for_seed = resolve_experiment_bundle(experiment_config, eval_seed)
                scaled_tasks, _ = _apply_budget_scale_to_tasks(
                    ordered_tasks=list(bundle_for_seed.ordered_tasks),
                    budget_scale=budget_scale,
                    budget_floor_ratio=config.budget_floor_ratio,
                )
                scenario = bundle_for_seed.scenario

            baseline_result = simulate_ordered_taskset_event_driven(
                ordered_tasks=scaled_tasks,
                scenario=scenario,
                config=RuntimeConfig(end_time=config.end_time, semantics=RuntimeSemantics.AMC_PLUS),
            )
            baseline_mode_changes_values.append(float(baseline_result.mode_change_count()))
            baseline_lo_cancellations_values.append(float(baseline_result.lo_job_cancellation_count()))
            baseline_deadline_misses_values.append(float(len(baseline_result.deadline_misses)))

            diagnostic_rows.append(
                _run_diagnostic_on_seed(
                    ordered_tasks=scaled_tasks,
                    scenario=scenario,
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
                    enable_ranked_pair_diagnostic=config.enable_ranked_pair_diagnostic,
                    ranked_pair_top_k_risk=config.ranked_pair_top_k_risk,
                    ranked_pair_top_k_surplus=config.ranked_pair_top_k_surplus,
                    ranked_pair_decrease_mode=config.ranked_pair_decrease_mode,
                    enable_constraint_guided_pair_diagnostic=config.enable_constraint_guided_pair_diagnostic,
                    constraint_guided_pair_top_k_risk=config.constraint_guided_pair_top_k_risk,
                    constraint_guided_pair_top_k_decrease=config.constraint_guided_pair_top_k_decrease,
                    constraint_guided_pair_prefer_lo=config.constraint_guided_pair_prefer_lo,
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
    baseline_lo_cancellation_ratio_total = baseline_lo_cancellations_mean / max(1.0, baseline_total_events_mean)

    valid_increase_count_mean = mean(item["valid_increase_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    valid_decrease_count_mean = mean(item["valid_decrease_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    valid_action_count_mean = mean(item["valid_action_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    safety_margin_min_p05 = mean(item["safety_margin_min_p05"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    increase_decrease_balance = valid_increase_count_mean / max(1.0, valid_decrease_count_mean)
    valid_ranked_pair_count_mean = mean(item["valid_ranked_pair_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    valid_ranked_pair_count_no_safety_mean = (
        mean(item["valid_ranked_pair_count_no_safety_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    )
    ranked_pair_candidate_count_mean = (
        mean(item["ranked_pair_candidate_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    )
    ranked_pair_reject_incremental_constraint_violation_mean = (
        mean(item["ranked_pair_reject_incremental_constraint_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    valid_constraint_guided_pair_count_mean = (
        mean(item["valid_constraint_guided_pair_count_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    )
    valid_constraint_guided_pair_count_no_safety_mean = (
        mean(item["valid_constraint_guided_pair_count_no_safety_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_incremental_constraint_violation_mean = (
        mean(item["constraint_guided_pair_reject_incremental_constraint_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_hi_lo_mode_violation_mean = (
        mean(item["constraint_guided_pair_reject_hi_lo_mode_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_hi_mode_switch_violation_mean = (
        mean(item["constraint_guided_pair_reject_hi_mode_switch_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_lo_mode_violation_mean = (
        mean(item["constraint_guided_pair_reject_lo_mode_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_budget_floor_violation_mean = (
        mean(item["constraint_guided_pair_reject_budget_floor_violation_mean"] for item in diagnostic_rows)
        if diagnostic_rows
        else 0.0
    )
    constraint_guided_pair_reject_unknown_mean = (
        mean(item["constraint_guided_pair_reject_unknown_mean"] for item in diagnostic_rows) if diagnostic_rows else 0.0
    )
    valid_ranked_pair_to_single_increase_ratio = valid_ranked_pair_count_mean / max(1.0, valid_increase_count_mean)

    # 先计算 baseline_total_events 对应的事件分组，后续所有推荐逻辑都依赖该变量。
    # 这里必须放在 recommended_for_dqn / recommended_for_constraint_guided_pair_dqn 之前，
    # 否则在某些分支中会出现 event_group 未定义错误。
    event_group = _event_group(
        baseline_total_events_mean,
        low_event_threshold=config.low_event_threshold,
        high_event_threshold=config.high_event_threshold,
    )
    slack_group = _slack_group(safety_margin_min_p05)

    # 在推荐逻辑之前统一计算 headroom 分组指标，保证所有分支都可用且口径一致。
    increase_headroom_group = _increase_headroom_group(valid_increase_count_mean)
    decrease_headroom_group = _decrease_headroom_group(valid_decrease_count_mean)
    balanced_headroom_group = _balanced_headroom_group(valid_increase_count_mean, valid_decrease_count_mean)

    recommended_for_constraint_guided_pair_dqn = (
        config.enable_constraint_guided_pair_diagnostic
        and baseline_deadline_misses_sum == 0.0
        and event_group == "medium_event"
        and valid_constraint_guided_pair_count_mean >= config.constraint_guided_pair_min_valid_count
    )
    # constraint-guided 诊断关闭时给出显式默认原因，便于后续 CSV 解释。
    if not config.enable_constraint_guided_pair_diagnostic:
        constraint_guided_pair_not_recommended_reason = "constraint_guided_pair_diagnostic_disabled"
    elif not recommended_for_constraint_guided_pair_dqn:
        cgp_reasons: list[str] = []
        if baseline_deadline_misses_sum > 0.0:
            cgp_reasons.append("deadline_misses")
        if valid_constraint_guided_pair_count_mean < config.constraint_guided_pair_min_valid_count:
            cgp_reasons.append("low_valid_constraint_guided_pair_headroom")
        if event_group == "low_event":
            cgp_reasons.append("too_few_baseline_events")
        elif event_group == "high_event":
            cgp_reasons.append("too_many_baseline_events")
        constraint_guided_pair_not_recommended_reason = ";".join(cgp_reasons)
    else:
        constraint_guided_pair_not_recommended_reason = ""

    if config.enable_ranked_pair_diagnostic:
        recommended_for_dqn = (
            baseline_deadline_misses_sum == 0.0
            and event_group == "medium_event"
            and valid_ranked_pair_count_mean >= config.ranked_pair_min_valid_count
            and valid_decrease_count_mean >= 3.0
        )
    else:
        recommended_for_dqn = (
            baseline_deadline_misses_sum == 0.0
            and event_group == "medium_event"
            and valid_increase_count_mean >= 3.0
            and valid_decrease_count_mean >= 3.0
            and increase_decrease_balance >= 0.2
        )

    not_recommended_reason = ""
    if not recommended_for_dqn:
        if config.enable_ranked_pair_diagnostic:
            reasons: list[str] = []
            if baseline_deadline_misses_sum > 0.0:
                reasons.append("deadline_misses")
            if event_group == "low_event":
                reasons.append("too_few_baseline_events")
            elif event_group == "high_event":
                reasons.append("too_many_baseline_events")
            if valid_ranked_pair_count_mean < config.ranked_pair_min_valid_count:
                reasons.append("reject_fast_ranked_pair_headroom")
            not_recommended_reason = ";".join(reasons)
        else:
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
        "workload": config.workload,
        "fixed_taskset_seed": taskset_seed,
        "candidate_seed": taskset_seed,
        "manifest_path": config.taskset_manifest or "",
        "manifest_row_index": (
            int(float(manifest_row.get("__manifest_row_index", "-1"))) if manifest_row else -1
        ),
        "source_generation_strategy": (
            str(manifest_row.get("generation_strategy", "two_stage_from_paper_exact"))
            if config.taskset_manifest is not None
            else ""
        ),
        "source_manifest_fast_total_events": (
            float(manifest_row.get("fast_baseline_total_events_mean", 0.0)) if manifest_row else 0.0
        ),
        "source_manifest_fast_valid_constraint_guided_pair_count": (
            float(manifest_row.get("fast_valid_constraint_guided_pair_count_mean", 0.0)) if manifest_row else 0.0
        ),
        "source_manifest_recommended_for_constraint_guided_pair_dqn": (
            _parse_bool_like(manifest_row.get("recommended_for_constraint_guided_pair_dqn", False))
            if manifest_row
            else False
        ),
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
        "baseline_lo_cancellation_ratio_total": baseline_lo_cancellation_ratio_total,
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
        "valid_ranked_pair_count_mean": valid_ranked_pair_count_mean,
        "valid_ranked_pair_count_no_safety_mean": valid_ranked_pair_count_no_safety_mean,
        "ranked_pair_candidate_count_mean": ranked_pair_candidate_count_mean,
        "valid_ranked_pair_to_single_increase_ratio": valid_ranked_pair_to_single_increase_ratio,
        "ranked_pair_reject_incremental_constraint_violation_mean": (
            ranked_pair_reject_incremental_constraint_violation_mean
        ),
        "valid_constraint_guided_pair_count_mean": valid_constraint_guided_pair_count_mean,
        "valid_constraint_guided_transfer_count_mean": valid_constraint_guided_pair_count_mean,
        "valid_constraint_guided_pair_count_no_safety_mean": valid_constraint_guided_pair_count_no_safety_mean,
        "constraint_guided_pair_reject_incremental_constraint_violation_mean": (
            constraint_guided_pair_reject_incremental_constraint_violation_mean
        ),
        "constraint_guided_pair_reject_hi_lo_mode_violation_mean": (
            constraint_guided_pair_reject_hi_lo_mode_violation_mean
        ),
        "constraint_guided_pair_reject_hi_mode_switch_violation_mean": (
            constraint_guided_pair_reject_hi_mode_switch_violation_mean
        ),
        "constraint_guided_pair_reject_lo_mode_violation_mean": (
            constraint_guided_pair_reject_lo_mode_violation_mean
        ),
        "constraint_guided_pair_reject_budget_floor_violation_mean": (
            constraint_guided_pair_reject_budget_floor_violation_mean
        ),
        "constraint_guided_pair_reject_unknown_mean": constraint_guided_pair_reject_unknown_mean,
        "recommended_for_constraint_guided_pair_dqn": bool(recommended_for_constraint_guided_pair_dqn),
        "recommended_for_constraint_guided_transfer_dqn": bool(recommended_for_constraint_guided_pair_dqn),
        "constraint_guided_pair_not_recommended_reason": constraint_guided_pair_not_recommended_reason,
        "constraint_guided_transfer_not_recommended_reason": constraint_guided_pair_not_recommended_reason,
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
    if config.workload == "automotive":
        row["automotive_mode"] = config.automotive_mode
        row["automotive_num_runnables"] = config.automotive_num_runnables
    if config.workload == "mc_fairgen":
        row["mc_fairgen_mode"] = config.mc_fairgen_mode
        row["mc_fairgen_num_tasks"] = config.mc_fairgen_num_tasks
        row["mc_fairgen_hi_ratio"] = config.mc_fairgen_hi_ratio
        row["mc_fairgen_u_hi_lo_min"] = config.mc_fairgen_u_hi_lo_min
        row["mc_fairgen_u_hi_lo_max"] = config.mc_fairgen_u_hi_lo_max
        row["mc_fairgen_u_hi_hi_min"] = config.mc_fairgen_u_hi_hi_min
        row["mc_fairgen_u_hi_hi_max"] = config.mc_fairgen_u_hi_hi_max
        row["mc_fairgen_u_lo_lo_min"] = config.mc_fairgen_u_lo_lo_min
        row["mc_fairgen_u_lo_lo_max"] = config.mc_fairgen_u_lo_lo_max
        row["mc_fairgen_hi_budget_rho_min"] = config.mc_fairgen_hi_budget_rho_min
        row["mc_fairgen_hi_budget_rho_max"] = config.mc_fairgen_hi_budget_rho_max
        row["mc_fairgen_lo_budget_rho_min"] = config.mc_fairgen_lo_budget_rho_min
        row["mc_fairgen_lo_budget_rho_max"] = config.mc_fairgen_lo_budget_rho_max

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
    parser.add_argument("--workload", choices=["automotive", "mc_fairgen"], default="automotive")
    parser.add_argument("--automotive-mode", type=str, default="paper_exact")
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)
    parser.add_argument("--automotive-num-runnables", type=int, default=150)
    parser.add_argument("--mc-fairgen-mode", type=str, default="paper_learnable_headroom")
    parser.add_argument("--mc-fairgen-num-tasks", type=int, default=16)
    parser.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    parser.add_argument("--mc-fairgen-period-source", type=str, default="automotive")
    parser.add_argument("--mc-fairgen-period-scale", type=int, default=100)
    parser.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    parser.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    parser.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    parser.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.60)
    parser.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    parser.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    parser.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.05)
    parser.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.25)
    parser.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    parser.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.40)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.05)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.80)
    parser.add_argument("--require-schedulable", action="store_true")
    parser.add_argument("--fixed-taskset-seeds", type=str, default="")
    parser.add_argument(
        "--taskset-manifest",
        type=str,
        default=None,
        help="CSV manifest from generate_learnable_tasksets.py. When set, seeds come from manifest.",
    )
    parser.add_argument(
        "--manifest-seed-column",
        type=str,
        default="candidate_seed",
        help="Seed column name in --taskset-manifest.",
    )
    parser.add_argument(
        "--manifest-seed-limit",
        type=int,
        default=None,
        help="Optional upper bound of selected manifest seeds.",
    )
    parser.add_argument(
        "--manifest-filter-recommended",
        action="store_true",
        help="Only keep manifest rows recommended by known recommendation columns.",
    )
    parser.add_argument(
        "--manifest-output-selected",
        type=str,
        default=None,
        help="Optional output path for selected manifest rows.",
    )
    parser.add_argument(
        "--manifest-strict-parameter-check",
        action="store_true",
        help="Fail when key CLI parameters differ from manifest parameters.",
    )
    parser.add_argument("--budget-scales", type=str, default="1.0")
    parser.add_argument("--eval-seeds", type=str, default="")
    parser.add_argument("--seeds", dest="seeds", type=str, default="")
    parser.add_argument("--end-time", type=int, default=10_000_000)
    parser.add_argument("--agent-period", type=int, default=100_000)
    parser.add_argument("--reward-mode", type=str, default="interval_v1")
    parser.add_argument(
        "--action-space",
        choices=["single", "pair", "triple", "constraint_guided_pair", "constraint_guided_transfer"],
        default="single",
    )
    parser.add_argument("--budget-increase-ratio", type=float, default=0.025)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.0125)
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
    parser.add_argument("--enable-constraint-guided-pair-diagnostic", action="store_true")
    parser.add_argument("--constraint-guided-pair-min-valid-count", type=float, default=1.0)
    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=4)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action="store_true")
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

    budget_scales = _parse_float_list(args.budget_scales)
    raw_eval_seeds = args.eval_seeds if args.eval_seeds.strip() else args.seeds
    eval_seeds = _parse_int_list_or_range(raw_eval_seeds)
    manifest_rows_by_seed: dict[int, dict[str, str]] = {}

    if args.taskset_manifest:
        manifest_rows = _load_manifest_rows(
            args.taskset_manifest,
            seed_column=args.manifest_seed_column,
            seed_limit=args.manifest_seed_limit,
            filter_recommended=args.manifest_filter_recommended,
        )
        for index, row in enumerate(manifest_rows):
            copied = dict(row)
            copied["__manifest_row_index"] = str(index)
            manifest_rows_by_seed[int(copied[args.manifest_seed_column])] = copied
        fixed_taskset_seeds = list(manifest_rows_by_seed.keys())
        _warn_or_raise_manifest_parameter_mismatch(args=args, manifest_rows=manifest_rows)
        if args.manifest_output_selected:
            _write_selected_manifest_rows(
                args.manifest_output_selected,
                [manifest_rows_by_seed[seed] for seed in fixed_taskset_seeds],
            )
    else:
        fixed_taskset_seeds = _parse_int_list_or_range(args.fixed_taskset_seeds)

    if not fixed_taskset_seeds:
        raise ValueError("--fixed-taskset-seeds 不能为空（或请提供 --taskset-manifest）")
    if not budget_scales:
        raise ValueError("--budget-scales 不能为空")
    if not eval_seeds:
        raise ValueError("--eval-seeds/--seeds 不能为空")

    config = ScanConfig.from_args(args, eval_seeds, budget_scales, manifest_rows_by_seed)
    scan_pairs = [(seed, scale) for seed in fixed_taskset_seeds for scale in budget_scales]
    rows = _run_parallel(scan_pairs, config, args.workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "automotive_mode",
        "automotive_num_runnables",
        "mc_fairgen_mode",
        "mc_fairgen_num_tasks",
        "mc_fairgen_hi_ratio",
        "mc_fairgen_u_hi_lo_min",
        "mc_fairgen_u_hi_lo_max",
        "mc_fairgen_u_hi_hi_min",
        "mc_fairgen_u_hi_hi_max",
        "mc_fairgen_u_lo_lo_min",
        "mc_fairgen_u_lo_lo_max",
        "mc_fairgen_hi_budget_rho_min",
        "mc_fairgen_hi_budget_rho_max",
        "mc_fairgen_lo_budget_rho_min",
        "mc_fairgen_lo_budget_rho_max",
        "fixed_taskset_seed",
        "candidate_seed",
        "manifest_path",
        "manifest_row_index",
        "source_generation_strategy",
        "source_manifest_fast_total_events",
        "source_manifest_fast_valid_constraint_guided_pair_count",
        "source_manifest_recommended_for_constraint_guided_pair_dqn",
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
        "baseline_lo_cancellation_ratio_total",
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
        "ranked_pair_candidate_count_mean",
        "valid_ranked_pair_count_mean",
        "valid_ranked_pair_count_no_safety_mean",
        "valid_ranked_pair_to_single_increase_ratio",
        "ranked_pair_reject_incremental_constraint_violation_mean",
        "valid_constraint_guided_pair_count_mean",
        "valid_constraint_guided_transfer_count_mean",
        "valid_constraint_guided_pair_count_no_safety_mean",
        "constraint_guided_pair_reject_incremental_constraint_violation_mean",
        "constraint_guided_pair_reject_hi_lo_mode_violation_mean",
        "constraint_guided_pair_reject_hi_mode_switch_violation_mean",
        "constraint_guided_pair_reject_lo_mode_violation_mean",
        "constraint_guided_pair_reject_budget_floor_violation_mean",
        "constraint_guided_pair_reject_unknown_mean",
        "recommended_for_constraint_guided_pair_dqn",
        "recommended_for_constraint_guided_transfer_dqn",
        "constraint_guided_pair_not_recommended_reason",
        "constraint_guided_transfer_not_recommended_reason",
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

    # 为了避免未来新增列导致 DictWriter 报错，这里动态补齐所有 row 的 key。
    all_fieldnames = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in all_fieldnames:
                all_fieldnames.append(key)

    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
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
