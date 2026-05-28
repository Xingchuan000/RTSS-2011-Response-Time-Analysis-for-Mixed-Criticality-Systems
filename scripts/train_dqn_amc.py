"""正式 DQN 训练命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from amc_py.dqn import (
    DqnBudgetAgent,
    DqnConfig,
    ExperimentConfig,
    Transition,
    build_automotive_experiment_config,
    build_mc_fairgen_experiment_config,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import (
    compute_service_quality_metrics,
    mean_optional as mean_optional_service_metric,
    safe_relative_reduction,
    service_metrics_to_row,
)
from amc_py.model_selection import (
    is_conservative_qos_valid,
    is_qos_best_valid,
    is_qos_stable_valid,
    qos_sort_key,
)
from amc_py.models import Task
from amc_py.rl.actions import describe_budget_action
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.reward_config import available_reward_modes, load_reward_mode_config
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult


STEP_LOG_FIELDNAMES = [
    "episode",
    "step",
    "sim_time",
    "reward",
    "episode_reward",
    "total_reward",
    "loss",
    "epsilon",
    "action_id",
    "accepted",
    "rejected",
    "reject_reason",
    "residual_guard_enabled",
    "residual_guard_rejected",
    "residual_guard_rejected_actions",
    "residual_guard_hi_pressure_delta_limit",
    "residual_guard_hi_pressure_abs_limit",
    "residual_action_type",
    "residual_rank",
    "residual_resolved_increase_task",
    "residual_resolved_decrease_tasks",
    "valid_action_count",
    "masked_action_count",
    "noop_due_to_no_valid_action",
    "is_noop",
    "is_explicit_noop",
    "is_budget_action",
    "is_increase_action",
    "is_decrease_action",
    "is_transfer_action",
    "decrease_hits_hi",
    "decrease_hits_lo",
    "decrease_task_count",
    "unsafe_decrease",
    "mode_changes",
    "lo_cancellations",
    "deadline_misses",
    "step_reward_total",
    "step_reward_job_start",
    "step_reward_lo_overrun",
    "step_reward_hi_overrun",
    "step_reward_mode_change",
    "mode_change_spike_penalty_value",
    "step_reward_lo_cancellation",
    "step_reward_deadline_miss",
    "step_reward_invalid_action",
    "paper_reward",
    "noop_reward_bonus",
    "lo_overrun_rate",
    "hi_overrun_rate",
    "mode_change_per_job",
    "lo_cancellation_rate",
    "deadline_miss_rate",
    "invalid_action",
    "budget_change_norm",
    "budget_change_penalty_value",
    "budget_drift_mean",
    "budget_drift_penalty_value",
    "lo_pressure_mean",
    "lo_pressure_max",
    "lo_near_cancel_rate",
    "hi_mode_pressure_mean",
    "lo_pressure_penalty_value",
    "lo_pressure_max_penalty_value",
    "lo_near_cancel_penalty_value",
    "hi_mode_pressure_penalty_value",
    "reward_after_regularization",
    "workload",
    "total_util",
    "num_tasks",
    "cf",
    "cp",
    "taskset_seed",
    "scenario_seed",
    "require_schedulable",
    "observation_mode",
    "state_dim",
]

NOOP_Q_DIAGNOSTIC_FIELDNAMES = [
    "noop_q_mean",
    "noop_q_std",
    "noop_q_rank_mean",
    "noop_q_rank_median",
    "noop_q_rank_min",
    "noop_q_rank_max",
    "noop_q_margin_to_best_mean",
    "noop_q_is_best_rate",
    "noop_valid_rate",
    "noop_q_sample_count",
]

QOS_VALIDATION_FIELDNAMES = [
    "released_lo_jobs_mean",
    "cancelled_lo_jobs_mean",
    "completed_lo_jobs_mean",
    "lo_deadline_misses_sum",
    "hi_deadline_misses_sum",
    "lc_service_loss_mean",
    "lc_qos_mean",
    "min_lc_service_mean",
    "budget_adjust_count_mean",
    "mean_abs_budget_change_mean",
    "baseline_released_lo_jobs_mean",
    "baseline_lc_service_loss_mean",
    "baseline_lc_qos_mean",
    "baseline_min_lc_service_mean",
    "baseline_hi_deadline_misses_sum",
    "relative_lc_loss_reduction",
    "lc_service_loss_delta_mean",
    "lc_qos_delta_mean",
    "mode_change_delta_ratio",
]


def _qos_validation_metadata_row(
    row: dict[str, int | float | None],
    *,
    qos_stable_mode_delta: float,
) -> dict[str, object]:
    """从 validation row 派生 QoS 选模元数据字段。"""

    metadata: dict[str, object] = {
        "hi_deadline_misses_sum": int(float(row.get("hi_deadline_misses_sum", row.get("deadline_misses_sum", 0)) or 0)),
        "lc_service_loss_mean": float(row["lc_service_loss_mean"]),
        "baseline_lc_service_loss_mean": float(row["baseline_lc_service_loss_mean"]),
        "lc_qos_mean": float(row["lc_qos_mean"]),
        "baseline_lc_qos_mean": float(row["baseline_lc_qos_mean"]),
        "relative_lc_loss_reduction": row.get("relative_lc_loss_reduction"),
        "min_lc_service_mean": row.get("min_lc_service_mean"),
        "mode_change_delta_ratio": float(row["mode_change_delta_ratio"]),
        "budget_adjust_count_mean": float(row["budget_adjust_count_mean"]),
        "mean_abs_budget_change_mean": row.get("mean_abs_budget_change_mean"),
        "is_conservative_qos_valid": is_conservative_qos_valid(row),
        "is_qos_stable_valid": is_qos_stable_valid(row, delta=qos_stable_mode_delta),
        "is_qos_best_valid": is_qos_best_valid(row),
    }
    return metadata


def _build_best_metadata(
    *,
    save_best_by: str,
    reward_mode: str,
    reward_definition: str,
    double_dqn: bool,
    relative_score_alpha: float,
    require_better_than_baseline_for_best: bool,
    qos_stable_mode_delta: float,
    best_type: str,
    best_row: dict[str, int | float | None] | None,
) -> dict[str, object]:
    """构建 best checkpoint metadata。"""

    if best_row is None:
        return {
            "save_best_by": save_best_by,
            "best_type": best_type,
            "selection_metric": best_type,
            "relative_score_alpha": relative_score_alpha,
            "require_better_than_baseline_for_best": require_better_than_baseline_for_best,
            "qos_stable_mode_delta": qos_stable_mode_delta,
            "reward_mode": reward_mode,
            "reward_definition": reward_definition,
            "double_dqn": double_dqn,
            "found_valid_checkpoint": False,
            "reason": "No checkpoint satisfied HI safety and mode stability constraints.",
        }
    best_relative_score = float(best_row.get("relative_score", 0.0))
    best_delta_mode = float(best_row.get("relative_delta_mode_changes", 0.0))
    best_delta_lo = float(best_row.get("relative_delta_lo_cancellations", 0.0))
    best_raw_delta_mode = float(
        best_row.get("raw_delta_mode_changes", best_row.get("dqn_mode_changes_delta_mean", 0.0))
    )
    best_raw_delta_lo = float(
        best_row.get("raw_delta_lo_cancellations", best_row.get("dqn_lo_cancellations_delta_mean", 0.0))
    )
    metadata: dict[str, object] = {
        "save_best_by": save_best_by,
        "best_type": best_type,
        "selection_metric": best_type if best_type != "primary" else save_best_by,
        "relative_score_alpha": relative_score_alpha,
        "require_better_than_baseline_for_best": require_better_than_baseline_for_best,
        "qos_stable_mode_delta": qos_stable_mode_delta,
        "best_validation_episode": int(best_row["episode"]),
        "best_relative_score": best_relative_score,
        "dqn_lo_cancellations_mean": float(best_row["lo_cancellations_mean"]),
        "baseline_lo_cancellations_mean": float(best_row["baseline_lo_cancellations_mean"]),
        "dqn_mode_changes_mean": float(best_row["mode_changes_mean"]),
        "baseline_mode_changes_mean": float(best_row["baseline_mode_changes_mean"]),
        "normalized_delta_lo_cancellations": best_delta_lo,
        "normalized_delta_mode_changes": best_delta_mode,
        "raw_delta_lo_cancellations_mean": best_raw_delta_lo,
        "raw_delta_mode_changes_mean": best_raw_delta_mode,
        "best_model_is_better_than_baseline": best_relative_score < 0.0,
        "reward_mode": reward_mode,
        "reward_definition": reward_definition,
        "double_dqn": double_dqn,
        "found_valid_checkpoint": True,
        "selection_reason": "Best checkpoint selected by configured criterion.",
    }
    metadata.update(_qos_validation_metadata_row(best_row, qos_stable_mode_delta=qos_stable_mode_delta))
    return metadata


def _merge_counter_json(rows: list[dict[str, int | float | None]], field: str) -> dict[str, float]:
    """把多行中的 JSON 计数字段聚合为单个字典。"""

    merged: Counter[str] = Counter()
    for row in rows:
        raw = row.get(field)
        if not raw:
            continue
        data = json.loads(str(raw))
        for key, value in data.items():
            merged[str(key)] += float(value)
    return dict(sorted(merged.items()))


def _parse_hidden_layers(raw_value: str | None) -> tuple[int, ...] | None:
    """将逗号分隔的隐藏层字符串解析为整数元组。"""

    if raw_value is None or raw_value == "":
        return None
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def _parse_seed_spec(raw_value: str) -> list[int]:
    """解析 seed 规格，支持 `0:9` 与 `0,1,2`。"""

    text = raw_value.strip()
    if not text:
        return []
    seeds: list[int] = []
    for part in (item.strip() for item in text.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"seed 区间必须满足 begin<=end，收到: {part}")
            seeds.extend(range(begin, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def _build_episode_seed_schedule(
    *,
    episodes: int,
    seed: int,
    mode: str,
    train_seeds: list[int],
) -> list[int]:
    """构建每个 episode 使用的 seed 调度表。"""

    if episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    if mode == "fixed":
        return [seed for _ in range(episodes)]
    if mode == "per-episode":
        return [seed + episode for episode in range(episodes)]
    if mode == "cycle":
        if not train_seeds:
            raise ValueError("train-seed-mode=cycle 时，--train-seeds 不能为空")
        return [train_seeds[episode % len(train_seeds)] for episode in range(episodes)]
    raise ValueError(f"不支持的 train-seed-mode: {mode}")


def _relative_lc_reduction_from_validation_row(row: dict[str, int | float | None]) -> float | None:
    """从 validation row 计算相对 LC cancellation reduction。

    这里使用 baseline 与 DQN 的 lo_cancellations_mean 做归一化比较：
    reduction = (baseline - dqn) / baseline。
    值越大表示相对 baseline 的改善越明显。
    """

    baseline = row.get("baseline_lo_cancellations_mean")
    dqn = row.get("lo_cancellations_mean")
    if baseline is None or dqn is None:
        return None
    baseline_f = float(baseline)
    if baseline_f <= 0.0:
        return None
    return (baseline_f - float(dqn)) / baseline_f


def _is_elite_replay_candidate(
    row: dict[str, int | float | None],
    *,
    current_reduction: float | None,
    current_best_reduction: float | None,
    elite_score_min: float,
    elite_score_ratio: float,
    elite_max_mode_delta: float,
    elite_require_no_hi_miss: bool,
    elite_require_qos_stable: bool,
) -> tuple[bool, float | None, str]:
    """判断当前 validation checkpoint 是否应把 recent transitions 加入 elite replay。"""

    if current_reduction is None:
        return False, None, "no_reduction"
    if current_reduction <= 0.0:
        return False, None, "non_positive_reduction"

    if elite_require_no_hi_miss:
        deadline_misses = float(row.get("deadline_misses_sum", 0.0) or 0.0)
        if deadline_misses > 0.0:
            return False, None, "hi_deadline_miss"

    if elite_require_qos_stable:
        baseline_mode = float(row.get("baseline_mode_changes_mean", 0.0) or 0.0)
        dqn_mode = float(row.get("mode_changes_mean", 0.0) or 0.0)
        denom = max(abs(baseline_mode), 1e-9)
        mode_delta_ratio = (dqn_mode - baseline_mode) / denom
        if mode_delta_ratio > elite_max_mode_delta:
            return False, None, "mode_delta_too_large"

    best_for_threshold = (
        current_best_reduction if current_best_reduction is not None else current_reduction
    )
    threshold = max(float(elite_score_min), float(elite_score_ratio) * float(best_for_threshold))
    if current_reduction + 1e-12 < threshold:
        return False, threshold, "below_threshold"

    return True, threshold, "accepted"


def _serialize_tasks(tasks: list[Task]) -> list[dict[str, int | str]]:
    """将任务集转换为可写入 JSON 的结构。"""

    return [
        {
            "name": task.name,
            "period": task.period,
            "deadline": task.deadline,
            "c_lo": task.c_lo,
            "c_hi": task.c_hi,
            "criticality": task.criticality.value,
        }
        for task in tasks
    ]


def _get_noop_action_id(env) -> int | None:
    """从环境动作空间中解析显式 noop 的 action_id。"""

    for action in env._actions:  # noqa: SLF001
        if bool(getattr(action, "is_noop", False)):
            return int(action.action_id)
    return None


def _get_increase_action_ids(env) -> tuple[int, ...]:
    """从动作空间中提取静态 increase-only 动作编号。

    口径说明：
    - 仅保留 `increase_idx` 非空且 `decrease_indices` 为空的动作；
    - 显式 noop 不纳入；
    - transfer / pair / triple 等含 decrease 语义的动作不纳入。
    """

    increase_ids: list[int] = []
    for action in env._actions:  # noqa: SLF001
        if bool(getattr(action, "is_noop", False)):
            continue
        increase_idx = getattr(action, "increase_idx", None)
        decrease_indices = tuple(getattr(action, "decrease_indices", ()) or ())
        if increase_idx is not None and len(decrease_indices) == 0:
            increase_ids.append(int(action.action_id))
    return tuple(sorted(increase_ids))


def _noop_q_diagnostics_to_row(agent: DqnBudgetAgent, states: list[tuple[float, ...]], masks: list[tuple[bool, ...]]) -> dict[str, float | int | None]:
    """把采集到的 validation 决策状态转换为 CSV 可写字段。

    `states` 与 `masks` 只来自 agent 做 greedy 决策前的同一时刻：
    - state 表示 policy network 实际看到的 observation；
    - mask 表示同一 observation 下环境允许的合法动作集合。

    这里不对 Q 值结果做额外修正，只调用 `DqnBudgetAgent.compute_noop_q_diagnostics()`，
    保持文档要求的诊断口径集中在 agent 内部。
    """

    if states:
        state_tensor = torch.tensor(states, dtype=torch.float32, device=agent.device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=agent.device)
    else:
        state_tensor = torch.empty((0, agent.observation_dim), dtype=torch.float32, device=agent.device)
        mask_tensor = torch.empty((0, agent.action_dim), dtype=torch.bool, device=agent.device)
    diagnostics = agent.compute_noop_q_diagnostics(state_tensor, mask_tensor)
    return {
        "noop_q_mean": diagnostics.noop_q_mean,
        "noop_q_std": diagnostics.noop_q_std,
        "noop_q_rank_mean": diagnostics.noop_q_rank_mean,
        "noop_q_rank_median": diagnostics.noop_q_rank_median,
        "noop_q_rank_min": diagnostics.noop_q_rank_min,
        "noop_q_rank_max": diagnostics.noop_q_rank_max,
        "noop_q_margin_to_best_mean": diagnostics.noop_q_margin_to_best_mean,
        "noop_q_is_best_rate": diagnostics.noop_q_is_best_rate,
        "noop_valid_rate": diagnostics.noop_valid_rate,
        "noop_q_sample_count": diagnostics.sample_count,
    }


def _mean_optional_metric(rows: list[dict[str, int | float | None]], key: str) -> float | None:
    """对可能为空的 noop Q 诊断字段求均值。

    baseline 行没有 DQN Q 值，且没有显式 noop 或 noop 全部无效时相关字段会是 None。
    因此聚合时只对真实数值求均值；若所有 seed 都为空，则继续输出 None。
    """

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    """计算分位数（线性插值），用于 episode 级特征诊断。

    这里不引入额外依赖，直接用稳定的线性插值实现：
    - q 取值范围是 [0, 1]；
    - 当样本为空时返回 0.0，保持 CSV 字段始终可写。
    """

    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - float(lo)
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """写入 jsonl 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_rows_from_runtime(result: SimulationResult) -> list[dict]:
    """将 runtime 结果转换为逐行 trace。"""

    rows: list[dict] = []
    for tick in result.trace:
        rows.append(
            {
                "event": "schedule_tick",
                "time": tick.time,
                "executing_task": tick.executing_task,
                "executing_release_index": tick.executing_release_index,
                "mode": tick.mode.name,
            }
        )
    rows.extend(result.debug_events)
    for miss in result.deadline_misses:
        rows.append(
            {
                "event": "deadline_miss",
                "task": miss.task,
                "release_index": miss.release_index,
                "release_time": miss.release_time,
                "absolute_deadline": miss.absolute_deadline,
                "mode_at_miss": miss.mode_at_miss.name,
                "executed_at_miss": miss.executed_at_miss,
            }
        )
    return rows


def _run_baseline_validation_seed_worker(args_tuple: tuple[ExperimentConfig, int, int]) -> dict[str, int | float | None]:
    """运行单个 validation seed 的 AMC+ baseline 仿真。

    这里必须使用模块顶层函数，而不能把逻辑写成 `_run_validation()` 的内部闭包。
    原因是 macOS 的 multiprocessing 默认采用 `spawn` 启动方式，
    子进程只能 pickle 并导入模块顶层对象；嵌套函数通常无法被子进程恢复。
    """

    experiment_config, seed, validation_end_time = args_tuple
    bundle = resolve_experiment_bundle(experiment_config, seed)
    baseline_result = simulate_ordered_taskset_event_driven(
        ordered_tasks=list(bundle.ordered_tasks),
        scenario=bundle.scenario,
        config=RuntimeConfig(
            end_time=validation_end_time,
            semantics=RuntimeSemantics.AMC_PLUS,
        ),
    )
    baseline_service_metrics = compute_service_quality_metrics(baseline_result)
    # worker 只返回单个 seed 的原始计数，主进程统一负责做均值聚合，
    # 这样可以保证串行路径和并行路径共用同一套聚合口径。
    return {
        "mode_changes": baseline_result.mode_change_count(),
        "lo_cancellations": baseline_result.lo_job_cancellation_count(),
        **service_metrics_to_row(baseline_service_metrics),
    }


def _evaluate_agent_on_validation_seed(
    *,
    agent: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    seed: int,
    validation_end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    feature_config: FeatureConfig,
    max_q_diagnostic_samples: int,
    constraint_guided_pair_top_k_risk: int,
    constraint_guided_pair_top_k_decrease: int,
    constraint_guided_pair_prefer_lo: bool,
    constraint_guided_pair_include_hi_risk_boost: bool,
    constraint_guided_pair_allow_increase_only_when_safe: bool,
    enable_residual_safety_fallback: bool,
    residual_guard_hi_pressure_delta_limit: float,
    residual_guard_hi_pressure_abs_limit: float,
    residual_guard_reject_decrease_pressure_threshold: float,
    residual_guard_use_hi_pressure_max: bool,
    log_validation_policy_actions: bool,
) -> dict[str, int | float | None]:
    """评估一个 validation seed，并返回该 seed 的完整 DQN 指标。

    这个 helper 是串行路径与并行路径共享的唯一统计实现：
    - 串行路径直接复用主进程中的 `agent`；
    - 并行路径在 worker 中先从磁盘加载 CPU agent，再调用本函数。

    这样做的目的是让两条路径在动作选择、环境推进、指标累加和 rate 计算上
    保持严格一致，避免未来只改了一条路径导致 validation 口径漂移。
    """

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=validation_end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
        enable_residual_safety_fallback=enable_residual_safety_fallback,
        residual_guard_hi_pressure_delta_limit=residual_guard_hi_pressure_delta_limit,
        residual_guard_hi_pressure_abs_limit=residual_guard_hi_pressure_abs_limit,
        residual_guard_reject_decrease_pressure_threshold=residual_guard_reject_decrease_pressure_threshold,
        residual_guard_use_hi_pressure_max=residual_guard_use_hi_pressure_max,
    )
    obs = env.reset(seed=seed)
    done = False
    # 下面这批变量全部是“单个 validation seed 的原始计数器”，
    # 最终由 `_run_validation()` 跨 seed 做统一均值聚合。
    accepted_actions = 0
    rejected_actions = 0
    step_count = 0
    selected_action_count = 0
    noop_actions = 0
    explicit_noop_actions = 0
    total_reward = 0.0
    last_info: dict[str, int | float | str | bool | None] = {
        "mode_changes": 0,
        "lo_cancellations": 0,
        "deadline_misses": 0,
    }
    # 保存每次 validation 决策前的状态和合法动作 mask，用于结束后统一诊断 explicit noop Q 值。
    # 这里不改变 agent 的动作选择，只额外记录 policy network 当时实际看到的输入。
    diagnostic_states: list[tuple[float, ...]] = []
    diagnostic_valid_masks: list[tuple[bool, ...]] = []
    validation_action_counter: Counter[int] = Counter()
    validation_action_accepted_counter: Counter[int] = Counter()
    validation_action_rejected_counter: Counter[int] = Counter()
    validation_action_type_counter: Counter[str] = Counter()
    validation_resolved_increase_task_counter: Counter[str] = Counter()
    validation_resolved_decrease_task_counter: Counter[str] = Counter()
    validation_action_reward_sum: defaultdict[int, float] = defaultdict(float)
    validation_action_lo_delta_sum: defaultdict[int, float] = defaultdict(float)
    validation_action_mode_delta_sum: defaultdict[int, float] = defaultdict(float)
    validation_action_is_increase_sum: defaultdict[int, int] = defaultdict(int)
    validation_action_is_decrease_sum: defaultdict[int, int] = defaultdict(int)
    validation_action_is_transfer_sum: defaultdict[int, int] = defaultdict(int)
    validation_action_decrease_hits_hi_sum: defaultdict[int, int] = defaultdict(int)
    validation_action_decrease_hits_lo_sum: defaultdict[int, int] = defaultdict(int)
    validation_action_unsafe_decrease_sum: defaultdict[int, int] = defaultdict(int)

    while not done:
        # step_count 明确定义为“完成了多少次 agent 决策循环”，
        # 因此所有 per-step rate 的分母都必须统一使用它。
        step_count += 1
        mask = env.valid_action_mask()
        # dynamic_v1 是状态相关特征，必须在每次决策前刷新。
        if agent.q_network_type == "action_aware" and agent.action_feature_mode == "dynamic_v1":
            action_features = env.get_action_feature_matrix(agent.action_feature_mode)
            action_feature_names = env.get_action_feature_names(agent.action_feature_mode)
            agent.set_action_features(action_features, action_feature_names)
        if len(diagnostic_states) < max_q_diagnostic_samples:
            diagnostic_states.append(tuple(float(value) for value in obs.state_vector))
            diagnostic_valid_masks.append(tuple(bool(value) for value in mask))
        action_id = agent.select_action_id(
            obs.state_vector,
            valid_action_mask=mask,
            training=False,
        )
        # selected_action_count 只表示 agent 是否产生了离散动作编号，
        # 不表示该动作是否最终被环境接受。
        selected_action_count += int(action_id is not None)
        result = env.step(action_id)
        total_reward += result.reward
        if log_validation_policy_actions and action_id is not None:
            action_key = int(action_id)
            validation_action_counter[action_key] += 1
            accepted = bool(result.info.get("accepted", False))
            if accepted:
                validation_action_accepted_counter[action_key] += 1
            else:
                validation_action_rejected_counter[action_key] += 1
            action_meta = env._actions[action_key]
            if bool(getattr(action_meta, "is_noop", False)):
                action_type = "noop"
            elif getattr(action_meta, "residual_action_type", None):
                action_type = str(action_meta.residual_action_type)
            else:
                action_type = str(action_meta.action_space_type)
            validation_action_type_counter[f"{action_key}:{action_type}"] += 1
            resolved_increase_task = result.info.get("resolved_increase_task")
            if resolved_increase_task is not None:
                validation_resolved_increase_task_counter[f"{action_key}:{resolved_increase_task}"] += 1
            for task_name in (result.info.get("resolved_decrease_tasks") or ()):
                validation_resolved_decrease_task_counter[f"{action_key}:{task_name}"] += 1
            validation_action_reward_sum[action_key] += float(result.reward)
            validation_action_lo_delta_sum[action_key] += float(result.info.get("delta_lo_cancellations", 0.0))
            validation_action_mode_delta_sum[action_key] += float(result.info.get("delta_mode_changes", 0.0))
            validation_action_is_increase_sum[action_key] += int(bool(result.info.get("is_increase_action", False)))
            validation_action_is_decrease_sum[action_key] += int(bool(result.info.get("is_decrease_action", False)))
            validation_action_is_transfer_sum[action_key] += int(bool(result.info.get("is_transfer_action", False)))
            validation_action_decrease_hits_hi_sum[action_key] += int(bool(result.info.get("decrease_hits_hi", False)))
            validation_action_decrease_hits_lo_sum[action_key] += int(bool(result.info.get("decrease_hits_lo", False)))
            validation_action_unsafe_decrease_sum[action_key] += int(bool(result.info.get("unsafe_decrease", False)))

        is_noop = bool(result.info.get("is_noop", False))
        if is_noop:
            noop_actions += 1
            if bool(result.info.get("is_explicit_noop_action", False)):
                explicit_noop_actions += 1

        if action_id is not None:
            if bool(result.info.get("accepted")):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done
        last_info = result.info

    runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
    service_metrics = compute_service_quality_metrics(runtime_result)
    debug_stats = env.debug_statistics()
    row = {
        "mode_changes": int(last_info.get("mode_changes", 0)),
        "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
        "deadline_misses": int(last_info.get("deadline_misses", 0)),
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "step_count": step_count,
        "selected_action_count": selected_action_count,
        "noop_actions": noop_actions,
        "explicit_noop_actions": explicit_noop_actions,
        # 所有 rate 一律只使用 step_count 做分母，避免一个 step 被重复计入多个类别后
        # 破坏概率解释。例如显式 noop 既可能是 accepted，也必须计入 noop。
        "noop_action_rate": noop_actions / step_count if step_count > 0 else 0.0,
        "explicit_noop_action_rate": explicit_noop_actions / step_count if step_count > 0 else 0.0,
        "accepted_action_rate": accepted_actions / step_count if step_count > 0 else 0.0,
        "rejected_action_rate": rejected_actions / step_count if step_count > 0 else 0.0,
        "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
        "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
        "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
        "reward": float(total_reward),
        "observation_mode": str(last_info.get("observation_mode", feature_config.observation_mode)),
        "state_dim": int(last_info.get("state_dim", len(obs.state_vector))),
        **service_metrics_to_row(service_metrics),
    }
    row.update(_noop_q_diagnostics_to_row(agent, diagnostic_states, diagnostic_valid_masks))
    if log_validation_policy_actions:
        action_definitions = {
            str(action.action_id): describe_budget_action(action)
            for action in env._actions
        }
        row["validation_action_definitions_json"] = json.dumps(action_definitions, ensure_ascii=False, sort_keys=True)
        row["validation_action_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_accepted_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_accepted_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_rejected_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_rejected_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_type_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_type_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_resolved_increase_task_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_resolved_increase_task_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_resolved_decrease_task_hist_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_resolved_decrease_task_counter.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_reward_sum_json"] = json.dumps(
            {str(k): float(v) for k, v in validation_action_reward_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_lo_delta_sum_json"] = json.dumps(
            {str(k): float(v) for k, v in validation_action_lo_delta_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_mode_delta_sum_json"] = json.dumps(
            {str(k): float(v) for k, v in validation_action_mode_delta_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_is_increase_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_is_increase_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_is_decrease_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_is_decrease_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_is_transfer_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_is_transfer_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_decrease_hits_hi_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_decrease_hits_hi_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_decrease_hits_lo_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_decrease_hits_lo_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        row["validation_action_unsafe_decrease_sum_json"] = json.dumps(
            {str(k): int(v) for k, v in validation_action_unsafe_decrease_sum.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    return row


def _run_dqn_validation_seed_worker(
    args_tuple: tuple[
        str,
        ExperimentConfig,
        int,
        int,
        int,
        str,
        str,
        float,
        float,
        bool,
        float,
        bool,
        str,
        FeatureConfig,
        int,
        int,
        int,
        bool,
        bool,
        bool,
        bool,
        float,
        float,
        float,
        bool,
        bool,
    ],
) -> dict[str, int | float | None]:
    """运行单个 validation seed 的 DQN policy evaluation。

    并行 validation 不直接把主进程里的 agent 实例发送给子进程，
    因为主进程中的 agent 可能绑定了 MPS/GPU 设备状态。这里严格按文档要求：
    先在主进程保存模型快照，再由子进程用 CPU 设备重新加载，只做推理与环境仿真。
    """

    (
        model_path,
        experiment_config,
        seed,
        validation_end_time,
        agent_period,
        reward_mode,
        action_space,
        budget_increase_ratio,
        budget_decrease_ratio,
        include_explicit_noop,
        budget_floor_ratio,
        forbid_decreasing_hi_budgets,
        mask_detail_mode,
        feature_config,
        max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe,
        enable_residual_safety_fallback,
        residual_guard_hi_pressure_delta_limit,
        residual_guard_hi_pressure_abs_limit,
        residual_guard_reject_decrease_pressure_threshold,
        residual_guard_use_hi_pressure_max,
        log_validation_policy_actions,
    ) = args_tuple
    agent = DqnBudgetAgent.load(Path(model_path), device="cpu")
    return _evaluate_agent_on_validation_seed(
        agent=agent,
        experiment_config=experiment_config,
        seed=seed,
        validation_end_time=validation_end_time,
        agent_period=agent_period,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        feature_config=feature_config,
        max_q_diagnostic_samples=max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
        enable_residual_safety_fallback=enable_residual_safety_fallback,
        residual_guard_hi_pressure_delta_limit=residual_guard_hi_pressure_delta_limit,
        residual_guard_hi_pressure_abs_limit=residual_guard_hi_pressure_abs_limit,
        residual_guard_reject_decrease_pressure_threshold=residual_guard_reject_decrease_pressure_threshold,
        residual_guard_use_hi_pressure_max=residual_guard_use_hi_pressure_max,
        log_validation_policy_actions=log_validation_policy_actions,
    )


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """按 CLI 参数构建实验配置。"""

    if args.workload == "small":
        return (
            build_small_nominal_experiment_config()
            if args.scenario == "nominal"
            else build_small_stress_experiment_config()
        )
    if args.workload == "rtss11":
        return build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            hi_overrun_prob=args.hi_overrun_prob,
            lo_overrun_prob=args.lo_overrun_prob,
            lo_overrun_factor=args.lo_overrun_factor,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
        )
    if args.workload == "mc_fairgen":
        return build_mc_fairgen_experiment_config(
            mode=args.mc_fairgen_mode,
            num_tasks=args.mc_fairgen_num_tasks,
            hi_ratio=args.mc_fairgen_hi_ratio,
            period_source=args.mc_fairgen_period_source,
            period_scale=args.mc_fairgen_period_scale,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            u_hi_lo_min=args.mc_fairgen_u_hi_lo_min,
            u_hi_lo_max=args.mc_fairgen_u_hi_lo_max,
            u_hi_hi_min=args.mc_fairgen_u_hi_hi_min,
            u_hi_hi_max=args.mc_fairgen_u_hi_hi_max,
            u_lo_lo_min=args.mc_fairgen_u_lo_lo_min,
            u_lo_lo_max=args.mc_fairgen_u_lo_lo_max,
            hi_budget_rho_min=args.mc_fairgen_hi_budget_rho_min,
            hi_budget_rho_max=args.mc_fairgen_hi_budget_rho_max,
            lo_budget_rho_min=args.mc_fairgen_lo_budget_rho_min,
            lo_budget_rho_max=args.mc_fairgen_lo_budget_rho_max,
            hi_overrun_prob=args.mc_fairgen_hi_overrun_prob,
            lo_overrun_prob=args.mc_fairgen_lo_overrun_prob,
            hi_overrun_factor_min=args.mc_fairgen_hi_overrun_factor_min,
            hi_overrun_factor_max=args.mc_fairgen_hi_overrun_factor_max,
            lo_overrun_factor_min=args.mc_fairgen_lo_overrun_factor_min,
            lo_overrun_factor_max=args.mc_fairgen_lo_overrun_factor_max,
        )
    if args.workload == "automotive":
        return build_automotive_experiment_config(
            num_runnables=args.automotive_num_runnables,
            mode=args.automotive_mode,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            learnable_target_budget_util_min=args.learnable_target_budget_util_min,
            learnable_target_budget_util_max=args.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=args.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=args.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=args.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=args.learnable_lo_budget_rho_max,
            budget_floor_ratio=args.budget_floor_ratio,
        )
    raise ValueError(f"unsupported workload: {args.workload}")


def _run_validation(
    *,
    agent: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    validation_seeds: list[int],
    validation_end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    feature_config: FeatureConfig = FeatureConfig(),
    validation_workers: int = 1,
    baseline_cache: dict[str, float] | None = None,
    max_q_diagnostic_samples: int = 1000,
    constraint_guided_pair_top_k_risk: int = 3,
    constraint_guided_pair_top_k_decrease: int = 5,
    constraint_guided_pair_prefer_lo: bool = False,
    constraint_guided_pair_include_hi_risk_boost: bool = False,
    constraint_guided_pair_allow_increase_only_when_safe: bool = False,
    enable_residual_safety_fallback: bool = False,
    residual_guard_hi_pressure_delta_limit: float = 0.03,
    residual_guard_hi_pressure_abs_limit: float = 0.30,
    residual_guard_reject_decrease_pressure_threshold: float = 0.05,
    residual_guard_use_hi_pressure_max: bool = False,
    log_validation_policy_actions: bool = False,
) -> tuple[dict[str, int | float | None], dict[str, int | float | None], bool]:
    """在验证集上评估当前 agent，并返回聚合指标。"""

    used_baseline_cache = baseline_cache is not None

    if baseline_cache is None:
        # baseline 的输入参数只依赖 experiment_config、seed 和验证时长，
        # 因此最适合按 seed 拆成完全独立的 worker 任务。
        baseline_worker_args = [
            (experiment_config, seed, validation_end_time)
            for seed in validation_seeds
        ]
        if validation_workers == 1:
            baseline_rows = [
                _run_baseline_validation_seed_worker(item)
                for item in baseline_worker_args
            ]
        else:
            try:
                with ProcessPoolExecutor(max_workers=validation_workers) as executor:
                    baseline_rows = list(executor.map(_run_baseline_validation_seed_worker, baseline_worker_args))
            except PermissionError:
                # 某些受限执行环境（如沙箱）禁止创建进程信号量，这里回退到串行路径，
                # 保持统计口径不变，仅牺牲并行加速。
                baseline_rows = [_run_baseline_validation_seed_worker(item) for item in baseline_worker_args]
        baseline_cache = {
            "baseline_mode_changes_mean": sum(row["mode_changes"] for row in baseline_rows) / len(validation_seeds),
            "baseline_lo_cancellations_mean": (
                sum(row["lo_cancellations"] for row in baseline_rows) / len(validation_seeds)
            ),
            "baseline_released_lo_jobs_mean": (
                sum(float(row["released_lo_jobs"]) for row in baseline_rows) / len(validation_seeds)
            ),
            "baseline_lc_service_loss_mean": (
                sum(float(row["lc_service_loss"]) for row in baseline_rows) / len(validation_seeds)
            ),
            "baseline_lc_qos_mean": (
                sum(float(row["lc_qos"]) for row in baseline_rows) / len(validation_seeds)
            ),
            "baseline_min_lc_service_mean": mean_optional_service_metric(baseline_rows, "min_lc_service"),
            "baseline_hi_deadline_misses_sum": sum(int(float(row["hi_deadline_misses"])) for row in baseline_rows),
            "baseline_lo_deadline_misses_sum": sum(int(float(row["lo_deadline_misses"])) for row in baseline_rows),
            "baseline_budget_adjust_count_mean": (
                sum(float(row["budget_adjust_count"]) for row in baseline_rows) / len(validation_seeds)
            ),
            "baseline_mean_abs_budget_change_mean": mean_optional_service_metric(
                baseline_rows,
                "mean_abs_budget_change",
            ),
        }

    if validation_workers == 1:
        # 串行路径直接复用当前主进程中的 agent，避免默认配置下每次 validation 都发生 save/load 开销。
        dqn_rows = [
            _evaluate_agent_on_validation_seed(
                agent=agent,
                experiment_config=experiment_config,
                seed=seed,
                validation_end_time=validation_end_time,
                agent_period=agent_period,
                reward_mode=reward_mode,
                action_space=action_space,
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                include_explicit_noop=include_explicit_noop,
                budget_floor_ratio=budget_floor_ratio,
                forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                mask_detail_mode=mask_detail_mode,
                feature_config=feature_config,
                max_q_diagnostic_samples=max_q_diagnostic_samples,
                constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
                enable_residual_safety_fallback=enable_residual_safety_fallback,
                residual_guard_hi_pressure_delta_limit=residual_guard_hi_pressure_delta_limit,
                residual_guard_hi_pressure_abs_limit=residual_guard_hi_pressure_abs_limit,
                residual_guard_reject_decrease_pressure_threshold=(
                    residual_guard_reject_decrease_pressure_threshold
                ),
                residual_guard_use_hi_pressure_max=residual_guard_use_hi_pressure_max,
                log_validation_policy_actions=log_validation_policy_actions,
            )
            for seed in validation_seeds
        ]
    else:
        # 并行路径在 validation 开始时冻结一次模型快照，确保所有 worker 使用完全相同的策略参数。
        with TemporaryDirectory(prefix="dqn_validation_") as tmp_dir:
            model_path = Path(tmp_dir) / "policy_snapshot.pt"
            agent.save(model_path)
            dqn_worker_args = [
                (
                    str(model_path),
                    experiment_config,
                    seed,
                    validation_end_time,
                    agent_period,
                    reward_mode,
                    action_space,
                    budget_increase_ratio,
                    budget_decrease_ratio,
                    include_explicit_noop,
                    budget_floor_ratio,
                    forbid_decreasing_hi_budgets,
                    mask_detail_mode,
                    feature_config,
                    max_q_diagnostic_samples,
                    constraint_guided_pair_top_k_risk,
                    constraint_guided_pair_top_k_decrease,
                    constraint_guided_pair_prefer_lo,
                    constraint_guided_pair_include_hi_risk_boost,
                    constraint_guided_pair_allow_increase_only_when_safe,
                    enable_residual_safety_fallback,
                    residual_guard_hi_pressure_delta_limit,
                    residual_guard_hi_pressure_abs_limit,
                    residual_guard_reject_decrease_pressure_threshold,
                    residual_guard_use_hi_pressure_max,
                    log_validation_policy_actions,
                )
                for seed in validation_seeds
            ]
            try:
                with ProcessPoolExecutor(max_workers=validation_workers) as executor:
                    dqn_rows = list(executor.map(_run_dqn_validation_seed_worker, dqn_worker_args))
            except PermissionError:
                # 与 baseline 并行分支同理：受限环境下保持口径一致地退回串行。
                dqn_rows = [_run_dqn_validation_seed_worker(item) for item in dqn_worker_args]

    seed_count = len(validation_seeds)
    baseline_mode_changes_mean = float(baseline_cache["baseline_mode_changes_mean"])
    baseline_lo_cancellations_mean = float(baseline_cache["baseline_lo_cancellations_mean"])
    baseline_lc_service_loss_mean = float(baseline_cache["baseline_lc_service_loss_mean"])
    baseline_lc_qos_mean = float(baseline_cache["baseline_lc_qos_mean"])
    mode_delta_sum = sum(row["mode_changes"] for row in dqn_rows) - baseline_mode_changes_mean * seed_count
    cancel_delta_sum = sum(row["lo_cancellations"] for row in dqn_rows) - baseline_lo_cancellations_mean * seed_count
    released_lo_jobs_mean = sum(float(row["released_lo_jobs"]) for row in dqn_rows) / seed_count
    lc_service_loss_mean = sum(float(row["lc_service_loss"]) for row in dqn_rows) / seed_count
    lc_qos_mean = sum(float(row["lc_qos"]) for row in dqn_rows) / seed_count
    validation_row = {
        "validation_seed_count": seed_count,
        "deadline_misses_sum": sum(int(row["deadline_misses"]) for row in dqn_rows),
        "mode_changes_mean": sum(row["mode_changes"] for row in dqn_rows) / seed_count,
        "lo_cancellations_mean": sum(row["lo_cancellations"] for row in dqn_rows) / seed_count,
        "baseline_mode_changes_mean": baseline_mode_changes_mean,
        "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
        "dqn_mode_changes_delta_mean": mode_delta_sum / seed_count,
        "dqn_lo_cancellations_delta_mean": cancel_delta_sum / seed_count,
        "accepted_actions_mean": sum(row["accepted_actions"] for row in dqn_rows) / seed_count,
        "rejected_actions_mean": sum(row["rejected_actions"] for row in dqn_rows) / seed_count,
        "step_count_mean": sum(row["step_count"] for row in dqn_rows) / seed_count,
        "selected_action_count_mean": sum(row["selected_action_count"] for row in dqn_rows) / seed_count,
        "noop_actions_mean": sum(row["noop_actions"] for row in dqn_rows) / seed_count,
        "explicit_noop_actions_mean": sum(row["explicit_noop_actions"] for row in dqn_rows) / seed_count,
        "noop_action_rate_mean": sum(row["noop_action_rate"] for row in dqn_rows) / seed_count,
        "explicit_noop_action_rate_mean": sum(row["explicit_noop_action_rate"] for row in dqn_rows) / seed_count,
        "accepted_action_rate_mean": sum(row["accepted_action_rate"] for row in dqn_rows) / seed_count,
        "rejected_action_rate_mean": sum(row["rejected_action_rate"] for row in dqn_rows) / seed_count,
        "valid_action_count_mean": sum(row["valid_action_count_mean"] for row in dqn_rows) / seed_count,
        "masked_action_count_mean": sum(row["masked_action_count_mean"] for row in dqn_rows) / seed_count,
        "no_safe_action_steps_mean": sum(row["no_safe_action_steps"] for row in dqn_rows) / seed_count,
        "reward_mean": sum(row["reward"] for row in dqn_rows) / seed_count,
        "observation_mode": str(feature_config.observation_mode),
        "state_dim_mean": sum(row["state_dim"] for row in dqn_rows) / seed_count,
        "released_lo_jobs_mean": released_lo_jobs_mean,
        "cancelled_lo_jobs_mean": sum(float(row["cancelled_lo_jobs"]) for row in dqn_rows) / seed_count,
        "completed_lo_jobs_mean": sum(float(row["completed_lo_jobs"]) for row in dqn_rows) / seed_count,
        "lo_deadline_misses_sum": sum(int(float(row["lo_deadline_misses"])) for row in dqn_rows),
        "hi_deadline_misses_sum": sum(int(float(row["hi_deadline_misses"])) for row in dqn_rows),
        "lc_service_loss_mean": lc_service_loss_mean,
        "lc_qos_mean": lc_qos_mean,
        "min_lc_service_mean": mean_optional_service_metric(dqn_rows, "min_lc_service"),
        "budget_adjust_count_mean": sum(float(row["budget_adjust_count"]) for row in dqn_rows) / seed_count,
        "mean_abs_budget_change_mean": mean_optional_service_metric(dqn_rows, "mean_abs_budget_change"),
        "baseline_released_lo_jobs_mean": float(baseline_cache["baseline_released_lo_jobs_mean"]),
        "baseline_lc_service_loss_mean": baseline_lc_service_loss_mean,
        "baseline_lc_qos_mean": baseline_lc_qos_mean,
        "baseline_min_lc_service_mean": baseline_cache["baseline_min_lc_service_mean"],
        "baseline_hi_deadline_misses_sum": int(float(baseline_cache["baseline_hi_deadline_misses_sum"])),
        "relative_lc_loss_reduction": safe_relative_reduction(baseline_lc_service_loss_mean, lc_service_loss_mean),
        "lc_service_loss_delta_mean": lc_service_loss_mean - baseline_lc_service_loss_mean,
        "lc_qos_delta_mean": lc_qos_mean - baseline_lc_qos_mean,
        "mode_change_delta_ratio": (
            (sum(row["mode_changes"] for row in dqn_rows) / seed_count) - baseline_mode_changes_mean
        ) / max(1.0, baseline_mode_changes_mean),
    }
    # validation 输出采用文档第 6 节的字段名，数值为各 validation seed 诊断结果的均值；
    # `noop_q_sample_count` 代表实际参与 Q 诊断的状态样本总数，便于核对采样是否达到上限。
    for fieldname in NOOP_Q_DIAGNOSTIC_FIELDNAMES:
        if fieldname == "noop_q_sample_count":
            validation_row[fieldname] = sum(int(row[fieldname] or 0) for row in dqn_rows)
        else:
            validation_row[fieldname] = _mean_optional_metric(dqn_rows, fieldname)
    if log_validation_policy_actions:
        validation_row["policy_action_definitions_json"] = str(dqn_rows[0]["validation_action_definitions_json"])
        validation_row["policy_action_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_hist_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_accepted_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_accepted_hist_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_rejected_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_rejected_hist_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_type_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_type_hist_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_resolved_increase_task_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_resolved_increase_task_hist_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_resolved_decrease_task_hist_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_resolved_decrease_task_hist_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_reward_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_reward_sum_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_lo_delta_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_lo_delta_sum_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_mode_delta_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_mode_delta_sum_json"), ensure_ascii=False, sort_keys=True
        )
        validation_row["policy_action_is_increase_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_is_increase_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_is_decrease_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_is_decrease_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_is_transfer_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_is_transfer_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_decrease_hits_hi_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_decrease_hits_hi_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_decrease_hits_lo_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_decrease_hits_lo_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        validation_row["policy_action_unsafe_decrease_sum_json"] = json.dumps(
            _merge_counter_json(dqn_rows, "validation_action_unsafe_decrease_sum_json"),
            ensure_ascii=False,
            sort_keys=True,
        )
    return validation_row, baseline_cache, used_baseline_cache


def _is_better_validation_row(
    *,
    candidate_row: dict[str, int | float],
    best_row: dict[str, int | float] | None,
    save_best_by: str,
    relative_score_alpha: float = 1.0,
    require_better_than_baseline_for_best: bool = False,
    qos_stable_mode_delta: float = 0.05,
) -> bool:
    """判断候选验证结果是否优于当前 best。"""

    if save_best_by in {"lo_cancellations", "mode_changes", "reward", "relative_score", "pareto_relative_score"}:
        if int(candidate_row["deadline_misses_sum"]) != 0:
            return False
    use_lo_cancellations_gate = save_best_by == "lo_cancellations"
    if best_row is None:
        if use_lo_cancellations_gate:
            return float(candidate_row["mode_changes_mean"]) <= float(
                candidate_row["baseline_mode_changes_mean"]
            )
        return True
    if save_best_by in {"lo_cancellations", "mode_changes", "reward", "relative_score", "pareto_relative_score"} and int(best_row["deadline_misses_sum"]) != 0:
        return True
    if use_lo_cancellations_gate:
        candidate_mode_ok = float(candidate_row["mode_changes_mean"]) <= float(
            candidate_row["baseline_mode_changes_mean"]
        )
        if not candidate_mode_ok:
            return False
        best_mode_ok = float(best_row["mode_changes_mean"]) <= float(best_row["baseline_mode_changes_mean"])
        if not best_mode_ok:
            return True
        return float(candidate_row["lo_cancellations_mean"]) < float(best_row["lo_cancellations_mean"])
    if save_best_by == "reward":
        # 阶段 3：reward 指标方向应为“越大越好”，与 cancellation/mode-change 相反。
        return float(candidate_row["reward_mean"]) > float(best_row["reward_mean"])
    if save_best_by == "qos_stable":
        if not is_qos_stable_valid(candidate_row, delta=qos_stable_mode_delta):
            return False
        if best_row is None or not is_qos_stable_valid(best_row, delta=qos_stable_mode_delta):
            return True
        return qos_sort_key(candidate_row) < qos_sort_key(best_row)
    if save_best_by == "conservative_qos":
        if not is_conservative_qos_valid(candidate_row):
            return False
        if best_row is None or not is_conservative_qos_valid(best_row):
            return True
        return qos_sort_key(candidate_row) < qos_sort_key(best_row)
    if save_best_by == "qos_best":
        if not is_qos_best_valid(candidate_row):
            return False
        if best_row is None or not is_qos_best_valid(best_row):
            return True
        return qos_sort_key(candidate_row) < qos_sort_key(best_row)
    if save_best_by == "relative_score":
        # relative_score 越小越好，<0 表示综合优于 baseline。
        # 本轮要求：即便 relative_score>=0，也要保留“验证集里最好的那个”checkpoint。
        _ = (relative_score_alpha, require_better_than_baseline_for_best)
        candidate_score = float(candidate_row["relative_score"])
        return candidate_score < float(best_row["relative_score"])
    if save_best_by == "pareto_relative_score":
        # 更温和的 Pareto 风格选模分数：
        # 1) 保留原有 relative_score（越小越好）作为主项；
        # 2) 对“比 baseline 更差”的维度施加软惩罚，而不是硬性 +inf；
        # 3) 两个惩罚项都基于归一化差值，保持与 relative_score 同量纲。
        #
        # 公式：
        # score = relative_score
        #       + 10 * max(0, relative_delta_mode_changes)
        #       + 10 * max(0, relative_delta_lo_cancellations)
        #
        # 解释：
        # - 当候选同时不劣于 baseline（两个 delta 都 <=0）时，惩罚为 0，退化为 relative_score 比较；
        # - 当某一维劣于 baseline 时，仍允许进入比较，但会因软惩罚降低被选中概率；
        # - 这比 strict gate 更平滑，适合你要求的“温和版本”。
        _ = (relative_score_alpha, require_better_than_baseline_for_best)
        candidate_score = (
            float(candidate_row["relative_score"])
            + 10.0 * max(0.0, float(candidate_row["relative_delta_mode_changes"]))
            + 10.0 * max(0.0, float(candidate_row["relative_delta_lo_cancellations"]))
        )
        best_score = (
            float(best_row["relative_score"])
            + 10.0 * max(0.0, float(best_row["relative_delta_mode_changes"]))
            + 10.0 * max(0.0, float(best_row["relative_delta_lo_cancellations"]))
        )
        return candidate_score < best_score
    metric_field = {
        "lo_cancellations": "lo_cancellations_mean",
        "mode_changes": "mode_changes_mean",
    }[save_best_by]
    return float(candidate_row[metric_field]) < float(best_row[metric_field])


def _is_pareto_valid_checkpoint(row: dict[str, int | float | None]) -> bool:
    """判断一个 validation checkpoint 是否满足 Pareto-valid 条件。

    Pareto-valid 的定义严格按你给出的规则：
    1. deadline_misses_sum == 0（硬约束）；
    2. dqn_mode_changes_mean <= baseline_mode_changes_mean；
    3. dqn_lo_cancellations_mean <= baseline_lo_cancellations_mean。

    该函数只负责“是否通过筛选”，不负责在候选之间排序。
    """

    return (
        int(row["deadline_misses_sum"]) == 0
        and float(row["mode_changes_mean"]) <= float(row["baseline_mode_changes_mean"])
        and float(row["lo_cancellations_mean"]) <= float(row["baseline_lo_cancellations_mean"])
    )


def _build_validation_unified_summary_rows(
    validation_rows: list[dict[str, int | float]],
) -> list[dict[str, int | float | str]]:
    """把训练期 validation 指标转换为阶段 0/1 的统一 summary 口径。"""

    summary_rows: list[dict[str, int | float | str]] = []
    for row in validation_rows:
        baseline_mode_changes_mean = float(row["baseline_mode_changes_mean"])
        baseline_lo_cancellations_mean = float(row["baseline_lo_cancellations_mean"])
        dqn_mode_changes_mean = float(row["mode_changes_mean"])
        dqn_lo_cancellations_mean = float(row["lo_cancellations_mean"])
        baseline_lc_service_loss_mean = float(row["baseline_lc_service_loss_mean"])
        baseline_lc_qos_mean = float(row["baseline_lc_qos_mean"])
        dqn_lc_service_loss_mean = float(row["lc_service_loss_mean"])
        dqn_lc_qos_mean = float(row["lc_qos_mean"])
        mode_change_ratio: float | str
        lo_cancellation_ratio: float | str
        if dqn_mode_changes_mean == 0.0:
            mode_change_ratio = "inf" if baseline_mode_changes_mean > 0.0 else "nan"
        else:
            mode_change_ratio = baseline_mode_changes_mean / dqn_mode_changes_mean
        if dqn_lo_cancellations_mean == 0.0:
            lo_cancellation_ratio = "inf" if baseline_lo_cancellations_mean > 0.0 else "nan"
        else:
            lo_cancellation_ratio = baseline_lo_cancellations_mean / dqn_lo_cancellations_mean

        summary_rows.append(
            {
                "episode": int(row["episode"]),
                "validation_seed_count": int(row["validation_seed_count"]),
                "baseline_mode_changes_mean": baseline_mode_changes_mean,
                "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
                "baseline_lc_service_loss_mean": baseline_lc_service_loss_mean,
                "baseline_lc_qos_mean": baseline_lc_qos_mean,
                "dqn_mode_changes_mean": dqn_mode_changes_mean,
                "dqn_lo_cancellations_mean": dqn_lo_cancellations_mean,
                "dqn_lc_service_loss_mean": dqn_lc_service_loss_mean,
                "dqn_lc_qos_mean": dqn_lc_qos_mean,
                # 这三列用于与文档要求一致地直观看到“相对 baseline 的差值与综合得分”。
                "relative_score": float(row["relative_score"]),
                "relative_score_alpha": float(row["relative_score_alpha"]),
                "relative_delta_mode_changes": float(row["relative_delta_mode_changes"]),
                "relative_delta_lo_cancellations": float(row["relative_delta_lo_cancellations"]),
                "raw_delta_mode_changes": float(
                    row.get("raw_delta_mode_changes", row["dqn_mode_changes_delta_mean"])
                ),
                "raw_delta_lo_cancellations": float(
                    row.get("raw_delta_lo_cancellations", row["dqn_lo_cancellations_delta_mean"])
                ),
                "is_better_than_baseline": bool(row["is_better_than_baseline"]),
                "mode_changes_delta_vs_baseline": float(row["dqn_mode_changes_delta_mean"]),
                "lo_cancellations_delta_vs_baseline": float(row["dqn_lo_cancellations_delta_mean"]),
                "mode_change_ratio": mode_change_ratio,
                "lo_cancellation_ratio": lo_cancellation_ratio,
                "relative_lc_loss_reduction": row.get("relative_lc_loss_reduction"),
                "lc_service_loss_delta_mean": float(row["lc_service_loss_delta_mean"]),
                "lc_qos_delta_mean": float(row["lc_qos_delta_mean"]),
                "min_lc_service_mean": row.get("min_lc_service_mean"),
                "mode_change_delta_ratio": float(row["mode_change_delta_ratio"]),
                "hi_deadline_misses_sum": int(float(row["hi_deadline_misses_sum"])),
                "qos_stable_valid_delta000": is_qos_stable_valid(row, 0.00),
                "qos_stable_valid_delta005": is_qos_stable_valid(row, 0.05),
                "qos_stable_valid_delta010": is_qos_stable_valid(row, 0.10),
                "best_candidate_rank_key": str(qos_sort_key(row)),
                "accepted_action_count_mean": float(row["accepted_actions_mean"]),
                "rejected_action_count_mean": float(row["rejected_actions_mean"]),
                "noop_action_count_mean": float(row["noop_actions_mean"]),
                "noop_action_rate_mean": float(row["noop_action_rate_mean"]),
                "explicit_noop_action_rate_mean": float(row["explicit_noop_action_rate_mean"]),
                "accepted_action_rate_mean": float(row["accepted_action_rate_mean"]),
                "rejected_action_rate_mean": float(row["rejected_action_rate_mean"]),
                "masked_action_count_mean": float(row["masked_action_count_mean"]),
                "valid_action_count_mean": float(row["valid_action_count_mean"]),
                # explicit noop 的 Q 值诊断字段直接透传到 unified summary，
                # 这样验证趋势表可以同时观察动作实际选择频率与 Q 值排名变化。
                "noop_q_mean": row.get("noop_q_mean"),
                "noop_q_std": row.get("noop_q_std"),
                "noop_q_rank_mean": row.get("noop_q_rank_mean"),
                "noop_q_rank_median": row.get("noop_q_rank_median"),
                "noop_q_rank_min": row.get("noop_q_rank_min"),
                "noop_q_rank_max": row.get("noop_q_rank_max"),
                "noop_q_margin_to_best_mean": row.get("noop_q_margin_to_best_mean"),
                "noop_q_is_best_rate": row.get("noop_q_is_best_rate"),
                "noop_valid_rate": row.get("noop_valid_rate"),
                "noop_q_sample_count": row.get("noop_q_sample_count"),
            }
        )
    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    """构建正式训练 CLI 的命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--exploration-seed", type=int, default=None)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--min-replay-size", type=int, default=500)
    # Elite Replay v1：仅在普通 replay 之外追加“精英样本池”并混合采样，不修改 DQN loss 与网络结构。
    parser.add_argument("--use-elite-replay", action="store_true")
    parser.add_argument("--elite-replay-capacity", type=int, default=2000)
    parser.add_argument("--elite-replay-min-size", type=int, default=128)
    parser.add_argument("--elite-batch-size", type=int, default=8)
    parser.add_argument("--elite-score-min", type=float, default=0.05)
    parser.add_argument("--elite-score-ratio", type=float, default=0.8)
    parser.add_argument("--elite-recent-episodes", type=int, default=10)
    parser.add_argument(
        "--elite-start-episode",
        type=int,
        default=0,
        help="在该 episode 序号之前禁用 elite replay；默认 0 表示保持 Elite Replay v1 原行为。",
    )
    parser.add_argument("--elite-max-mode-delta", type=float, default=0.05)
    parser.add_argument(
        "--elite-require-no-hi-miss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--elite-require-qos-stable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--elite-max-add-per-validation", type=int, default=2000)
    # Two-Pool Elite Replay：第二层 best elite 池参数，默认全部关闭或保守值，保证旧命令兼容。
    parser.add_argument("--use-best-elite-replay", action="store_true")
    parser.add_argument("--best-elite-replay-capacity", type=int, default=2000)
    parser.add_argument("--best-elite-replay-min-size", type=int, default=128)
    parser.add_argument("--best-elite-batch-size", type=int, default=4)
    parser.add_argument("--best-elite-min-improvement", type=float, default=0.001)
    parser.add_argument("--best-elite-recent-episodes", type=int, default=10)
    parser.add_argument("--best-elite-start-episode", type=int, default=100)
    parser.add_argument("--best-elite-max-add-per-validation", type=int, default=2000)
    parser.add_argument("--best-elite-replace-on-new-best", action="store_true")
    parser.add_argument("--hidden-layers", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    # 稳定性修改 C：默认 target network 更新频率改为 5（优化步为单位）。
    # 该值可继续通过 CLI 覆盖，用于后续对照实验。
    parser.add_argument("--target-update-freq", type=int, default=5)
    parser.add_argument("--target-update-frequency", type=int, default=None)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="DQN 梯度裁剪阈值（L2 norm），用于限制反向传播后的梯度范数。",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5000)
    parser.add_argument(
        "--noop-exploration-prob",
        type=float,
        default=0.0,
        help=(
            "During epsilon exploration, if explicit noop is valid, choose noop "
            "with this probability before sampling other valid actions."
        ),
    )
    parser.add_argument(
        "--exploration-mode",
        choices=[
            "epsilon_greedy",
            "epsilon_safe_increase_mixture",
            "epsilon_increase_coverage",
            "epsilon_plateau_soft_target_balanced",
        ],
        default="epsilon_greedy",
        help=(
            "Training-time epsilon exploration action sampling mode. "
            "epsilon_greedy preserves the original behavior. "
            "epsilon_safe_increase_mixture samples from valid increase-only actions "
            "with --safe-increase-explore-prob when epsilon exploration is triggered. "
            "epsilon_increase_coverage samples the least-visited valid increase-only action "
            "with --safe-increase-explore-prob when epsilon exploration is triggered. "
            "epsilon_plateau_soft_target_balanced only mixes in coverage-balanced sampling "
            "during plateau-triggered bursts."
        ),
    )
    parser.add_argument(
        "--safe-increase-explore-prob",
        type=float,
        default=0.0,
        help=(
            "Only used when --exploration-mode epsilon_safe_increase_mixture or epsilon_increase_coverage. "
            "During epsilon exploration, probability of entering the increase-only branch."
        ),
    )
    parser.add_argument(
        "--plateau-balanced-start-episode",
        type=int,
        default=40,
        help="Minimum episode before plateau-triggered balanced bursts can be activated.",
    )
    parser.add_argument(
        "--plateau-balanced-window",
        type=int,
        default=3,
        help="Number of consecutive validation checks without best improvement before triggering a burst.",
    )
    parser.add_argument(
        "--plateau-balanced-burst-episodes",
        type=int,
        default=20,
        help="Number of training episodes for each plateau-triggered balanced exploration burst.",
    )
    parser.add_argument(
        "--plateau-balanced-mix-prob",
        type=float,
        default=0.3,
        help=(
            "During an active burst, probability of using coverage-balanced increase sampling "
            "inside epsilon exploration."
        ),
    )
    parser.add_argument(
        "--plateau-balanced-max-best-reduction",
        type=float,
        default=0.08,
        help=(
            "Only trigger plateau-balanced bursts while current best relative LC reduction is below "
            "this threshold. Set <=0 to disable this protection."
        ),
    )
    parser.add_argument(
        "--plateau-balanced-reset-counts-on-burst",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to reset target coverage counts when a new burst starts.",
    )
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Double DQN target calculation: policy net selects next action, target net evaluates it.",
    )
    parser.add_argument(
        "--max-q-diagnostic-samples",
        type=int,
        default=1000,
        help="Maximum validation decision states sampled for noop Q diagnostics per validation seed.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive", "mc_fairgen"], default="small")
    # automotive workload 允许从 CLI 显式切换 runnable 数量与 workload 语义模式，
    # 这样训练入口就不再把 automotive 固定写死为 150 + paper_like。
    parser.add_argument("--automotive-num-runnables", type=int, choices=[150, 250], default=150)
    parser.add_argument(
        "--automotive-mode",
        choices=["fast", "paper_like", "paper_exact", "paper_learnable_headroom"],
        default="paper_like",
    )
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)
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
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hi-overrun-prob", type=float, default=0.05)
    parser.add_argument("--lo-overrun-prob", type=float, default=0.10)
    parser.add_argument("--lo-overrun-factor", type=float, default=1.5)
    parser.add_argument("--log-train-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--train-seed-mode", choices=["fixed", "per-episode", "cycle"], default="fixed")
    parser.add_argument(
        "--fixed-taskset-seed",
        type=int,
        default=None,
        help=(
            "如果设置该参数，RTSS11 任务集生成固定使用该 seed；"
            "每个 episode 变化的 seed 仅用于 scenario 生成。"
        ),
    )
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--validation-seeds", type=str, default="100:129")
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--validation-end-time", type=int, default=10000)
    parser.add_argument(
        "--log-validation-policy-actions",
        action="store_true",
        help="Log per-checkpoint validation policy action histogram and resolved residual targets.",
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=1,
        help="并行 validation 的进程数；1 表示保持串行。",
    )
    parser.add_argument(
        "--dqn-device",
        type=str,
        default=None,
        help=(
            "Torch device for DQN training. Examples: cpu, cuda, cuda:0, mps. "
            "Default keeps existing behavior: mps on macOS if available, otherwise cpu. "
            "Use --dqn-device cuda to enable NVIDIA GPU training."
        ),
    )
    parser.add_argument(
        "--log-step-every",
        type=int,
        default=1,
        help="每隔多少个 global step 记录一行 step-level 日志；1 表示每步记录，0 表示关闭 step-level 日志。",
    )
    parser.add_argument(
        "--save-best-by",
        choices=[
            "lo_cancellations",
            "mode_changes",
            "reward",
            "relative_score",
            "pareto_relative_score",
            "qos_stable",
            "conservative_qos",
            "qos_best",
        ],
        default="mode_changes",
    )
    parser.add_argument("--qos-stable-mode-delta", type=float, default=0.05)
    parser.add_argument("--save-all-best-types", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=list(available_reward_modes()),
        default="mendes",
    )
    parser.add_argument(
        "--relative-score-alpha",
        type=float,
        default=1.0,
        help="relative_score 中 mode_changes 差值的权重 alpha。",
    )
    parser.add_argument(
        "--require-better-than-baseline-for-best",
        action="store_true",
        help="仅当 relative_score < 0 时才允许刷新 model_best.pt。",
    )
    parser.add_argument(
        "--action-space",
        choices=[
            "triple",
            "pair",
            "single",
            "constraint_guided_pair",
            "constraint_guided_transfer",
            "residual_ranked",
            "residual_safe_ranked",
            "residual_anchor_mc_lo_2",
            "residual_safe_adjust_15a",
        ],
        default="triple",
    )
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=5)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--constraint-guided-pair-include-hi-risk-boost", action="store_true")
    parser.add_argument(
        "--constraint-guided-pair-allow-increase-only-when-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--q-network-type",
        choices=["mlp", "action_aware"],
        default="mlp",
        help="DQN Q 网络类型：mlp 为原始 Q(s)->all actions；action_aware 为共享 Q(s,a)。",
    )
    parser.add_argument(
        "--action-feature-mode",
        choices=["static_v1", "dynamic_v1"],
        default="static_v1",
        help="action-aware 模式的动作描述符配置。",
    )
    parser.add_argument(
        "--action-aware-mask-mode",
        choices=["none", "increase_noop"],
        default="none",
        help="action-aware 诊断 mask 模式：none(旧行为) 或 increase_noop(屏蔽 decrease)。",
    )
    parser.add_argument(
        "--enable-residual-safety-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable residual safety fallback: reject DQN budget actions that increase HI mode risk.",
    )
    parser.add_argument("--residual-guard-hi-pressure-delta-limit", type=float, default=0.03)
    parser.add_argument("--residual-guard-hi-pressure-abs-limit", type=float, default=0.30)
    parser.add_argument("--residual-guard-reject-decrease-pressure-threshold", type=float, default=0.05)
    parser.add_argument(
        "--residual-guard-use-hi-pressure-max",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--budget-floor-ratio",
        type=float,
        default=0.0,
        help=(
            "Reject budget actions that would reduce any task budget below "
            "initial_budget * this ratio. 0 disables the floor."
        ),
    )
    parser.add_argument(
        "--forbid-decreasing-hi-budgets",
        action="store_true",
        # 阶段 3：开启后会在 action mask 中屏蔽“decrease 命中 HI 任务”的动作。
        # 这是硬约束，不是软惩罚，不依赖 reward 权重调节。
        help="If set, action masks reject budget actions whose decrease tasks include any HI-criticality task.",
    )
    parser.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument(
        "--observation-mode",
        choices=[
            "v10_basic",
            "v11_full_10d",
            "v11_no_risk_9d",
            "v11_no_util_9d",
            "v11_no_max_9d",
            "v11_no_priority_9d",
            "v11_no_risk_no_util_8d",
            "v11_lite_6d",
            "v12_full_14d",
        ],
        default="v10_basic",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """运行正式 DQN 训练并产出完整目录结构。"""

    args = build_parser().parse_args()
    requested_action_space = args.action_space
    if requested_action_space == "constraint_guided_pair":
        # 兼容旧参数名：内部统一走 constraint_guided_transfer 语义。
        args.action_space = "constraint_guided_transfer"
    if args.trace_every < 0:
        raise ValueError("--trace-every 必须为非负整数")
    if args.trace_every > 0 and args.trace_dir is None:
        raise ValueError("设置 --trace-every 时必须同时提供 --trace-dir")
    if args.validate_every < 0:
        raise ValueError("--validate-every 必须为非负整数")
    if args.validation_workers < 1:
        raise ValueError("--validation-workers 必须为正整数")
    if args.log_step_every < 0:
        raise ValueError("--log-step-every 必须为非负整数")
    if args.max_q_diagnostic_samples < 0:
        raise ValueError("--max-q-diagnostic-samples 必须为非负整数")
    if args.elite_replay_capacity <= 0:
        raise ValueError("--elite-replay-capacity 必须为正整数")
    if args.elite_replay_min_size < 0:
        raise ValueError("--elite-replay-min-size 必须为非负整数")
    if args.elite_batch_size < 0:
        raise ValueError("--elite-batch-size 必须为非负整数")
    if args.elite_batch_size > args.batch_size:
        raise ValueError("--elite-batch-size 不能大于 --batch-size")
    if args.best_elite_replay_capacity <= 0:
        raise ValueError("--best-elite-replay-capacity 必须为正整数")
    if args.best_elite_replay_min_size < 0:
        raise ValueError("--best-elite-replay-min-size 必须为非负整数")
    if args.best_elite_batch_size < 0:
        raise ValueError("--best-elite-batch-size 必须为非负整数")
    if args.elite_batch_size + args.best_elite_batch_size > args.batch_size:
        raise ValueError("--elite-batch-size + --best-elite-batch-size 不能大于 --batch-size")
    if not 0.0 <= args.elite_score_min <= 1.0:
        raise ValueError("--elite-score-min 必须在 [0, 1] 内")
    if not 0.0 <= args.elite_score_ratio <= 1.0:
        raise ValueError("--elite-score-ratio 必须在 [0, 1] 内")
    if args.elite_recent_episodes < 0:
        raise ValueError("--elite-recent-episodes 必须为非负整数")
    if args.elite_start_episode < 0:
        raise ValueError("--elite-start-episode 必须为非负整数")
    if args.elite_max_mode_delta < 0.0:
        raise ValueError("--elite-max-mode-delta 必须为非负数")
    if args.elite_max_add_per_validation <= 0:
        raise ValueError("--elite-max-add-per-validation 必须为正整数")
    if args.use_elite_replay and args.elite_batch_size == 0:
        raise ValueError("启用 --use-elite-replay 时 --elite-batch-size 必须大于 0")
    if args.use_elite_replay and args.elite_recent_episodes == 0:
        raise ValueError("启用 --use-elite-replay 时 --elite-recent-episodes 必须大于 0")
    if args.best_elite_min_improvement < 0.0:
        raise ValueError("--best-elite-min-improvement 必须为非负数")
    if args.best_elite_recent_episodes < 0:
        raise ValueError("--best-elite-recent-episodes 必须为非负整数")
    if args.best_elite_start_episode < 0:
        raise ValueError("--best-elite-start-episode 必须为非负整数")
    if args.best_elite_max_add_per_validation <= 0:
        raise ValueError("--best-elite-max-add-per-validation 必须为正整数")
    if args.use_best_elite_replay and args.best_elite_batch_size == 0:
        raise ValueError("启用 --use-best-elite-replay 时 --best-elite-batch-size 必须大于 0")
    if args.use_best_elite_replay and args.best_elite_recent_episodes == 0:
        raise ValueError("启用 --use-best-elite-replay 时 --best-elite-recent-episodes 必须大于 0")
    if args.plateau_balanced_start_episode < 0:
        raise ValueError("--plateau-balanced-start-episode 必须为非负整数")
    if args.plateau_balanced_window <= 0:
        raise ValueError("--plateau-balanced-window 必须为正整数")
    if args.plateau_balanced_burst_episodes < 0:
        raise ValueError("--plateau-balanced-burst-episodes 必须为非负整数")
    if not 0.0 <= args.plateau_balanced_mix_prob <= 1.0:
        raise ValueError("--plateau-balanced-mix-prob must be in [0, 1]")
    if args.target_update_frequency is not None:
        args.target_update_freq = int(args.target_update_frequency)
    if args.budget_floor_ratio < 0.0 or args.budget_floor_ratio > 1.0:
        raise ValueError("--budget-floor-ratio must be in [0, 1]")
    if args.q_network_type == "action_aware" and args.action_space != "single":
        raise ValueError("第一版 action_aware 仅支持 --action-space single；请勿用于其他动作空间。")
    if args.action_aware_mask_mode != "none" and (
        args.q_network_type != "action_aware" or args.action_space != "single"
    ):
        raise ValueError("--action-aware-mask-mode 仅支持 q_network_type=action_aware 且 action_space=single。")
    reward_mode_config = load_reward_mode_config(args.reward_mode)
    feature_config = FeatureConfig(
        observation_mode=args.observation_mode,
        ema_alpha=args.ema_alpha,
        overrun_ema_alpha=args.overrun_ema_alpha,
        history_k=args.history_k,
        event_window=args.event_window,
        max_cost_weight=args.max_cost_weight,
        risk_max_scale=args.risk_max_scale,
        include_safety_margin=args.include_safety_margin,
    )

    experiment_config = _build_experiment_config(args)
    train_seed_candidates = _parse_seed_spec(args.train_seeds)
    episode_seed_schedule = _build_episode_seed_schedule(
        episodes=args.episodes,
        seed=args.seed,
        mode=args.train_seed_mode,
        train_seeds=train_seed_candidates,
    )
    validation_seeds = _parse_seed_spec(args.validation_seeds)

    hidden_layers = _parse_hidden_layers(args.hidden_layers)
    # 阶段 0：显式拆分 DQN 侧随机源，保证同配置可复现且可追踪。
    # - network_seed:     只用于网络初始化；
    # - exploration_seed: 只用于 epsilon-greedy 探索；
    # - replay_seed:      只用于 replay 抽样。
    network_seed = args.seed if args.network_seed is None else int(args.network_seed)
    exploration_seed = args.seed if args.exploration_seed is None else int(args.exploration_seed)
    replay_seed = args.seed if args.replay_seed is None else int(args.replay_seed)
    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        use_elite_replay=args.use_elite_replay,
        elite_replay_capacity=args.elite_replay_capacity,
        elite_replay_min_size=args.elite_replay_min_size,
        elite_batch_size=args.elite_batch_size,
        use_best_elite_replay=args.use_best_elite_replay,
        best_elite_replay_capacity=args.best_elite_replay_capacity,
        best_elite_replay_min_size=args.best_elite_replay_min_size,
        best_elite_batch_size=args.best_elite_batch_size,
        target_update_freq=args.target_update_freq,
        grad_clip_norm=args.grad_clip_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        noop_exploration_prob=args.noop_exploration_prob,
        exploration_mode=args.exploration_mode,
        safe_increase_explore_prob=args.safe_increase_explore_prob,
        plateau_balanced_mix_prob=args.plateau_balanced_mix_prob,
        hidden_layers=hidden_layers,
        seed=args.seed,
        network_seed=network_seed,
        exploration_seed=exploration_seed,
        replay_seed=replay_seed,
        q_network_type=args.q_network_type,
        action_feature_mode=args.action_feature_mode,
        action_aware_mask_mode=args.action_aware_mask_mode,
    )

    initial_seed = episode_seed_schedule[0]
    initial_bundle = resolve_experiment_bundle(experiment_config, initial_seed)
    initial_env = build_env_from_experiment_config(
        experiment_config,
        seed=initial_seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=args.reward_mode,
        action_space=args.action_space,
        budget_increase_ratio=args.budget_increase_ratio,
        budget_decrease_ratio=args.budget_decrease_ratio,
        include_explicit_noop=args.include_explicit_noop,
        budget_floor_ratio=args.budget_floor_ratio,
        forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
        mask_detail_mode=args.mask_detail_mode,
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=(
            args.constraint_guided_pair_allow_increase_only_when_safe
        ),
        enable_residual_safety_fallback=args.enable_residual_safety_fallback,
        residual_guard_hi_pressure_delta_limit=args.residual_guard_hi_pressure_delta_limit,
        residual_guard_hi_pressure_abs_limit=args.residual_guard_hi_pressure_abs_limit,
        residual_guard_reject_decrease_pressure_threshold=(
            args.residual_guard_reject_decrease_pressure_threshold
        ),
        residual_guard_use_hi_pressure_max=args.residual_guard_use_hi_pressure_max,
    )
    initial_obs = initial_env.reset(seed=initial_seed)
    action_features = None
    action_feature_names = None
    if args.q_network_type == "action_aware":
        action_features = initial_env.get_action_feature_matrix(args.action_feature_mode)
        action_feature_names = initial_env.get_action_feature_names(args.action_feature_mode)
    increase_action_ids = _get_increase_action_ids(initial_env)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=initial_env.action_space_size,
        config=config,
        noop_action_id=_get_noop_action_id(initial_env),
        increase_action_ids=increase_action_ids,
        hidden_layers=hidden_layers,
        device=args.dqn_device,
        double_dqn=args.double_dqn,
        action_features=action_features,
        action_feature_names=action_feature_names,
    )

    if args.output_dir is None:
        output_dir = (
            Path(f"outputs/dqn_rtss11/u{int(round(args.total_util * 1000)):03d}_seed{args.seed}")
            if args.workload == "rtss11"
            else Path("outputs/dqn_amc")
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    if args.checkpoint > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    # 训练启动时把 device 解析结果明确打印出来，方便在终端里直接确认是否真的用了 GPU。
    print(f"[DQN] requested device: {args.dqn_device}")
    print(f"[DQN] resolved device: {agent.device}")
    print(f"[DQN] torch version: {torch.__version__}")
    if agent.device.type == "cuda":
        cuda_index = agent.device.index if agent.device.index is not None else 0
        print(f"[DQN] cuda available: {torch.cuda.is_available()}")
        print(f"[DQN] cuda device count: {torch.cuda.device_count()}")
        print(f"[DQN] cuda device name: {torch.cuda.get_device_name(cuda_index)}")

    step_rows: list[dict[str, int | float | str | bool]] = []
    train_metric_rows: list[dict[str, int | float | str]] = []
    action_hist_rows: list[dict[str, int]] = []
    validation_rows: list[dict[str, int | float | None]] = []
    best_validation_row: dict[str, int | float | None] | None = None
    model_best_path = output_dir / "model_best.pt"
    best_model_metadata_path = output_dir / "best_model_metadata.json"
    best_model_saved = False
    best_rows_by_type: dict[str, dict[str, int | float | None] | None] = {
        "conservative_qos": None,
        "qos_stable": None,
        "qos_best": None,
    }
    baseline_validation_cache: dict[str, int | float | None] | None = None
    # `pareto_relative_score` 两阶段选模状态：
    # - False：尚未见到任何 Pareto-valid checkpoint，允许 fallback 到软分数；
    # - True ：已经见到至少一个 Pareto-valid checkpoint，后续仅在 Pareto-valid 集合内比较。
    pareto_valid_seen: bool = False
    plateau_no_improve_count = 0
    plateau_best_reduction: float | None = None
    plateau_last_trigger_episode: int | None = None
    plateau_trigger_rows: list[dict[str, int | float | str | bool | None]] = []
    # Elite Replay v1：缓存最近若干训练 episode 的 transition。
    # 当 validation checkpoint 满足精英条件时，把这一窗口内的 transition 批量写入 elite buffer。
    # recent transition 窗口要同时覆盖 candidate elite 和 best elite 的回看范围。
    recent_window_len = max(args.elite_recent_episodes, args.best_elite_recent_episodes)
    recent_episode_transition_buffers: deque[tuple[int, list[Transition]]] = deque(
        maxlen=max(1, recent_window_len)
    )
    elite_replay_rows: list[dict[str, int | float | str | bool | None]] = []
    elite_current_best_reduction: float | None = None
    best_elite_current_best_reduction: float | None = None

    global_step = 0
    for episode in range(args.episodes):
        # elite_active 由“是否启用 elite replay”与“是否达到起始 episode”共同决定。
        # 这里按 0-based episode 索引判断：episode >= elite_start_episode 时激活。
        elite_active = bool(args.use_elite_replay and episode >= args.elite_start_episode)
        agent.set_elite_replay_runtime_enabled(elite_active)
        # best elite 按 1-based episode 序号控制激活时机，便于与日志展示一致。
        best_elite_active = bool(args.use_best_elite_replay and (episode + 1) >= args.best_elite_start_episode)
        agent.set_best_elite_replay_runtime_enabled(best_elite_active)
        collect_elite_transitions = bool(elite_active or best_elite_active)
        episode_seed = episode_seed_schedule[episode]
        env = build_env_from_experiment_config(
            experiment_config,
            seed=episode_seed,
            end_time=args.end_time,
            agent_period=args.agent_period,
            semantics=RuntimeSemantics.AMC_PLUS,
            reward_mode=args.reward_mode,
            action_space=args.action_space,
            budget_increase_ratio=args.budget_increase_ratio,
            budget_decrease_ratio=args.budget_decrease_ratio,
            include_explicit_noop=args.include_explicit_noop,
            budget_floor_ratio=args.budget_floor_ratio,
            forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
            mask_detail_mode=args.mask_detail_mode,
            feature_config=feature_config,
            constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
            constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
            constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
            constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
            constraint_guided_pair_allow_increase_only_when_safe=(
                args.constraint_guided_pair_allow_increase_only_when_safe
            ),
            enable_residual_safety_fallback=args.enable_residual_safety_fallback,
            residual_guard_hi_pressure_delta_limit=args.residual_guard_hi_pressure_delta_limit,
            residual_guard_hi_pressure_abs_limit=args.residual_guard_hi_pressure_abs_limit,
            residual_guard_reject_decrease_pressure_threshold=(
                args.residual_guard_reject_decrease_pressure_threshold
            ),
            residual_guard_use_hi_pressure_max=args.residual_guard_use_hi_pressure_max,
        )
        bundle = resolve_experiment_bundle(experiment_config, episode_seed)
        obs = env.reset(seed=episode_seed)
        if args.q_network_type == "action_aware" and args.action_feature_mode == "static_v1":
            episode_action_features = env.get_action_feature_matrix(args.action_feature_mode)
            if episode_action_features != action_features:
                raise RuntimeError(
                    "action_aware static action features changed across episodes. "
                    "第一版要求固定任务集与固定动作描述符。"
                )
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_accepted_actions = 0
        episode_rejected_actions = 0
        episode_step_count = 0
        episode_selected_action_count = 0
        episode_noop_actions = 0
        episode_explicit_noop_actions = 0
        # Pareto-aware 奖励诊断计数器：
        # - budget_action_count 仅统计 agent 明确给出的预算动作（排除显式/隐式 noop）；
        # - 其余计数器全部由 env.info 的动作语义字段驱动，避免脚本侧重复推断口径漂移。
        episode_budget_action_count = 0
        episode_increase_action_count = 0
        episode_decrease_action_count = 0
        episode_transfer_action_count = 0
        episode_hi_decrease_count = 0
        episode_unsafe_decrease_count = 0
        reward_job_start_sum = 0.0
        reward_lo_overrun_sum = 0.0
        reward_hi_overrun_sum = 0.0
        reward_mode_change_sum = 0.0
        reward_mode_change_spike_penalty_value_sum = 0.0
        reward_lo_cancellation_sum = 0.0
        reward_deadline_miss_sum = 0.0
        reward_paper_sum = 0.0
        reward_noop_bonus_sum = 0.0
        reward_budget_change_penalty_sum = 0.0
        reward_budget_change_norm_sum = 0.0
        reward_budget_drift_penalty_sum = 0.0
        reward_budget_drift_mean_sum = 0.0
        # v11 诊断：记录每个 step 的 safety margin，再聚合为 episode 级 mean/p05。
        feature_safety_margin_min_values: list[float] = []
        # Elite Replay v1：按 episode 收集 transition，供 validation 后按窗口批量入池。
        episode_transitions: list[Transition] = []
        exploration_action_count_before = int(agent.exploration_action_count)
        exploration_noop_action_count_before = int(agent.exploration_noop_action_count)
        exploration_safe_increase_action_count_before = int(agent.exploration_safe_increase_action_count)
        exploration_all_valid_action_count_before = int(agent.exploration_all_valid_action_count)
        exploration_safe_increase_fallback_count_before = int(agent.exploration_safe_increase_fallback_count)
        exploration_increase_coverage_action_count_before = int(
            agent.exploration_increase_coverage_action_count
        )
        exploration_increase_coverage_tie_count_before = int(agent.exploration_increase_coverage_tie_count)
        plateau_balanced_action_count_before = int(agent.plateau_balanced_action_count)
        plateau_balanced_fallback_count_before = int(agent.plateau_balanced_fallback_count)
        plateau_balanced_burst_count_before = int(agent.plateau_balanced_burst_count)
        episode_action_hist: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "accepted": 0, "rejected": 0})
        last_info: dict[str, int | float | str | bool | None] = {
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        }

        while not done:
            # episode_step_count 是训练期指标的主分母，定义为“完成了多少个环境 step”。
            episode_step_count += 1
            mask = env.valid_action_mask()
            current_action_features = None
            if args.q_network_type == "action_aware":
                current_action_features = env.get_action_feature_matrix(args.action_feature_mode)
                if args.action_feature_mode == "dynamic_v1":
                    agent.set_action_features(
                        current_action_features,
                        action_feature_names=action_feature_names,
                    )
            valid_action_count = sum(mask)
            masked_action_count = len(mask) - valid_action_count
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=True,
            )
            # selected_action_count 只表示“agent 有没有给出 action_id”，
            # 与是否 accepted/noop 是正交维度。
            episode_selected_action_count += int(action_id is not None)
            result = env.step(action_id)
            if action_id is not None and not bool(result.info.get("selected_action_was_mask_valid", False)):
                raise RuntimeError(f"检测到动作选择与 mask 语义不一致，episode={episode}, action_id={action_id}")
            next_mask = (
                env.valid_action_mask()
                if not result.done
                else tuple(False for _ in range(env.action_space_size))
            )
            next_action_features = None
            if args.q_network_type == "action_aware":
                if result.done:
                    if current_action_features is not None:
                        next_action_features = tuple(
                            tuple(0.0 for _ in row) for row in current_action_features
                        )
                else:
                    next_action_features = env.get_action_feature_matrix(args.action_feature_mode)

            loss: float | None = None
            if action_id is not None:
                transition = Transition(
                    state=obs.state_vector,
                    action_id=action_id,
                    reward=result.reward,
                    next_state=result.observation.state_vector,
                    done=result.done,
                    valid_action_mask=tuple(mask),
                    next_valid_action_mask=tuple(next_mask),
                    action_features=(
                        current_action_features
                        if args.q_network_type == "action_aware" and args.action_feature_mode == "dynamic_v1"
                        else None
                    ),
                    next_action_features=(
                        next_action_features
                        if args.q_network_type == "action_aware" and args.action_feature_mode == "dynamic_v1"
                        else None
                    ),
                )
                agent.remember(transition)
                if collect_elite_transitions:
                    episode_transitions.append(transition)
                loss = agent.optimize_one_step()
                if loss is not None:
                    episode_losses.append(loss)
                episode_action_hist[action_id]["count"] += 1
                if bool(result.info.get("accepted")):
                    episode_action_hist[action_id]["accepted"] += 1
                else:
                    episode_action_hist[action_id]["rejected"] += 1

            accepted = bool(result.info.get("accepted"))
            rejected = action_id is not None and not accepted
            noop = bool(result.info.get("is_noop", False))
            explicit_noop = bool(result.info.get("is_explicit_noop_action", False))
            is_budget_action = bool(result.info.get("is_budget_action", False))
            is_increase_action = bool(result.info.get("is_increase_action", False))
            is_decrease_action = bool(result.info.get("is_decrease_action", False))
            is_transfer_action = bool(result.info.get("is_transfer_action", False))
            decrease_hits_hi = bool(result.info.get("decrease_hits_hi", False))
            unsafe_decrease = bool(result.info.get("unsafe_decrease", False))
            # 统计口径说明：
            # - explicit noop 既可能是 accepted，也必须计入 noop；
            # - 因此 rate 分母必须用 step_count，不能用 accepted+rejected+noop。
            if accepted:
                episode_accepted_actions += 1
            if rejected:
                episode_rejected_actions += 1
            if noop:
                episode_noop_actions += 1
            if explicit_noop:
                episode_explicit_noop_actions += 1
            if is_budget_action:
                episode_budget_action_count += 1
            if is_increase_action:
                episode_increase_action_count += 1
            if is_decrease_action:
                episode_decrease_action_count += 1
            if is_transfer_action:
                episode_transfer_action_count += 1
            if is_decrease_action and decrease_hits_hi:
                episode_hi_decrease_count += 1
            if unsafe_decrease:
                episode_unsafe_decrease_count += 1

            episode_reward += result.reward
            reward_job_start_sum += float(result.info.get("step_reward_job_start", 0.0))
            reward_lo_overrun_sum += float(result.info.get("step_reward_lo_overrun", 0.0))
            reward_hi_overrun_sum += float(result.info.get("step_reward_hi_overrun", 0.0))
            reward_mode_change_sum += float(result.info.get("step_reward_mode_change", 0.0))
            reward_mode_change_spike_penalty_value_sum += float(
                result.info.get("mode_change_spike_penalty_value", 0.0)
            )
            reward_lo_cancellation_sum += float(result.info.get("step_reward_lo_cancellation", 0.0))
            reward_deadline_miss_sum += float(result.info.get("step_reward_deadline_miss", 0.0))
            reward_paper_sum += float(result.info.get("paper_reward", 0.0))
            reward_noop_bonus_sum += float(result.info.get("noop_reward_bonus", 0.0))
            reward_budget_change_penalty_sum += float(result.info.get("budget_change_penalty_value", 0.0))
            reward_budget_change_norm_sum += float(result.info.get("budget_change_norm", 0.0))
            reward_budget_drift_penalty_sum += float(result.info.get("budget_drift_penalty_value", 0.0))
            reward_budget_drift_mean_sum += float(result.info.get("budget_drift_mean", 0.0))
            feature_safety_margin_min_values.append(float(result.info.get("feature_safety_margin_min", 1.0)))
            # `global_step` 是跨 episode 的全局步号。
            # 文档要求只对 step-level 明细日志做采样，不改变任何训练统计与优化逻辑。
            should_log_step = args.log_step_every > 0 and global_step % args.log_step_every == 0
            if should_log_step:
                step_rows.append(
                    {
                        "episode": episode,
                        "step": global_step,
                        "sim_time": int(result.info.get("time", 0)),
                        "reward": float(result.reward),
                        "episode_reward": float(episode_reward),
                        "total_reward": float(episode_reward),
                        "loss": "" if loss is None else loss,
                        "epsilon": agent.current_epsilon,
                        "action_id": "" if action_id is None else action_id,
                        "accepted": accepted,
                        "rejected": rejected,
                        "reject_reason": (
                            "no_valid_action" if action_id is None else str(result.info.get("reject_reason", ""))
                        ),
                        "residual_guard_enabled": bool(result.info.get("residual_guard_enabled", False)),
                        "residual_guard_rejected": bool(result.info.get("residual_guard_rejected", False)),
                        "residual_guard_rejected_actions": int(
                            result.info.get("residual_guard_rejected_actions", 0)
                        ),
                        "residual_guard_hi_pressure_delta_limit": float(
                            result.info.get("residual_guard_hi_pressure_delta_limit", 0.0)
                        ),
                        "residual_guard_hi_pressure_abs_limit": float(
                            result.info.get("residual_guard_hi_pressure_abs_limit", 0.0)
                        ),
                        "residual_action_type": result.info.get("residual_action_type", ""),
                        "residual_rank": "" if result.info.get("residual_rank") is None else int(result.info.get("residual_rank", 0)),
                        "residual_resolved_increase_task": result.info.get("residual_resolved_increase_task", ""),
                        "residual_resolved_decrease_tasks": str(
                            result.info.get("residual_resolved_decrease_tasks", ())
                        ),
                        "valid_action_count": valid_action_count,
                        "masked_action_count": masked_action_count,
                        "noop_due_to_no_valid_action": action_id is None,
                        "is_noop": noop,
                        "is_explicit_noop": explicit_noop,
                        "is_budget_action": is_budget_action,
                        "is_increase_action": is_increase_action,
                        "is_decrease_action": is_decrease_action,
                        "is_transfer_action": is_transfer_action,
                        "decrease_hits_hi": decrease_hits_hi,
                        "decrease_hits_lo": bool(result.info.get("decrease_hits_lo", False)),
                        "decrease_task_count": int(result.info.get("decrease_task_count", 0)),
                        "unsafe_decrease": unsafe_decrease,
                        "mode_changes": int(result.info.get("mode_changes", 0)),
                        "lo_cancellations": int(result.info.get("lo_cancellations", 0)),
                        "deadline_misses": int(result.info.get("deadline_misses", 0)),
                        "step_reward_total": float(result.info.get("step_reward_total", 0.0)),
                        "step_reward_job_start": float(result.info.get("step_reward_job_start", 0.0)),
                        "step_reward_lo_overrun": float(result.info.get("step_reward_lo_overrun", 0.0)),
                        "step_reward_hi_overrun": float(result.info.get("step_reward_hi_overrun", 0.0)),
                        "step_reward_mode_change": float(result.info.get("step_reward_mode_change", 0.0)),
                        # 单独落盘二次项惩罚，便于区分线性 mode-change 惩罚和 spike 惩罚来源。
                        "mode_change_spike_penalty_value": float(
                            result.info.get("mode_change_spike_penalty_value", 0.0)
                        ),
                        "step_reward_lo_cancellation": float(result.info.get("step_reward_lo_cancellation", 0.0)),
                        "step_reward_deadline_miss": float(result.info.get("step_reward_deadline_miss", 0.0)),
                        "step_reward_invalid_action": float(result.info.get("step_reward_invalid_action", 0.0)),
                        "paper_reward": float(result.info.get("paper_reward", 0.0)),
                        "noop_reward_bonus": float(result.info.get("noop_reward_bonus", 0.0)),
                        # 新 reward 公式关键变量写入 step log，便于后续定位“哪一项在主导训练”。
                        "lo_overrun_rate": float(result.info.get("lo_overrun_rate", 0.0)),
                        "hi_overrun_rate": float(result.info.get("hi_overrun_rate", 0.0)),
                        "mode_change_per_job": float(result.info.get("mode_change_per_job", 0.0)),
                        "lo_cancellation_rate": float(result.info.get("lo_cancellation_rate", 0.0)),
                        "deadline_miss_rate": float(result.info.get("deadline_miss_rate", 0.0)),
                        "invalid_action": float(result.info.get("invalid_action", 0.0)),
                        "budget_change_norm": float(result.info.get("budget_change_norm", 0.0)),
                        "budget_change_penalty_value": float(result.info.get("budget_change_penalty_value", 0.0)),
                        "budget_drift_mean": float(result.info.get("budget_drift_mean", 0.0)),
                        "budget_drift_penalty_value": float(result.info.get("budget_drift_penalty_value", 0.0)),
                        "lo_pressure_mean": float(result.info.get("lo_pressure_mean", 0.0)),
                        "lo_pressure_max": float(result.info.get("lo_pressure_max", 0.0)),
                        "lo_near_cancel_rate": float(result.info.get("lo_near_cancel_rate", 0.0)),
                        "hi_mode_pressure_mean": float(result.info.get("hi_mode_pressure_mean", 0.0)),
                        "lo_pressure_penalty_value": float(result.info.get("lo_pressure_penalty_value", 0.0)),
                        "lo_pressure_max_penalty_value": float(
                            result.info.get("lo_pressure_max_penalty_value", 0.0)
                        ),
                        "lo_near_cancel_penalty_value": float(
                            result.info.get("lo_near_cancel_penalty_value", 0.0)
                        ),
                        "hi_mode_pressure_penalty_value": float(
                            result.info.get("hi_mode_pressure_penalty_value", 0.0)
                        ),
                        "reward_after_regularization": float(result.info.get("reward_after_regularization", 0.0)),
                        "workload": args.workload,
                        "total_util": args.total_util,
                        "num_tasks": len(bundle.ordered_tasks),
                        "cf": args.cf,
                        "cp": args.cp,
                        "taskset_seed": bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed,
                        "scenario_seed": bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed,
                        "require_schedulable": args.require_schedulable,
                        "observation_mode": str(result.info.get("observation_mode", args.observation_mode)),
                        "state_dim": int(result.info.get("state_dim", len(result.observation.state_vector))),
                    }
                )
            obs = result.observation
            done = result.done
            last_info = result.info
            global_step += 1

        debug_stats = env.debug_statistics()
        if int(debug_stats["selected_invalid_mask_actions"]) > 0:
            raise RuntimeError(
                f"selected_invalid_mask_actions 必须为 0，episode={episode}, value={debug_stats['selected_invalid_mask_actions']}"
            )
        # 在进入 validation 判断前先刷新 recent episode 窗口，确保本轮 episode 可参与 elite 判定。
        if collect_elite_transitions:
            recent_episode_transition_buffers.append((episode + 1, list(episode_transitions)))

        loss_mean = sum(episode_losses) / len(episode_losses) if episode_losses else ""
        loss_last: float | str = episode_losses[-1] if episode_losses else ""
        taskset_seed = bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed
        scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed
        exploration_action_delta = int(agent.exploration_action_count - exploration_action_count_before)
        exploration_noop_delta = int(agent.exploration_noop_action_count - exploration_noop_action_count_before)
        exploration_safe_increase_delta = int(
            agent.exploration_safe_increase_action_count - exploration_safe_increase_action_count_before
        )
        exploration_all_valid_delta = int(
            agent.exploration_all_valid_action_count - exploration_all_valid_action_count_before
        )
        exploration_safe_increase_fallback_delta = int(
            agent.exploration_safe_increase_fallback_count - exploration_safe_increase_fallback_count_before
        )
        exploration_increase_coverage_delta = int(
            agent.exploration_increase_coverage_action_count - exploration_increase_coverage_action_count_before
        )
        exploration_increase_coverage_tie_delta = int(
            agent.exploration_increase_coverage_tie_count - exploration_increase_coverage_tie_count_before
        )
        plateau_balanced_action_delta = int(
            agent.plateau_balanced_action_count - plateau_balanced_action_count_before
        )
        plateau_balanced_fallback_delta = int(
            agent.plateau_balanced_fallback_count - plateau_balanced_fallback_count_before
        )
        plateau_balanced_burst_delta = int(
            agent.plateau_balanced_burst_count - plateau_balanced_burst_count_before
        )
        coverage_counts = [
            int(agent.increase_exploration_visit_counts.get(int(action_id), 0))
            for action_id in increase_action_ids
        ]
        if coverage_counts:
            coverage_min = int(min(coverage_counts))
            coverage_max = int(max(coverage_counts))
            coverage_mean = float(sum(coverage_counts) / len(coverage_counts))
            coverage_var = float(
                sum((count - coverage_mean) ** 2 for count in coverage_counts) / len(coverage_counts)
            )
            coverage_std = coverage_var ** 0.5
        else:
            coverage_min = 0
            coverage_max = 0
            coverage_mean = 0.0
            coverage_std = 0.0
        train_metric_rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "taskset_seed": taskset_seed,
                "scenario_seed": scenario_seed,
                "network_seed": network_seed,
                "exploration_seed": exploration_seed,
                "replay_seed": replay_seed,
                "total_util": args.total_util,
                "num_tasks": len(bundle.ordered_tasks),
                "taskset_fingerprint": bundle.taskset_fingerprint or "",
                "steps": episode_accepted_actions + episode_rejected_actions + episode_noop_actions,
                # `steps` 保留旧字段以兼容历史脚本；其值可能与 step_count 不同（存在重复口径）。
                # 新分析请优先使用 `step_count`。
                "step_count": episode_step_count,
                "selected_action_count": episode_selected_action_count,
                "total_reward": episode_reward,
                "epsilon": agent.current_epsilon,
                "loss_mean": loss_mean,
                "loss_last": loss_last,
                "accepted_actions": episode_accepted_actions,
                "rejected_actions": episode_rejected_actions,
                "noop_actions": episode_noop_actions,
                "explicit_noop_actions": episode_explicit_noop_actions,
                "budget_action_count": episode_budget_action_count,
                "increase_action_count": episode_increase_action_count,
                "decrease_action_count": episode_decrease_action_count,
                "transfer_action_count": episode_transfer_action_count,
                "hi_decrease_count": episode_hi_decrease_count,
                "unsafe_decrease_count": episode_unsafe_decrease_count,
                "noop_action_rate": (
                    episode_noop_actions
                    / episode_step_count
                    if episode_step_count > 0
                    else 0.0
                ),
                # 显式 noop rate 单独输出，便于区分“主动不调预算”和“无动作可用”。
                "explicit_noop_action_rate": (
                    episode_explicit_noop_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "accepted_action_rate": (
                    episode_accepted_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "rejected_action_rate": (
                    episode_rejected_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                # 文档要求这些 rate 使用 budget_action_count/decrease_action_count 作为分母。
                "unsafe_decrease_rate": (
                    episode_unsafe_decrease_count / max(1, episode_budget_action_count)
                ),
                "decrease_action_rate": (
                    episode_decrease_action_count / max(1, episode_budget_action_count)
                ),
                "hi_decrease_rate": (
                    episode_hi_decrease_count / max(1, episode_decrease_action_count)
                ),
                "increase_action_rate": (
                    episode_increase_action_count / max(1, episode_budget_action_count)
                ),
                "transfer_action_rate": (
                    episode_transfer_action_count / max(1, episode_budget_action_count)
                ),
                "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
                "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
                "action_space_type": str(debug_stats["action_space_type"]),
                "action_count": int(debug_stats["action_count"]),
                "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
                "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
                "budget_floor_ratio": float(debug_stats["budget_floor_ratio"]),
                "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
                "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
                "masked_decrease_hi_forbidden_count": int(debug_stats["masked_decrease_hi_forbidden_count"]),
                "masked_decrease_hi_forbidden_rate": float(debug_stats["masked_decrease_hi_forbidden_rate"]),
                "masked_budget_floor_violation_count": int(debug_stats["masked_budget_floor_violation_count"]),
                "masked_budget_floor_violation_rate": float(debug_stats["masked_budget_floor_violation_rate"]),
                "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
                "selected_explicit_noop_actions": int(debug_stats["selected_explicit_noop_actions"]),
                "selected_explicit_noop_rate": float(debug_stats["selected_explicit_noop_rate"]),
                "mode_changes": int(last_info.get("mode_changes", 0)),
                "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
                "deadline_misses": int(last_info.get("deadline_misses", 0)),
                "job_starts": int(env._monitor.job_start_count),
                "lo_overruns": int(env._monitor.lo_overrun_count),
                "hi_overruns": int(env._monitor.hi_overrun_count),
                "reward_job_start_sum": reward_job_start_sum,
                "reward_lo_overrun_sum": reward_lo_overrun_sum,
                "reward_hi_overrun_sum": reward_hi_overrun_sum,
                "reward_mode_change_sum": reward_mode_change_sum,
                "reward_mode_change_spike_penalty_value_sum": reward_mode_change_spike_penalty_value_sum,
                "reward_lo_cancellation_sum": reward_lo_cancellation_sum,
                "reward_deadline_miss_sum": reward_deadline_miss_sum,
                "reward_paper_sum": reward_paper_sum,
                "reward_noop_bonus_sum": reward_noop_bonus_sum,
                "reward_budget_change_penalty_sum": reward_budget_change_penalty_sum,
                "reward_budget_change_norm_sum": reward_budget_change_norm_sum,
                "reward_budget_drift_penalty_sum": reward_budget_drift_penalty_sum,
                "reward_budget_drift_mean_sum": reward_budget_drift_mean_sum,
                # Elite Replay v1 训练期统计：用于观察 elite buffer 是否生效以及采样占比。
                "elite_replay_enabled": bool(args.use_elite_replay),
                "elite_replay_active": bool(elite_active),
                "elite_replay_buffer_size": int(agent.elite_replay_size),
                "elite_transitions_added_total": int(agent.elite_transitions_added_total),
                "elite_samples_used_total": int(agent.elite_samples_used_total),
                "best_elite_replay_enabled": bool(args.use_best_elite_replay),
                "best_elite_replay_active": bool(best_elite_active),
                "best_elite_replay_buffer_size": int(agent.best_elite_replay_size),
                "best_elite_transitions_added_total": int(agent.best_elite_transitions_added_total),
                "best_elite_samples_used_total": int(agent.best_elite_samples_used_total),
                "normal_samples_used_total": int(agent.normal_samples_used_total),
                "exploration_mode": args.exploration_mode,
                "noop_exploration_prob": args.noop_exploration_prob,
                "safe_increase_explore_prob": args.safe_increase_explore_prob,
                "increase_action_id_count": len(increase_action_ids),
                "exploration_action_count": exploration_action_delta,
                "exploration_noop_action_count": exploration_noop_delta,
                "exploration_safe_increase_action_count": exploration_safe_increase_delta,
                "exploration_all_valid_action_count": exploration_all_valid_delta,
                "exploration_safe_increase_fallback_count": exploration_safe_increase_fallback_delta,
                "exploration_increase_coverage_action_count": exploration_increase_coverage_delta,
                "exploration_increase_coverage_tie_count": exploration_increase_coverage_tie_delta,
                "plateau_balanced_active": bool(agent.plateau_balanced_is_active),
                "plateau_balanced_active_episodes_remaining": int(
                    agent.plateau_balanced_active_episodes_remaining
                ),
                "plateau_balanced_burst_count_total": int(agent.plateau_balanced_burst_count),
                "plateau_balanced_burst_count_delta": plateau_balanced_burst_delta,
                "plateau_balanced_action_count": plateau_balanced_action_delta,
                "plateau_balanced_fallback_count": plateau_balanced_fallback_delta,
                "exploration_noop_action_rate": (
                    float(exploration_noop_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "exploration_safe_increase_action_rate": (
                    float(exploration_safe_increase_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "exploration_all_valid_action_rate": (
                    float(exploration_all_valid_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "exploration_safe_increase_fallback_rate": (
                    float(exploration_safe_increase_fallback_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "exploration_increase_coverage_action_rate": (
                    float(exploration_increase_coverage_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "exploration_increase_coverage_tie_rate": (
                    float(exploration_increase_coverage_tie_delta) / float(exploration_increase_coverage_delta)
                    if exploration_increase_coverage_delta > 0
                    else 0.0
                ),
                "plateau_balanced_action_rate": (
                    float(plateau_balanced_action_delta) / float(exploration_action_delta)
                    if exploration_action_delta > 0
                    else 0.0
                ),
                "increase_coverage_min_count": coverage_min,
                "increase_coverage_max_count": coverage_max,
                "increase_coverage_mean_count": coverage_mean,
                "increase_coverage_std_count": coverage_std,
                "observation_mode": str(last_info.get("observation_mode", args.observation_mode)),
                "state_dim": int(last_info.get("state_dim", len(obs.state_vector))),
                "feature_safety_margin_min_mean": (
                    sum(feature_safety_margin_min_values) / len(feature_safety_margin_min_values)
                    if feature_safety_margin_min_values
                    else 0.0
                ),
                "feature_safety_margin_min_p05": _percentile(feature_safety_margin_min_values, 0.05),
            }
        )
        for action_id in sorted(episode_action_hist):
            action_hist = episode_action_hist[action_id]
            action_hist_rows.append(
                {
                    "episode": episode,
                    "action_id": action_id,
                    "count": action_hist["count"],
                    "accepted_count": action_hist["accepted"],
                    "rejected_count": action_hist["rejected"],
                }
            )
        action_counts = [int(action_hist["count"]) for action_hist in episode_action_hist.values()]
        total_action_count = sum(action_counts)
        action_entropy = (
            -sum((count / total_action_count) * math.log((count / total_action_count) + 1e-12) for count in action_counts)
            if total_action_count > 0
            else 0.0
        )
        single_increase_ids = [int(action.action_id) for action in env._actions if action.increase_idx is not None]
        single_decrease_ids = [int(action.action_id) for action in env._actions if action.decrease_indices]
        action7_count = int(episode_action_hist.get(7, {}).get("count", 0))
        action8_11_count = sum(int(episode_action_hist.get(action_id, {}).get("count", 0)) for action_id in (8, 9, 10, 11))
        increase_action_count = sum(int(episode_action_hist.get(action_id, {}).get("count", 0)) for action_id in single_increase_ids)
        decrease_action_count = sum(int(episode_action_hist.get(action_id, {}).get("count", 0)) for action_id in single_decrease_ids)
        train_metric_rows[-1]["action_entropy"] = action_entropy
        train_metric_rows[-1]["action7_usage_rate"] = (
            float(action7_count) / float(total_action_count) if total_action_count > 0 else 0.0
        )
        train_metric_rows[-1]["action8_11_usage_rate"] = (
            float(action8_11_count) / float(total_action_count) if total_action_count > 0 else 0.0
        )
        train_metric_rows[-1]["increase_action_usage_rate"] = (
            float(increase_action_count) / float(total_action_count) if total_action_count > 0 else 0.0
        )
        train_metric_rows[-1]["decrease_action_usage_rate"] = (
            float(decrease_action_count) / float(total_action_count) if total_action_count > 0 else 0.0
        )

        # 先推进上一轮 burst 的剩余 episode 计数，再进行本 episode 的 checkpoint/validation。
        # 这样如果 validation 在当前 episode 结束时触发新的 burst，新的 burst 不会被当场递减。
        agent.on_episode_end()

        if args.trace_every > 0 and (episode + 1) % args.trace_every == 0 and args.trace_dir is not None:
            runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_action_log.jsonl", env.action_log)
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_mask_log.jsonl", env.mask_log)
            _write_jsonl(
                args.trace_dir / f"episode_{episode + 1:04d}_runtime_trace.jsonl",
                _trace_rows_from_runtime(runtime_result),
            )

        if args.checkpoint > 0 and (episode + 1) % args.checkpoint == 0:
            agent.save(checkpoint_dir / f"model_episode_{episode + 1:04d}.pt")

        if args.validate_every > 0 and validation_seeds and (episode + 1) % args.validate_every == 0:
            validation_row, baseline_validation_cache, used_baseline_cache = _run_validation(
                agent=agent,
                experiment_config=experiment_config,
                validation_seeds=validation_seeds,
                validation_end_time=args.validation_end_time,
                agent_period=args.agent_period,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                mask_detail_mode=args.mask_detail_mode,
                feature_config=feature_config,
                validation_workers=args.validation_workers,
                baseline_cache=baseline_validation_cache,
                max_q_diagnostic_samples=args.max_q_diagnostic_samples,
                constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=(
                    args.constraint_guided_pair_allow_increase_only_when_safe
                ),
                enable_residual_safety_fallback=args.enable_residual_safety_fallback,
                residual_guard_hi_pressure_delta_limit=args.residual_guard_hi_pressure_delta_limit,
                residual_guard_hi_pressure_abs_limit=args.residual_guard_hi_pressure_abs_limit,
                residual_guard_reject_decrease_pressure_threshold=(
                    args.residual_guard_reject_decrease_pressure_threshold
                ),
                residual_guard_use_hi_pressure_max=args.residual_guard_use_hi_pressure_max,
                log_validation_policy_actions=args.log_validation_policy_actions,
            )
            if used_baseline_cache:
                print("Using cached baseline validation metrics")
            validation_row["episode"] = episode + 1
            # 阶段 2：显式记录相对 baseline 的两类 delta，并按 alpha 计算综合分数。
            # 当前版本采用“归一化 validation 口径”，使 validation 选模尺度与 interval reward 更一致：
            # - raw_delta_lo   = dqn_lo_cancellations_mean - baseline_lo_cancellations_mean
            # - raw_delta_mode = dqn_mode_changes_mean    - baseline_mode_changes_mean
            # - normalized_delta_lo   = raw_delta_lo   / max(1, baseline_lo_cancellations_mean)
            # - normalized_delta_mode = raw_delta_mode / max(1, baseline_mode_changes_mean)
            # 公式：relative_score = normalized_delta_lo + alpha * normalized_delta_mode
            # 因此 relative_score 越小越好，<0 表示按归一化综合指标优于 baseline。
            # raw delta 仍写入额外字段，便于人工查看实际 count 差异。
            raw_delta_lo = (
                float(validation_row["lo_cancellations_mean"])
                - float(validation_row["baseline_lo_cancellations_mean"])
            )
            raw_delta_mode = float(validation_row["mode_changes_mean"]) - float(
                validation_row["baseline_mode_changes_mean"]
            )

            baseline_lo_denominator = max(
                1.0,
                float(validation_row["baseline_lo_cancellations_mean"]),
            )
            baseline_mode_denominator = max(
                1.0,
                float(validation_row["baseline_mode_changes_mean"]),
            )

            normalized_delta_lo = raw_delta_lo / baseline_lo_denominator
            normalized_delta_mode = raw_delta_mode / baseline_mode_denominator

            validation_row["relative_score_alpha"] = args.relative_score_alpha
            validation_row["raw_delta_lo_cancellations"] = raw_delta_lo
            validation_row["raw_delta_mode_changes"] = raw_delta_mode
            validation_row["relative_delta_lo_cancellations"] = normalized_delta_lo
            validation_row["relative_delta_mode_changes"] = normalized_delta_mode
            validation_row["relative_score"] = (
                normalized_delta_lo
                + args.relative_score_alpha * normalized_delta_mode
            )
            validation_row["is_better_than_baseline"] = float(validation_row["relative_score"]) < 0.0
            # 标记当前候选是否满足 Pareto-valid，用于“先筛选，再排序”的两阶段策略。
            validation_row["is_pareto_valid"] = _is_pareto_valid_checkpoint(validation_row)

            current_reduction = _relative_lc_reduction_from_validation_row(validation_row)
            # Elite Replay v1：仅用于“精英阈值”历史参考，不影响原有 best checkpoint 选择逻辑。
            if current_reduction is not None and current_reduction > 0.0:
                candidate_for_best, _, _ = _is_elite_replay_candidate(
                    validation_row,
                    current_reduction=current_reduction,
                    current_best_reduction=current_reduction,
                    elite_score_min=0.0,
                    elite_score_ratio=0.0,
                    elite_max_mode_delta=args.elite_max_mode_delta,
                    elite_require_no_hi_miss=args.elite_require_no_hi_miss,
                    elite_require_qos_stable=args.elite_require_qos_stable,
                )
                if candidate_for_best and (
                    elite_current_best_reduction is None or current_reduction > elite_current_best_reduction
                ):
                    elite_current_best_reduction = current_reduction

            elite_added_count = 0
            elite_candidate = False
            elite_threshold: float | None = None
            elite_reason = "disabled"
            elite_recent_episode_start: int | None = None
            elite_recent_episode_end: int | None = None
            if elite_active:
                elite_candidate, elite_threshold, elite_reason = _is_elite_replay_candidate(
                    validation_row,
                    current_reduction=current_reduction,
                    current_best_reduction=elite_current_best_reduction,
                    elite_score_min=args.elite_score_min,
                    elite_score_ratio=args.elite_score_ratio,
                    elite_max_mode_delta=args.elite_max_mode_delta,
                    elite_require_no_hi_miss=args.elite_require_no_hi_miss,
                    elite_require_qos_stable=args.elite_require_qos_stable,
                )
                if elite_candidate:
                    recent_items = list(recent_episode_transition_buffers)
                    if recent_items:
                        elite_recent_episode_start = int(recent_items[0][0])
                        elite_recent_episode_end = int(recent_items[-1][0])
                    transitions_to_add: list[Transition] = []
                    for _, transitions in recent_items:
                        transitions_to_add.extend(transitions)
                    # 防止单次 validation 向 elite buffer 注入过量样本。
                    if len(transitions_to_add) > args.elite_max_add_per_validation:
                        transitions_to_add = transitions_to_add[-args.elite_max_add_per_validation :]
                    elite_added_count = agent.remember_elite_many(transitions_to_add)
            else:
                elite_reason = "not_started"

            best_elite_candidate = False
            best_elite_reason = "disabled"
            best_elite_added_count = 0
            best_elite_recent_episode_start: int | None = None
            best_elite_recent_episode_end: int | None = None
            if best_elite_active:
                safe_candidate, _, safe_reason = _is_elite_replay_candidate(
                    validation_row,
                    current_reduction=current_reduction,
                    current_best_reduction=current_reduction,
                    elite_score_min=0.0,
                    elite_score_ratio=0.0,
                    elite_max_mode_delta=args.elite_max_mode_delta,
                    elite_require_no_hi_miss=args.elite_require_no_hi_miss,
                    elite_require_qos_stable=args.elite_require_qos_stable,
                )
                if not safe_candidate:
                    best_elite_reason = safe_reason
                elif best_elite_current_best_reduction is None:
                    best_elite_candidate = True
                    best_elite_reason = "accepted_new_best"
                elif current_reduction is not None and current_reduction > (
                    best_elite_current_best_reduction + args.best_elite_min_improvement
                ):
                    best_elite_candidate = True
                    best_elite_reason = "accepted_new_best"
                else:
                    best_elite_reason = "not_new_best"
            else:
                best_elite_reason = "not_started" if args.use_best_elite_replay else "disabled"

            if best_elite_candidate:
                recent_items = list(recent_episode_transition_buffers)[-args.best_elite_recent_episodes :]
                if recent_items:
                    best_elite_recent_episode_start = int(recent_items[0][0])
                    best_elite_recent_episode_end = int(recent_items[-1][0])
                transitions_to_add: list[Transition] = []
                for _, transitions in recent_items:
                    transitions_to_add.extend(transitions)
                if len(transitions_to_add) > args.best_elite_max_add_per_validation:
                    transitions_to_add = transitions_to_add[-args.best_elite_max_add_per_validation :]
                if args.best_elite_replace_on_new_best:
                    agent.clear_best_elite_replay()
                best_elite_added_count = agent.remember_best_elite_many(transitions_to_add)
                # 只要触发了本次 best elite 新最优判定，就刷新 best reduction，避免重复触发。
                best_elite_current_best_reduction = current_reduction

            plateau_triggered = False
            plateau_trigger_reason = ""
            if current_reduction is not None:
                # plateau 统计只在新探索模式下驱动训练状态，避免影响旧主线行为。
                if args.exploration_mode == "epsilon_plateau_soft_target_balanced":
                    if plateau_best_reduction is None or current_reduction > plateau_best_reduction + 1e-12:
                        plateau_best_reduction = current_reduction
                        plateau_no_improve_count = 0
                    else:
                        plateau_no_improve_count += 1

                    episode_number = episode + 1
                    enough_episode = episode_number >= args.plateau_balanced_start_episode
                    plateau_reached = plateau_no_improve_count >= args.plateau_balanced_window
                    burst_len_positive = args.plateau_balanced_burst_episodes > 0
                    below_reduction_gate = True
                    if (
                        args.plateau_balanced_max_best_reduction > 0.0
                        and plateau_best_reduction is not None
                    ):
                        below_reduction_gate = (
                            plateau_best_reduction < args.plateau_balanced_max_best_reduction
                        )
                    if enough_episode and plateau_reached and burst_len_positive and below_reduction_gate:
                        agent.start_plateau_balanced_burst(
                            args.plateau_balanced_burst_episodes,
                            reset_counts=args.plateau_balanced_reset_counts_on_burst,
                        )
                        plateau_triggered = True
                        plateau_last_trigger_episode = episode_number
                        plateau_trigger_reason = "plateau_low_reduction"
                        plateau_no_improve_count = 0

            plateau_status = {
                "episode": episode + 1,
                "current_reduction": current_reduction,
                "plateau_best_reduction": plateau_best_reduction,
                "plateau_no_improve_count": plateau_no_improve_count,
                "plateau_triggered": plateau_triggered,
                "plateau_trigger_reason": plateau_trigger_reason,
                "plateau_active_episodes_remaining": agent.plateau_balanced_active_episodes_remaining,
                "plateau_burst_count": agent.plateau_balanced_burst_count,
            }
            plateau_trigger_rows.append(plateau_status)
            validation_row.update(
                {
                    "plateau_current_reduction": current_reduction,
                    "plateau_best_reduction": plateau_best_reduction,
                    "plateau_no_improve_count": plateau_no_improve_count,
                    "plateau_balanced_triggered": plateau_triggered,
                    "plateau_balanced_active_episodes_remaining": agent.plateau_balanced_active_episodes_remaining,
                    "plateau_balanced_burst_count": agent.plateau_balanced_burst_count,
                    "elite_replay_enabled": bool(args.use_elite_replay),
                    "elite_replay_active": bool(elite_active),
                    "elite_replay_candidate": bool(elite_candidate),
                    "elite_replay_reason": str(elite_reason),
                    "elite_replay_threshold": elite_threshold,
                    "elite_replay_current_reduction": current_reduction,
                    "elite_replay_best_reduction": elite_current_best_reduction,
                    "elite_replay_added_count": int(elite_added_count),
                    "elite_replay_buffer_size": int(agent.elite_replay_size),
                    "elite_replay_recent_episode_start": elite_recent_episode_start,
                    "elite_replay_recent_episode_end": elite_recent_episode_end,
                    "elite_samples_used_total": int(agent.elite_samples_used_total),
                    "normal_samples_used_total": int(agent.normal_samples_used_total),
                    "best_elite_replay_enabled": bool(args.use_best_elite_replay),
                    "best_elite_replay_active": bool(best_elite_active),
                    "best_elite_replay_candidate": bool(best_elite_candidate),
                    "best_elite_replay_reason": str(best_elite_reason),
                    "best_elite_replay_current_reduction": current_reduction,
                    "best_elite_replay_best_reduction": best_elite_current_best_reduction,
                    "best_elite_replay_added_count": int(best_elite_added_count),
                    "best_elite_replay_buffer_size": int(agent.best_elite_replay_size),
                    "best_elite_replay_recent_episode_start": best_elite_recent_episode_start,
                    "best_elite_replay_recent_episode_end": best_elite_recent_episode_end,
                    "best_elite_samples_used_total": int(agent.best_elite_samples_used_total),
                    "best_elite_transitions_added_total": int(agent.best_elite_transitions_added_total),
                }
            )
            elite_replay_rows.append(
                {
                    "episode": episode + 1,
                    "enabled": bool(args.use_elite_replay),
                    "active": bool(elite_active),
                    "candidate": bool(elite_candidate),
                    "reason": str(elite_reason),
                    "threshold": elite_threshold,
                    "current_reduction": current_reduction,
                    "best_reduction": elite_current_best_reduction,
                    "added_count": int(elite_added_count),
                    "buffer_size": int(agent.elite_replay_size),
                    "recent_episode_start": elite_recent_episode_start,
                    "recent_episode_end": elite_recent_episode_end,
                    "elite_samples_used_total": int(agent.elite_samples_used_total),
                    "normal_samples_used_total": int(agent.normal_samples_used_total),
                    "best_enabled": bool(args.use_best_elite_replay),
                    "best_active": bool(best_elite_active),
                    "best_candidate": bool(best_elite_candidate),
                    "best_reason": str(best_elite_reason),
                    "best_current_reduction": current_reduction,
                    "best_best_reduction": best_elite_current_best_reduction,
                    "best_added_count": int(best_elite_added_count),
                    "best_buffer_size": int(agent.best_elite_replay_size),
                    "best_recent_episode_start": best_elite_recent_episode_start,
                    "best_recent_episode_end": best_elite_recent_episode_end,
                    "best_samples_used_total": int(agent.best_elite_samples_used_total),
                    "best_transitions_added_total": int(agent.best_elite_transitions_added_total),
                }
            )
            validation_rows.append(validation_row)

            should_update_best = False
            if args.save_best_by == "pareto_relative_score":
                # 方向 2：Pareto 选模改为“先筛选，再排序”
                # ------------------------------------------
                # 阶段一（筛选）：
                # - 先判断当前候选是否 Pareto-valid。
                # - 一旦历史中出现过 Pareto-valid，后续只允许在 Pareto-valid 集合中竞争 best。
                #
                # 阶段二（排序）：
                # - 在 Pareto-valid 集合内，按 relative_score（越小越好）排序。
                #
                # fallback（仅在阶段一尚未出现任何 Pareto-valid 时）：
                # - 临时使用温和软分数 `pareto_relative_score` 维持“有 best 可保存”的行为。
                is_pareto_valid = bool(validation_row["is_pareto_valid"])
                if is_pareto_valid:
                    if not pareto_valid_seen:
                        # 首次出现 Pareto-valid：立刻切换到“只看 Pareto-valid”阶段，
                        # 并直接把当前候选设为新的 best 基准。
                        pareto_valid_seen = True
                        should_update_best = True
                    else:
                        # 已进入 Pareto-only 阶段：
                        # - 如果当前 best 不是 Pareto-valid（理论上只会在切换边界出现），直接替换；
                        # - 否则只比较 relative_score。
                        if best_validation_row is None or not _is_pareto_valid_checkpoint(best_validation_row):
                            should_update_best = True
                        else:
                            should_update_best = float(validation_row["relative_score"]) < float(
                                best_validation_row["relative_score"]
                            )
                else:
                    if not pareto_valid_seen:
                        # 还没见到任何 Pareto-valid，允许 fallback 到温和软分数排序。
                        should_update_best = _is_better_validation_row(
                            candidate_row=validation_row,
                            best_row=best_validation_row,
                            save_best_by="pareto_relative_score",
                            relative_score_alpha=args.relative_score_alpha,
                            require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                            qos_stable_mode_delta=args.qos_stable_mode_delta,
                        )
                    else:
                        # 已见到 Pareto-valid 后，非 Pareto-valid 候选直接丢弃，不参与 best 竞争。
                        should_update_best = False
            else:
                should_update_best = _is_better_validation_row(
                    candidate_row=validation_row,
                    best_row=best_validation_row,
                    save_best_by=args.save_best_by,
                    relative_score_alpha=args.relative_score_alpha,
                    require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                    qos_stable_mode_delta=args.qos_stable_mode_delta,
                )

            if should_update_best:
                best_validation_row = validation_row
                agent.save(model_best_path)
                best_model_saved = True
            if args.save_all_best_types:
                for best_type in ("conservative_qos", "qos_stable", "qos_best"):
                    should_update_aux_best = _is_better_validation_row(
                        candidate_row=validation_row,
                        best_row=best_rows_by_type[best_type],
                        save_best_by=best_type,
                        relative_score_alpha=args.relative_score_alpha,
                        require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                        qos_stable_mode_delta=args.qos_stable_mode_delta,
                    )
                    if should_update_aux_best:
                        best_rows_by_type[best_type] = validation_row
                        agent.save(output_dir / f"model_best_{best_type}.pt")
            print(
                {
                    "episode": episode + 1,
                    "validation_reward_mean": float(validation_row["reward_mean"]),
                    "train_loss_last": loss_last,
                }
            )

    train_log_path = output_dir / "train_log.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    action_hist_path = output_dir / "train_action_histogram.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    elite_replay_log_path = output_dir / "elite_replay_log.csv"
    validation_unified_summary_path = output_dir / "validation_unified_summary.csv"
    validation_policy_actions_path = output_dir / "validation_policy_actions.csv"
    model_path = output_dir / "model_final.pt"
    config_path = output_dir / "config.json"

    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        # 即使 `--log-step-every 0` 让 step_rows 为空，也仍然写出固定 header，
        # 这样下游脚本可以明确识别“文件存在但没有逐步日志”。
        writer = csv.DictWriter(f, fieldnames=STEP_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(step_rows)

    if args.log_train_metrics:
        metric_fieldnames = [
            "episode",
            "episode_seed",
            "taskset_seed",
            "scenario_seed",
            "network_seed",
            "exploration_seed",
            "replay_seed",
            "total_util",
            "num_tasks",
            "taskset_fingerprint",
            "steps",
            "step_count",
            "selected_action_count",
            "total_reward",
            "epsilon",
            "loss_mean",
            "loss_last",
            "accepted_actions",
            "rejected_actions",
            "noop_actions",
            "explicit_noop_actions",
            "budget_action_count",
            "increase_action_count",
            "decrease_action_count",
            "transfer_action_count",
            "hi_decrease_count",
            "unsafe_decrease_count",
            "noop_action_rate",
            "explicit_noop_action_rate",
            "accepted_action_rate",
            "rejected_action_rate",
            "unsafe_decrease_rate",
            "decrease_action_rate",
            "hi_decrease_rate",
            "increase_action_rate",
            "transfer_action_rate",
            "safety_checked_actions",
            "selected_invalid_mask_actions",
            "action_space_type",
            "action_count",
            "budget_increase_ratio",
            "budget_decrease_ratio",
            "budget_floor_ratio",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "masked_decrease_hi_forbidden_count",
            "masked_decrease_hi_forbidden_rate",
            "masked_budget_floor_violation_count",
            "masked_budget_floor_violation_rate",
            "no_safe_action_steps",
            "selected_explicit_noop_actions",
            "selected_explicit_noop_rate",
            "mode_changes",
            "lo_cancellations",
            "deadline_misses",
            "job_starts",
            "lo_overruns",
            "hi_overruns",
            "reward_job_start_sum",
            "reward_lo_overrun_sum",
            "reward_hi_overrun_sum",
            "reward_mode_change_sum",
            "reward_mode_change_spike_penalty_value_sum",
            "reward_lo_cancellation_sum",
            "reward_deadline_miss_sum",
            "reward_paper_sum",
            "reward_noop_bonus_sum",
            "reward_budget_change_penalty_sum",
            "reward_budget_change_norm_sum",
            "reward_budget_drift_penalty_sum",
            "reward_budget_drift_mean_sum",
            "elite_replay_enabled",
            "elite_replay_active",
            "elite_replay_buffer_size",
            "elite_transitions_added_total",
            "elite_samples_used_total",
            "best_elite_replay_enabled",
            "best_elite_replay_active",
            "best_elite_replay_buffer_size",
            "best_elite_transitions_added_total",
            "best_elite_samples_used_total",
            "normal_samples_used_total",
            "exploration_mode",
            "noop_exploration_prob",
            "safe_increase_explore_prob",
            "increase_action_id_count",
            "exploration_action_count",
            "exploration_noop_action_count",
            "exploration_safe_increase_action_count",
            "exploration_all_valid_action_count",
            "exploration_safe_increase_fallback_count",
            "exploration_increase_coverage_action_count",
            "exploration_increase_coverage_tie_count",
            "plateau_balanced_active",
            "plateau_balanced_active_episodes_remaining",
            "plateau_balanced_burst_count_total",
            "plateau_balanced_burst_count_delta",
            "plateau_balanced_action_count",
            "plateau_balanced_fallback_count",
            "exploration_noop_action_rate",
            "exploration_safe_increase_action_rate",
            "exploration_all_valid_action_rate",
            "exploration_safe_increase_fallback_rate",
            "exploration_increase_coverage_action_rate",
            "exploration_increase_coverage_tie_rate",
            "plateau_balanced_action_rate",
            "increase_coverage_min_count",
            "increase_coverage_max_count",
            "increase_coverage_mean_count",
            "increase_coverage_std_count",
            "observation_mode",
            "state_dim",
            "feature_safety_margin_min_mean",
            "feature_safety_margin_min_p05",
            "action_entropy",
            "action7_usage_rate",
            "action8_11_usage_rate",
            "increase_action_usage_rate",
            "decrease_action_usage_rate",
        ]
        with train_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writeheader()
            writer.writerows(train_metric_rows)

    action_hist_fieldnames = ["episode", "action_id", "count", "accepted_count", "rejected_count"]
    with action_hist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=action_hist_fieldnames)
        writer.writeheader()
        writer.writerows(action_hist_rows)

    if validation_rows:
        validation_fieldnames = [
            "episode",
            "validation_seed_count",
            "deadline_misses_sum",
            "mode_changes_mean",
            "lo_cancellations_mean",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "dqn_mode_changes_delta_mean",
            "dqn_lo_cancellations_delta_mean",
            "accepted_actions_mean",
            "rejected_actions_mean",
            "step_count_mean",
            "selected_action_count_mean",
            "noop_actions_mean",
            "explicit_noop_actions_mean",
            "noop_action_rate_mean",
            "explicit_noop_action_rate_mean",
            "accepted_action_rate_mean",
            "rejected_action_rate_mean",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "no_safe_action_steps_mean",
            "reward_mean",
            "observation_mode",
            "state_dim_mean",
        ]
        validation_fieldnames.extend(QOS_VALIDATION_FIELDNAMES)
        validation_fieldnames.extend(
            [
                "plateau_current_reduction",
                "plateau_best_reduction",
                "plateau_no_improve_count",
                "plateau_balanced_triggered",
                "plateau_balanced_active_episodes_remaining",
                "plateau_balanced_burst_count",
                "elite_replay_enabled",
                "elite_replay_active",
                "elite_replay_candidate",
                "elite_replay_reason",
                "elite_replay_threshold",
                "elite_replay_current_reduction",
                "elite_replay_best_reduction",
                "elite_replay_added_count",
                "elite_replay_buffer_size",
                "elite_replay_recent_episode_start",
                "elite_replay_recent_episode_end",
                "elite_samples_used_total",
                "normal_samples_used_total",
                "best_elite_replay_enabled",
                "best_elite_replay_active",
                "best_elite_replay_candidate",
                "best_elite_replay_reason",
                "best_elite_replay_current_reduction",
                "best_elite_replay_best_reduction",
                "best_elite_replay_added_count",
                "best_elite_replay_buffer_size",
                "best_elite_replay_recent_episode_start",
                "best_elite_replay_recent_episode_end",
                "best_elite_samples_used_total",
                "best_elite_transitions_added_total",
                "relative_score_alpha",
                "raw_delta_lo_cancellations",
                "raw_delta_mode_changes",
                "relative_delta_lo_cancellations",
                "relative_delta_mode_changes",
                "relative_score",
                "is_better_than_baseline",
                "is_pareto_valid",
            ]
        )
        validation_fieldnames.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
        if args.log_validation_policy_actions:
            validation_fieldnames.extend(
                [
                    "policy_action_definitions_json",
                    "policy_action_hist_json",
                    "policy_action_accepted_hist_json",
                    "policy_action_rejected_hist_json",
                    "policy_action_type_hist_json",
                    "policy_resolved_increase_task_hist_json",
                    "policy_resolved_decrease_task_hist_json",
                    "policy_action_reward_sum_json",
                    "policy_action_lo_delta_sum_json",
                    "policy_action_mode_delta_sum_json",
                    "policy_action_is_increase_sum_json",
                    "policy_action_is_decrease_sum_json",
                    "policy_action_is_transfer_sum_json",
                    "policy_action_decrease_hits_hi_sum_json",
                    "policy_action_decrease_hits_lo_sum_json",
                    "policy_action_unsafe_decrease_sum_json",
                ]
            )
        with validation_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_fieldnames)
            writer.writeheader()
            writer.writerows(validation_rows)
        if args.log_validation_policy_actions:
            policy_rows: list[dict[str, int | float | str]] = []
            for validation_row in validation_rows:
                episode = int(validation_row["episode"])
                definitions = json.loads(str(validation_row["policy_action_definitions_json"]))
                hist = json.loads(str(validation_row["policy_action_hist_json"]))
                accepted_hist = json.loads(str(validation_row["policy_action_accepted_hist_json"]))
                rejected_hist = json.loads(str(validation_row["policy_action_rejected_hist_json"]))
                reward_sum_hist = json.loads(str(validation_row["policy_action_reward_sum_json"]))
                lo_delta_sum_hist = json.loads(str(validation_row["policy_action_lo_delta_sum_json"]))
                mode_delta_sum_hist = json.loads(str(validation_row["policy_action_mode_delta_sum_json"]))
                is_increase_sum_hist = json.loads(str(validation_row["policy_action_is_increase_sum_json"]))
                is_decrease_sum_hist = json.loads(str(validation_row["policy_action_is_decrease_sum_json"]))
                is_transfer_sum_hist = json.loads(str(validation_row["policy_action_is_transfer_sum_json"]))
                decrease_hits_hi_sum_hist = json.loads(
                    str(validation_row["policy_action_decrease_hits_hi_sum_json"])
                )
                decrease_hits_lo_sum_hist = json.loads(
                    str(validation_row["policy_action_decrease_hits_lo_sum_json"])
                )
                unsafe_decrease_sum_hist = json.loads(str(validation_row["policy_action_unsafe_decrease_sum_json"]))
                increase_hist = json.loads(str(validation_row["policy_resolved_increase_task_hist_json"]))
                decrease_hist = json.loads(str(validation_row["policy_resolved_decrease_task_hist_json"]))
                for action_id_text, action_name in sorted(definitions.items(), key=lambda item: int(item[0])):
                    action_id = int(action_id_text)
                    count = int(hist.get(action_id_text, 0))
                    accepted_count = int(accepted_hist.get(action_id_text, 0))
                    rejected_count = int(rejected_hist.get(action_id_text, 0))
                    reward_sum = float(reward_sum_hist.get(action_id_text, 0.0))
                    lo_delta_sum = float(lo_delta_sum_hist.get(action_id_text, 0.0))
                    mode_delta_sum = float(mode_delta_sum_hist.get(action_id_text, 0.0))
                    is_increase_count = int(is_increase_sum_hist.get(action_id_text, 0))
                    is_decrease_count = int(is_decrease_sum_hist.get(action_id_text, 0))
                    is_transfer_count = int(is_transfer_sum_hist.get(action_id_text, 0))
                    decrease_hits_hi_count = int(decrease_hits_hi_sum_hist.get(action_id_text, 0))
                    decrease_hits_lo_count = int(decrease_hits_lo_sum_hist.get(action_id_text, 0))
                    unsafe_decrease_count = int(unsafe_decrease_sum_hist.get(action_id_text, 0))
                    action_type = "noop" if action_name == "noop" else action_name.split("|", maxsplit=1)[0]
                    if is_increase_count > 0:
                        action_direction = "increase"
                    elif is_decrease_count > 0:
                        action_direction = "decrease"
                    elif is_transfer_count > 0:
                        action_direction = "transfer"
                    elif action_type == "noop":
                        action_direction = "noop"
                    else:
                        action_direction = "other"
                    resolved_increase_tasks = {
                        key.split(":", maxsplit=1)[1]: int(value)
                        for key, value in increase_hist.items()
                        if key.startswith(f"{action_id_text}:")
                    }
                    resolved_decrease_tasks = {
                        key.split(":", maxsplit=1)[1]: int(value)
                        for key, value in decrease_hist.items()
                        if key.startswith(f"{action_id_text}:")
                    }
                    # 兼容 single action space 的可读列：
                    # - increase 取出现次数最多的 resolved 任务；
                    # - decrease 取出现次数最多的 resolved 任务。
                    resolved_increase_task = ""
                    if resolved_increase_tasks:
                        resolved_increase_task = max(
                            resolved_increase_tasks.items(),
                            key=lambda item: (item[1], item[0]),
                        )[0]
                    resolved_decrease_task = ""
                    if resolved_decrease_tasks:
                        resolved_decrease_task = max(
                            resolved_decrease_tasks.items(),
                            key=lambda item: (item[1], item[0]),
                        )[0]
                    policy_rows.append(
                        {
                            "episode": episode,
                            "action_id": action_id,
                            "action_name": str(action_name),
                            "action_type": action_type,
                            "action_direction": action_direction,
                            "count": count,
                            "accepted_count": accepted_count,
                            "rejected_count": rejected_count,
                            "accepted_rate": (float(accepted_count) / float(count)) if count > 0 else 0.0,
                            "is_increase_action": bool(is_increase_count > 0),
                            "is_decrease_action": bool(is_decrease_count > 0),
                            "is_transfer_action": bool(is_transfer_count > 0),
                            "decrease_hits_hi": bool(decrease_hits_hi_count > 0),
                            "decrease_hits_lo": bool(decrease_hits_lo_count > 0),
                            "unsafe_decrease_count": unsafe_decrease_count,
                            "unsafe_decrease_rate": (
                                float(unsafe_decrease_count) / float(max(1, count))
                            ),
                            "reward_sum": reward_sum,
                            "reward_mean": (reward_sum / float(count)) if count > 0 else 0.0,
                            "lo_delta_sum": lo_delta_sum,
                            "lo_delta_mean": (lo_delta_sum / float(count)) if count > 0 else 0.0,
                            "mode_delta_sum": mode_delta_sum,
                            "mode_delta_mean": (mode_delta_sum / float(count)) if count > 0 else 0.0,
                            "resolved_increase_task": resolved_increase_task,
                            "resolved_decrease_task": resolved_decrease_task,
                            "resolved_increase_tasks_json": json.dumps(
                                resolved_increase_tasks, ensure_ascii=False, sort_keys=True
                            ),
                            "resolved_decrease_tasks_json": json.dumps(
                                resolved_decrease_tasks, ensure_ascii=False, sort_keys=True
                            ),
                        }
                    )
            with validation_policy_actions_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "action_id",
                        "action_name",
                        "action_type",
                        "action_direction",
                        "count",
                        "accepted_count",
                        "rejected_count",
                        "accepted_rate",
                        "is_increase_action",
                        "is_decrease_action",
                        "is_transfer_action",
                        "decrease_hits_hi",
                        "decrease_hits_lo",
                        "unsafe_decrease_count",
                        "unsafe_decrease_rate",
                        "reward_sum",
                        "reward_mean",
                        "lo_delta_sum",
                        "lo_delta_mean",
                        "mode_delta_sum",
                        "mode_delta_mean",
                        "resolved_increase_task",
                        "resolved_decrease_task",
                        "resolved_increase_tasks_json",
                        "resolved_decrease_tasks_json",
                    ],
                )
                writer.writeheader()
                writer.writerows(policy_rows)
        # 阶段 0/1 统一 summary：保持与评估脚本一致的核心字段命名，便于直接横向对比。
        validation_unified_summary_rows = _build_validation_unified_summary_rows(validation_rows)
        validation_unified_summary_fields = [
            "episode",
            "validation_seed_count",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "baseline_lc_service_loss_mean",
            "baseline_lc_qos_mean",
            "dqn_mode_changes_mean",
            "dqn_lo_cancellations_mean",
            "dqn_lc_service_loss_mean",
            "dqn_lc_qos_mean",
            "relative_score",
            "relative_score_alpha",
            "relative_delta_mode_changes",
            "relative_delta_lo_cancellations",
            "raw_delta_mode_changes",
            "raw_delta_lo_cancellations",
            "is_better_than_baseline",
            "mode_changes_delta_vs_baseline",
            "lo_cancellations_delta_vs_baseline",
            "mode_change_ratio",
            "lo_cancellation_ratio",
            "relative_lc_loss_reduction",
            "lc_service_loss_delta_mean",
            "lc_qos_delta_mean",
            "min_lc_service_mean",
            "mode_change_delta_ratio",
            "hi_deadline_misses_sum",
            "qos_stable_valid_delta000",
            "qos_stable_valid_delta005",
            "qos_stable_valid_delta010",
            "best_candidate_rank_key",
            "accepted_action_count_mean",
            "rejected_action_count_mean",
            "noop_action_count_mean",
            "noop_action_rate_mean",
            "explicit_noop_action_rate_mean",
            "accepted_action_rate_mean",
            "rejected_action_rate_mean",
            "masked_action_count_mean",
            "valid_action_count_mean",
        ]
        validation_unified_summary_fields.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
        with validation_unified_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_unified_summary_fields)
            writer.writeheader()
            writer.writerows(validation_unified_summary_rows)

    # 无论是否启用 elite replay，都固定写出该日志文件与 header，方便下游脚本稳态解析。
    elite_replay_fieldnames = [
        "episode",
        "enabled",
        "active",
        "candidate",
        "reason",
        "threshold",
        "current_reduction",
        "best_reduction",
        "added_count",
        "buffer_size",
        "recent_episode_start",
        "recent_episode_end",
        "elite_samples_used_total",
        "normal_samples_used_total",
        "best_enabled",
        "best_active",
        "best_candidate",
        "best_reason",
        "best_current_reduction",
        "best_best_reduction",
        "best_added_count",
        "best_buffer_size",
        "best_recent_episode_start",
        "best_recent_episode_end",
        "best_samples_used_total",
        "best_transitions_added_total",
    ]
    with elite_replay_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=elite_replay_fieldnames)
        writer.writeheader()
        writer.writerows(elite_replay_rows)

    agent.save(model_path)
    if validation_rows:
        best_model_metadata = _build_best_metadata(
            save_best_by=args.save_best_by,
            reward_mode=args.reward_mode,
            reward_definition=reward_mode_config.describe(),
            double_dqn=args.double_dqn,
            relative_score_alpha=args.relative_score_alpha,
            require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
            qos_stable_mode_delta=args.qos_stable_mode_delta,
            best_type=args.save_best_by,
            best_row=best_validation_row,
        )
        with best_model_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(best_model_metadata, f, ensure_ascii=False, indent=2)
        if best_validation_row is None:
            print(f"WARNING: No valid checkpoint found for save_best_by={args.save_best_by}.")
        elif float(best_validation_row.get("relative_score", 0.0)) >= 0.0:
            # 文档要求：即使当前 best 仍劣于 baseline，也必须保留 best checkpoint。
            # 因此这里只打印提示，不中断、不回滚、不跳过保存。
            print("WARNING: Best available checkpoint is still worse than baseline on validation.")
            print("Saved anyway for trend analysis: model_best.pt")
        if args.save_all_best_types:
            for best_type in ("conservative_qos", "qos_stable", "qos_best"):
                metadata_path = output_dir / f"best_model_metadata_{best_type}.json"
                metadata = _build_best_metadata(
                    save_best_by=args.save_best_by,
                    reward_mode=args.reward_mode,
                    reward_definition=reward_mode_config.describe(),
                    double_dqn=args.double_dqn,
                    relative_score_alpha=args.relative_score_alpha,
                    require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                    qos_stable_mode_delta=args.qos_stable_mode_delta,
                    best_type=best_type,
                    best_row=best_rows_by_type[best_type],
                )
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
    config_payload = {
        "dqn_config": asdict(config),
        "workload": args.workload,
        "scenario": args.scenario,
        "scenario_name": initial_bundle.scenario.name,
        "total_util": args.total_util,
        "num_tasks": (
            args.mc_fairgen_num_tasks if args.workload == "mc_fairgen" else args.num_tasks
        ),
        "cf": args.cf,
        "cp": args.cp,
        "require_schedulable": args.require_schedulable,
        "hi_overrun_prob": args.hi_overrun_prob,
        "lo_overrun_prob": args.lo_overrun_prob,
        "lo_overrun_factor": args.lo_overrun_factor,
        "train_seed_mode": args.train_seed_mode,
        "train_seeds": episode_seed_schedule,
        "scenario_seed_offset": args.scenario_seed_offset,
        "fixed_taskset_seed": args.fixed_taskset_seed,
        "mc_fairgen_mode": args.mc_fairgen_mode,
        "mc_fairgen_num_tasks": args.mc_fairgen_num_tasks,
        "mc_fairgen_hi_ratio": args.mc_fairgen_hi_ratio,
        "mc_fairgen_period_source": args.mc_fairgen_period_source,
        "mc_fairgen_period_scale": args.mc_fairgen_period_scale,
        "mc_fairgen_tick_ns": 10,
        "mc_fairgen_u_hi_lo_min": args.mc_fairgen_u_hi_lo_min,
        "mc_fairgen_u_hi_lo_max": args.mc_fairgen_u_hi_lo_max,
        "mc_fairgen_u_hi_hi_min": args.mc_fairgen_u_hi_hi_min,
        "mc_fairgen_u_hi_hi_max": args.mc_fairgen_u_hi_hi_max,
        "mc_fairgen_u_lo_lo_min": args.mc_fairgen_u_lo_lo_min,
        "mc_fairgen_u_lo_lo_max": args.mc_fairgen_u_lo_lo_max,
        "mc_fairgen_hi_budget_rho_min": args.mc_fairgen_hi_budget_rho_min,
        "mc_fairgen_hi_budget_rho_max": args.mc_fairgen_hi_budget_rho_max,
        "mc_fairgen_lo_budget_rho_min": args.mc_fairgen_lo_budget_rho_min,
        "mc_fairgen_lo_budget_rho_max": args.mc_fairgen_lo_budget_rho_max,
        "mc_fairgen_hi_overrun_prob": args.mc_fairgen_hi_overrun_prob,
        "mc_fairgen_lo_overrun_prob": args.mc_fairgen_lo_overrun_prob,
        "mc_fairgen_hi_overrun_factor_min": args.mc_fairgen_hi_overrun_factor_min,
        "mc_fairgen_hi_overrun_factor_max": args.mc_fairgen_hi_overrun_factor_max,
        "mc_fairgen_lo_overrun_factor_min": args.mc_fairgen_lo_overrun_factor_min,
        "mc_fairgen_lo_overrun_factor_max": args.mc_fairgen_lo_overrun_factor_max,
        "validation_seeds": validation_seeds,
        "validate_every": args.validate_every,
        "validation_end_time": args.validation_end_time,
        "validation_workers": args.validation_workers,
        "max_q_diagnostic_samples": args.max_q_diagnostic_samples,
        "save_best_by": args.save_best_by,
        "qos_stable_mode_delta": args.qos_stable_mode_delta,
        "save_all_best_types": args.save_all_best_types,
        "relative_score_alpha": args.relative_score_alpha,
        "relative_score_normalization": "baseline_mean_denominator",
        "reward_mode": args.reward_mode,
        "reward_definition": reward_mode_config.describe(),
        "double_dqn": args.double_dqn,
        "use_elite_replay": args.use_elite_replay,
        "elite_start_episode": args.elite_start_episode,
        "elite_replay_capacity": args.elite_replay_capacity,
        "elite_replay_min_size": args.elite_replay_min_size,
        "elite_batch_size": args.elite_batch_size,
        "elite_score_min": args.elite_score_min,
        "elite_score_ratio": args.elite_score_ratio,
        "elite_recent_episodes": args.elite_recent_episodes,
        "elite_max_mode_delta": args.elite_max_mode_delta,
        "elite_require_no_hi_miss": args.elite_require_no_hi_miss,
        "elite_require_qos_stable": args.elite_require_qos_stable,
        "elite_max_add_per_validation": args.elite_max_add_per_validation,
        "elite_transitions_added_total": int(agent.elite_transitions_added_total),
        "elite_samples_used_total": int(agent.elite_samples_used_total),
        "use_best_elite_replay": args.use_best_elite_replay,
        "best_elite_replay_capacity": args.best_elite_replay_capacity,
        "best_elite_replay_min_size": args.best_elite_replay_min_size,
        "best_elite_batch_size": args.best_elite_batch_size,
        "best_elite_min_improvement": args.best_elite_min_improvement,
        "best_elite_recent_episodes": args.best_elite_recent_episodes,
        "best_elite_start_episode": args.best_elite_start_episode,
        "best_elite_max_add_per_validation": args.best_elite_max_add_per_validation,
        "best_elite_replace_on_new_best": args.best_elite_replace_on_new_best,
        "best_elite_transitions_added_total": int(agent.best_elite_transitions_added_total),
        "best_elite_samples_used_total": int(agent.best_elite_samples_used_total),
        "best_elite_replay_final_buffer_size": int(agent.best_elite_replay_size),
        "normal_samples_used_total": int(agent.normal_samples_used_total),
        "elite_replay_final_buffer_size": int(agent.elite_replay_size),
        "dqn_device_requested": args.dqn_device,
        "dqn_device_resolved": str(agent.device),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_device_name": (
            torch.cuda.get_device_name(agent.device.index if agent.device.index is not None else 0)
            if agent.device.type == "cuda"
            else None
        ),
        "q_network_type": args.q_network_type,
        "action_feature_mode": args.action_feature_mode,
        "action_aware_mask_mode": args.action_aware_mask_mode,
        "exploration_mode": args.exploration_mode,
        "safe_increase_explore_prob": args.safe_increase_explore_prob,
        "plateau_balanced_start_episode": args.plateau_balanced_start_episode,
        "plateau_balanced_window": args.plateau_balanced_window,
        "plateau_balanced_burst_episodes": args.plateau_balanced_burst_episodes,
        "plateau_balanced_mix_prob": args.plateau_balanced_mix_prob,
        "plateau_balanced_max_best_reduction": args.plateau_balanced_max_best_reduction,
        "plateau_balanced_reset_counts_on_burst": args.plateau_balanced_reset_counts_on_burst,
        "plateau_balanced_total_bursts": agent.plateau_balanced_burst_count,
        "plateau_balanced_total_actions": agent.plateau_balanced_action_count,
        "plateau_balanced_total_fallbacks": agent.plateau_balanced_fallback_count,
        "noop_exploration_prob": args.noop_exploration_prob,
        "action_feature_names": list(action_feature_names or ()),
        "action_feature_dim": 0 if action_feature_names is None else len(action_feature_names),
        "increase_action_ids": list(increase_action_ids),
        "increase_action_id_count": len(increase_action_ids),
        "requested_action_space": requested_action_space,
        "action_space": args.action_space,
        "constraint_guided_transfer_top_k_risk": args.constraint_guided_pair_top_k_risk,
        "constraint_guided_transfer_top_k_decrease": args.constraint_guided_pair_top_k_decrease,
        "constraint_guided_transfer_prefer_lo": args.constraint_guided_pair_prefer_lo,
        "constraint_guided_transfer_include_hi_risk_boost": args.constraint_guided_pair_include_hi_risk_boost,
        "constraint_guided_pair_top_k_risk": args.constraint_guided_pair_top_k_risk,
        "constraint_guided_pair_top_k_decrease": args.constraint_guided_pair_top_k_decrease,
        "constraint_guided_pair_prefer_lo": args.constraint_guided_pair_prefer_lo,
        "constraint_guided_pair_include_hi_risk_boost": args.constraint_guided_pair_include_hi_risk_boost,
        "constraint_guided_pair_allow_increase_only_when_safe": (
            args.constraint_guided_pair_allow_increase_only_when_safe
        ),
        "enable_residual_safety_fallback": args.enable_residual_safety_fallback,
        "residual_guard_hi_pressure_delta_limit": args.residual_guard_hi_pressure_delta_limit,
        "residual_guard_hi_pressure_abs_limit": args.residual_guard_hi_pressure_abs_limit,
        "residual_guard_reject_decrease_pressure_threshold": (
            args.residual_guard_reject_decrease_pressure_threshold
        ),
        "residual_guard_use_hi_pressure_max": args.residual_guard_use_hi_pressure_max,
        "budget_increase_ratio": args.budget_increase_ratio,
        "budget_decrease_ratio": args.budget_decrease_ratio,
        "budget_floor_ratio": args.budget_floor_ratio,
        "include_explicit_noop": args.include_explicit_noop,
        "forbid_decreasing_hi_budgets": args.forbid_decreasing_hi_budgets,
        "mask_detail_mode": args.mask_detail_mode,
        "observation_mode": args.observation_mode,
        "feature_config": {
            "ema_alpha": args.ema_alpha,
            "overrun_ema_alpha": args.overrun_ema_alpha,
            "history_k": args.history_k,
            "event_window": args.event_window,
            "max_cost_weight": args.max_cost_weight,
            "risk_max_scale": args.risk_max_scale,
            "include_safety_margin": args.include_safety_margin,
        },
        "log_step_every": args.log_step_every,
        "log_train_metrics": args.log_train_metrics,
        "trace_every": args.trace_every,
        "seed_metadata": {
            "base_seed": args.seed,
            "network_seed": network_seed,
            "exploration_seed": exploration_seed,
            "replay_seed": replay_seed,
        },
        "normalization_bounds": {
            task_name: {"min_cost": bound.min_cost, "max_cost": bound.max_cost}
            for task_name, bound in initial_bundle.normalization_bounds.items()
        },
        "action_space_size": initial_env.action_space_size,
        "observation_dim": len(initial_obs.state_vector),
        "tasks": _serialize_tasks(list(initial_bundle.ordered_tasks)),
        "runtime_config": {
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "semantics": RuntimeSemantics.AMC_PLUS.value,
        },
        "effective_taskset_seed": (
            (initial_bundle.metadata or {}).get("workload_metadata", {}).get("effective_taskset_seed")
            if isinstance((initial_bundle.metadata or {}).get("workload_metadata", {}), dict)
            else None
        ),
        "provider_attempts": (
            (initial_bundle.metadata or {}).get("workload_metadata", {}).get("attempts")
            if isinstance((initial_bundle.metadata or {}).get("workload_metadata", {}), dict)
            else None
        ),
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
