"""正式 DQN 评估命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 与训练脚本保持一致：当用户直接执行
    # `python scripts/evaluate_dqn_amc.py ...` 时，确保仓库根目录进入
    # `sys.path`，这样 `amc_py` 包导入行为与测试/README 中的调用方式一致。
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from amc_py.dqn import (
    DqnBudgetAgent,
    build_automotive_experiment_config,
    build_mc_fairgen_experiment_config,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.experiments import evaluate_taskset
from amc_py.metrics import (
    compute_lo_quality_weighted_metrics,
    compute_lo_job_loss_breakdown_metrics,
    compute_runtime_degradation_metrics,
    compute_service_quality_metrics,
    lo_job_loss_breakdown_to_row,
    lo_quality_weighted_metrics_to_row,
    mean_optional as mean_optional_service_metric,
    safe_relative_reduction,
    service_metrics_to_row,
)
from amc_py.rl.actions import (
    build_budget_action_space,
    compute_action_space_fingerprint,
)
from amc_py.rl.agents import HeuristicBudgetAgent, NoOpBudgetAgent, RandomBudgetAgent
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.reward_config import available_reward_modes
from amc_py.rl.runtime_wrapper import (
    AgentRuntimeConfig,
    AgentRuntimeResult,
    simulate_ordered_taskset_with_agent,
)
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult
from amc_py.qamc.metrics_support import compute_qamc_metrics, qamc_metrics_to_row
from amc_py.qamc.loss_metrics import (
    compute_qamc_loss_metrics,
    qamc_loss_metrics_to_row,
)
from amc_py.qamc.profiles import load_profile_bundle_from_manifest
from amc_py.qamc.reference_config import (
    assert_reference_matches_values,
    load_and_validate_frozen_reference,
    validate_qamc_model_artifact,
)
from amc_py.qamc.profile_spec import load_profile_spec
from amc_py.qamc.heuristic import QAmcBudgetPressureHeuristic
from amc_py.qamc.rl_contract import validate_qamc_rl_semantics


QAMC_METHOD_ALIASES = {
    "q_amc_budget_heuristic": "heuristic_agent",
    "q_amc_dqn_budget_overlay": "dqn_agent",
    "q_amc_viper_budget_overlay": "viper_tree_agent",
}
QAMC_OUTPUT_METHOD_NAMES = {
    value: key for key, value in QAMC_METHOD_ALIASES.items()
}


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

DEGRADATION_FIELDNAMES = [
    "hdm",
    "jne",
    "ldm",
    "nid",
    "tid",
    "total_time",
    "tid_ratio",
    "nid_per_1e6_time",
    "mean_degraded_interval",
    "safety_feasible",
    "jne_plus_ldm",
    "lo_job_losses_total",
    "lo_budget_cancellations",
    "lo_release_dropped_in_degraded_mode",
    "lo_active_dropped_on_mode_switch",
    "jne_residual_not_in_cancellations",
    "active_drop_share_of_jne",
]

QOS_FIELDNAMES = [
    "released_lo_jobs",
    "cancelled_lo_jobs",
    "completed_lo_jobs",
    "lo_deadline_misses",
    "hi_deadline_misses",
    "lc_service_loss",
    "lc_qos",
    "min_lc_service",
    "budget_adjust_count",
    "mean_abs_budget_change",
]

LO_QUALITY_WEIGHTED_FIELDNAMES = [
    "lo_equiv_jne",
    "lo_equiv_jne_rate",
    "lo_quality_qos",
    "lo_quality_loss",
    "lo_full_quality_completed",
    "lo_full_quality_ratio",
    "lo_degraded_released",
    "lo_degraded_completed",
    "lo_degraded_cancelled",
    "lo_degraded_deadline_missed",
    "lo_degraded_not_completed",
    "lo_degraded_release_ratio",
    "lo_degraded_completion_ratio",
    "lo_degraded_among_completed_ratio",
    "lo_degraded_quality_sum",
    "lo_degraded_budget_sum",
    "lo_degraded_original_budget_sum",
    "lo_degraded_budget_ratio_mean",
    "lo_degraded_exec_time_sum",
    "lo_degraded_exec_time_ratio",
    "lo_zero_service_jobs",
    "lo_zero_service_ratio",
    "lo_full_quality_service_sum",
    "lo_total_service_sum",
]

LEGACY_DEGRADED_FIELDS = (
    "lo_degraded_released",
    "lo_degraded_completed",
    "lo_degraded_cancelled",
    "lo_degraded_deadline_missed",
    "lo_degraded_not_completed",
    "lo_degraded_release_ratio",
    "lo_degraded_completion_ratio",
    "lo_degraded_among_completed_ratio",
    "lo_degraded_quality_sum",
    "lo_degraded_budget_sum",
    "lo_degraded_original_budget_sum",
    "lo_degraded_budget_ratio_mean",
    "lo_degraded_exec_time_sum",
    "lo_degraded_exec_time_ratio",
)


def _blank_legacy_degraded_fields(row: dict[str, object]) -> None:
    for name in LEGACY_DEGRADED_FIELDS:
        row[name] = None

TASK_LEVEL_INFO_KEYS = [
    "final_budget_ratio_by_task_json",
    "max_budget_ratio_by_task_json",
    "min_budget_ratio_by_task_json",
    "increase_count_by_task_json",
    "decrease_count_by_task_json",
    "recovery_decrease_count_by_task_json",
    "over_increase_count_by_task_json",
    "consecutive_increase_max_by_task_json",
    "over_budget_dwell_steps_by_task_json",
    "soft_cap_dwell_steps_by_task_json",
]


def _parse_seeds(raw_value: str) -> list[int]:
    """将种子字符串解析为整数列表，支持 `a:b` 与逗号列表。"""

    seeds: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
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


def _parse_baselines(raw_value: str) -> list[str]:
    """将逗号分隔 baseline 列表解析为方法名列表。"""

    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _to_float(row: dict[str, int | float | str | bool], key: str) -> float:
    """把评估行中的数值字段统一转成 float。"""

    value = row.get(key, 0.0)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return 0.0
    return float(text)


def _safe_ratio(numerator: float, denominator: float) -> str | float:
    """计算比值并显式处理分母为 0 的情况。"""

    if denominator == 0.0:
        if numerator > 0.0:
            return "inf"
        return "nan"
    return numerator / denominator


def _optional_delta(left: object, right: object) -> float | None:
    if left in (None, "") or right in (None, ""):
        return None
    return float(left) - float(right)


def _budget_overruns_from_result(result: SimulationResult) -> int:
    """在 AMC_PLUS 口径下估算 budget_overruns。"""

    return result.mode_change_count() + result.lo_job_cancellation_count()


def _empty_noop_q_diagnostics_row() -> dict[str, float | int | None]:
    """生成非 DQN 方法使用的空 noop Q 诊断字段。

    baseline、random、heuristic、noop agent 没有 policy network Q 值，因此这些字段按文档要求写空值。
    """

    return {fieldname: None for fieldname in NOOP_Q_DIAGNOSTIC_FIELDNAMES}


def _degradation_metrics_to_row(result: SimulationResult) -> dict[str, int | float | None]:
    """把论文 degraded-service 指标展平成 evaluate CSV 行字段。

    这里严格复用统一指标实现，避免在评估脚本中重复推导 hdm/jne/ldm/nid/tid，
    从而保证与 pre-DQN baseline 脚本和其他分析工具的统计口径完全一致。
    """

    degradation = compute_runtime_degradation_metrics(result)
    loss_breakdown = compute_lo_job_loss_breakdown_metrics(result, degradation)
    return {
        "hdm": degradation.hdm,
        "jne": degradation.jne,
        "ldm": degradation.ldm,
        "nid": degradation.nid,
        "tid": degradation.tid,
        "total_time": degradation.total_time,
        "tid_ratio": degradation.tid_ratio,
        "nid_per_1e6_time": degradation.nid_per_1e6_time,
        "mean_degraded_interval": degradation.mean_degraded_interval,
        "safety_feasible": degradation.safety_feasible,
        "jne_plus_ldm": degradation.jne + degradation.ldm,
        **lo_job_loss_breakdown_to_row(loss_breakdown),
    }


def _lo_quality_weighted_metrics_to_row_from_result(
    result: SimulationResult,
) -> dict[str, int | float | None]:
    """把单次 runtime 结果转成质量加权 LO 指标行。"""

    metrics = compute_lo_quality_weighted_metrics(result)
    return lo_quality_weighted_metrics_to_row(metrics)


def _baseline_runtime_config(
    *,
    end_time: int,
    semantics: RuntimeSemantics,
    capture_trace: bool = False,
    capture_debug_events: bool = False,
    c_amc_sem_xf: float = 0.5,
) -> RuntimeConfig:
    """构造正式 HOUT 评估使用的 baseline runtime 配置。

    正式长时域评估必须显式关闭逐 tick trace，避免 `end_time=2e7/5e7` 时产生
    巨量内存与 IO 开销。同时所有语义都统一记录 degraded mode 中被丢弃/抑制的
    LO release，确保 `JNE + LDM` 在 AMC+、RA、RH 之间可公平横向比较。
    """

    return RuntimeConfig(
        end_time=end_time,
        semantics=semantics,
        capture_trace=capture_trace,
        capture_debug_events=capture_debug_events,
        record_dropped_lo_releases=True,
        drop_lo_jobs_on_hi_switch=(semantics is not RuntimeSemantics.C_AMC_SEM),
        c_amc_sem_lo_degradation_ratio=c_amc_sem_xf,
        c_amc_sem_primary_on_switch_time=(semantics is RuntimeSemantics.C_AMC_SEM),
    )


def _formal_agent_runtime_config(
    *,
    end_time: int,
    semantics: RuntimeSemantics,
    capture_trace: bool = False,
    capture_debug_events: bool = False,
    c_amc_sem_xf: float = 0.5,
) -> RuntimeConfig:
    """构造正式评估下 agent/DQN 共用的 runtime 配置。"""

    return RuntimeConfig(
        end_time=end_time,
        semantics=semantics,
        capture_trace=capture_trace,
        capture_debug_events=capture_debug_events,
        record_dropped_lo_releases=True,
        drop_lo_jobs_on_hi_switch=(semantics is not RuntimeSemantics.C_AMC_SEM),
        c_amc_sem_lo_degradation_ratio=c_amc_sem_xf,
        c_amc_sem_primary_on_switch_time=(semantics is RuntimeSemantics.C_AMC_SEM),
    )


def _noop_q_diagnostics_to_row(agent: DqnBudgetAgent, states: list[tuple[float, ...]], masks: list[tuple[bool, ...]]) -> dict[str, float | int | None]:
    """把评估期采集的决策状态转换为 explicit noop Q 诊断字段。

    这些状态均在 agent 调用 `select_action_id()` 之前记录，因此 Q 值诊断反映的是
    实际 greedy 决策时刻 noop 在合法动作集合中的相对位置。
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


def _mean_optional_metric(rows: list[dict[str, int | float | str | bool]], key: str) -> float | None:
    """对可能为空的 noop Q 诊断字段求均值。"""

    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return mean(values)


def _task_level_info_row(info: dict[str, int | float | str | bool | None]) -> dict[str, str]:
    """从环境 info 中提取 task-level JSON 日志。

    这里统一把缺失字段写成空字符串，保证 CSV 列稳定且兼容旧 checkpoint / 旧评估结果。
    """

    return {key: str(info.get(key, "")) if info.get(key, "") is not None else "" for key in TASK_LEVEL_INFO_KEYS}


def _eval_summary_fieldnames() -> list[str]:
    """返回 evaluate 输出 CSV 的固定 header。

    抽成 helper 的目的是让测试可以直接断言列集合，而无需真正跑完整 evaluate 流程。
    """

    fieldnames = [
        "workload",
        "total_util",
        "num_tasks",
        "cf",
        "cp",
        "seed",
        "taskset_seed",
        "scenario_seed",
        "method",
        "q_network_type",
        "amc_rtb_schedulable",
        "attempts",
        "end_time",
        "agent_period",
        "dqn_runtime_semantics",
        "c_amc_sem_xf",
        "reward_mode",
        "forbid_decreasing_hi_budgets",
        "enable_deploy_cap_mask",
        "deploy_cap_mask_ratio",
        "deploy_cap_mask_criticality",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "budget_overruns",
    ]
    fieldnames.extend(DEGRADATION_FIELDNAMES)
    fieldnames.extend(
        [
            "accepted_actions",
            "rejected_actions",
            "step_count",
            "selected_action_count",
            "noop_actions",
            "explicit_noop_actions",
            "noop_action_rate",
            "explicit_noop_action_rate",
            "accepted_action_rate",
            "rejection_rate",
            "total_reward",
            "check_safety",
            "safety_checked_actions",
            "safety_accepted_actions",
            "safety_rejected_actions",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "masked_decrease_hi_forbidden_count",
            "masked_decrease_hi_forbidden_rate",
            "masked_action_count_max",
            "mask_rejection_rate_mean",
            "selected_invalid_mask_actions",
            "selected_explicit_noop_actions",
            "selected_explicit_noop_rate",
            "action_space_type",
            "action_count",
            "budget_increase_ratio",
            "budget_decrease_ratio",
            "budget_floor_ratio",
            "no_safe_action_steps",
            "masked_budget_floor_violation_count",
            "masked_budget_floor_violation_rate",
            "masked_deploy_cap_increase_count",
            "masked_deploy_cap_increase_rate",
            "observation_mode",
            "state_dim",
            "tree_runtime_policy_type",
            "tree_state_encoding",
            "tree_fixed_point_scale",
            "tree_fixed_point_config_hash",
            "tree_artifact_schema_version",
            "integer_equivalence_verified",
            "mean_over_increase_excess",
            "over_increase_action_count",
            "mean_budget_soft_cap_increase_excess",
            "soft_cap_increase_action_count",
            "mean_budget_soft_cap_penalty_value",
            "mean_budget_soft_cap_dwell_excess_mean",
            "mean_budget_soft_cap_dwell_excess_max",
            "max_budget_soft_cap_dwell_excess_max",
            "soft_cap_dwell_state_count",
            "soft_cap_dwell_state_rate",
            "mean_budget_soft_cap_dwell_penalty_value",
            "mean_budget_soft_cap_dwell_max_penalty_value",
            "mean_budget_soft_cap_dwell_total_penalty_value",
            "safe_recovery_decrease_count",
            "unsafe_decrease_full_count",
            "mean_budget_over_drift_deadzone",
            "mean_increase_concentration_excess",
            "pingpong_action_count",
            "mean_active_lo_under_hi_pressure",
            "mean_active_lo_work_ratio",
            "mean_active_lo_job_rate",
            "mean_active_lo_under_hi_pressure_penalty_value",
        ]
    )
    fieldnames.extend(TASK_LEVEL_INFO_KEYS)
    fieldnames.extend(QOS_FIELDNAMES)
    fieldnames.extend(LO_QUALITY_WEIGHTED_FIELDNAMES)
    fieldnames.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
    fieldnames.extend(
        [
            "tree_id",
            "tree_method",
            "tree_depth",
            "tree_node_count",
            "tree_leaf_count",
            "tree_max_depth_param",
            "tree_min_samples_leaf",
            "tree_raw_top1_invalid_count",
            "tree_raw_top1_invalid_rate",
            "tree_fallback_count",
            "tree_fallback_rate",
            "tree_no_valid_action_count",
            "tree_no_valid_action_rate",
            "tree_selected_action_count",
            "tree_selected_action_match_teacher_count",
            "tree_selected_action_match_teacher_rate",
            "tree_raw_action_match_teacher_rate",
            "tree_q_regret_mean",
            "tree_q_regret_p95",
        ]
    )
    fieldnames.extend(
        [
            "qamc_release_count",
            "qamc_managed_release_count",
            "qamc_paper_quality_sum",
            "qamc_paper_quality_per_release",
            "qamc_normalized_provided_quality_sum",
            "qamc_normalized_quality_qos",
            "qamc_zero_service_count",
            "qamc_zero_service_ratio",
            "qamc_release_target_rank_mean",
            "qamc_release_target_normalized_mean",
            "qamc_completed_quality_conditional_mean",
            "qamc_completed_quality_unconditional_per_release",
            "qamc_overrun_stop_count",
            "qamc_quality_transition_count",
            "qamc_min_quality_exhaustion_count",
            "qamc_tasks_ever_degraded",
            "qamc_first_degradation_time",
            "qamc_runtime_level_depth_mean",
            "qamc_runtime_level_depth_max",
            "qamc_raw_rank_drop_mean",
            "qamc_raw_rank_drop_max",
            "qamc_release_count_by_raw_rank_json",
            "qamc_completed_count_by_raw_rank_json",
            "qamc_task_time_at_raw_rank_ratio_json",
            "qamc_non_degradable_task_count",
            "qamc_trigger_budget_mean_ratio_to_c_lo",
            "qamc_trigger_below_design_count",
            "qamc_trigger_equal_design_count",
            "qamc_trigger_above_design_count",
            "qamc_would_overrun_design_count",
            "qamc_dqn_budget_update_event_count",
            "qamc_dqn_budget_update_task_count",
            "qamc_viper_budget_update_event_count",
            "qamc_viper_budget_update_task_count",
            "qamc_heuristic_budget_update_event_count",
            "qamc_heuristic_budget_update_task_count",
            "qamc_offline_budget_update_event_count",
            "qamc_offline_budget_update_task_count",
            "qamc_unspecified_budget_update_event_count",
            "qamc_unspecified_budget_update_task_count",
            "qamc_profile_fingerprint",
            "qamc_legacy_degraded_metrics_applicable",
            "qamc_loss_released_lo_jobs",
            "qamc_loss_completed_positive_quality_jobs",
            "qamc_loss_overrun_stopped_zero_quality_jobs",
            "qamc_loss_deadline_lost_zero_quality_jobs",
            "qamc_loss_min_threshold_fallback_zero_quality_jobs",
            "qamc_loss_hi_mode_discard_zero_quality_jobs",
            "qamc_loss_other_zero_quality_jobs",
        ]
    )
    return fieldnames


def _empty_tree_diagnostics_row() -> dict[str, float | int | str | None]:
    """为非 tree 方法补齐空 tree 字段，避免 CSV 列口径漂移。"""

    return {
        "tree_id": None,
        "tree_method": None,
        "tree_depth": None,
        "tree_node_count": None,
        "tree_leaf_count": None,
        "tree_max_depth_param": None,
        "tree_min_samples_leaf": None,
        "tree_runtime_policy_type": None,
        "tree_state_encoding": None,
        "tree_fixed_point_scale": None,
        "tree_fixed_point_config_hash": None,
        "tree_artifact_schema_version": None,
        "integer_equivalence_verified": None,
        "tree_raw_top1_invalid_count": None,
        "tree_raw_top1_invalid_rate": None,
        "tree_fallback_count": None,
        "tree_fallback_rate": None,
        "tree_no_valid_action_count": None,
        "tree_no_valid_action_rate": None,
        "tree_selected_action_count": None,
        "tree_selected_action_match_teacher_count": None,
        "tree_selected_action_match_teacher_rate": None,
        "tree_raw_action_match_teacher_rate": None,
        "tree_q_regret_mean": None,
        "tree_q_regret_p95": None,
    }


def _build_pure_runtime_baseline_row(
    *,
    row_base: dict[str, int | float | str | bool],
    method: str,
    runtime_result: SimulationResult,
    action_space: str,
    action_count: int,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    budget_floor_ratio: float,
) -> dict[str, int | float | str | bool | None]:
    """构造不经过 agent 决策的纯 runtime baseline 结果行。

    AMC+、AMC-RA、AMC-RH 都属于“仅切换 runtime 语义、不引入动作决策”的 baseline。
    因此这里统一把动作、mask、Q 诊断相关字段置为 0 或空值，避免三处复制粘贴后口径漂移。
    """

    service_metrics = compute_service_quality_metrics(runtime_result)
    return {
        **row_base,
        **_empty_noop_q_diagnostics_row(),
        **_empty_tree_diagnostics_row(),
        "method": method,
        "mode_changes": runtime_result.mode_change_count(),
        "lo_cancellations": runtime_result.lo_job_cancellation_count(),
        "deadline_misses": len(runtime_result.deadline_misses),
        "budget_overruns": _budget_overruns_from_result(runtime_result),
        "accepted_actions": 0,
        "rejected_actions": 0,
        "step_count": 0,
        "selected_action_count": 0,
        "noop_actions": 0,
        "explicit_noop_actions": 0,
        "noop_action_rate": 0.0,
        "explicit_noop_action_rate": 0.0,
        "accepted_action_rate": 0.0,
        "rejection_rate": 0.0,
        "total_reward": 0.0,
        "check_safety": True,
        "safety_checked_actions": 0,
        "safety_accepted_actions": 0,
        "safety_rejected_actions": 0,
        "valid_action_count_mean": 0.0,
        "masked_action_count_mean": 0.0,
        "masked_decrease_hi_forbidden_count": 0,
        "masked_decrease_hi_forbidden_rate": 0.0,
        "masked_action_count_max": 0,
        "mask_rejection_rate_mean": 0.0,
        "selected_invalid_mask_actions": 0,
        "selected_explicit_noop_actions": 0,
        "selected_explicit_noop_rate": 0.0,
        "action_space_type": action_space,
        "action_count": action_count,
        "budget_increase_ratio": budget_increase_ratio,
        "budget_decrease_ratio": budget_decrease_ratio,
        "budget_floor_ratio": budget_floor_ratio,
        "no_safe_action_steps": 0,
        "masked_budget_floor_violation_count": 0,
        "masked_budget_floor_violation_rate": 0.0,
        "masked_deploy_cap_increase_count": 0,
        "masked_deploy_cap_increase_rate": 0.0,
        **_degradation_metrics_to_row(runtime_result),
        **service_metrics_to_row(service_metrics),
        **_lo_quality_weighted_metrics_to_row_from_result(runtime_result),
    }


def _aggregate_action_log_metrics(action_log: list[dict[str, object]]) -> dict[str, float | int]:
    """从 action_log 中聚合 single recovery reward 的关键诊断指标。

    这些字段直接来自 env.info，因此这里不做额外推断，只做简单均值/计数汇总。
    """

    step_count = len(action_log)
    if step_count == 0:
        return {
            "mean_over_increase_excess": 0.0,
            "over_increase_action_count": 0,
            "mean_budget_soft_cap_increase_excess": 0.0,
            "soft_cap_increase_action_count": 0,
            "mean_budget_soft_cap_penalty_value": 0.0,
            "mean_budget_soft_cap_dwell_excess_mean": 0.0,
            "mean_budget_soft_cap_dwell_excess_max": 0.0,
            "max_budget_soft_cap_dwell_excess_max": 0.0,
            "soft_cap_dwell_state_count": 0,
            "soft_cap_dwell_state_rate": 0.0,
            "mean_budget_soft_cap_dwell_penalty_value": 0.0,
            "mean_budget_soft_cap_dwell_max_penalty_value": 0.0,
            "mean_budget_soft_cap_dwell_total_penalty_value": 0.0,
            "safe_recovery_decrease_count": 0,
            "unsafe_decrease_full_count": 0,
            "mean_budget_over_drift_deadzone": 0.0,
            "mean_increase_concentration_excess": 0.0,
            "pingpong_action_count": 0,
            "mean_active_lo_under_hi_pressure": 0.0,
            "mean_active_lo_work_ratio": 0.0,
            "mean_active_lo_job_rate": 0.0,
            "mean_active_lo_under_hi_pressure_penalty_value": 0.0,
        }

    mean_over_increase_excess = mean(float(row.get("over_increase_excess", 0.0)) for row in action_log)
    over_increase_action_count = sum(int(bool(row.get("is_over_increase_action", False))) for row in action_log)
    mean_budget_soft_cap_increase_excess = mean(
        float(row.get("budget_soft_cap_increase_excess", 0.0)) for row in action_log
    )
    soft_cap_increase_action_count = sum(
        int(bool(row.get("is_soft_cap_increase_action", False))) for row in action_log
    )
    mean_budget_soft_cap_penalty_value = mean(
        float(row.get("budget_soft_cap_penalty_value", 0.0)) for row in action_log
    )
    mean_budget_soft_cap_dwell_excess_mean = mean(
        float(row.get("budget_soft_cap_dwell_excess_mean", 0.0)) for row in action_log
    )
    mean_budget_soft_cap_dwell_excess_max = mean(
        float(row.get("budget_soft_cap_dwell_excess_max", 0.0)) for row in action_log
    )
    max_budget_soft_cap_dwell_excess_max = max(
        float(row.get("budget_soft_cap_dwell_excess_max", 0.0)) for row in action_log
    )
    soft_cap_dwell_state_count = sum(
        int(bool(row.get("is_soft_cap_dwell_state", False))) for row in action_log
    )
    soft_cap_dwell_state_rate = soft_cap_dwell_state_count / float(max(step_count, 1))
    mean_budget_soft_cap_dwell_penalty_value = mean(
        float(row.get("budget_soft_cap_dwell_penalty_value", 0.0)) for row in action_log
    )
    mean_budget_soft_cap_dwell_max_penalty_value = mean(
        float(row.get("budget_soft_cap_dwell_max_penalty_value", 0.0)) for row in action_log
    )
    mean_budget_soft_cap_dwell_total_penalty_value = mean(
        float(row.get("budget_soft_cap_dwell_total_penalty_value", 0.0)) for row in action_log
    )
    safe_recovery_decrease_count = sum(int(bool(row.get("safe_recovery_decrease", False))) for row in action_log)
    unsafe_decrease_full_count = sum(int(bool(row.get("unsafe_decrease_full", False))) for row in action_log)
    mean_budget_over_drift_deadzone = mean(
        float(row.get("budget_over_drift_deadzone_mean", 0.0)) for row in action_log
    )
    mean_increase_concentration_excess = mean(
        float(row.get("increase_concentration_excess", 0.0)) for row in action_log
    )
    pingpong_action_count = sum(int(float(row.get("pingpong_action", 0.0)) > 0.0) for row in action_log)
    mean_active_lo_under_hi_pressure = mean(
        float(row.get("active_lo_under_hi_pressure", 0.0)) for row in action_log
    )
    mean_active_lo_work_ratio = mean(float(row.get("active_lo_work_ratio", 0.0)) for row in action_log)
    mean_active_lo_job_rate = mean(float(row.get("active_lo_job_rate", 0.0)) for row in action_log)
    mean_active_lo_under_hi_pressure_penalty_value = mean(
        float(row.get("active_lo_under_hi_pressure_penalty_value", 0.0)) for row in action_log
    )
    return {
        "mean_over_increase_excess": mean_over_increase_excess,
        "over_increase_action_count": over_increase_action_count,
        "mean_budget_soft_cap_increase_excess": mean_budget_soft_cap_increase_excess,
        "soft_cap_increase_action_count": soft_cap_increase_action_count,
        "mean_budget_soft_cap_penalty_value": mean_budget_soft_cap_penalty_value,
        "mean_budget_soft_cap_dwell_excess_mean": mean_budget_soft_cap_dwell_excess_mean,
        "mean_budget_soft_cap_dwell_excess_max": mean_budget_soft_cap_dwell_excess_max,
        "max_budget_soft_cap_dwell_excess_max": max_budget_soft_cap_dwell_excess_max,
        "soft_cap_dwell_state_count": soft_cap_dwell_state_count,
        "soft_cap_dwell_state_rate": soft_cap_dwell_state_rate,
        "mean_budget_soft_cap_dwell_penalty_value": mean_budget_soft_cap_dwell_penalty_value,
        "mean_budget_soft_cap_dwell_max_penalty_value": mean_budget_soft_cap_dwell_max_penalty_value,
        "mean_budget_soft_cap_dwell_total_penalty_value": (
            mean_budget_soft_cap_dwell_total_penalty_value
        ),
        "safe_recovery_decrease_count": safe_recovery_decrease_count,
        "unsafe_decrease_full_count": unsafe_decrease_full_count,
        "mean_budget_over_drift_deadzone": mean_budget_over_drift_deadzone,
        "mean_increase_concentration_excess": mean_increase_concentration_excess,
        "pingpong_action_count": pingpong_action_count,
        "mean_active_lo_under_hi_pressure": mean_active_lo_under_hi_pressure,
        "mean_active_lo_work_ratio": mean_active_lo_work_ratio,
        "mean_active_lo_job_rate": mean_active_lo_job_rate,
        "mean_active_lo_under_hi_pressure_penalty_value": (
            mean_active_lo_under_hi_pressure_penalty_value
        ),
    }


def _evaluate_dqn_once(
    *,
    model_path: Path,
    experiment_config,
    agent_period: int,
    seed: int,
    end_time: int,
    row_base: dict[str, int | float | str | bool],
    reward_mode: str,
    dqn_runtime_semantics: RuntimeSemantics,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    feature_config: FeatureConfig,
    c_amc_sem_xf: float = 0.5,
    trace_dir: Path | None = None,
    debug_log_dir: Path | None = None,
    trace_enabled: bool = False,
    capture_trace: bool = False,
    capture_debug_events: bool = False,
    agent_device: str | None = None,
    double_dqn: bool = True,
    max_q_diagnostic_samples: int = 1000,
    constraint_guided_pair_top_k_risk: int = 3,
    constraint_guided_pair_top_k_decrease: int = 5,
    constraint_guided_pair_prefer_lo: bool = False,
    constraint_guided_pair_include_hi_risk_boost: bool = False,
    constraint_guided_pair_allow_increase_only_when_safe: bool = False,
) -> tuple[dict[str, int | float | str | bool], SimulationResult, list[dict[str, object]]]:
    """以评估模式运行一次 DQN agent。"""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=dqn_runtime_semantics,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        enable_deploy_cap_mask=enable_deploy_cap_mask,
        deploy_cap_mask_ratio=deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=deploy_cap_mask_criticality,
        capture_trace=capture_trace,
        capture_debug_events=capture_debug_events,
        record_dropped_lo_releases=True,
        c_amc_sem_xf=c_amc_sem_xf,
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
    )
    # 评估阶段既支持主进程串行加载，也支持并行 worker 中按需加载。
    # 并行 worker 会显式传入 `agent_device="cpu"`，避免子进程去占用 MPS/GPU。
    agent = DqnBudgetAgent.load(model_path, device=agent_device)
    # 评估本身不执行 bootstrap target 计算，但仍保存 CLI 传入的 Double DQN 开关，
    # 让本次评估进程中的 agent 配置与训练/消融命令保持同一语义入口。
    agent.double_dqn = bool(double_dqn)
    if agent.action_dim != env.action_space_size:
        raise ValueError(
            "模型动作空间与环境不兼容："
            f"model.action_dim={agent.action_dim}, env.action_space_size={env.action_space_size}"
        )

    obs = env.reset(seed=seed)
    if getattr(agent, "q_network_type", "mlp") == "action_aware":
        action_features = env.get_action_feature_matrix(agent.action_feature_mode)
        action_feature_names = env.get_action_feature_names(agent.action_feature_mode)
        agent.set_action_features(action_features, action_feature_names)
    done = False
    # 评估期统计口径与训练期保持一致：
    # - step_count：总决策步；
    # - selected_action_count：action_id 非空步数；
    # - accepted/rejected：仅对非空 action_id 统计；
    # - noop/explicit_noop：由环境 info 提供事实标签。
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
    # 评估期 noop Q 诊断缓存：每一行对应一次 DQN agent 决策前的 observation 和合法动作 mask。
    # 结束后统一送入 agent.compute_noop_q_diagnostics，避免在每个 step 中重复做额外统计。
    diagnostic_states: list[tuple[float, ...]] = []
    diagnostic_valid_masks: list[tuple[bool, ...]] = []

    while not done:
        # 每次循环对应一次环境 step，所有 rate 都以该值为主分母。
        step_count += 1
        mask = env.valid_action_mask()
        # dynamic_v1 为状态相关特征，评估时也必须逐步刷新。
        if getattr(agent, "q_network_type", "mlp") == "action_aware" and agent.action_feature_mode == "dynamic_v1":
            action_features = env.get_action_feature_matrix(agent.action_feature_mode)
            action_feature_names = env.get_action_feature_names(agent.action_feature_mode)
            agent.set_action_features(action_features, action_feature_names)
        if len(diagnostic_states) < max_q_diagnostic_samples:
            diagnostic_states.append(tuple(float(value) for value in obs.state_vector))
            diagnostic_valid_masks.append(tuple(bool(value) for value in mask))
        action_id = agent.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
        # 是否提供 action_id 与动作是否 accepted/noop 是两个独立维度。
        selected_action_count += int(action_id is not None)

        result = env.step(action_id)
        total_reward += result.reward
        last_info = result.info

        if bool(result.info.get("is_noop", False)):
            noop_actions += 1
            if bool(result.info.get("is_explicit_noop_action", False)):
                # 只有环境显式标记为 explicit noop，才计入 explicit_noop_actions。
                explicit_noop_actions += 1

        if action_id is not None:
            if bool(result.info.get("accepted")):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done

    budget_overruns = 0
    if hasattr(env, "_engine") and env._engine is not None:
        budget_overruns = _budget_overruns_from_result(env._engine.finish())
    debug_stats = env.debug_statistics()
    action_log_metrics = _aggregate_action_log_metrics(env.action_log)
    # 阶段 2：所有动作 rate 都基于 step_count，避免 explicit noop 双重计数导致分母膨胀。
    # rejected_action_rate 沿用历史字段名 `rejection_rate`，语义等价于 rejected/step_count。
    rejection_rate = (rejected_actions / step_count) if step_count > 0 else 0.0
    noop_action_rate = (noop_actions / step_count) if step_count > 0 else 0.0
    explicit_noop_action_rate = (explicit_noop_actions / step_count) if step_count > 0 else 0.0
    accepted_action_rate = (accepted_actions / step_count) if step_count > 0 else 0.0
    if trace_enabled and hasattr(env, "_engine") and env._engine is not None:
        _write_agent_debug_files(
            trace_dir=trace_dir,
            debug_log_dir=debug_log_dir,
            seed=seed,
            method="dqn_agent",
            action_log=env.action_log,
            mask_log=env.mask_log,
            runtime_result=env._engine.finish(),
        )

    runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
    dqn_service_metrics = compute_service_quality_metrics(runtime_result)
    row = {
            **row_base,
            **_empty_tree_diagnostics_row(),
            "method": "dqn_agent",
            "q_network_type": str(getattr(agent, "q_network_type", "mlp")),
            "mode_changes": int(last_info.get("mode_changes", 0)),
            "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
            "deadline_misses": int(last_info.get("deadline_misses", 0)),
            "budget_overruns": budget_overruns,
            "accepted_actions": accepted_actions,
            "rejected_actions": rejected_actions,
            "step_count": step_count,
            "selected_action_count": selected_action_count,
            "noop_actions": noop_actions,
            "explicit_noop_actions": explicit_noop_actions,
            "noop_action_rate": noop_action_rate,
            "explicit_noop_action_rate": explicit_noop_action_rate,
            "accepted_action_rate": accepted_action_rate,
            "rejection_rate": rejection_rate,
            "total_reward": total_reward,
            "check_safety": bool(debug_stats["check_safety"]),
            "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
            "safety_accepted_actions": int(debug_stats["safety_accepted_actions"]),
            "safety_rejected_actions": int(debug_stats["safety_rejected_actions"]),
            "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
            "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
            "masked_decrease_hi_forbidden_count": int(debug_stats["masked_decrease_hi_forbidden_count"]),
            "masked_decrease_hi_forbidden_rate": float(debug_stats["masked_decrease_hi_forbidden_rate"]),
            "masked_action_count_max": int(debug_stats["masked_action_count_max"]),
            "mask_rejection_rate_mean": float(debug_stats["mask_rejection_rate_mean"]),
            "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
            "selected_explicit_noop_actions": int(debug_stats["selected_explicit_noop_actions"]),
            "selected_explicit_noop_rate": float(debug_stats["selected_explicit_noop_rate"]),
            "action_space_type": str(debug_stats["action_space_type"]),
            "action_count": int(debug_stats["action_count"]),
            "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
            "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
            "budget_floor_ratio": float(debug_stats["budget_floor_ratio"]),
            "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
            "masked_budget_floor_violation_count": int(debug_stats["masked_budget_floor_violation_count"]),
            "masked_budget_floor_violation_rate": float(debug_stats["masked_budget_floor_violation_rate"]),
            "masked_deploy_cap_increase_count": int(debug_stats["masked_deploy_cap_increase_count"]),
            "masked_deploy_cap_increase_rate": float(debug_stats["masked_deploy_cap_increase_rate"]),
            "observation_mode": str(last_info.get("observation_mode", feature_config.observation_mode)),
            "state_dim": int(last_info.get("state_dim", len(obs.state_vector))),
            "mean_over_increase_excess": float(action_log_metrics["mean_over_increase_excess"]),
            "over_increase_action_count": int(action_log_metrics["over_increase_action_count"]),
            "mean_budget_soft_cap_increase_excess": float(
                action_log_metrics["mean_budget_soft_cap_increase_excess"]
            ),
            "soft_cap_increase_action_count": int(action_log_metrics["soft_cap_increase_action_count"]),
            "mean_budget_soft_cap_penalty_value": float(
                action_log_metrics["mean_budget_soft_cap_penalty_value"]
            ),
            "mean_budget_soft_cap_dwell_excess_mean": float(
                action_log_metrics["mean_budget_soft_cap_dwell_excess_mean"]
            ),
            "mean_budget_soft_cap_dwell_excess_max": float(
                action_log_metrics["mean_budget_soft_cap_dwell_excess_max"]
            ),
            "max_budget_soft_cap_dwell_excess_max": float(
                action_log_metrics["max_budget_soft_cap_dwell_excess_max"]
            ),
            "soft_cap_dwell_state_count": int(action_log_metrics["soft_cap_dwell_state_count"]),
            "soft_cap_dwell_state_rate": float(action_log_metrics["soft_cap_dwell_state_rate"]),
            "mean_budget_soft_cap_dwell_penalty_value": float(
                action_log_metrics["mean_budget_soft_cap_dwell_penalty_value"]
            ),
            "mean_budget_soft_cap_dwell_max_penalty_value": float(
                action_log_metrics["mean_budget_soft_cap_dwell_max_penalty_value"]
            ),
            "mean_budget_soft_cap_dwell_total_penalty_value": float(
                action_log_metrics["mean_budget_soft_cap_dwell_total_penalty_value"]
            ),
            "safe_recovery_decrease_count": int(action_log_metrics["safe_recovery_decrease_count"]),
            "unsafe_decrease_full_count": int(action_log_metrics["unsafe_decrease_full_count"]),
            "mean_budget_over_drift_deadzone": float(action_log_metrics["mean_budget_over_drift_deadzone"]),
            "mean_increase_concentration_excess": float(
                action_log_metrics["mean_increase_concentration_excess"]
            ),
            "pingpong_action_count": int(action_log_metrics["pingpong_action_count"]),
            "mean_active_lo_under_hi_pressure": float(
                action_log_metrics["mean_active_lo_under_hi_pressure"]
            ),
            "mean_active_lo_work_ratio": float(action_log_metrics["mean_active_lo_work_ratio"]),
            "mean_active_lo_job_rate": float(action_log_metrics["mean_active_lo_job_rate"]),
            "mean_active_lo_under_hi_pressure_penalty_value": float(
                action_log_metrics["mean_active_lo_under_hi_pressure_penalty_value"]
            ),
            **_degradation_metrics_to_row(runtime_result),
            **service_metrics_to_row(dqn_service_metrics),
            **_lo_quality_weighted_metrics_to_row_from_result(runtime_result),
            **_task_level_info_row(last_info),
        }
    row.update(_noop_q_diagnostics_to_row(agent, diagnostic_states, diagnostic_valid_masks))
    return (
        row,
        runtime_result,
        env.action_log,
    )


def _write_tree_audit_files(
    *,
    tree_audit_dir: Path,
    seed: int,
    method: str,
    action_log: list[dict[str, object]],
    row_base: dict[str, object],
    tree_metadata: dict[str, object],
) -> None:
    """写出单个 seed/method 的 leaf audit JSONL 与 per-seed leaf summary CSV。

    输出文件：
    - {tree_audit_dir}/taskset{taskset_seed}_seed{seed}_{method}_leaf_audit.jsonl
    - {tree_audit_dir}/taskset{taskset_seed}_seed{seed}_{method}_leaf_summary.csv

    JSONL 只包含 action_log 中有 tree_leaf_id 的行（即 tree 决策行）。
    写入 JSONL 前先补齐 seed / method / tree 元数据，避免后续跨 seed 汇总时 key 缺失。
    """

    import numpy as np
    from collections import Counter

    tree_audit_dir.mkdir(parents=True, exist_ok=True)

    # 筛选出包含 leaf audit 信息的行
    audit_rows = [row for row in action_log if "tree_leaf_id" in row]
    if not audit_rows:
        return

    # 构造写入 JSONL 时每行统一补齐的元数据前缀
    metadata_prefix = {
        "seed": seed,
        "taskset_seed": row_base.get("taskset_seed"),
        "scenario_seed": row_base.get("scenario_seed"),
        "method": method,
        "tree_id": tree_metadata.get("tree_id"),
        "tree_method": tree_metadata.get("method"),
        "tree_depth": tree_metadata.get("tree_depth"),
        "tree_node_count": tree_metadata.get("tree_node_count"),
        "tree_leaf_count": tree_metadata.get("tree_leaf_count"),
        "tree_max_depth_param": tree_metadata.get("max_depth"),
        "tree_min_samples_leaf": tree_metadata.get("min_samples_leaf"),
    }
    # 过滤掉 None 值，避免 JSONL 中出现 null 字段导致后续解析歧义
    metadata_prefix = {k: v for k, v in metadata_prefix.items() if v is not None}

    # 补齐每行元数据后写入 JSONL（文件名包含 taskset seed，避免跨 taskset 覆盖）
    taskset_seed = row_base.get("taskset_seed", "unknown")
    stem = f"taskset{taskset_seed}_seed{seed}_{method}"
    jsonl_path = tree_audit_dir / f"{stem}_leaf_audit.jsonl"
    enriched_rows: list[dict[str, object]] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            enriched = {**metadata_prefix, **row}
            enriched_rows.append(enriched)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    # 按 leaf_id 聚合，写出 per-seed leaf summary CSV（使用补齐后的行）
    groups: dict[int, list[dict[str, object]]] = {}
    for row in enriched_rows:
        leaf_id = int(row.get("tree_leaf_id", -1))
        groups.setdefault(leaf_id, []).append(row)

    total_count = len(enriched_rows)
    summary_rows: list[dict[str, object]] = []
    for leaf_id, leaf_rows in sorted(groups.items()):
        hit_count = len(leaf_rows)
        hit_rate = hit_count / total_count if total_count > 0 else 0.0

        # 取第一个 leaf row 中的静态信息
        first = leaf_rows[0]
        path_depth = first.get("tree_path_depth")
        leaf_n_node_samples = first.get("tree_leaf_n_node_samples")
        leaf_impurity = first.get("tree_leaf_impurity")
        path_predicates_json = first.get("tree_path_predicates_json")

        # raw_top1_action_id 的众数
        raw_top1_ids = [row.get("tree_raw_top1_action_id") for row in leaf_rows]
        raw_top1_counter = Counter(r for r in raw_top1_ids if r is not None)
        raw_top1_mode = raw_top1_counter.most_common(1)[0][0] if raw_top1_counter else None

        # selected_action_id 的众数
        sel_ids = [row.get("tree_selected_action_id") for row in leaf_rows]
        sel_counter = Counter(s for s in sel_ids if s is not None)
        sel_mode = sel_counter.most_common(1)[0][0] if sel_counter else None

        # 对应的动作语义描述（取该 action_id mode 对应行的第一条 action_def_json）
        raw_top1_def_json = None
        for row in leaf_rows:
            if row.get("tree_raw_top1_action_id") == raw_top1_mode and raw_top1_mode is not None:
                raw_top1_def_json = row.get("tree_raw_top1_action_def_json")
                break
        selected_def_json = None
        for row in leaf_rows:
            if row.get("tree_selected_action_id") == sel_mode and sel_mode is not None:
                selected_def_json = row.get("tree_selected_action_def_json")
                break

        # fallback 统计
        fallback_count = sum(int(bool(row.get("tree_fallback_used", False))) for row in leaf_rows)
        fallback_rate = fallback_count / hit_count if hit_count > 0 else 0.0

        # raw_invalid 统计
        raw_invalid_count = sum(int(bool(row.get("tree_raw_top1_invalid", False))) for row in leaf_rows)
        raw_invalid_rate = raw_invalid_count / hit_count if hit_count > 0 else 0.0

        # teacher match 统计
        teacher_match_vals = [row.get("teacher_selected_action_match") for row in leaf_rows]
        match_count = sum(1 for v in teacher_match_vals if v is True)
        match_rate = match_count / hit_count if hit_count > 0 else 0.0

        # q_regret 统计
        q_regrets = [
            float(row["teacher_q_regret_selected"])
            for row in leaf_rows
            if row.get("teacher_q_regret_selected") is not None
        ]
        q_regret_mean = float(np.mean(q_regrets)) if q_regrets else None
        q_regret_p95 = float(np.percentile(q_regrets, 95)) if q_regrets else None

        # reward 统计
        rewards = [float(row.get("reward", 0.0)) for row in leaf_rows]
        reward_sum = float(np.sum(rewards))
        reward_mean = float(np.mean(rewards)) if rewards else 0.0

        # accepted 统计
        accepted_count = sum(int(bool(row.get("accepted", False))) for row in leaf_rows)
        accepted_rate = accepted_count / hit_count if hit_count > 0 else 0.0

        # outcome delta 统计
        delta_deadline_misses_sum = sum(int(row.get("delta_deadline_misses", 0) or 0) for row in leaf_rows)
        delta_mode_changes_sum = sum(int(row.get("delta_mode_changes", 0) or 0) for row in leaf_rows)
        delta_lo_cancellations_sum = sum(int(row.get("delta_lo_cancellations", 0) or 0) for row in leaf_rows)

        summary_rows.append({
            "seed": seed,
            "method": method,
            "tree_id": first.get("tree_id", ""),
            "tree_leaf_id": leaf_id,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "path_depth": path_depth,
            "leaf_n_node_samples": leaf_n_node_samples,
            "leaf_impurity": leaf_impurity,
            "raw_top1_action_id_mode": raw_top1_mode,
            "raw_top1_action_def_json_mode": raw_top1_def_json if raw_top1_def_json is not None else "",
            "selected_action_id_mode": sel_mode,
            "selected_action_def_json_mode": selected_def_json if selected_def_json is not None else "",
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "raw_invalid_count": raw_invalid_count,
            "raw_invalid_rate": raw_invalid_rate,
            "teacher_match_count": match_count,
            "teacher_match_rate": match_rate,
            "q_regret_selected_mean": q_regret_mean,
            "q_regret_selected_p95": q_regret_p95,
            "reward_sum": reward_sum,
            "reward_mean": reward_mean,
            "accepted_count": accepted_count,
            "accepted_rate": accepted_rate,
            "delta_deadline_misses_sum": delta_deadline_misses_sum,
            "delta_mode_changes_sum": delta_mode_changes_sum,
            "delta_lo_cancellations_sum": delta_lo_cancellations_sum,
            "path_predicates_json": path_predicates_json if path_predicates_json is not None else "",
        })

    csv_path = tree_audit_dir / f"{stem}_leaf_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


def _evaluate_tree_once(
    *,
    tree_artifact_dir: Path,
    method_name: str,
    experiment_config,
    agent_period: int,
    seed: int,
    end_time: int,
    row_base: dict[str, int | float | str | bool],
    reward_mode: str,
    dqn_runtime_semantics: RuntimeSemantics,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    feature_config: FeatureConfig,
    c_amc_sem_xf: float = 0.5,
    teacher_model_path: Path | None = None,
    require_integer_tree: bool = False,
    leaf_audit_enabled: bool = False,
    leaf_audit_state_mode: str = "split",
    leaf_audit_top_k_actions: int = 5,
) -> tuple[dict[str, int | float | str | bool | None], SimulationResult, list[dict[str, object]]]:
    """在正式 HOUT 入口中执行 tree policy。"""

    from amc_py.viper.artifacts import load_tree_policy_artifact
    from amc_py.viper.metrics import evaluate_tree_policy_once

    tree_policy = load_tree_policy_artifact(tree_artifact_dir, require_integer_tree=require_integer_tree)
    metadata = tree_policy.metadata
    if dqn_runtime_semantics is RuntimeSemantics.Q_AMC:
        if (
            experiment_config.qamc_reference_config_path is None
            or experiment_config.qamc_profile_manifest_path is None
            or experiment_config.qamc_profile_spec_path is None
        ):
            raise ValueError("QAMC_REFERENCE_PROFILE_ARTIFACTS_REQUIRED")
        frozen = load_and_validate_frozen_reference(
            experiment_config.qamc_reference_config_path
        )
        manifest = json.loads(
            Path(experiment_config.qamc_profile_manifest_path).read_text(
                encoding="utf-8"
            )
        )
        spec = load_profile_spec(experiment_config.qamc_profile_spec_path)
        expected = {
            "reference_config_fingerprint": frozen["fingerprint"],
            "profile_manifest_fingerprint": manifest.get("fingerprint"),
            "profile_spec_fingerprint": spec.fingerprint,
        }
        qamc_metadata = metadata.get("qamc")
        if not isinstance(qamc_metadata, dict) or any(
            qamc_metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("QAMC_TREE_ARTIFACT_FINGERPRINT_MISMATCH")
    teacher = DqnBudgetAgent.load(teacher_model_path) if teacher_model_path is not None else None
    tree_metrics, runtime_result, action_log = evaluate_tree_policy_once(
        tree_policy=tree_policy,
        experiment_config=experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        runtime_semantics=dqn_runtime_semantics,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        enable_deploy_cap_mask=enable_deploy_cap_mask,
        deploy_cap_mask_ratio=deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=deploy_cap_mask_criticality,
        feature_config=feature_config,
        c_amc_sem_xf=c_amc_sem_xf,
        teacher=teacher,
        leaf_audit_enabled=leaf_audit_enabled,
        leaf_audit_state_mode=leaf_audit_state_mode,
        leaf_audit_top_k_actions=leaf_audit_top_k_actions,
    )
    step_count = int(tree_metrics.get("step_count", 0))
    accepted_actions = int(tree_metrics.get("accepted_actions", 0))
    row = {
        **row_base,
        **_empty_noop_q_diagnostics_row(),
        "method": method_name,
        "q_network_type": "",
        "budget_overruns": int(tree_metrics["mode_changes"]) + int(tree_metrics["lo_cancellations"]),
        "noop_actions": 0,
        "explicit_noop_actions": 0,
        "noop_action_rate": 0.0,
        "explicit_noop_action_rate": 0.0,
        "accepted_action_rate": ((accepted_actions / step_count) if step_count > 0 else 0.0),
        "rejection_rate": 0.0,
        "observation_mode": feature_config.observation_mode,
        "state_dim": int(metadata.get("state_dim", 0)),
        **tree_metrics,
        "tree_id": metadata.get("tree_id"),
        "tree_method": metadata.get("method"),
        "tree_depth": metadata.get("tree_depth"),
        "tree_node_count": metadata.get("tree_node_count"),
        "tree_leaf_count": metadata.get("tree_leaf_count"),
        "tree_max_depth_param": metadata.get("max_depth"),
        "tree_min_samples_leaf": metadata.get("min_samples_leaf"),
    }
    return row, runtime_result, action_log


UNIFIED_SUMMARY_FIELDNAMES = [
    "workload",
    "total_util",
    "num_tasks",
    "cf",
    "cp",
    "end_time",
    "agent_period",
    "dqn_runtime_semantics",
    "c_amc_sem_xf",
    "reward_mode",
    "action_space_type",
    "budget_increase_ratio",
    "budget_decrease_ratio",
    "budget_floor_ratio",
    "forbid_decreasing_hi_budgets",
    "enable_deploy_cap_mask",
    "deploy_cap_mask_ratio",
    "deploy_cap_mask_criticality",
    "tree_runtime_policy_type",
    "tree_state_encoding",
    "tree_fixed_point_scale",
    "tree_fixed_point_config_hash",
    "tree_artifact_schema_version",
    "integer_equivalence_verified",
    "row_type",
    "method",
    "reference_method",
    "seed_count",
    "mode_changes_mean",
    "lo_cancellations_mean",
    "released_lo_jobs_mean",
    "lc_service_loss_mean",
    "lc_qos_mean",
    "min_lc_service_mean",
    "deadline_misses_sum",
    "hi_deadline_misses_sum",
    "lo_deadline_misses_sum",
    "jne_mean",
    "ldm_mean",
    "jne_plus_ldm_mean",
    "nid_mean",
    "tid_mean",
    "tid_ratio_mean",
    "nid_per_1e6_time_mean",
    "mean_degraded_interval_mean",
    "safety_feasible_sum",
    "safety_feasible_rate",
    "lo_job_losses_total_mean",
    "lo_budget_cancellations_mean",
    "lo_release_dropped_in_degraded_mode_mean",
    "lo_active_dropped_on_mode_switch_mean",
    "jne_residual_not_in_cancellations_mean",
    "active_drop_share_of_jne_mean",
    "lo_equiv_jne_mean",
    "lo_equiv_jne_rate_mean",
    "lo_quality_qos_mean",
    "lo_quality_loss_mean",
    "lo_full_quality_completed_mean",
    "lo_full_quality_ratio_mean",
    "lo_degraded_released_mean",
    "lo_degraded_completed_mean",
    "lo_degraded_cancelled_mean",
    "lo_degraded_deadline_missed_mean",
    "lo_degraded_not_completed_mean",
    "lo_degraded_release_ratio_mean",
    "lo_degraded_completion_ratio_mean",
    "lo_degraded_among_completed_ratio_mean",
    "lo_degraded_quality_sum_mean",
    "lo_degraded_budget_sum_mean",
    "lo_degraded_original_budget_sum_mean",
    "lo_degraded_budget_ratio_mean",
    "lo_degraded_exec_time_sum_mean",
    "lo_degraded_exec_time_ratio_mean",
    "lo_zero_service_jobs_mean",
    "lo_zero_service_ratio_mean",
    "lo_full_quality_service_sum_mean",
    "lo_total_service_sum_mean",
    "delta_lc_service_loss",
    "relative_lc_loss_reduction",
    "delta_lo_cancellations",
    "relative_lo_cancellation_reduction",
    "delta_jne_plus_ldm",
    "relative_jne_plus_ldm_reduction",
    "delta_nid",
    "relative_nid_reduction",
    "delta_tid",
    "relative_tid_reduction",
    "delta_lo_budget_cancellations",
    "delta_lo_release_dropped_in_degraded_mode",
    "delta_lo_active_dropped_on_mode_switch",
    "delta_jne_residual_not_in_cancellations",
    "delta_lo_equiv_jne_rate",
    "relative_lo_equiv_jne_rate_reduction",
    "delta_lo_quality_qos",
    "delta_lo_quality_loss",
    "relative_lo_quality_loss_reduction",
    "delta_lo_full_quality_ratio",
    "delta_lo_degraded_release_ratio",
    "delta_lo_degraded_completion_ratio",
    "delta_lo_zero_service_ratio",
    "delta_tid_ratio",
    "accepted_action_count_mean",
    "rejected_action_count_mean",
    "noop_action_count_mean",
    "explicit_noop_action_count_mean",
    "accepted_action_rate_mean",
    "rejected_action_rate_mean",
    "noop_action_rate_mean",
    "explicit_noop_action_rate_mean",
    "masked_action_count_mean",
    "valid_action_count_mean",
]


def _mean_metric(rows: list[dict[str, int | float | str | bool]], key: str) -> float:
    """对非空数值字段求均值。"""

    return mean(_to_float(row, key) for row in rows)


def _group_summary_context(
    sample_row: dict[str, int | float | str | bool],
) -> dict[str, int | float | str | bool]:
    """抽取 unified summary 每个 workload group 共享的上下文字段。"""

    return {
        "workload": str(sample_row.get("workload", "")),
        "total_util": str(sample_row.get("total_util", "")),
        "num_tasks": str(sample_row.get("num_tasks", "")),
        "cf": str(sample_row.get("cf", "")),
        "cp": str(sample_row.get("cp", "")),
        "end_time": int(_to_float(sample_row, "end_time")),
        "agent_period": int(_to_float(sample_row, "agent_period")),
        "dqn_runtime_semantics": str(sample_row.get("dqn_runtime_semantics", "")),
        "c_amc_sem_xf": _to_float(sample_row, "c_amc_sem_xf"),
        "reward_mode": str(sample_row.get("reward_mode", "")),
        "action_space_type": str(sample_row.get("action_space_type", "")),
        "budget_increase_ratio": _to_float(sample_row, "budget_increase_ratio"),
        "budget_decrease_ratio": _to_float(sample_row, "budget_decrease_ratio"),
        "budget_floor_ratio": _to_float(sample_row, "budget_floor_ratio"),
        "forbid_decreasing_hi_budgets": bool(sample_row.get("forbid_decreasing_hi_budgets", False)),
        "enable_deploy_cap_mask": bool(sample_row.get("enable_deploy_cap_mask", False)),
        "deploy_cap_mask_ratio": _to_float(sample_row, "deploy_cap_mask_ratio"),
        "deploy_cap_mask_criticality": str(sample_row.get("deploy_cap_mask_criticality", "")),
    }


def _tree_summary_context(
    sample_row: dict[str, int | float | str | bool],
) -> dict[str, int | float | str | bool | None]:
    """抽取 tree artifact 相关的统一摘要字段。"""

    return {
        "tree_runtime_policy_type": sample_row.get("tree_runtime_policy_type"),
        "tree_state_encoding": sample_row.get("tree_state_encoding"),
        "tree_fixed_point_scale": sample_row.get("tree_fixed_point_scale"),
        "tree_fixed_point_config_hash": sample_row.get("tree_fixed_point_config_hash"),
        "tree_artifact_schema_version": sample_row.get("tree_artifact_schema_version"),
        "integer_equivalence_verified": sample_row.get("integer_equivalence_verified"),
    }


def _aggregate_method_summary_rows(
    group_rows: list[dict[str, int | float | str | bool]],
) -> list[dict[str, int | float | str | bool | None]]:
    """对每个 method 聚合一行 method_summary。"""

    rows_by_method: dict[str, list[dict[str, int | float | str | bool]]] = {}
    for row in group_rows:
        rows_by_method.setdefault(str(row.get("method", "")), []).append(row)

    summary_rows: list[dict[str, int | float | str | bool | None]] = []
    context = _group_summary_context(group_rows[0])
    for method, method_rows in sorted(rows_by_method.items()):
        summary_row: dict[str, int | float | str | bool | None] = {
            **context,
            **_tree_summary_context(method_rows[0]),
            "row_type": "method_summary",
            "method": method,
            "reference_method": "",
            "seed_count": len(method_rows),
            "mode_changes_mean": _mean_metric(method_rows, "mode_changes"),
            "lo_cancellations_mean": _mean_metric(method_rows, "lo_cancellations"),
            "released_lo_jobs_mean": _mean_metric(method_rows, "released_lo_jobs"),
            "lc_service_loss_mean": _mean_metric(method_rows, "lc_service_loss"),
            "lc_qos_mean": _mean_metric(method_rows, "lc_qos"),
            "min_lc_service_mean": mean_optional_service_metric(method_rows, "min_lc_service"),
            "deadline_misses_sum": sum(int(_to_float(row, "deadline_misses")) for row in method_rows),
            "hi_deadline_misses_sum": sum(int(_to_float(row, "hi_deadline_misses")) for row in method_rows),
            "lo_deadline_misses_sum": sum(int(_to_float(row, "lo_deadline_misses")) for row in method_rows),
            "jne_mean": _mean_metric(method_rows, "jne"),
            "ldm_mean": _mean_metric(method_rows, "ldm"),
            "jne_plus_ldm_mean": _mean_metric(method_rows, "jne_plus_ldm"),
            "nid_mean": _mean_metric(method_rows, "nid"),
            "tid_mean": _mean_metric(method_rows, "tid"),
            "tid_ratio_mean": _mean_metric(method_rows, "tid_ratio"),
            "nid_per_1e6_time_mean": _mean_metric(method_rows, "nid_per_1e6_time"),
            "mean_degraded_interval_mean": mean_optional_service_metric(method_rows, "mean_degraded_interval"),
            "safety_feasible_sum": sum(int(_to_float(row, "safety_feasible")) for row in method_rows),
            "safety_feasible_rate": _mean_metric(method_rows, "safety_feasible"),
            "lo_job_losses_total_mean": _mean_metric(method_rows, "lo_job_losses_total"),
            "lo_budget_cancellations_mean": _mean_metric(method_rows, "lo_budget_cancellations"),
            "lo_release_dropped_in_degraded_mode_mean": _mean_metric(
                method_rows,
                "lo_release_dropped_in_degraded_mode",
            ),
            "lo_active_dropped_on_mode_switch_mean": _mean_metric(
                method_rows,
                "lo_active_dropped_on_mode_switch",
            ),
            "jne_residual_not_in_cancellations_mean": _mean_metric(
                method_rows,
                "jne_residual_not_in_cancellations",
            ),
            "active_drop_share_of_jne_mean": mean_optional_service_metric(
                method_rows,
                "active_drop_share_of_jne",
            ),
            "lo_equiv_jne_mean": _mean_metric(method_rows, "lo_equiv_jne"),
            "lo_equiv_jne_rate_mean": _mean_metric(method_rows, "lo_equiv_jne_rate"),
            "lo_quality_qos_mean": _mean_metric(method_rows, "lo_quality_qos"),
            "lo_quality_loss_mean": _mean_metric(method_rows, "lo_quality_loss"),
            "lo_full_quality_completed_mean": _mean_metric(method_rows, "lo_full_quality_completed"),
            "lo_full_quality_ratio_mean": _mean_metric(method_rows, "lo_full_quality_ratio"),
            "lo_degraded_released_mean": _mean_optional_metric(method_rows, "lo_degraded_released"),
            "lo_degraded_completed_mean": _mean_optional_metric(method_rows, "lo_degraded_completed"),
            "lo_degraded_cancelled_mean": _mean_optional_metric(method_rows, "lo_degraded_cancelled"),
            "lo_degraded_deadline_missed_mean": _mean_optional_metric(
                method_rows,
                "lo_degraded_deadline_missed",
            ),
            "lo_degraded_not_completed_mean": _mean_optional_metric(method_rows, "lo_degraded_not_completed"),
            "lo_degraded_release_ratio_mean": _mean_optional_metric(method_rows, "lo_degraded_release_ratio"),
            "lo_degraded_completion_ratio_mean": _mean_optional_metric(
                method_rows,
                "lo_degraded_completion_ratio",
            ),
            "lo_degraded_among_completed_ratio_mean": mean_optional_service_metric(
                method_rows,
                "lo_degraded_among_completed_ratio",
            ),
            "lo_degraded_quality_sum_mean": _mean_optional_metric(method_rows, "lo_degraded_quality_sum"),
            "lo_degraded_budget_sum_mean": _mean_optional_metric(method_rows, "lo_degraded_budget_sum"),
            "lo_degraded_original_budget_sum_mean": _mean_optional_metric(
                method_rows,
                "lo_degraded_original_budget_sum",
            ),
            "lo_degraded_budget_ratio_mean": mean_optional_service_metric(
                method_rows,
                "lo_degraded_budget_ratio_mean",
            ),
            "lo_degraded_exec_time_sum_mean": _mean_optional_metric(method_rows, "lo_degraded_exec_time_sum"),
            "lo_degraded_exec_time_ratio_mean": mean_optional_service_metric(
                method_rows,
                "lo_degraded_exec_time_ratio",
            ),
            "lo_zero_service_jobs_mean": _mean_metric(method_rows, "lo_zero_service_jobs"),
            "lo_zero_service_ratio_mean": _mean_metric(method_rows, "lo_zero_service_ratio"),
            "lo_full_quality_service_sum_mean": _mean_metric(
                method_rows,
                "lo_full_quality_service_sum",
            ),
            "lo_total_service_sum_mean": _mean_metric(method_rows, "lo_total_service_sum"),
            "delta_lc_service_loss": None,
            "relative_lc_loss_reduction": None,
            "delta_lo_cancellations": None,
            "relative_lo_cancellation_reduction": None,
            "delta_jne_plus_ldm": None,
            "relative_jne_plus_ldm_reduction": None,
            "delta_nid": None,
            "relative_nid_reduction": None,
            "delta_tid": None,
            "relative_tid_reduction": None,
            "delta_lo_budget_cancellations": None,
            "delta_lo_release_dropped_in_degraded_mode": None,
            "delta_lo_active_dropped_on_mode_switch": None,
            "delta_jne_residual_not_in_cancellations": None,
            "delta_lo_equiv_jne_rate": None,
            "relative_lo_equiv_jne_rate_reduction": None,
            "delta_lo_quality_qos": None,
            "delta_lo_quality_loss": None,
            "relative_lo_quality_loss_reduction": None,
            "delta_lo_full_quality_ratio": None,
            "delta_lo_degraded_release_ratio": None,
            "delta_lo_degraded_completion_ratio": None,
            "delta_lo_zero_service_ratio": None,
            "delta_tid_ratio": None,
            "accepted_action_count_mean": _mean_metric(method_rows, "accepted_actions"),
            "rejected_action_count_mean": _mean_metric(method_rows, "rejected_actions"),
            "noop_action_count_mean": _mean_metric(method_rows, "noop_actions"),
            "explicit_noop_action_count_mean": _mean_metric(method_rows, "explicit_noop_actions"),
            "accepted_action_rate_mean": _mean_metric(method_rows, "accepted_action_rate"),
            "rejected_action_rate_mean": _mean_metric(method_rows, "rejection_rate"),
            "noop_action_rate_mean": _mean_metric(method_rows, "noop_action_rate"),
            "explicit_noop_action_rate_mean": _mean_metric(method_rows, "explicit_noop_action_rate"),
            "masked_action_count_mean": _mean_metric(method_rows, "masked_action_count_mean"),
            "valid_action_count_mean": _mean_metric(method_rows, "valid_action_count_mean"),
            **_task_level_info_row(method_rows[0]),
        }
        for fieldname in NOOP_Q_DIAGNOSTIC_FIELDNAMES:
            summary_row[fieldname] = (
                sum(int(row.get(fieldname) or 0) for row in method_rows)
                if fieldname == "noop_q_sample_count"
                else _mean_optional_metric(method_rows, fieldname)
            )
        qamc_numeric_fields = (
            "paper_quality_sum",
            "paper_quality_per_release",
            "normalized_provided_quality_sum",
            "normalized_quality_qos",
            "zero_service_count",
            "zero_service_ratio",
            "release_target_rank_mean",
            "release_target_normalized_mean",
            "completed_quality_conditional_mean",
            "completed_quality_unconditional_per_release",
            "overrun_stop_count",
            "quality_transition_count",
            "min_quality_exhaustion_count",
            "tasks_ever_degraded",
            "runtime_level_depth_mean",
            "runtime_level_depth_max",
            "raw_rank_drop_mean",
            "raw_rank_drop_max",
            "non_degradable_task_count",
            "trigger_budget_mean_ratio_to_c_lo",
            "trigger_below_design_count",
            "trigger_equal_design_count",
            "trigger_above_design_count",
            "would_overrun_design_count",
            "dqn_budget_update_event_count",
            "dqn_budget_update_task_count",
            "viper_budget_update_event_count",
            "viper_budget_update_task_count",
            "heuristic_budget_update_event_count",
            "heuristic_budget_update_task_count",
            "offline_budget_update_event_count",
            "offline_budget_update_task_count",
            "unspecified_budget_update_event_count",
            "unspecified_budget_update_task_count",
        )
        if "qamc_normalized_quality_qos" in method_rows[0]:
            for field in qamc_numeric_fields:
                summary_row[f"qamc_{field}_mean"] = _mean_metric(
                    method_rows, f"qamc_{field}"
                )
            summary_row["qamc_first_degradation_time_mean"] = _mean_optional_metric(
                method_rows, "qamc_first_degradation_time"
            )
            summary_row["qamc_profile_fingerprint"] = method_rows[0].get(
                "qamc_profile_fingerprint", ""
            )
            for field_name in sorted(
                key for key in method_rows[0] if key.startswith("qamc_loss_")
            ):
                summary_row[f"{field_name}_mean"] = _mean_metric(
                    method_rows,
                    field_name,
                )
            summary_row["qamc_legacy_degraded_metrics_applicable"] = False
        summary_rows.append(summary_row)
    return summary_rows


def _build_dqn_reference_comparison_rows(
    method_summary_rows: list[dict[str, int | float | str | bool | None]],
) -> list[dict[str, int | float | str | bool | None]]:
    """基于 method_summary 生成 dqn_vs_reference 行。"""

    summary_by_method = {
        str(row["method"]): row
        for row in method_summary_rows
        if str(row.get("row_type", "")) == "method_summary"
    }
    dqn_row = summary_by_method.get("q_amc_dqn_budget_overlay") or summary_by_method.get(
        "dqn_agent"
    )
    if dqn_row is None:
        return []

    comparison_rows: list[dict[str, int | float | str | bool | None]] = []
    for reference_method in [
        "q_amc_native",
        "amc_same_full_sample_native",
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
        "c_amc_sem_baseline",
        "noop_agent",
    ]:
        reference_row = summary_by_method.get(reference_method)
        if reference_row is None:
            continue
        comparison_rows.append(
            {
                **{
                    key: dqn_row[key]
                    for key in (
                        "workload",
                        "total_util",
                        "num_tasks",
                        "cf",
                        "cp",
                        "end_time",
                        "agent_period",
                        "dqn_runtime_semantics",
                        "c_amc_sem_xf",
                        "reward_mode",
                        "action_space_type",
                        "budget_increase_ratio",
                        "budget_decrease_ratio",
                        "budget_floor_ratio",
                        "forbid_decreasing_hi_budgets",
                        "enable_deploy_cap_mask",
                        "deploy_cap_mask_ratio",
                        "deploy_cap_mask_criticality",
                    )
                },
                **_tree_summary_context(dqn_row),
                "row_type": "dqn_vs_reference",
                "method": str(dqn_row["method"]),
                "reference_method": reference_method,
                "seed_count": dqn_row["seed_count"],
                "mode_changes_mean": dqn_row["mode_changes_mean"],
                "lo_cancellations_mean": dqn_row["lo_cancellations_mean"],
                "released_lo_jobs_mean": dqn_row["released_lo_jobs_mean"],
                "lc_service_loss_mean": dqn_row["lc_service_loss_mean"],
                "lc_qos_mean": dqn_row["lc_qos_mean"],
                "min_lc_service_mean": dqn_row["min_lc_service_mean"],
                "deadline_misses_sum": dqn_row["deadline_misses_sum"],
                "hi_deadline_misses_sum": dqn_row["hi_deadline_misses_sum"],
                "lo_deadline_misses_sum": dqn_row["lo_deadline_misses_sum"],
                "jne_mean": dqn_row["jne_mean"],
                "ldm_mean": dqn_row["ldm_mean"],
                "jne_plus_ldm_mean": dqn_row["jne_plus_ldm_mean"],
                "nid_mean": dqn_row["nid_mean"],
                "tid_mean": dqn_row["tid_mean"],
                "tid_ratio_mean": dqn_row["tid_ratio_mean"],
                "nid_per_1e6_time_mean": dqn_row["nid_per_1e6_time_mean"],
                "mean_degraded_interval_mean": dqn_row["mean_degraded_interval_mean"],
                "safety_feasible_sum": dqn_row["safety_feasible_sum"],
                "safety_feasible_rate": dqn_row["safety_feasible_rate"],
                "lo_job_losses_total_mean": dqn_row["lo_job_losses_total_mean"],
                "lo_budget_cancellations_mean": dqn_row["lo_budget_cancellations_mean"],
                "lo_release_dropped_in_degraded_mode_mean": dqn_row[
                    "lo_release_dropped_in_degraded_mode_mean"
                ],
                "lo_active_dropped_on_mode_switch_mean": dqn_row[
                    "lo_active_dropped_on_mode_switch_mean"
                ],
                "jne_residual_not_in_cancellations_mean": dqn_row[
                    "jne_residual_not_in_cancellations_mean"
                ],
                "active_drop_share_of_jne_mean": dqn_row["active_drop_share_of_jne_mean"],
                "lo_equiv_jne_mean": dqn_row["lo_equiv_jne_mean"],
                "lo_equiv_jne_rate_mean": dqn_row["lo_equiv_jne_rate_mean"],
                "lo_quality_qos_mean": dqn_row["lo_quality_qos_mean"],
                "lo_quality_loss_mean": dqn_row["lo_quality_loss_mean"],
                "lo_full_quality_completed_mean": dqn_row["lo_full_quality_completed_mean"],
                "lo_full_quality_ratio_mean": dqn_row["lo_full_quality_ratio_mean"],
                "lo_degraded_released_mean": dqn_row["lo_degraded_released_mean"],
                "lo_degraded_completed_mean": dqn_row["lo_degraded_completed_mean"],
                "lo_degraded_cancelled_mean": dqn_row["lo_degraded_cancelled_mean"],
                "lo_degraded_deadline_missed_mean": dqn_row["lo_degraded_deadline_missed_mean"],
                "lo_degraded_not_completed_mean": dqn_row["lo_degraded_not_completed_mean"],
                "lo_degraded_release_ratio_mean": dqn_row["lo_degraded_release_ratio_mean"],
                "lo_degraded_completion_ratio_mean": dqn_row["lo_degraded_completion_ratio_mean"],
                "lo_degraded_among_completed_ratio_mean": dqn_row[
                    "lo_degraded_among_completed_ratio_mean"
                ],
                "lo_degraded_quality_sum_mean": dqn_row["lo_degraded_quality_sum_mean"],
                "lo_degraded_budget_sum_mean": dqn_row["lo_degraded_budget_sum_mean"],
                "lo_degraded_original_budget_sum_mean": dqn_row[
                    "lo_degraded_original_budget_sum_mean"
                ],
                "lo_degraded_budget_ratio_mean": dqn_row["lo_degraded_budget_ratio_mean"],
                "lo_degraded_exec_time_sum_mean": dqn_row["lo_degraded_exec_time_sum_mean"],
                "lo_degraded_exec_time_ratio_mean": dqn_row["lo_degraded_exec_time_ratio_mean"],
                "lo_zero_service_jobs_mean": dqn_row["lo_zero_service_jobs_mean"],
                "lo_zero_service_ratio_mean": dqn_row["lo_zero_service_ratio_mean"],
                "lo_full_quality_service_sum_mean": dqn_row["lo_full_quality_service_sum_mean"],
                "lo_total_service_sum_mean": dqn_row["lo_total_service_sum_mean"],
                "delta_lc_service_loss": float(dqn_row["lc_service_loss_mean"]) - float(reference_row["lc_service_loss_mean"]),
                "relative_lc_loss_reduction": safe_relative_reduction(
                    float(reference_row["lc_service_loss_mean"]),
                    float(dqn_row["lc_service_loss_mean"]),
                ),
                "delta_lo_cancellations": float(dqn_row["lo_cancellations_mean"]) - float(reference_row["lo_cancellations_mean"]),
                "relative_lo_cancellation_reduction": safe_relative_reduction(
                    float(reference_row["lo_cancellations_mean"]),
                    float(dqn_row["lo_cancellations_mean"]),
                ),
                "delta_jne_plus_ldm": float(dqn_row["jne_plus_ldm_mean"]) - float(reference_row["jne_plus_ldm_mean"]),
                "relative_jne_plus_ldm_reduction": safe_relative_reduction(
                    float(reference_row["jne_plus_ldm_mean"]),
                    float(dqn_row["jne_plus_ldm_mean"]),
                ),
                "delta_nid": float(dqn_row["nid_mean"]) - float(reference_row["nid_mean"]),
                "relative_nid_reduction": safe_relative_reduction(
                    float(reference_row["nid_mean"]),
                    float(dqn_row["nid_mean"]),
                ),
                "delta_tid": float(dqn_row["tid_mean"]) - float(reference_row["tid_mean"]),
                "relative_tid_reduction": safe_relative_reduction(
                    float(reference_row["tid_mean"]),
                    float(dqn_row["tid_mean"]),
                ),
                "delta_lo_budget_cancellations": float(dqn_row["lo_budget_cancellations_mean"]) - float(reference_row["lo_budget_cancellations_mean"]),
                "delta_lo_release_dropped_in_degraded_mode": float(
                    dqn_row["lo_release_dropped_in_degraded_mode_mean"]
                ) - float(reference_row["lo_release_dropped_in_degraded_mode_mean"]),
                "delta_lo_active_dropped_on_mode_switch": float(
                    dqn_row["lo_active_dropped_on_mode_switch_mean"]
                ) - float(reference_row["lo_active_dropped_on_mode_switch_mean"]),
                "delta_jne_residual_not_in_cancellations": float(
                    dqn_row["jne_residual_not_in_cancellations_mean"]
                ) - float(reference_row["jne_residual_not_in_cancellations_mean"]),
                "delta_lo_equiv_jne_rate": float(dqn_row["lo_equiv_jne_rate_mean"]) - float(reference_row["lo_equiv_jne_rate_mean"]),
                "relative_lo_equiv_jne_rate_reduction": safe_relative_reduction(
                    float(reference_row["lo_equiv_jne_rate_mean"]),
                    float(dqn_row["lo_equiv_jne_rate_mean"]),
                ),
                "delta_lo_quality_qos": float(dqn_row["lo_quality_qos_mean"]) - float(reference_row["lo_quality_qos_mean"]),
                "delta_lo_quality_loss": float(dqn_row["lo_quality_loss_mean"]) - float(reference_row["lo_quality_loss_mean"]),
                "relative_lo_quality_loss_reduction": safe_relative_reduction(
                    float(reference_row["lo_quality_loss_mean"]),
                    float(dqn_row["lo_quality_loss_mean"]),
                ),
                "delta_lo_full_quality_ratio": float(dqn_row["lo_full_quality_ratio_mean"]) - float(reference_row["lo_full_quality_ratio_mean"]),
                "delta_lo_degraded_release_ratio": _optional_delta(
                    dqn_row["lo_degraded_release_ratio_mean"],
                    reference_row["lo_degraded_release_ratio_mean"],
                ),
                "delta_lo_degraded_completion_ratio": _optional_delta(
                    dqn_row["lo_degraded_completion_ratio_mean"],
                    reference_row["lo_degraded_completion_ratio_mean"],
                ),
                "delta_lo_zero_service_ratio": float(dqn_row["lo_zero_service_ratio_mean"]) - float(reference_row["lo_zero_service_ratio_mean"]),
                "delta_tid_ratio": float(dqn_row["tid_ratio_mean"]) - float(reference_row["tid_ratio_mean"]),
                "accepted_action_count_mean": dqn_row["accepted_action_count_mean"],
                "rejected_action_count_mean": dqn_row["rejected_action_count_mean"],
                "noop_action_count_mean": dqn_row["noop_action_count_mean"],
                "explicit_noop_action_count_mean": dqn_row["explicit_noop_action_count_mean"],
                "accepted_action_rate_mean": dqn_row["accepted_action_rate_mean"],
                "rejected_action_rate_mean": dqn_row["rejected_action_rate_mean"],
                "noop_action_rate_mean": dqn_row["noop_action_rate_mean"],
                "explicit_noop_action_rate_mean": dqn_row["explicit_noop_action_rate_mean"],
                "masked_action_count_mean": dqn_row["masked_action_count_mean"],
                "valid_action_count_mean": dqn_row["valid_action_count_mean"],
                **_task_level_info_row(dqn_row),
                **{fieldname: dqn_row.get(fieldname) for fieldname in NOOP_Q_DIAGNOSTIC_FIELDNAMES},
            }
        )
        comparison = comparison_rows[-1]
        qamc_qos_key = "qamc_normalized_quality_qos_mean"
        if qamc_qos_key in dqn_row and qamc_qos_key in reference_row:
            qamc_summary_fields = (
                "qamc_paper_quality_sum_mean",
                "qamc_paper_quality_per_release_mean",
                "qamc_normalized_provided_quality_sum_mean",
                "qamc_normalized_quality_qos_mean",
                "qamc_zero_service_count_mean",
                "qamc_zero_service_ratio_mean",
                "qamc_overrun_stop_count_mean",
                "qamc_quality_transition_count_mean",
                "qamc_min_quality_exhaustion_count_mean",
                "qamc_runtime_level_depth_mean_mean",
                "qamc_raw_rank_drop_mean_mean",
                "qamc_trigger_budget_mean_ratio_to_c_lo_mean",
            )
            comparison.update(
                {field: dqn_row.get(field) for field in qamc_summary_fields}
            )
            comparison["delta_qamc_normalized_quality_qos"] = float(
                dqn_row[qamc_qos_key]
            ) - float(reference_row[qamc_qos_key])
            comparison["delta_qamc_zero_service_ratio"] = float(
                dqn_row["qamc_zero_service_ratio_mean"]
            ) - float(reference_row["qamc_zero_service_ratio_mean"])
    return comparison_rows


def _build_unified_summary_rows(
    rows: list[dict[str, int | float | str | bool]],
) -> list[dict[str, int | float | str | bool | None]]:
    """从评估明细构建新的长表 unified summary。"""

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, int | float | str | bool]]] = {}
    for row in rows:
        key = (
            str(row.get("workload", "")),
            str(row.get("total_util", "")),
            str(row.get("num_tasks", "")),
            str(row.get("cf", "")),
            str(row.get("cp", "")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, int | float | str | bool | None]] = []
    for _, group_rows in sorted(grouped.items()):
        method_summary_rows = _aggregate_method_summary_rows(group_rows)
        summary_rows.extend(method_summary_rows)
        summary_rows.extend(_build_dqn_reference_comparison_rows(method_summary_rows))
    return summary_rows


def _write_unified_summary_csv(
    output_path: Path,
    rows: list[dict[str, int | float | str | bool]],
) -> None:
    """把新的长表 unified summary 写成独立 CSV 文件。"""

    summary_rows = _build_unified_summary_rows(rows)
    summary_path = output_path.with_name(f"{output_path.stem}_unified_summary.csv")
    fieldnames = list(UNIFIED_SUMMARY_FIELDNAMES)
    fieldnames.extend(TASK_LEVEL_INFO_KEYS)
    fieldnames.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
    known = set(fieldnames)
    fieldnames.extend(
        sorted({key for row in summary_rows for key in row} - known)
    )
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _parse_csv_set(raw_value: str) -> set[str]:
    """将逗号分隔字符串转为去空白集合。"""

    return {part.strip() for part in raw_value.split(",") if part.strip()}


def _parse_int_set_or_ranges(raw_value: str) -> set[int]:
    """解析逗号分隔的整数或闭区间。

    支持：
    - "1550"
    - "1550,1551,1552"
    - "1550:1599"
    - "1550:1555,1590,1595:1599"
    区间为闭区间，包含 end。
    """

    result: set[int] = set()
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            start_raw, end_raw = token.split(":", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                raise ValueError(f"invalid seed range (end < start): {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 jsonl 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_rows_from_runtime(result: SimulationResult) -> list[dict]:
    """将 runtime 结果转为 jsonl trace 行。

    输出顺序上先写逐 tick 调度快照，再追加事件级 debug 日志。这样一份文件里
    同时包含“CPU 在跑谁”和“为什么发生切换/更新/miss”的两条视角。
    """

    rows: list[dict] = []
    # 这里提前建 job 索引，是为了在 deadline_miss 行中把“该 miss 对应的 job
    # 到底是不是 degraded、释放时处于什么 mode、原始预算是多少”一起补齐。
    # 这样 trace/debug JSONL 可以直接支撑计划文档要求的后处理，不需要再二次 join。
    job_by_key = {(job.task.name, job.release_index): job for job in result.jobs}
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
        job = job_by_key.get((miss.task, miss.release_index))
        rows.append(
            {
                "event": "deadline_miss",
                "task": miss.task,
                "release_index": miss.release_index,
                "release_time": miss.release_time,
                "absolute_deadline": miss.absolute_deadline,
                "mode_at_miss": miss.mode_at_miss.name,
                "executed_at_miss": miss.executed_at_miss,
                "released_in_mode": None if job is None else job.released_in_mode.name,
                "is_degraded": None if job is None else job.is_degraded,
                "service_quality_if_completed": None if job is None else job.service_quality_if_completed,
                "original_actual_cost": None if job is None else job.original_actual_cost,
                "original_runtime_budget_at_release": (
                    None if job is None else job.original_runtime_budget_at_release
                ),
            }
        )
    return rows


def _build_runtime_budget_timeline(result: SimulationResult) -> dict[str, list[tuple[int, int]]]:
    """为每个任务构造预算时间线，供 deadline miss 详情回溯使用。"""

    if not result.jobs:
        return {}
    initial_budgets = {job.task.name: job.task.c_lo for job in result.jobs}
    timeline: dict[str, list[tuple[int, int]]] = {task_name: [(0, budget)] for task_name, budget in initial_budgets.items()}
    for update in result.budget_update_events:
        for task_name, budget in update.updates.items():
            timeline.setdefault(task_name, [(0, budget)])
            timeline[task_name].append((update.time, budget))
    return timeline


def _budget_at_time(timeline: dict[str, list[tuple[int, int]]], task_name: str, time: int) -> int | None:
    """查询某任务在指定时刻的全局预算值。"""

    budget = None
    for update_time, candidate_budget in timeline.get(task_name, []):
        if update_time > time:
            break
        budget = candidate_budget
    return budget


def _last_action_before(action_log: list[dict], time: int) -> dict | None:
    """返回 miss 发生前最近一次 agent 决策。"""

    last_action: dict | None = None
    for row in action_log:
        if int(row.get("time", -1)) <= time:
            last_action = row
        else:
            break
    return last_action


def _deadline_miss_detail_rows(
    *,
    row_base: dict[str, int | float | str | bool],
    method: str,
    runtime_result: SimulationResult,
    action_log: list[dict],
) -> list[dict[str, object]]:
    """展开 deadline miss 详情。

    这里不只输出 miss 数量，而是把 task/job/budget/action 四条信息链拼到一起，
    这样看到某条 miss 记录时就能直接回答：
    - 是哪个 job miss；
    - 释放时预算是多少；
    - miss 当刻全局预算又是多少；
    - miss 前最近一次动作是谁、何时发生、是否被接受。
    """

    jobs_by_key = {(job.task.name, job.release_index): job for job in runtime_result.jobs}
    budget_timeline = _build_runtime_budget_timeline(runtime_result)
    detail_rows: list[dict[str, object]] = []
    for miss in runtime_result.deadline_misses:
        job = jobs_by_key[(miss.task, miss.release_index)]
        last_action = _last_action_before(action_log, miss.absolute_deadline)
        last_budget_update_time = None
        for event in runtime_result.budget_update_events:
            if event.time <= miss.absolute_deadline:
                last_budget_update_time = event.time
            else:
                break
        detail_rows.append(
            {
                "workload": row_base["workload"],
                "total_util": row_base["total_util"],
                "seed": row_base["seed"],
                "method": method,
                "time": miss.absolute_deadline,
                "task": job.task.name,
                "criticality": job.task.criticality.value,
                "release_index": job.release_index,
                "release_time": job.release_time,
                "absolute_deadline": job.absolute_deadline,
                "actual_cost": job.actual_cost,
                "executed_at_miss": miss.executed_at_miss,
                "runtime_budget_at_release": job.runtime_budget_at_release,
                "current_global_budget": _budget_at_time(budget_timeline, job.task.name, miss.absolute_deadline),
                "completion_time": job.completion_time,
                "dropped": job.dropped,
                "drop_time": job.drop_time,
                "mode_at_miss": miss.mode_at_miss.name,
                "last_action_time": None if last_action is None else last_action.get("time"),
                "last_action_id": None if last_action is None else last_action.get("action_id"),
                "last_action_accepted": None if last_action is None else last_action.get("accepted"),
                "last_action_updates": None if last_action is None else last_action.get("updates"),
                "last_budget_update_time": last_budget_update_time,
            }
        )
    return detail_rows


def _write_agent_debug_files(
    *,
    trace_dir: Path | None,
    debug_log_dir: Path | None,
    seed: int,
    method: str,
    action_log: list[dict],
    runtime_result: SimulationResult,
    mask_log: list[dict] | None = None,
) -> None:
    """按方法/seed 写出 action、trace、debug 三类文件。"""

    if trace_dir is not None:
        _write_jsonl(trace_dir / f"seed{seed}_{method}_action_log.jsonl", action_log)
        _write_jsonl(trace_dir / f"seed{seed}_{method}_runtime_trace.jsonl", _trace_rows_from_runtime(runtime_result))
        if mask_log is not None:
            _write_jsonl(trace_dir / f"seed{seed}_{method}_mask_log.jsonl", mask_log)
    if debug_log_dir is not None:
        _write_jsonl(debug_log_dir / f"seed{seed}_{method}_debug_events.jsonl", runtime_result.debug_events)


def _deadline_miss_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    """筛选 deadline_misses > 0 的记录。"""

    return [row for row in rows if int(row["deadline_misses"]) > 0]


def _run_qamc_budget_heuristic(
    *,
    experiment_config,
    seed: int,
    end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    feature_config: FeatureConfig,
    c_amc_sem_xf: float,
) -> AgentRuntimeResult:
    """Run the q-AMC heuristic through the same env executor as DQN/VIPER."""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.Q_AMC,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        enable_deploy_cap_mask=enable_deploy_cap_mask,
        deploy_cap_mask_ratio=deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=deploy_cap_mask_criticality,
        feature_config=feature_config,
        record_dropped_lo_releases=True,
        c_amc_sem_xf=c_amc_sem_xf,
        budget_update_source="HEURISTIC_ACTION",
    )
    policy = QAmcBudgetPressureHeuristic()
    observation = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        mask = env.valid_action_mask()
        action_id = policy.select_action_id(observation, env.actions, mask)
        step = env.step(action_id)
        observation = step.observation
        total_reward += float(step.reward)
        done = step.done
    action_log = list(env.action_log)
    debug = env.debug_statistics()
    accepted = sum(int(bool(row.get("accepted", False))) for row in action_log)
    rejected = sum(
        int(not bool(row.get("accepted", False)) and not bool(row.get("noop", False)))
        for row in action_log
    )
    noop = sum(int(bool(row.get("noop", False))) for row in action_log)
    return AgentRuntimeResult(
        runtime_result=env.runtime_result,
        accepted_actions=accepted,
        rejected_actions=rejected,
        noop_actions=noop,
        total_reward=total_reward,
        action_log=action_log,
        safety_checked_actions=int(debug["safety_checked_actions"]),
        safety_accepted_actions=int(debug["safety_accepted_actions"]),
        safety_rejected_actions=int(debug["safety_rejected_actions"]),
    )


def _evaluate_enabled_methods_for_seed(
    *,
    seed: int,
    experiment_config,
    workload: str,
    total_util: float,
    num_tasks: int,
    cf: float,
    cp: float,
    require_schedulable: bool,
    enabled_methods: set[str],
    model_path: Path,
    end_time: int,
    agent_period: int,
    reward_mode: str,
    dqn_runtime_semantics: RuntimeSemantics,
    c_amc_sem_xf: float,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    feature_config: FeatureConfig,
    trace_dir: Path | None,
    debug_log_dir: Path | None,
    trace_seed_set: set[int],
    trace_method_set: set[str],
    dqn_agent_device: str | None = None,
    double_dqn: bool = True,
    max_q_diagnostic_samples: int = 1000,
    constraint_guided_pair_top_k_risk: int = 3,
    constraint_guided_pair_top_k_decrease: int = 5,
    constraint_guided_pair_prefer_lo: bool = False,
    constraint_guided_pair_include_hi_risk_boost: bool = False,
    constraint_guided_pair_allow_increase_only_when_safe: bool = False,
    bc_tree_model: Path | None = None,
    dagger_tree_model: Path | None = None,
    viper_tree_model: Path | None = None,
    tree_compare_teacher_model: Path | None = None,
    tree_audit_dir: Path | None = None,
    tree_audit_seed_set: set[int] | None = None,
    tree_audit_method_set: set[str] | None = None,
    tree_audit_state_mode: str = "split",
    tree_audit_top_k_actions: int = 5,
    require_integer_tree_artifact: bool = False,
) -> tuple[list[dict[str, int | float | str | bool]], list[dict[str, object]]]:
    """评估单个 seed 下的所有启用方法。

    这里把"一个 seed 对应的整组评估工作"封装成独立 helper，有两个目的：
    - 串行路径直接在主进程里调用，保持现有行为与口径；
    - 并行路径把 seed 分发给多个子进程时，也复用同一份实现，避免两套逻辑漂移。

    返回值拆成两部分：
    - rows：最终写入评估 CSV 的按方法统计行；
    - deadline_miss_details：若存在 miss，则展开写入 jsonl 的详情行。
    """

    bundle = resolve_experiment_bundle(experiment_config, seed)
    qamc_profile_bundle = None
    if dqn_runtime_semantics is RuntimeSemantics.Q_AMC:
        manifest_path = experiment_config.qamc_profile_manifest_path
        if manifest_path is None:
            raise ValueError("QAMC_PROFILE_MANIFEST_REQUIRED")
        if experiment_config.qamc_profile_spec_path is None:
            raise ValueError("QAMC_PROFILE_SPEC_REQUIRED")
        qamc_profile_bundle = load_profile_bundle_from_manifest(
            manifest_path,
            taskset_fingerprint=str(bundle.taskset_fingerprint),
            spec_path=experiment_config.qamc_profile_spec_path,
        )
    actions = build_budget_action_space(
        list(bundle.ordered_tasks),
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
    )

    if require_schedulable:
        amc_rtb_schedulable = True
    else:
        amc_rtb_schedulable = evaluate_taskset(
            list(bundle.ordered_tasks),
            method="amc_rtb",
            priority_policy="opa",
        ).schedulable
    attempts = bundle.taskset_attempts
    taskset_seed = bundle.taskset_seed if bundle.taskset_seed is not None else seed
    scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else seed

    row_base: dict[str, int | float | str | bool] = {
        "workload": workload,
        "total_util": total_util,
        "num_tasks": num_tasks,
        "cf": cf,
        "cp": cp,
        "seed": seed,
        "taskset_seed": taskset_seed,
        "scenario_seed": scenario_seed,
        "amc_rtb_schedulable": amc_rtb_schedulable,
        "attempts": attempts,
        "end_time": end_time,
        "agent_period": agent_period,
        "dqn_runtime_semantics": dqn_runtime_semantics.value,
        "c_amc_sem_xf": c_amc_sem_xf,
        "reward_mode": reward_mode,
        "action_space_type": action_space,
        "budget_increase_ratio": budget_increase_ratio,
        "budget_decrease_ratio": budget_decrease_ratio,
        "budget_floor_ratio": budget_floor_ratio,
        "forbid_decreasing_hi_budgets": forbid_decreasing_hi_budgets,
        "enable_deploy_cap_mask": enable_deploy_cap_mask,
        "deploy_cap_mask_ratio": deploy_cap_mask_ratio,
        "deploy_cap_mask_criticality": deploy_cap_mask_criticality,
    }
    # trace / debug 的开关粒度仍然保持“按 seed、按 method 控制”，
    # 这样并行后也不会改变原有调试文件的生成规则，同时正式 HOUT 默认不积累
    # 逐 tick trace 与事件级 debug 日志，仅在显式调试时再打开。
    trace_enabled_for_seed = (
        (trace_dir is not None or debug_log_dir is not None)
        and (not trace_seed_set or seed in trace_seed_set)
    )
    capture_runtime_trace_for_seed = trace_dir is not None and trace_enabled_for_seed
    capture_debug_events_for_seed = (trace_dir is not None or debug_log_dir is not None) and trace_enabled_for_seed

    # 计算当前 seed 是否应启用 leaf audit（独立于 --trace-dir）
    tree_audit_enabled_for_seed = (
        tree_audit_dir is not None
        and (not tree_audit_seed_set or seed in tree_audit_seed_set)
    )

    runtime_config = _formal_agent_runtime_config(
        end_time=end_time,
        semantics=dqn_runtime_semantics,
        capture_trace=capture_runtime_trace_for_seed,
        capture_debug_events=capture_debug_events_for_seed,
        c_amc_sem_xf=c_amc_sem_xf,
    )

    rows: list[dict[str, int | float | str | bool]] = []
    deadline_miss_details: list[dict[str, object]] = []
    qamc_runtime_results: dict[str, SimulationResult] = {}

    if "amc_same_full_sample_native" in enabled_methods:
        baseline_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=_baseline_runtime_config(
                end_time=end_time,
                semantics=RuntimeSemantics.AMC,
                capture_trace=capture_runtime_trace_for_seed,
                capture_debug_events=capture_debug_events_for_seed,
                c_amc_sem_xf=c_amc_sem_xf,
            ),
        )
        rows.append(
            _build_pure_runtime_baseline_row(
                row_base=row_base,
                method="amc_same_full_sample_native",
                runtime_result=baseline_result,
                action_space=action_space,
                action_count=len(actions),
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
            )
        )

    if "q_amc_native" in enabled_methods:
        if qamc_profile_bundle is None:
            raise ValueError("QAMC_NATIVE_BASELINE_REQUIRES_QAMC_RUNTIME")
        native_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=_baseline_runtime_config(
                end_time=end_time,
                semantics=RuntimeSemantics.Q_AMC,
                capture_trace=capture_runtime_trace_for_seed,
                capture_debug_events=capture_debug_events_for_seed,
                c_amc_sem_xf=c_amc_sem_xf,
            ),
            qamc_profile_bundle=qamc_profile_bundle,
        )
        rows.append(
            _build_pure_runtime_baseline_row(
                row_base=row_base,
                method="q_amc_native",
                runtime_result=native_result,
                action_space=action_space,
                action_count=len(actions),
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
            )
        )
        qamc_runtime_results["q_amc_native"] = native_result

    if "amc_plus_baseline" in enabled_methods:
        baseline_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=_baseline_runtime_config(
                end_time=end_time,
                semantics=RuntimeSemantics.AMC_PLUS,
                capture_trace=capture_runtime_trace_for_seed,
                capture_debug_events=capture_debug_events_for_seed,
                c_amc_sem_xf=c_amc_sem_xf,
            ),
        )
        rows.append(
            _build_pure_runtime_baseline_row(
                row_base=row_base,
                method="amc_plus_baseline",
                runtime_result=baseline_result,
                action_space=action_space,
                action_count=len(actions),
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
            )
        )
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method="amc_plus_baseline",
                runtime_result=baseline_result,
                action_log=[],
            )
        )
        if trace_enabled_for_seed and (not trace_method_set or "amc_plus_baseline" in trace_method_set):
            _write_agent_debug_files(
                trace_dir=trace_dir,
                debug_log_dir=debug_log_dir,
                seed=seed,
                method="amc_plus_baseline",
                action_log=[],
                runtime_result=baseline_result,
            )

    runtime_baseline_specs = [
        ("amc_ra_baseline", RuntimeSemantics.AMC_RA),
        ("amc_rh_baseline", RuntimeSemantics.AMC_RH),
        ("c_amc_sem_baseline", RuntimeSemantics.C_AMC_SEM),
    ]
    for method_name, semantics in runtime_baseline_specs:
        if method_name not in enabled_methods:
            continue
        runtime_baseline_result = simulate_ordered_taskset_event_driven(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            config=_baseline_runtime_config(
                end_time=end_time,
                semantics=semantics,
                capture_trace=capture_runtime_trace_for_seed,
                capture_debug_events=capture_debug_events_for_seed,
                c_amc_sem_xf=c_amc_sem_xf,
            ),
        )
        rows.append(
            _build_pure_runtime_baseline_row(
                row_base=row_base,
                method=method_name,
                runtime_result=runtime_baseline_result,
                action_space=action_space,
                action_count=len(actions),
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                budget_floor_ratio=budget_floor_ratio,
            )
        )
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method=method_name,
                runtime_result=runtime_baseline_result,
                action_log=[],
            )
        )
        if trace_enabled_for_seed and (not trace_method_set or method_name in trace_method_set):
            _write_agent_debug_files(
                trace_dir=trace_dir,
                debug_log_dir=debug_log_dir,
                seed=seed,
                method=method_name,
                action_log=[],
                runtime_result=runtime_baseline_result,
            )

    if "noop_agent" in enabled_methods:
        noop_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            agent=NoOpBudgetAgent(),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(
                agent_period=agent_period,
                end_time=end_time,
                check_safety=True,
                reward_mode=reward_mode,
                forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                budget_floor_ratio=budget_floor_ratio,
                budget_rounding_mode="ceil_floor",
                min_budget_delta=1,
                enable_deploy_cap_mask=enable_deploy_cap_mask,
                deploy_cap_mask_ratio=deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=deploy_cap_mask_criticality,
            ),
            bounds=bundle.normalization_bounds,
            qamc_profile_bundle=qamc_profile_bundle,
        )
        qamc_runtime_results["noop_agent"] = noop_result.runtime_result
        # wrapper 类 baseline 自己不返回显式的 step_count 字段，
        # 因此统一用 action_log 行数来定义“发生了多少次 agent 决策”。
        noop_step_count = len(noop_result.action_log)
        noop_selected_action_count = sum(int(row.get("action_id") is not None) for row in noop_result.action_log)
        noop_explicit_noop_actions = sum(int(bool(row.get("is_explicit_noop", False))) for row in noop_result.action_log)
        noop_rejection_rate = (noop_result.rejected_actions / noop_step_count) if noop_step_count > 0 else 0.0
        noop_service_metrics = compute_service_quality_metrics(noop_result.runtime_result)
        rows.append(
            {
                **row_base,
                **_empty_noop_q_diagnostics_row(),
                "method": "noop_agent",
                "mode_changes": noop_result.runtime_result.mode_change_count(),
                "lo_cancellations": noop_result.runtime_result.lo_job_cancellation_count(),
                "deadline_misses": len(noop_result.runtime_result.deadline_misses),
                "budget_overruns": _budget_overruns_from_result(noop_result.runtime_result),
                "accepted_actions": noop_result.accepted_actions,
                "rejected_actions": noop_result.rejected_actions,
                "step_count": noop_step_count,
                "selected_action_count": noop_selected_action_count,
                "noop_actions": noop_result.noop_actions,
                "explicit_noop_actions": noop_explicit_noop_actions,
                "noop_action_rate": ((noop_result.noop_actions / noop_step_count) if noop_step_count > 0 else 0.0),
                "explicit_noop_action_rate": (
                    (noop_explicit_noop_actions / noop_step_count) if noop_step_count > 0 else 0.0
                ),
                "accepted_action_rate": (
                    (noop_result.accepted_actions / noop_step_count) if noop_step_count > 0 else 0.0
                ),
                "rejection_rate": noop_rejection_rate,
                "total_reward": noop_result.total_reward,
                "check_safety": True,
                "safety_checked_actions": noop_result.safety_checked_actions,
                "safety_accepted_actions": noop_result.safety_accepted_actions,
                "safety_rejected_actions": noop_result.safety_rejected_actions,
                "valid_action_count_mean": 0.0,
                "masked_action_count_mean": 0.0,
                "masked_decrease_hi_forbidden_count": 0,
                "masked_decrease_hi_forbidden_rate": 0.0,
                "masked_action_count_max": 0,
                "mask_rejection_rate_mean": 0.0,
                "selected_invalid_mask_actions": 0,
                "selected_explicit_noop_actions": 0,
                "selected_explicit_noop_rate": 0.0,
                "action_space_type": action_space,
                "action_count": len(actions),
                "budget_increase_ratio": budget_increase_ratio,
                "budget_decrease_ratio": budget_decrease_ratio,
                "budget_floor_ratio": budget_floor_ratio,
                "no_safe_action_steps": 0,
                "masked_budget_floor_violation_count": 0,
                "masked_budget_floor_violation_rate": 0.0,
                "masked_deploy_cap_increase_count": 0,
                "masked_deploy_cap_increase_rate": 0.0,
                **_degradation_metrics_to_row(noop_result.runtime_result),
                **service_metrics_to_row(noop_service_metrics),
                **_lo_quality_weighted_metrics_to_row_from_result(noop_result.runtime_result),
            }
        )
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method="noop_agent",
                runtime_result=noop_result.runtime_result,
                action_log=noop_result.action_log,
            )
        )
        if trace_enabled_for_seed and (not trace_method_set or "noop_agent" in trace_method_set):
            _write_agent_debug_files(
                trace_dir=trace_dir,
                debug_log_dir=debug_log_dir,
                seed=seed,
                method="noop_agent",
                action_log=noop_result.action_log,
                runtime_result=noop_result.runtime_result,
            )

    if "random_agent" in enabled_methods:
        random_result = simulate_ordered_taskset_with_agent(
            ordered_tasks=list(bundle.ordered_tasks),
            scenario=bundle.scenario,
            agent=RandomBudgetAgent(actions=actions, seed=seed),
            runtime_config=runtime_config,
            agent_config=AgentRuntimeConfig(
                agent_period=agent_period,
                end_time=end_time,
                check_safety=True,
                reward_mode=reward_mode,
                forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                budget_floor_ratio=budget_floor_ratio,
                budget_rounding_mode="ceil_floor",
                min_budget_delta=1,
                enable_deploy_cap_mask=enable_deploy_cap_mask,
                deploy_cap_mask_ratio=deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=deploy_cap_mask_criticality,
            ),
            bounds=bundle.normalization_bounds,
            qamc_profile_bundle=qamc_profile_bundle,
        )
        qamc_runtime_results["random_agent"] = random_result.runtime_result
        random_step_count = len(random_result.action_log)
        random_selected_action_count = sum(int(row.get("action_id") is not None) for row in random_result.action_log)
        random_explicit_noop_actions = sum(
            int(bool(row.get("is_explicit_noop", False))) for row in random_result.action_log
        )
        random_rejection_rate = (random_result.rejected_actions / random_step_count) if random_step_count > 0 else 0.0
        random_service_metrics = compute_service_quality_metrics(random_result.runtime_result)
        rows.append(
            {
                **row_base,
                **_empty_noop_q_diagnostics_row(),
                "method": "random_agent",
                "mode_changes": random_result.runtime_result.mode_change_count(),
                "lo_cancellations": random_result.runtime_result.lo_job_cancellation_count(),
                "deadline_misses": len(random_result.runtime_result.deadline_misses),
                "budget_overruns": _budget_overruns_from_result(random_result.runtime_result),
                "accepted_actions": random_result.accepted_actions,
                "rejected_actions": random_result.rejected_actions,
                "step_count": random_step_count,
                "selected_action_count": random_selected_action_count,
                "noop_actions": random_result.noop_actions,
                "explicit_noop_actions": random_explicit_noop_actions,
                "noop_action_rate": ((random_result.noop_actions / random_step_count) if random_step_count > 0 else 0.0),
                "explicit_noop_action_rate": (
                    (random_explicit_noop_actions / random_step_count) if random_step_count > 0 else 0.0
                ),
                "accepted_action_rate": (
                    (random_result.accepted_actions / random_step_count) if random_step_count > 0 else 0.0
                ),
                "rejection_rate": random_rejection_rate,
                "total_reward": random_result.total_reward,
                "check_safety": True,
                "safety_checked_actions": random_result.safety_checked_actions,
                "safety_accepted_actions": random_result.safety_accepted_actions,
                "safety_rejected_actions": random_result.safety_rejected_actions,
                "valid_action_count_mean": 0.0,
                "masked_action_count_mean": 0.0,
                "masked_decrease_hi_forbidden_count": 0,
                "masked_decrease_hi_forbidden_rate": 0.0,
                "masked_action_count_max": 0,
                "mask_rejection_rate_mean": 0.0,
                "selected_invalid_mask_actions": 0,
                "selected_explicit_noop_actions": 0,
                "selected_explicit_noop_rate": 0.0,
                "action_space_type": action_space,
                "action_count": len(actions),
                "budget_increase_ratio": budget_increase_ratio,
                "budget_decrease_ratio": budget_decrease_ratio,
                "budget_floor_ratio": budget_floor_ratio,
                "no_safe_action_steps": 0,
                "masked_budget_floor_violation_count": 0,
                "masked_budget_floor_violation_rate": 0.0,
                "masked_deploy_cap_increase_count": 0,
                "masked_deploy_cap_increase_rate": 0.0,
                **_degradation_metrics_to_row(random_result.runtime_result),
                **service_metrics_to_row(random_service_metrics),
                **_lo_quality_weighted_metrics_to_row_from_result(random_result.runtime_result),
            }
        )
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method="random_agent",
                runtime_result=random_result.runtime_result,
                action_log=random_result.action_log,
            )
        )
        if trace_enabled_for_seed and (not trace_method_set or "random_agent" in trace_method_set):
            _write_agent_debug_files(
                trace_dir=trace_dir,
                debug_log_dir=debug_log_dir,
                seed=seed,
                method="random_agent",
                action_log=random_result.action_log,
                runtime_result=random_result.runtime_result,
            )

    if "heuristic_agent" in enabled_methods:
        if qamc_profile_bundle is not None:
            heuristic_result = _run_qamc_budget_heuristic(
                experiment_config=experiment_config,
                seed=seed,
                end_time=end_time,
                agent_period=agent_period,
                reward_mode=reward_mode,
                action_space=action_space,
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                include_explicit_noop=include_explicit_noop,
                budget_floor_ratio=budget_floor_ratio,
                forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                mask_detail_mode=mask_detail_mode,
                enable_deploy_cap_mask=enable_deploy_cap_mask,
                deploy_cap_mask_ratio=deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=deploy_cap_mask_criticality,
                feature_config=feature_config,
                c_amc_sem_xf=c_amc_sem_xf,
            )
        else:
            heuristic_result = simulate_ordered_taskset_with_agent(
                ordered_tasks=list(bundle.ordered_tasks),
                scenario=bundle.scenario,
                agent=HeuristicBudgetAgent(actions=actions),
                runtime_config=runtime_config,
                agent_config=AgentRuntimeConfig(
                    agent_period=agent_period,
                    end_time=end_time,
                    check_safety=True,
                    reward_mode=reward_mode,
                    forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                    budget_floor_ratio=budget_floor_ratio,
                    budget_rounding_mode="ceil_floor",
                    min_budget_delta=1,
                    enable_deploy_cap_mask=enable_deploy_cap_mask,
                    deploy_cap_mask_ratio=deploy_cap_mask_ratio,
                    deploy_cap_mask_criticality=deploy_cap_mask_criticality,
                    budget_update_source="HEURISTIC_ACTION",
                ),
                bounds=bundle.normalization_bounds,
            )
        qamc_runtime_results["heuristic_agent"] = heuristic_result.runtime_result
        heuristic_step_count = len(heuristic_result.action_log)
        heuristic_selected_action_count = sum(
            int(row.get("action_id") is not None) for row in heuristic_result.action_log
        )
        heuristic_explicit_noop_actions = sum(
            int(bool(row.get("is_explicit_noop", False))) for row in heuristic_result.action_log
        )
        heuristic_rejection_rate = (
            (heuristic_result.rejected_actions / heuristic_step_count)
            if heuristic_step_count > 0
            else 0.0
        )
        heuristic_service_metrics = compute_service_quality_metrics(heuristic_result.runtime_result)
        rows.append(
            {
                **row_base,
                **_empty_noop_q_diagnostics_row(),
                "method": "heuristic_agent",
                "mode_changes": heuristic_result.runtime_result.mode_change_count(),
                "lo_cancellations": heuristic_result.runtime_result.lo_job_cancellation_count(),
                "deadline_misses": len(heuristic_result.runtime_result.deadline_misses),
                "budget_overruns": _budget_overruns_from_result(heuristic_result.runtime_result),
                "accepted_actions": heuristic_result.accepted_actions,
                "rejected_actions": heuristic_result.rejected_actions,
                "step_count": heuristic_step_count,
                "selected_action_count": heuristic_selected_action_count,
                "noop_actions": heuristic_result.noop_actions,
                "explicit_noop_actions": heuristic_explicit_noop_actions,
                "noop_action_rate": (
                    (heuristic_result.noop_actions / heuristic_step_count)
                    if heuristic_step_count > 0
                    else 0.0
                ),
                "explicit_noop_action_rate": (
                    (heuristic_explicit_noop_actions / heuristic_step_count)
                    if heuristic_step_count > 0
                    else 0.0
                ),
                "accepted_action_rate": (
                    (heuristic_result.accepted_actions / heuristic_step_count)
                    if heuristic_step_count > 0
                    else 0.0
                ),
                "rejection_rate": heuristic_rejection_rate,
                "total_reward": heuristic_result.total_reward,
                "check_safety": True,
                "safety_checked_actions": heuristic_result.safety_checked_actions,
                "safety_accepted_actions": heuristic_result.safety_accepted_actions,
                "safety_rejected_actions": heuristic_result.safety_rejected_actions,
                "valid_action_count_mean": 0.0,
                "masked_action_count_mean": 0.0,
                "masked_decrease_hi_forbidden_count": 0,
                "masked_decrease_hi_forbidden_rate": 0.0,
                "masked_action_count_max": 0,
                "mask_rejection_rate_mean": 0.0,
                "selected_invalid_mask_actions": 0,
                "selected_explicit_noop_actions": 0,
                "selected_explicit_noop_rate": 0.0,
                "action_space_type": action_space,
                "action_count": len(actions),
                "budget_increase_ratio": budget_increase_ratio,
                "budget_decrease_ratio": budget_decrease_ratio,
                "budget_floor_ratio": budget_floor_ratio,
                "no_safe_action_steps": 0,
                "masked_budget_floor_violation_count": 0,
                "masked_budget_floor_violation_rate": 0.0,
                "masked_deploy_cap_increase_count": 0,
                "masked_deploy_cap_increase_rate": 0.0,
                **_degradation_metrics_to_row(heuristic_result.runtime_result),
                **service_metrics_to_row(heuristic_service_metrics),
                **_lo_quality_weighted_metrics_to_row_from_result(heuristic_result.runtime_result),
            }
        )
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method="heuristic_agent",
                runtime_result=heuristic_result.runtime_result,
                action_log=heuristic_result.action_log,
            )
        )
        if trace_enabled_for_seed and (not trace_method_set or "heuristic_agent" in trace_method_set):
            _write_agent_debug_files(
                trace_dir=trace_dir,
                debug_log_dir=debug_log_dir,
                seed=seed,
                method="heuristic_agent",
                action_log=heuristic_result.action_log,
                runtime_result=heuristic_result.runtime_result,
            )

    if "dqn_agent" in enabled_methods:
        dqn_row, dqn_runtime_result, dqn_action_log = _evaluate_dqn_once(
            model_path=model_path,
            experiment_config=experiment_config,
            agent_period=agent_period,
            seed=seed,
            end_time=end_time,
            row_base=row_base,
            reward_mode=reward_mode,
            dqn_runtime_semantics=dqn_runtime_semantics,
            action_space=action_space,
            budget_increase_ratio=budget_increase_ratio,
            budget_decrease_ratio=budget_decrease_ratio,
            include_explicit_noop=include_explicit_noop,
            budget_floor_ratio=budget_floor_ratio,
            forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
            mask_detail_mode=mask_detail_mode,
            enable_deploy_cap_mask=enable_deploy_cap_mask,
            deploy_cap_mask_ratio=deploy_cap_mask_ratio,
            deploy_cap_mask_criticality=deploy_cap_mask_criticality,
            trace_dir=trace_dir,
            debug_log_dir=debug_log_dir,
            trace_enabled=trace_enabled_for_seed and (not trace_method_set or "dqn_agent" in trace_method_set),
            capture_trace=capture_runtime_trace_for_seed,
            capture_debug_events=capture_debug_events_for_seed,
            agent_device=dqn_agent_device,
            double_dqn=double_dqn,
            max_q_diagnostic_samples=max_q_diagnostic_samples,
            feature_config=feature_config,
            c_amc_sem_xf=c_amc_sem_xf,
            constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
            constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
            constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
            constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
            constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
        )
        rows.append(dqn_row)
        qamc_runtime_results["dqn_agent"] = dqn_runtime_result
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method="dqn_agent",
                runtime_result=dqn_runtime_result,
                action_log=dqn_action_log,
            )
        )

    tree_specs = [
        ("bc_tree_agent", bc_tree_model),
        ("dagger_tree_agent", dagger_tree_model),
        ("viper_tree_agent", viper_tree_model),
    ]
    for method_name, artifact_dir in tree_specs:
        if method_name not in enabled_methods:
            continue
        if artifact_dir is None:
            raise ValueError(f"未提供 {method_name} 对应的 tree artifact 路径")
        # 计算当前 seed/method 是否应启用 leaf audit
        leaf_audit_enabled = (
            tree_audit_enabled_for_seed
            and (not tree_audit_method_set or method_name in tree_audit_method_set)
        )
        tree_row, tree_runtime_result, tree_action_log = _evaluate_tree_once(
            tree_artifact_dir=artifact_dir,
            method_name=method_name,
            experiment_config=experiment_config,
            agent_period=agent_period,
            seed=seed,
            end_time=end_time,
            row_base=row_base,
            reward_mode=reward_mode,
            dqn_runtime_semantics=dqn_runtime_semantics,
            action_space=action_space,
            budget_increase_ratio=budget_increase_ratio,
            budget_decrease_ratio=budget_decrease_ratio,
            include_explicit_noop=include_explicit_noop,
            budget_floor_ratio=budget_floor_ratio,
            forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
            mask_detail_mode=mask_detail_mode,
            enable_deploy_cap_mask=enable_deploy_cap_mask,
            deploy_cap_mask_ratio=deploy_cap_mask_ratio,
            deploy_cap_mask_criticality=deploy_cap_mask_criticality,
            feature_config=feature_config,
            c_amc_sem_xf=c_amc_sem_xf,
            teacher_model_path=tree_compare_teacher_model,
            require_integer_tree=require_integer_tree_artifact,
            leaf_audit_enabled=leaf_audit_enabled,
            leaf_audit_state_mode=tree_audit_state_mode,
            leaf_audit_top_k_actions=tree_audit_top_k_actions,
        )
        rows.append(tree_row)
        qamc_runtime_results[method_name] = tree_runtime_result
        deadline_miss_details.extend(
            _deadline_miss_detail_rows(
                row_base=row_base,
                method=method_name,
                runtime_result=tree_runtime_result,
                action_log=tree_action_log,
            )
        )
        # 写出 leaf audit 文件（独立于 trace）
        if leaf_audit_enabled and tree_audit_dir is not None:
            # 从 tree_row 中提取 tree metadata，不改变 _evaluate_tree_once 返回签名
            tree_metadata = {
                "tree_id": tree_row.get("tree_id"),
                "method": tree_row.get("tree_method"),
                "tree_depth": tree_row.get("tree_depth"),
                "tree_node_count": tree_row.get("tree_node_count"),
                "tree_leaf_count": tree_row.get("tree_leaf_count"),
                "max_depth": tree_row.get("tree_max_depth_param"),
                "min_samples_leaf": tree_row.get("tree_min_samples_leaf"),
            }
            _write_tree_audit_files(
                tree_audit_dir=tree_audit_dir,
                seed=seed,
                method=method_name,
                action_log=tree_action_log,
                row_base=row_base,
                tree_metadata=tree_metadata,
            )

    if qamc_profile_bundle is not None:
        for row in rows:
            runtime_result = qamc_runtime_results.get(str(row.get("method")))
            if runtime_result is None:
                continue
            row.update(qamc_metrics_to_row(compute_qamc_metrics(runtime_result, qamc_profile_bundle)))
            row.update(qamc_loss_metrics_to_row(compute_qamc_loss_metrics(runtime_result)))
            row["qamc_legacy_degraded_metrics_applicable"] = False
            _blank_legacy_degraded_fields(row)
            row["method"] = QAMC_OUTPUT_METHOD_NAMES.get(
                str(row.get("method")), str(row.get("method"))
            )

    return rows, deadline_miss_details


def _evaluate_seed_worker(
    args_tuple: tuple[
        int,
        object,
        str,
        float,
        int,
        float,
        float,
        bool,
        set[str],
        Path,
        int,
        int,
        str,
        RuntimeSemantics,
        float,
        str,
        float,
        float,
        bool,
        float,
        bool,
        str,
        bool,
        float,
        str,
        FeatureConfig,
        Path | None,
        Path | None,
        set[int],
        set[str],
        bool,
        int,
        int,
        int,
        bool,
        bool,
        bool,
        Path | None,
        Path | None,
        Path | None,
        Path | None,
        Path | None,
        set[int] | None,
        set[str] | None,
        str,
        int,
        bool,
    ],
) -> tuple[list[dict[str, int | float | str | bool]], list[dict[str, object]]]:
    """并行 worker：完成单个 seed 的全部评估方法。

    这里仍然按"一个 seed 内部顺序执行多个方法"的粒度并行，而不是把每个方法单独拆开。
    这样可以保持当前输出结构、trace 文件命名以及每个 seed 的局部执行顺序都不变，
    同时也避免把同一个任务集/场景在多个子进程里重复构造多次。
    """

    (
        seed,
        experiment_config,
        workload,
        total_util,
        num_tasks,
        cf,
        cp,
        require_schedulable,
        enabled_methods,
        model_path,
        end_time,
        agent_period,
        reward_mode,
        dqn_runtime_semantics,
        c_amc_sem_xf,
        action_space,
        budget_increase_ratio,
        budget_decrease_ratio,
        include_explicit_noop,
        budget_floor_ratio,
        forbid_decreasing_hi_budgets,
        mask_detail_mode,
        enable_deploy_cap_mask,
        deploy_cap_mask_ratio,
        deploy_cap_mask_criticality,
        feature_config,
        trace_dir,
        debug_log_dir,
        trace_seed_set,
        trace_method_set,
        double_dqn,
        max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe,
        bc_tree_model,
        dagger_tree_model,
        viper_tree_model,
        tree_compare_teacher_model,
        tree_audit_dir,
        tree_audit_seed_set,
        tree_audit_method_set,
        tree_audit_state_mode,
        tree_audit_top_k_actions,
        require_integer_tree_artifact,
    ) = args_tuple
    return _evaluate_enabled_methods_for_seed(
        seed=seed,
        experiment_config=experiment_config,
        workload=workload,
        total_util=total_util,
        num_tasks=num_tasks,
        cf=cf,
        cp=cp,
        require_schedulable=require_schedulable,
        enabled_methods=enabled_methods,
        model_path=model_path,
        end_time=end_time,
        agent_period=agent_period,
        reward_mode=reward_mode,
        dqn_runtime_semantics=dqn_runtime_semantics,
        c_amc_sem_xf=c_amc_sem_xf,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        enable_deploy_cap_mask=enable_deploy_cap_mask,
        deploy_cap_mask_ratio=deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=deploy_cap_mask_criticality,
        feature_config=feature_config,
        trace_dir=trace_dir,
        debug_log_dir=debug_log_dir,
        trace_seed_set=trace_seed_set,
        trace_method_set=trace_method_set,
        dqn_agent_device="cpu",
        double_dqn=double_dqn,
        max_q_diagnostic_samples=max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
        bc_tree_model=bc_tree_model,
        dagger_tree_model=dagger_tree_model,
        viper_tree_model=viper_tree_model,
        tree_compare_teacher_model=tree_compare_teacher_model,
        tree_audit_dir=tree_audit_dir,
        tree_audit_seed_set=tree_audit_seed_set,
        tree_audit_method_set=tree_audit_method_set,
        tree_audit_state_mode=tree_audit_state_mode,
        tree_audit_top_k_actions=tree_audit_top_k_actions,
        require_integer_tree_artifact=require_integer_tree_artifact,
    )


def build_parser() -> argparse.ArgumentParser:
    """构建正式评估 CLI 的参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive", "mc_fairgen"], default="small")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument(
        "--fixed-taskset-seed",
        type=int,
        default=None,
        help=(
            "如果设置该参数，支持 fixed taskset 的 workload 会固定使用该 seed 生成任务集；"
            "评估 seed 仅用于 scenario 生成。"
        ),
    )
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=1,
        help="并行按 seed 评估的进程数；1 表示保持串行。",
    )
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument(
        "--dqn-runtime-semantics",
        choices=["AMC_PLUS", "AMC_RA", "AMC_RH", "C_AMC_SEM", "Q_AMC"],
        default="AMC_PLUS",
        help="Runtime semantics used by dqn_agent and wrapper-based agent baselines.",
    )
    parser.add_argument(
        "--c-amc-sem-xf",
        type=float,
        default=0.5,
        help="LO-task degraded budget ratio used by c_amc_sem_baseline in HI mode.",
    )
    parser.add_argument("--qamc-reference-config-path", type=Path, default=None)
    parser.add_argument("--qamc-profile-manifest-path", type=Path, default=None)
    parser.add_argument("--qamc-profile-spec-path", type=Path, default=None)
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument(
        "--baselines",
        type=str,
        default="amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,noop_agent,dqn_agent",
    )
    parser.add_argument("--bc-tree-model", type=Path, default=None)
    parser.add_argument("--dagger-tree-model", type=Path, default=None)
    parser.add_argument("--viper-tree-model", type=Path, default=None)
    parser.add_argument("--tree-compare-teacher-model", type=Path, default=None)
    parser.add_argument("--require-integer-tree-artifact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tree-audit-dir",
        type=Path,
        default=None,
        help="单独输出 leaf audit 文件，不开启 runtime tick trace。",
    )
    parser.add_argument(
        "--tree-audit-seeds",
        type=str,
        default="",
        help="只对指定 HOUT scenario seed 写 audit；空字符串表示所有 seed。",
    )
    parser.add_argument(
        "--tree-audit-methods",
        type=str,
        default="",
        help="只对指定 tree method 写 audit；空字符串表示所有 tree methods。",
    )
    parser.add_argument(
        "--tree-audit-state-mode",
        choices=["none", "split", "all"],
        default="split",
        help="控制 leaf audit 中状态特征的记录粒度。",
    )
    parser.add_argument(
        "--tree-audit-top-k-actions",
        type=int,
        default=5,
        help="leaf audit 中记录 top-k 个动作信息。",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn_amc/eval_summary.csv"))
    parser.add_argument("--fail-on-deadline-miss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-seeds", type=str, default="")
    parser.add_argument("--trace-methods", type=str, default="")
    parser.add_argument("--debug-log-dir", type=Path, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=list(available_reward_modes()),
        default="mendes",
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
        "--noop-exploration-prob",
        type=float,
        default=0.0,
        help=(
            "During epsilon exploration, if explicit noop is valid, choose noop "
            "with this probability before sampling other valid actions."
        ),
    )
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Double DQN target calculation flag for consistency with training checkpoints.",
    )
    parser.add_argument(
        "--max-q-diagnostic-samples",
        type=int,
        default=1000,
        help="Maximum DQN decision states sampled for noop Q diagnostics per evaluation seed.",
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
        # 与训练脚本同名同义参数，保证“训练怎么约束，评估就怎么约束”。
        # 如果训练时开启但评估时关闭，会导致动作可行域不一致，比较结果失真。
        help="If set, action masks reject budget actions whose decrease tasks include any HI-criticality task.",
    )
    parser.add_argument("--enable-deploy-cap-mask", action="store_true")
    parser.add_argument("--deploy-cap-mask-ratio", type=float, default=4.0)
    parser.add_argument("--deploy-cap-mask-criticality", choices=["lo", "all"], default="lo")
    # automotive workload 允许从 CLI 显式切换 runnable 数量与 workload 语义模式，
    # 保证评估入口与训练入口能使用相同的 automotive 配置。
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
            "v13_rh_17d",
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
    """运行正式 DQN 评估，并输出统一 CSV。"""

    args = build_parser().parse_args()
    if args.action_space == "constraint_guided_pair":
        # 兼容旧参数名：内部统一走 constraint_guided_transfer。
        args.action_space = "constraint_guided_transfer"
    if args.evaluation_workers < 1:
        raise ValueError("--evaluation-workers 必须为正整数")
    if args.max_q_diagnostic_samples < 0:
        raise ValueError("--max-q-diagnostic-samples 必须为非负整数")
    if args.budget_floor_ratio < 0.0 or args.budget_floor_ratio > 1.0:
        raise ValueError("--budget-floor-ratio must be in [0, 1]")
    if not (0.0 < args.c_amc_sem_xf <= 1.0):
        raise ValueError("--c-amc-sem-xf must be in (0, 1]")
    if args.deploy_cap_mask_ratio <= 1.0:
        raise ValueError("--deploy-cap-mask-ratio must be > 1.0")
    if args.tree_audit_top_k_actions < 1:
        raise ValueError("--tree-audit-top-k-actions must be >= 1")
    tree_audit_seed_set = _parse_int_set_or_ranges(args.tree_audit_seeds)
    tree_audit_method_set = _parse_csv_set(args.tree_audit_methods)
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
    dqn_runtime_semantics = RuntimeSemantics(args.dqn_runtime_semantics)
    if args.workload == "small":
        experiment_config = (
            build_small_nominal_experiment_config()
            if args.scenario == "nominal"
            else build_small_stress_experiment_config()
        )
    elif args.workload == "rtss11":
        experiment_config = build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
        )
    elif args.workload == "automotive":
        experiment_config = build_automotive_experiment_config(
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
    elif args.workload == "mc_fairgen":
        experiment_config = build_mc_fairgen_experiment_config(
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
    else:
        raise ValueError(f"unsupported workload: {args.workload}")
    if dqn_runtime_semantics is RuntimeSemantics.Q_AMC:
        if args.qamc_reference_config_path is None:
            raise ValueError("QAMC_REFERENCE_CONFIG_REQUIRED")
        if not args.qamc_reference_config_path.is_file():
            raise ValueError("QAMC_REFERENCE_CONFIG_MISSING")
        frozen_reference = load_and_validate_frozen_reference(args.qamc_reference_config_path)
        validate_qamc_rl_semantics(
            semantics=dqn_runtime_semantics,
            action_space=args.action_space,
            check_safety=True,
            step_guard_semantics="checked",
            nonvacuity_disabled_guards=(),
            budget_rounding_mode="ceil_floor",
            min_budget_delta=1,
        )
        assert_reference_matches_values(
            frozen_reference,
            {
                "action_space": args.action_space,
                "include_explicit_noop": args.include_explicit_noop,
                "budget_increase_ratio": args.budget_increase_ratio,
                "budget_decrease_ratio": args.budget_decrease_ratio,
                "budget_rounding_mode": "ceil_floor",
                "min_budget_delta": 1,
                "budget_floor_ratio": args.budget_floor_ratio,
                "observation_mode": args.observation_mode,
                "reward_mode": args.reward_mode,
                "agent_period": args.agent_period,
                "check_safety": True,
            },
        )
        if args.qamc_profile_manifest_path is None or args.qamc_profile_spec_path is None:
            raise ValueError("QAMC_PROFILE_MANIFEST_AND_SPEC_REQUIRED")
        qamc_spec = load_profile_spec(args.qamc_profile_spec_path)
        experiment_config = replace(
            experiment_config,
            qamc_reference_config_path=str(args.qamc_reference_config_path),
            qamc_profile_manifest_path=str(args.qamc_profile_manifest_path),
            qamc_profile_spec_path=str(args.qamc_profile_spec_path),
        )
    effective_num_tasks = (
        args.mc_fairgen_num_tasks if args.workload == "mc_fairgen" else args.num_tasks
    )

    enabled_methods = set(_parse_baselines(args.baselines))
    if dqn_runtime_semantics is RuntimeSemantics.Q_AMC:
        enabled_methods = {
            QAMC_METHOD_ALIASES.get(method, method) for method in enabled_methods
        }
        if "dqn_agent" in enabled_methods:
            binding_seeds = _parse_seeds(args.seeds)
            if not binding_seeds:
                raise ValueError("QAMC_MODEL_BINDING_SEED_REQUIRED")
            binding_seed = binding_seeds[0]
            binding_bundle = resolve_experiment_bundle(
                experiment_config,
                binding_seed,
            )
            binding_profile = load_profile_bundle_from_manifest(
                args.qamc_profile_manifest_path,
                taskset_fingerprint=str(binding_bundle.taskset_fingerprint),
                spec_path=args.qamc_profile_spec_path,
            )
            binding_env = build_env_from_experiment_config(
                experiment_config,
                seed=binding_seed,
                end_time=args.end_time,
                agent_period=args.agent_period,
                semantics=dqn_runtime_semantics,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                mask_detail_mode=args.mask_detail_mode,
                enable_deploy_cap_mask=args.enable_deploy_cap_mask,
                deploy_cap_mask_ratio=args.deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=args.deploy_cap_mask_criticality,
                record_dropped_lo_releases=True,
                c_amc_sem_xf=args.c_amc_sem_xf,
                feature_config=feature_config,
                constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=(
                    args.constraint_guided_pair_allow_increase_only_when_safe
                ),
            )
            binding_observation = binding_env.reset(seed=binding_seed)
            validate_qamc_model_artifact(
                args.model,
                frozen_reference=frozen_reference,
                profile_manifest_path=args.qamc_profile_manifest_path,
                profile_spec_fingerprint=qamc_spec.fingerprint,
                expected_taskset_fingerprint=str(
                    binding_bundle.taskset_fingerprint
                ),
                expected_profile_fingerprint=binding_profile.fingerprint,
                expected_action_dim=binding_env.action_count,
                expected_observation_dim=len(binding_observation.state_vector),
                expected_action_space_fingerprint=compute_action_space_fingerprint(
                    binding_env.actions
                ),
                expected_semantic_version=qamc_spec.semantic_version,
                expected_demand_mapping_version=qamc_spec.demand_mapping_version,
            )
    trace_seed_set = {int(s) for s in _parse_csv_set(args.trace_seeds)}
    trace_method_set = _parse_csv_set(args.trace_methods)
    valid_methods = {
        "amc_same_full_sample_native",
        "q_amc_native",
        "q_amc_budget_heuristic",
        "q_amc_dqn_budget_overlay",
        "q_amc_viper_budget_overlay",
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
        "c_amc_sem_baseline",
        "noop_agent",
        "random_agent",
        "heuristic_agent",
        "dqn_agent",
        "bc_tree_agent",
        "dagger_tree_agent",
        "viper_tree_agent",
    }
    unsupported_methods = sorted(enabled_methods - valid_methods)
    if unsupported_methods:
        raise ValueError(f"不支持的 baselines: {unsupported_methods}")

    rows: list[dict[str, int | float | str | bool]] = []
    deadline_miss_details: list[dict[str, object]] = []
    seeds = _parse_seeds(args.seeds)
    if args.evaluation_workers == 1:
        per_seed_results = [
            _evaluate_enabled_methods_for_seed(
                seed=seed,
                experiment_config=experiment_config,
                workload=args.workload,
                total_util=args.total_util,
                num_tasks=effective_num_tasks,
                cf=args.cf,
                cp=args.cp,
                require_schedulable=args.require_schedulable,
                enabled_methods=enabled_methods,
                model_path=args.model,
                end_time=args.end_time,
                agent_period=args.agent_period,
                reward_mode=args.reward_mode,
                dqn_runtime_semantics=dqn_runtime_semantics,
                c_amc_sem_xf=args.c_amc_sem_xf,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                mask_detail_mode=args.mask_detail_mode,
                enable_deploy_cap_mask=args.enable_deploy_cap_mask,
                deploy_cap_mask_ratio=args.deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=args.deploy_cap_mask_criticality,
                feature_config=feature_config,
                trace_dir=args.trace_dir,
                debug_log_dir=args.debug_log_dir,
                trace_seed_set=trace_seed_set,
                trace_method_set=trace_method_set,
                double_dqn=args.double_dqn,
                max_q_diagnostic_samples=args.max_q_diagnostic_samples,
                constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=(
                    args.constraint_guided_pair_allow_increase_only_when_safe
                ),
                bc_tree_model=args.bc_tree_model,
                dagger_tree_model=args.dagger_tree_model,
                viper_tree_model=args.viper_tree_model,
                tree_compare_teacher_model=args.tree_compare_teacher_model,
                tree_audit_dir=args.tree_audit_dir,
                tree_audit_seed_set=tree_audit_seed_set,
                tree_audit_method_set=tree_audit_method_set,
                tree_audit_state_mode=args.tree_audit_state_mode,
                tree_audit_top_k_actions=args.tree_audit_top_k_actions,
                require_integer_tree_artifact=args.require_integer_tree_artifact,
            )
            for seed in seeds
        ]
    else:
        # 按 seed 并行时，每个 worker 都独立完成该 seed 的全部方法评估。
        # executor.map 会保持输入 seeds 的顺序，因此最终 CSV 仍然是稳定的。
        worker_args = [
            (
                seed,
                experiment_config,
                args.workload,
                args.total_util,
                effective_num_tasks,
                args.cf,
                args.cp,
                args.require_schedulable,
                enabled_methods,
                args.model,
                args.end_time,
                args.agent_period,
                args.reward_mode,
                dqn_runtime_semantics,
                args.c_amc_sem_xf,
                args.action_space,
                args.budget_increase_ratio,
                args.budget_decrease_ratio,
                args.include_explicit_noop,
                args.budget_floor_ratio,
                args.forbid_decreasing_hi_budgets,
                args.mask_detail_mode,
                args.enable_deploy_cap_mask,
                args.deploy_cap_mask_ratio,
                args.deploy_cap_mask_criticality,
                feature_config,
                args.trace_dir,
                args.debug_log_dir,
                trace_seed_set,
                trace_method_set,
                args.double_dqn,
                args.max_q_diagnostic_samples,
                args.constraint_guided_pair_top_k_risk,
                args.constraint_guided_pair_top_k_decrease,
                args.constraint_guided_pair_prefer_lo,
                args.constraint_guided_pair_include_hi_risk_boost,
                args.constraint_guided_pair_allow_increase_only_when_safe,
                args.bc_tree_model,
                args.dagger_tree_model,
                args.viper_tree_model,
                args.tree_compare_teacher_model,
                args.tree_audit_dir,
                tree_audit_seed_set,
                tree_audit_method_set,
                args.tree_audit_state_mode,
                args.tree_audit_top_k_actions,
                args.require_integer_tree_artifact,
            )
            for seed in seeds
        ]
        try:
            with ProcessPoolExecutor(max_workers=args.evaluation_workers) as executor:
                per_seed_results = list(executor.map(_evaluate_seed_worker, worker_args))
        except PermissionError:
            # 受限执行环境下可能禁止创建并行进程资源，此时回退到串行执行，
            # 保持输出字段与统计口径一致，仅不使用并行加速。
            per_seed_results = [_evaluate_seed_worker(item) for item in worker_args]

    for seed_rows, seed_deadline_miss_details in per_seed_results:
        rows.extend(seed_rows)
        deadline_miss_details.extend(seed_deadline_miss_details)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _eval_summary_fieldnames()
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_unified_summary_csv(args.output, rows)

    miss_rows = _deadline_miss_rows(rows)
    if miss_rows:
        detail_output_path = args.output.with_name(f"{args.output.stem}_deadline_misses.jsonl")
        _write_jsonl(detail_output_path, deadline_miss_details)
    if args.fail_on_deadline_miss and miss_rows:
        print("Deadline miss detected under check_safety=True:")
        for row in miss_rows:
            print(
                "  "
                f"total_util={row['total_util']} seed={row['seed']} "
                f"method={row['method']} deadline_misses={row['deadline_misses']}"
            )
        print("First deadline miss details:")
        printed = 0
        for detail_row in deadline_miss_details:
            if printed >= 3:
                break
            print(
                "  "
                f"seed={detail_row['seed']} method={detail_row['method']} task={detail_row['task']} "
                f"rel={detail_row['release_index']} deadline={detail_row['absolute_deadline']} "
                f"executed={detail_row['executed_at_miss']} "
                f"budget_at_release={detail_row['runtime_budget_at_release']}"
            )
            printed += 1
        print("Evaluation failed because --fail-on-deadline-miss is enabled.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
