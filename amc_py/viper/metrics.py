"""VIPER 训练与评估指标。"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
import math

import numpy as np

from amc_py.dqn import DqnBudgetAgent, ExperimentConfig, build_env_from_experiment_config
from amc_py.metrics import (
    compute_lo_quality_weighted_metrics,
    compute_runtime_degradation_metrics,
    compute_service_quality_metrics,
    lo_quality_weighted_metrics_to_row,
    service_metrics_to_row,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics, SimulationResult
from amc_py.viper.dataset import ViperSample
from amc_py.viper.tree_policy import TreeBudgetPolicy


def retention_higher_is_better(parent: float, teacher: float, tree: float) -> float | None:
    if teacher <= parent:
        return None
    return (tree - parent) / (teacher - parent)


def retention_lower_is_better(parent: float, teacher: float, tree: float) -> float | None:
    if teacher >= parent:
        return None
    return (parent - tree) / (parent - teacher)


def compute_offline_tree_metrics(tree_policy: TreeBudgetPolicy, samples: Sequence[ViperSample]) -> dict[str, float]:
    """在离线 dataset 上统计 tree fidelity 与 q-regret。"""

    labeled_samples = [sample for sample in samples if sample.teacher_action_id is not None]
    if not labeled_samples:
        raise ValueError("没有可评估的 labeled samples")
    correct = 0
    weighted_correct = 0.0
    total_weight = 0.0
    q_regrets: list[float] = []
    raw_invalid_count = 0
    mask_aware_match_count = 0
    for sample in labeled_samples:
        selected_action_id, info = tree_policy.select_action_id(sample.state_vector, sample.valid_action_mask)
        raw_invalid_count += int(bool(info["tree_raw_top1_invalid"]))
        if selected_action_id == sample.teacher_action_id:
            correct += 1
            mask_aware_match_count += 1
        weight = float(sample.viper_weight if sample.viper_weight is not None else 1.0)
        total_weight += weight
        if selected_action_id == sample.teacher_action_id:
            weighted_correct += weight
        if selected_action_id is not None and sample.q_best is not None:
            q_regrets.append(float(sample.q_best) - float(sample.raw_q_values[selected_action_id]))
    return {
        "offline_accuracy": correct / len(labeled_samples),
        "weighted_fidelity": (weighted_correct / total_weight) if total_weight > 0.0 else 0.0,
        "q_regret_mean": mean(q_regrets) if q_regrets else 0.0,
        "q_regret_p95": float(np.percentile(q_regrets, 95)) if q_regrets else 0.0,
        "raw_top1_invalid_rate_on_dataset": raw_invalid_count / len(labeled_samples),
        "mask_aware_match_rate_on_dataset": mask_aware_match_count / len(labeled_samples),
    }


def evaluate_tree_policy_once(
    *,
    tree_policy: TreeBudgetPolicy,
    experiment_config: ExperimentConfig,
    seed: int,
    end_time: int,
    agent_period: int,
    runtime_semantics: RuntimeSemantics,
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
    c_amc_sem_xf: float = 0.5,
    teacher: DqnBudgetAgent | None = None,
) -> tuple[dict[str, object], SimulationResult, list[dict[str, object]]]:
    """在真实 runtime 中评估一棵 tree policy。"""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=runtime_semantics,
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
        capture_trace=False,
        capture_debug_events=False,
        record_dropped_lo_releases=True,
        c_amc_sem_xf=c_amc_sem_xf,
        feature_config=feature_config,
    )
    obs = env.reset(seed=seed)
    done = False
    step_count = 0
    total_reward = 0.0
    raw_invalid_count = 0
    fallback_count = 0
    no_valid_action_count = 0
    selected_action_count = 0
    selected_action_match_teacher_count = 0
    raw_action_match_teacher_count = 0
    q_regrets: list[float] = []
    while not done:
        step_count += 1
        mask = env.valid_action_mask()
        teacher_diag = None
        if teacher is not None:
            if getattr(teacher, "q_network_type", "mlp") == "action_aware":
                action_features = env.get_action_feature_matrix(teacher.action_feature_mode)
                action_feature_names = env.get_action_feature_names(teacher.action_feature_mode)
                teacher.set_action_features(action_features, action_feature_names)
            teacher_diag = teacher.compute_q_diagnostics(obs.state_vector, mask)
        action_id, info = tree_policy.select_action_id(obs.state_vector, mask)
        raw_invalid_count += int(bool(info["tree_raw_top1_invalid"]))
        fallback_count += int(bool(info["tree_fallback_used"]))
        no_valid_action_count += int(bool(info["tree_no_valid_action"]))
        selected_action_count += int(action_id is not None)
        if teacher_diag is not None:
            raw_action_match_teacher_count += int(info["tree_raw_top1_action_id"] == teacher_diag["best_action_id"])
            selected_action_match_teacher_count += int(action_id == teacher_diag["best_action_id"])
            if action_id is not None and teacher_diag["q_best"] is not None:
                q_regrets.append(float(teacher_diag["q_best"]) - float(teacher_diag["raw_q_values"][action_id]))
        result = env.step(action_id)
        total_reward += float(result.reward)
        obs = result.observation
        done = result.done
    runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
    service_metrics = compute_service_quality_metrics(runtime_result)
    lo_quality_metrics = compute_lo_quality_weighted_metrics(runtime_result)
    degradation = compute_runtime_degradation_metrics(runtime_result)
    debug_stats = env.debug_statistics()
    row = {
        **service_metrics_to_row(service_metrics),
        **lo_quality_weighted_metrics_to_row(lo_quality_metrics),
        "deadline_misses": len(runtime_result.deadline_misses),
        "hi_deadline_misses": service_metrics.hi_deadline_misses,
        "lo_deadline_misses": service_metrics.lo_deadline_misses,
        "mode_changes": runtime_result.mode_change_count(),
        "lo_cancellations": runtime_result.lo_job_cancellation_count(),
        "hdm": degradation.hdm,
        "jne": degradation.jne,
        "ldm": degradation.ldm,
        "nid": degradation.nid,
        "tid": degradation.tid,
        "total_time": degradation.total_time,
        "tid_ratio": degradation.tid_ratio,
        "jne_plus_ldm": degradation.jne + degradation.ldm,
        "accepted_actions": selected_action_count,
        "rejected_actions": 0,
        "step_count": step_count,
        "selected_action_count": selected_action_count,
        "total_reward": total_reward,
        "tree_raw_top1_invalid_count": raw_invalid_count,
        "tree_raw_top1_invalid_rate": (raw_invalid_count / step_count) if step_count > 0 else 0.0,
        "tree_fallback_count": fallback_count,
        "tree_fallback_rate": (fallback_count / step_count) if step_count > 0 else 0.0,
        "tree_no_valid_action_count": no_valid_action_count,
        "tree_no_valid_action_rate": (no_valid_action_count / step_count) if step_count > 0 else 0.0,
        "tree_selected_action_count": selected_action_count,
        "tree_selected_action_match_teacher_count": selected_action_match_teacher_count if teacher is not None else None,
        "tree_selected_action_match_teacher_rate": (
            (selected_action_match_teacher_count / step_count) if teacher is not None and step_count > 0 else None
        ),
        "tree_raw_action_match_teacher_rate": (
            (raw_action_match_teacher_count / step_count) if teacher is not None and step_count > 0 else None
        ),
        "tree_q_regret_mean": (mean(q_regrets) if q_regrets else None),
        "tree_q_regret_p95": (float(np.percentile(q_regrets, 95)) if q_regrets else None),
        "action_space_type": str(debug_stats["action_space_type"]),
        "action_count": int(debug_stats["action_count"]),
        "check_safety": bool(debug_stats["check_safety"]),
        "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
        "safety_accepted_actions": int(debug_stats["safety_accepted_actions"]),
        "safety_rejected_actions": int(debug_stats["safety_rejected_actions"]),
        "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
        "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
        "masked_action_count_max": int(debug_stats["masked_action_count_max"]),
        "mask_rejection_rate_mean": float(debug_stats["mask_rejection_rate_mean"]),
        "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
        "selected_explicit_noop_actions": int(debug_stats["selected_explicit_noop_actions"]),
        "selected_explicit_noop_rate": float(debug_stats["selected_explicit_noop_rate"]),
        "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
    }
    return row, runtime_result, env.action_log
