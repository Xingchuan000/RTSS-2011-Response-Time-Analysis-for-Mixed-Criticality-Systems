"""VIPER 训练与评估指标。

新增 leaf-level execution audit 能力：
- _build_leaf_audit_fields(): 构造单步 leaf audit 诊断字段，合并到 env.action_log 中。
- evaluate_tree_policy_once(): 新增 leaf_audit_enabled / leaf_audit_state_mode / leaf_audit_top_k_actions 参数。
"""

from __future__ import annotations

from collections.abc import Sequence
import json
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
from amc_py.viper.tree_policy import TreeBudgetPolicy, IntegerTreeBudgetPolicy


def retention_higher_is_better(parent: float, teacher: float, tree: float) -> float | None:
    if teacher <= parent:
        return None
    return (tree - parent) / (teacher - parent)


def retention_lower_is_better(parent: float, teacher: float, tree: float) -> float | None:
    if teacher >= parent:
        return None
    return (parent - tree) / (parent - teacher)


def compute_offline_tree_metrics(tree_policy: TreeBudgetPolicy | IntegerTreeBudgetPolicy, samples: Sequence[ViperSample]) -> dict[str, float]:
    """在离线 dataset 上统计 tree fidelity 与 q-regret。"""

    labeled_samples = [sample for sample in samples if sample.teacher_action_id is not None]
    if not labeled_samples:
        raise ValueError("没有可评估的 labeled samples")
    correct = 0
    raw_correct = 0
    weighted_correct = 0.0
    total_weight = 0.0
    q_regrets: list[float] = []
    raw_invalid_count = 0
    mask_aware_match_count = 0
    noop_fallback_count = 0
    teacher_none_count = sum(int(sample.teacher_action_id is None) for sample in samples)
    for sample in labeled_samples:
        selected_action_id, info = tree_policy.select_action_id(sample.state_vector, sample.valid_action_mask)
        raw_invalid_count += int(bool(info["tree_raw_top1_invalid"]))
        noop_fallback_count += int(bool(info.get("tree_fallback_used", False) and selected_action_id is None))
        raw_top1 = info.get("tree_raw_top1_action_id")
        if raw_top1 == sample.teacher_action_id:
            raw_correct += 1
        if selected_action_id == sample.teacher_action_id:
            correct += 1
            mask_aware_match_count += 1
        weight = float(sample.viper_weight if sample.viper_weight is not None else 1.0)
        total_weight += weight
        if raw_top1 == sample.teacher_action_id:
            weighted_correct += weight
        raw_top1 = info.get("tree_raw_top1_action_id")
        if isinstance(raw_top1, int) and sample.q_best is not None:
            q_regrets.append(float(sample.q_best) - float(sample.raw_q_values[raw_top1]))
    raw_accuracy = raw_correct / len(labeled_samples)
    deployed_match = mask_aware_match_count / len(labeled_samples)
    return {
        "offline_accuracy": raw_accuracy,
        "offline_raw_top1_accuracy": raw_accuracy,
        "offline_deployed_match_rate": deployed_match,
        "weighted_fidelity": (weighted_correct / total_weight) if total_weight > 0.0 else 0.0,
        "q_regret_mean": mean(q_regrets) if q_regrets else 0.0,
        "q_regret_p95": float(np.percentile(q_regrets, 95)) if q_regrets else 0.0,
        "raw_top1_invalid_rate_on_dataset": raw_invalid_count / len(labeled_samples),
        "mask_aware_match_rate_on_dataset": mask_aware_match_count / len(labeled_samples),
        "noop_fallback_rate_on_dataset": noop_fallback_count / len(labeled_samples),
        "teacher_none_rate": teacher_none_count / len(samples) if samples else 0.0,
        "q_regret_raw_top1_mean": mean(q_regrets) if q_regrets else 0.0,
        "q_regret_raw_top1_p95": float(np.percentile(q_regrets, 95)) if q_regrets else 0.0,
    }


def _build_leaf_audit_fields(
    *,
    step_index: int,
    state_vector: tuple[float, ...],
    feature_names: tuple[str, ...],
    tree_policy: TreeBudgetPolicy | IntegerTreeBudgetPolicy,
    tree_info: dict[str, object],
    selected_action_id: int | None,
    valid_action_mask: tuple[bool, ...],
    teacher_diag: dict[str, object] | None,
    leaf_audit_state_mode: str,
    leaf_audit_top_k_actions: int,
) -> dict[str, object]:
    """构造单步 leaf audit 诊断字段，写入 env.action_log 中。

    参数：
    - step_index: 当前步序号（从 0 开始）。
    - state_vector: 当前状态向量。
    - feature_names: 特征名元组。
    - tree_policy: TreeBudgetPolicy 实例。
    - tree_info: select_action_id 返回的 info 字典（已包含 trace 信息）。
    - selected_action_id: 实际选中的动作编号。
    - valid_action_mask: 合法动作 mask 元组。
    - teacher_diag: teacher 的 Q 诊断信息（可为 None）。
    - leaf_audit_state_mode: "none" / "split" / "all"。
    - leaf_audit_top_k_actions: 记录 top-k 个动作。

    返回：
    - 可直接 update 到 action_log 条目的 dict，所有值均为 JSON 可序列化类型。
    """

    fields: dict[str, object] = {}

    # 基础 step 字段
    fields["tree_audit_step_index"] = step_index

    # 叶子 / 路径基本信息（直接从 tree_info 中提取）
    fields["tree_leaf_id"] = tree_info.get("tree_leaf_id")
    fields["tree_path_depth"] = tree_info.get("tree_path_depth")
    fields["tree_path_node_ids_json"] = json.dumps(
        list(tree_info.get("tree_path_node_ids", ())), ensure_ascii=False
    )
    path_predicates = tree_info.get("tree_path_predicates")
    if isinstance(path_predicates, list):
        fields["tree_path_predicates_json"] = json.dumps(path_predicates, ensure_ascii=False)

    # 叶子训练统计字段
    fields["tree_leaf_n_node_samples"] = tree_info.get("tree_leaf_n_node_samples")
    fields["tree_leaf_weighted_n_node_samples"] = tree_info.get("tree_leaf_weighted_n_node_samples")
    fields["tree_leaf_impurity"] = tree_info.get("tree_leaf_impurity")
    leaf_value = tree_info.get("tree_leaf_value")
    if leaf_value is not None:
        fields["tree_leaf_value_json"] = json.dumps(leaf_value, ensure_ascii=False)
    fields["tree_leaf_predicted_class_id"] = tree_info.get("tree_leaf_predicted_class_id")

    # 动作字段
    raw_top1_action_id = tree_info.get("tree_raw_top1_action_id")
    fields["tree_raw_top1_action_id"] = raw_top1_action_id
    fields["tree_selected_action_id"] = selected_action_id
    fields["tree_leaf_predicted_action_id"] = tree_info.get("tree_leaf_predicted_class_id")
    fields["tree_raw_top1_invalid"] = tree_info.get("tree_raw_top1_invalid")
    fields["tree_fallback_used"] = tree_info.get("tree_fallback_used")
    fields["tree_fallback_mode"] = tree_info.get("tree_fallback_mode", tree_policy.metadata.get("fallback_mode", "ranked_valid_or_none"))
    fields["tree_fallback_reason"] = tree_info.get("tree_fallback_reason")
    fields["tree_selected_action_kind"] = tree_info.get("tree_selected_action_kind", "budget_action" if selected_action_id is not None else "noop_fallback")
    fields["student_state_vector_int"] = json.dumps(tree_info.get("student_state_vector_int", ()), ensure_ascii=False)
    fields["tree_path_json"] = json.dumps(path_predicates or [], ensure_ascii=False)
    fields["candidate_budget_json"] = json.dumps(tree_info.get("candidate_budgets", {}), ensure_ascii=False, sort_keys=True)
    fields["active_release_budget_max_json"] = json.dumps(tree_info.get("active_release_budget_max", {}), ensure_ascii=False, sort_keys=True)
    fields["effective_check_budget_json"] = json.dumps(tree_info.get("effective_check_budgets", {}), ensure_ascii=False, sort_keys=True)
    fields["safety_reject_reason"] = tree_info.get("safety_reject_reason")
    fields["safety_diagnostics_json"] = json.dumps(tree_info.get("safety_diagnostics", []), ensure_ascii=False, sort_keys=True)
    fields["tree_no_valid_action"] = tree_info.get("tree_no_valid_action")

    # 动作语义描述
    raw_top1_def = tree_policy.action_definition(raw_top1_action_id)
    if raw_top1_def is not None:
        fields["tree_raw_top1_action_def_json"] = json.dumps(raw_top1_def, ensure_ascii=False)
    selected_def = tree_policy.action_definition(selected_action_id)
    if selected_def is not None:
        fields["tree_selected_action_def_json"] = json.dumps(selected_def, ensure_ascii=False)

    # action ranking / probability 字段
    ranking = tree_info.get("tree_action_ranking")
    proba = tree_info.get("tree_action_proba")
    top_k = int(leaf_audit_top_k_actions)
    if ranking is not None and proba is not None:
        ranking_tuple = tuple(ranking) if isinstance(ranking, list) else ranking
        top_k_ids = ranking_tuple[:top_k]
        fields["tree_topk_action_ids_json"] = json.dumps(list(top_k_ids), ensure_ascii=False)
        top_k_probs = [proba[aid] if aid < len(proba) else 0.0 for aid in top_k_ids]
        fields["tree_topk_action_probs_json"] = json.dumps(top_k_probs, ensure_ascii=False)

    valid_count = sum(1 for v in valid_action_mask if v)
    masked_count = len(valid_action_mask) - valid_count
    fields["tree_valid_action_count"] = valid_count
    fields["tree_masked_action_count"] = masked_count

    # teacher 对比字段
    if teacher_diag is not None:
        teacher_best = teacher_diag.get("best_action_id")
        fields["teacher_best_action_id"] = teacher_best
        teacher_best_def = tree_policy.action_definition(teacher_best)
        if teacher_best_def is not None:
            fields["teacher_best_action_def_json"] = json.dumps(teacher_best_def, ensure_ascii=False)
        q_best = teacher_diag.get("q_best")
        fields["teacher_q_best"] = float(q_best) if q_best is not None else None
        raw_q_values = teacher_diag.get("raw_q_values")
        if raw_q_values is not None and selected_action_id is not None and selected_action_id < len(raw_q_values):
            q_selected = float(raw_q_values[selected_action_id])
            fields["teacher_q_selected"] = q_selected
            if q_best is not None:
                fields["teacher_q_regret_selected"] = float(q_best) - q_selected
        if raw_q_values is not None and raw_top1_action_id is not None and raw_top1_action_id < len(raw_q_values):
            q_raw_top1 = float(raw_q_values[raw_top1_action_id])
            fields["teacher_q_raw_top1"] = q_raw_top1
            if q_best is not None:
                fields["teacher_q_regret_raw_top1"] = float(q_best) - q_raw_top1
        fields["teacher_selected_action_match"] = (
            (selected_action_id is not None and selected_action_id == teacher_best) if teacher_best is not None else None
        )
        fields["teacher_raw_action_match"] = (
            (raw_top1_action_id is not None and raw_top1_action_id == teacher_best) if teacher_best is not None else None
        )
        # teacher top-k：只在合法动作中排序
        if raw_q_values is not None:
            q_pairs = [
                (aid, float(raw_q_values[aid]))
                for aid in range(len(raw_q_values))
                if aid < len(valid_action_mask) and valid_action_mask[aid]
            ]
            q_pairs.sort(key=lambda x: (-x[1], x[0]))
            teacher_topk_ids = [aid for aid, _ in q_pairs[:top_k]]
            teacher_topk_qs = [q for _, q in q_pairs[:top_k]]
            fields["teacher_topk_action_ids_json"] = json.dumps(teacher_topk_ids, ensure_ascii=False)
            fields["teacher_topk_q_values_json"] = json.dumps(teacher_topk_qs, ensure_ascii=False)

    # 状态特征字段
    if leaf_audit_state_mode == "split" and isinstance(path_predicates, list):
        # 只写本次 path 上用到的 split feature 值
        split_features: dict[str, float] = {}
        for pred in path_predicates:
            fname = str(pred["feature_name"])
            fvalue = pred.get("value_int", pred.get("value"))
            split_features[fname] = fvalue
        fields["tree_path_feature_values_json"] = json.dumps(split_features, ensure_ascii=False)
    elif leaf_audit_state_mode == "all":
        fields["state_vector_json"] = json.dumps(list(state_vector), ensure_ascii=False)
        if isinstance(path_predicates, list):
            split_features = {}
            for pred in path_predicates:
                fname = str(pred["feature_name"])
                fvalue = pred.get("value_int", pred.get("value"))
                split_features[fname] = fvalue
            fields["tree_path_feature_values_json"] = json.dumps(split_features, ensure_ascii=False)

    return fields


def evaluate_tree_policy_once(
    *,
    tree_policy: TreeBudgetPolicy | IntegerTreeBudgetPolicy,
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
    leaf_audit_enabled: bool = False,
    leaf_audit_state_mode: str = "split",
    leaf_audit_top_k_actions: int = 5,
) -> tuple[dict[str, object], SimulationResult, list[dict[str, object]]]:
    """在真实 runtime 中评估一棵 tree policy。

    新增参数：
    - leaf_audit_enabled: 是否开启 leaf-level audit 日志。
    - leaf_audit_state_mode: "none" / "split" / "all"，控制状态特征记录粒度。
    - leaf_audit_top_k_actions: 记录 top-k 个动作信息。
    """

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
    noop_fallback_count = 0
    no_valid_action_count = 0
    selected_action_count = 0
    selected_action_match_teacher_count = 0
    raw_action_match_teacher_count = 0
    q_regrets: list[float] = []
    while not done:
        step_count += 1
        mask_raw = env.valid_action_mask()
        mask = tuple(bool(v) for v in mask_raw)
        teacher_diag = None
        if teacher is not None:
            if getattr(teacher, "q_network_type", "mlp") == "action_aware":
                action_features = env.get_action_feature_matrix(teacher.action_feature_mode)
                action_feature_names = env.get_action_feature_names(teacher.action_feature_mode)
                teacher.set_action_features(action_features, action_feature_names)
            teacher_diag = teacher.compute_q_diagnostics(obs.state_vector, mask)
        # 在 step 之前将 state_vector 转为 tuple，避免后续引用 obs 时 observation 已被更新
        state_vector = tuple(float(value) for value in obs.state_vector)
        action_id, info = tree_policy.select_action_id(
            state_vector, mask, include_decision_trace=leaf_audit_enabled
        )
        raw_invalid_count += int(bool(info["tree_raw_top1_invalid"]))
        fallback_count += int(bool(info["tree_fallback_used"]))
        noop_fallback_count += int(bool(info["tree_fallback_used"] and action_id is None))
        no_valid_action_count += int(bool(info["tree_no_valid_action"]))
        selected_action_count += int(action_id is not None)
        if teacher_diag is not None:
            raw_action_match_teacher_count += int(info["tree_raw_top1_action_id"] == teacher_diag["best_action_id"])
            selected_action_match_teacher_count += int(action_id == teacher_diag["best_action_id"])
            raw_action = info.get("tree_raw_top1_action_id")
            if isinstance(raw_action, int) and teacher_diag["q_best"] is not None:
                q_regrets.append(float(teacher_diag["q_best"]) - float(teacher_diag["raw_q_values"][raw_action]))
        result = env.step(action_id)
        # 在 env.step() 之后将 leaf audit 字段合并到 action_log
        if leaf_audit_enabled and env.action_log:
            # 当 raw top-1 被拒绝且 selected action 为 None（noop fallback）时，
            # env.action_log[-1] 记录的是 noop 本身的预算（当前预算），
            # 而不是被拒绝 raw top-1 的真实候选、安全包络和拒绝原因。
            # 此时应从 env 的 mask detail 中获取 raw action 的完整评估数据。
            raw_top1_id = info.get("tree_raw_top1_action_id")
            raw_top1_invalid = bool(info.get("tree_raw_top1_invalid", False))
            mask_detail = None
            if raw_top1_invalid and action_id is None and raw_top1_id is not None:
                mask_detail = env.get_last_mask_detail(int(raw_top1_id))
            # 优先使用 raw action 的 mask detail 中的候选/安全数据；缺失时回退到 action_log 末尾。
            evaluation_override: dict[str, object] = {}
            if mask_detail is not None:
                evaluation_override = {
                    "candidate_budgets": mask_detail.get("candidate_budgets", {}),
                    "active_release_budget_max": mask_detail.get("active_release_budget_max", {}),
                    "effective_check_budgets": mask_detail.get("effective_check_budgets", {}),
                    "safety_reject_reason": mask_detail.get("reject_reason_detail") or mask_detail.get("reject_reason"),
                    "safety_diagnostics": [],
                    "fallback_reason": "raw_top1_invalid",
                }
            log_entry = {key: env.action_log[-1].get(key) for key in ("candidate_budgets", "active_release_budget_max", "effective_check_budgets", "safety_reject_reason", "safety_diagnostics")}
            audit_fields = _build_leaf_audit_fields(
                step_index=step_count - 1,
                state_vector=state_vector,
                feature_names=tree_policy.feature_names,
                tree_policy=tree_policy,
                tree_info={**info, **{**log_entry, **evaluation_override}},
                selected_action_id=action_id,
                valid_action_mask=mask,
                teacher_diag=teacher_diag,
                leaf_audit_state_mode=leaf_audit_state_mode,
                leaf_audit_top_k_actions=leaf_audit_top_k_actions,
            )
            env.action_log[-1].update(audit_fields)
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
        "tree_noop_fallback_count": noop_fallback_count,
        "tree_noop_fallback_rate": (noop_fallback_count / step_count) if step_count > 0 else 0.0,
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
        "strict_candidate_deploy_cap": bool(debug_stats.get("strict_candidate_deploy_cap", False)),
        "deploy_cap_candidate_violation_count": int(debug_stats.get("deploy_cap_candidate_violation_count", 0)),
        "carry_over_aware_safety": bool(debug_stats.get("carry_over_aware_safety", False)),
        "lo_budget_overrun_guard_units": int(debug_stats.get("lo_budget_overrun_guard_units", 0)),
        "carry_over_envelope_applied_count": int(debug_stats.get("carry_over_envelope_applied_count", 0)),
        "carry_over_changed_candidate_count": int(debug_stats.get("carry_over_changed_candidate_count", 0)),
        "formal_v1_evaluation_count": int(debug_stats.get("formal_v1_evaluation_count", 0)),
        "formal_v1_cache_hit_count": int(debug_stats.get("formal_v1_cache_hit_count", 0)),
        "formal_v1_cache_miss_count": int(debug_stats.get("formal_v1_cache_miss_count", 0)),
        "formal_v1_mask_step_mismatch_count": int(debug_stats.get("formal_v1_mask_step_mismatch_count", 0)),
    }
    return row, runtime_result, env.action_log
