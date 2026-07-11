"""teacher rollout 数据采集。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import json

from amc_py.dqn import DqnBudgetAgent, ExperimentConfig, build_env_from_experiment_config, resolve_experiment_bundle
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.dataset import ViperSample
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash, quantize_state_vector
from amc_py.viper.schema import resolve_deployment_semantics_version
from amc_py.viper.tree_policy import TreeBudgetPolicy, IntegerTreeBudgetPolicy


def collect_teacher_labeled_rollouts(
    *,
    teacher: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    seeds: Sequence[int],
    end_time: int,
    agent_period: int,
    runtime_semantics: RuntimeSemantics,
    c_amc_sem_xf: float,
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
    teacher_id: str,
    taskset_seed: int | None,
    scenario_split: str,
    behavior_policy: TreeBudgetPolicy | None = None,
    tree_iteration: int | None = None,
    fixed_point_config: FixedPointConfig = FixedPointConfig(),
    tree_fallback_mode: str = "top1_or_noop",
) -> tuple[list[ViperSample], dict]:
    """采集 rollout 上的 teacher 标注样本。

    这里严格遵守计划中的顺序：当前状态先查 mask，再查 teacher Q，再决定 behavior action，
    最后把该行为动作送入 `env.step(...)`。
    """

    samples: list[ViperSample] = []
    if tree_fallback_mode not in {"top1_or_noop", "ranked_valid_or_none"}:
        raise ValueError(f"不支持的 tree_fallback_mode: {tree_fallback_mode}")
    if behavior_policy is not None and isinstance(behavior_policy, IntegerTreeBudgetPolicy):
        if fixed_point_config_hash(behavior_policy.fixed_point_config) != fixed_point_config_hash(fixed_point_config):
            raise ValueError("behavior policy fixed-point config hash 与采集配置不一致")
        if behavior_policy.metadata.get("fallback_mode") != tree_fallback_mode:
            raise ValueError("behavior policy fallback_mode 与新部署语义不一致")
    manifest_mask_reasons: Counter[str] = Counter()
    scenario_seeds: list[int] = []
    feature_names: tuple[str, ...] | None = None
    action_definitions: list[dict[str, object]] | None = None
    for seed in seeds:
        bundle = resolve_experiment_bundle(experiment_config, seed)
        scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else seed
        scenario_seeds.append(int(scenario_seed))
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
        feature_names = env.get_observation_feature_names()
        action_definitions = env.get_action_definitions()
        done = False
        decision_index = 0
        while not done:
            mask = env.valid_action_mask()
            if getattr(teacher, "q_network_type", "mlp") == "action_aware":
                action_features = env.get_action_feature_matrix(teacher.action_feature_mode)
                action_feature_names = env.get_action_feature_names(teacher.action_feature_mode)
                teacher.set_action_features(action_features, action_feature_names)
            q_diag = teacher.compute_q_diagnostics(obs.state_vector, mask)
            teacher_action_id = q_diag["best_action_id"]
            if teacher_action_id is not None and not bool(q_diag["effective_valid_action_mask"][teacher_action_id]):
                raise RuntimeError("teacher action 与当前 valid_action_mask 不一致")
            teacher_action_valid = True if teacher_action_id is None else bool(mask[teacher_action_id])
            if behavior_policy is None:
                behavior_action_id = teacher_action_id
            else:
                behavior_action_id, _ = behavior_policy.select_action_id(obs.state_vector, mask)
            samples.append(
                ViperSample(
                    teacher_id=teacher_id,
                    taskset_seed=taskset_seed,
                    scenario_seed=int(scenario_seed),
                    scenario_split=scenario_split,
                    horizon=int(end_time),
                    decision_index=decision_index,
                    time=int(obs.time),
                    state_vector=tuple(float(value) for value in obs.state_vector),
                    student_state_vector_int=quantize_state_vector(obs.state_vector, fixed_point_config),
                    valid_action_mask=tuple(bool(value) for value in mask),
                    teacher_action_id=(None if teacher_action_id is None else int(teacher_action_id)),
                    teacher_action_valid=bool(teacher_action_valid),
                    raw_q_values=tuple(float(value) for value in q_diag["raw_q_values"]),
                    q_best=(None if q_diag["q_best"] is None else float(q_diag["q_best"])),
                    q_second_best=(
                        None if q_diag["q_second_best"] is None else float(q_diag["q_second_best"])
                    ),
                    q_worst=(None if q_diag["q_worst"] is None else float(q_diag["q_worst"])),
                    q_margin_second=(
                        None if q_diag["q_margin_second"] is None else float(q_diag["q_margin_second"])
                    ),
                    viper_weight=(None if q_diag["viper_weight"] is None else float(q_diag["viper_weight"])),
                    behavior_policy=("oracle" if behavior_policy is None else str(behavior_policy.metadata.get("fallback_mode", tree_fallback_mode))),
                    behavior_action_id=(None if behavior_action_id is None else int(behavior_action_id)),
                    tree_iteration=tree_iteration,
                    raw_budgets_json=json.dumps(dict(obs.raw_budgets), ensure_ascii=False, sort_keys=True),
                    raw_recent_costs_json=json.dumps(dict(obs.raw_recent_costs), ensure_ascii=False, sort_keys=True),
                    mask_reject_reasons_json=json.dumps(env.mask_log[-1].get("reject_reason_counts", {}), ensure_ascii=False, sort_keys=True),
                )
            )
            manifest_mask_reasons.update(env.mask_log[-1].get("reject_reason_counts", {}))
            result = env.step(behavior_action_id)
            obs = result.observation
            done = result.done
            decision_index += 1
    manifest = {
        "dataset_id": f"{teacher_id}_{scenario_split}_{len(samples)}",
        "teacher_id": teacher_id,
        "teacher_model_path": "",
        "teacher_config_path": None,
        "taskset_seed": taskset_seed,
        "scenario_split": scenario_split,
        "scenario_seeds": scenario_seeds,
        "horizon": int(end_time),
        "agent_period": int(agent_period),
        "workload_args": {"experiment_name": experiment_config.name},
        "runtime_args": {
            "runtime_semantics": runtime_semantics.value,
            "c_amc_sem_xf": c_amc_sem_xf,
            "reward_mode": reward_mode,
            "action_space": action_space,
            "budget_increase_ratio": budget_increase_ratio,
            "budget_decrease_ratio": budget_decrease_ratio,
            "include_explicit_noop": include_explicit_noop,
            "budget_floor_ratio": budget_floor_ratio,
            "forbid_decreasing_hi_budgets": forbid_decreasing_hi_budgets,
            "mask_detail_mode": mask_detail_mode,
            "enable_deploy_cap_mask": enable_deploy_cap_mask,
            "deploy_cap_mask_ratio": deploy_cap_mask_ratio,
            "deploy_cap_mask_criticality": deploy_cap_mask_criticality,
        },
        "feature_config": {
            "observation_mode": feature_config.observation_mode,
            "ema_alpha": feature_config.ema_alpha,
            "overrun_ema_alpha": feature_config.overrun_ema_alpha,
            "history_k": feature_config.history_k,
            "event_window": feature_config.event_window,
            "max_cost_weight": feature_config.max_cost_weight,
            "risk_max_scale": feature_config.risk_max_scale,
            "include_safety_margin": feature_config.include_safety_margin,
        },
        "action_space_args": {"action_space": action_space},
        "sample_count": len(samples),
        "valid_labeled_sample_count": sum(int(sample.teacher_action_id is not None) for sample in samples),
        "no_valid_action_count": sum(int(sample.teacher_action_id is None) for sample in samples),
        "feature_names": list(feature_names or ()),
        "action_definitions": action_definitions or [],
        "mask_reject_reasons": dict(manifest_mask_reasons),
        "dataset_schema_version": "viper_fixed_ranked_v2" if tree_fallback_mode == "ranked_valid_or_none" else "viper_fixed_v1",
        "student_observation_encoding": "fixed_point_int",
        "fixed_point_config": {"scale": fixed_point_config.scale, "min_int": fixed_point_config.min_int, "max_int": fixed_point_config.max_int, "rounding_mode": fixed_point_config.rounding_mode, "input_min": fixed_point_config.input_min, "input_max": fixed_point_config.input_max, "schema_version": fixed_point_config.schema_version},
        "fixed_point_config_hash": fixed_point_config_hash(fixed_point_config),
        "teacher_observation_encoding": "float32",
        "tree_fallback_mode": tree_fallback_mode,
        "tree_state_encoding": "fixed_point_int",
        "action_validation_mode": experiment_config.action_validation_mode,
        "strict_candidate_deploy_cap": experiment_config.strict_candidate_deploy_cap,
        "carry_over_aware_safety": experiment_config.carry_over_aware_safety,
        "lo_budget_overrun_guard_units": experiment_config.lo_budget_overrun_guard_units,
        "budget_overrun_semantics": experiment_config.budget_overrun_semantics,
        "deployment_semantics_version": resolve_deployment_semantics_version(
            tree_state_encoding="fixed_point_int",
            tree_fallback_mode=tree_fallback_mode,
            action_validation_mode=experiment_config.action_validation_mode,
            strict_candidate_deploy_cap=experiment_config.strict_candidate_deploy_cap,
            carry_over_aware_safety=experiment_config.carry_over_aware_safety,
            lo_budget_overrun_guard_units=experiment_config.lo_budget_overrun_guard_units,
        ),
        "source_behavior_fallback_mode": "teacher_only" if behavior_policy is None else tree_fallback_mode,
        "behavior_rollout_modes": ["teacher_only" if behavior_policy is None else tree_fallback_mode],
        "dataset_contains_tree_behavior": behavior_policy is not None,
        "upgrade_history": [],
    }
    return samples, manifest
