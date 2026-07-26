"""teacher rollout 数据采集。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path

from amc_py.dqn import DqnBudgetAgent, ExperimentConfig, build_env_from_experiment_config, resolve_experiment_bundle
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.qamc.reference_config import load_and_validate_frozen_reference
from amc_py.qamc.profile_spec import load_profile_spec
from amc_py.viper.dataset import ViperSample
from amc_py.viper.fixed_point import (
    FixedPointConfig,
    fixed_point_config_hash,
    fixed_point_config_to_dict,
    quantize_state_vector,
)
from amc_py.viper.tree_policy import TreePolicyProtocol


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
    behavior_policy: TreePolicyProtocol | None = None,
    tree_iteration: int | None = None,
    student_state_encoding: str = "legacy_float32",
    fixed_point_config: FixedPointConfig | None = None,
) -> tuple[list[ViperSample], dict]:
    """采集 rollout 上的 teacher 标注样本。

    这里严格遵守计划中的顺序：当前状态先查 mask，再查 teacher Q，再决定 behavior action，
    最后把该行为动作送入 `env.step(...)`。
    """

    if student_state_encoding not in {"legacy_float32", "fixed_point_int"}:
        raise ValueError(f"不支持的 student_state_encoding: {student_state_encoding}")
    if student_state_encoding == "fixed_point_int" and fixed_point_config is None:
        raise ValueError("fixed_point_int 模式必须提供 fixed_point_config")

    samples: list[ViperSample] = []
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
                    behavior_policy=("oracle" if behavior_policy is None else str(behavior_policy.metadata.get("method"))),
                    behavior_action_id=(None if behavior_action_id is None else int(behavior_action_id)),
                    tree_iteration=tree_iteration,
                    raw_budgets_json=json.dumps(dict(obs.raw_budgets), ensure_ascii=False, sort_keys=True),
                    raw_recent_costs_json=json.dumps(dict(obs.raw_recent_costs), ensure_ascii=False, sort_keys=True),
                    mask_reject_reasons_json=json.dumps(env.mask_log[-1].get("reject_reason_counts", {}), ensure_ascii=False, sort_keys=True),
                    student_state_vector_int=(
                        None
                        if student_state_encoding == "legacy_float32"
                        else quantize_state_vector(obs.state_vector, fixed_point_config)  # type: ignore[arg-type]
                    ),
                )
            )
            manifest_mask_reasons.update(env.mask_log[-1].get("reject_reason_counts", {}))
            result = env.step(behavior_action_id)
            obs = result.observation
            done = result.done
            decision_index += 1
    manifest = {
        "dataset_schema_version": "viper_dataset_fixed_int_v1",
        "teacher_state_encoding": "float32",
        "student_state_encoding": student_state_encoding,
        "fixed_point_config": (
            None if fixed_point_config is None else fixed_point_config_to_dict(fixed_point_config)
        ),
        "fixed_point_config_hash": (
            None
            if fixed_point_config is None
            else fixed_point_config_hash(fixed_point_config)
        ),
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
    }
    if runtime_semantics is RuntimeSemantics.Q_AMC:
        if (
            experiment_config.qamc_reference_config_path is None
            or experiment_config.qamc_profile_manifest_path is None
            or experiment_config.qamc_profile_spec_path is None
        ):
            raise ValueError("QAMC_REFERENCE_PROFILE_ARTIFACTS_REQUIRED")
        frozen = load_and_validate_frozen_reference(
            experiment_config.qamc_reference_config_path
        )
        profile_manifest = json.loads(
            Path(experiment_config.qamc_profile_manifest_path).read_text(
                encoding="utf-8"
            )
        )
        spec = load_profile_spec(experiment_config.qamc_profile_spec_path)
        manifest["qamc"] = {
            "reference_config_fingerprint": frozen["fingerprint"],
            "profile_manifest_fingerprint": profile_manifest.get("fingerprint"),
            "profile_spec_fingerprint": spec.fingerprint,
            "quality_visible_to_agent": False,
            "formal_safety_claim": False,
        }
    return samples, manifest
