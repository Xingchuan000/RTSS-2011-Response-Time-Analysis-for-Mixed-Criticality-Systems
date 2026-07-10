"""BC / DAGGER / VIPER 树训练器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import shutil

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier

from amc_py.dqn import DqnBudgetAgent, ExperimentConfig
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.artifacts import save_tree_policy_artifact, required_artifact_files, validate_artifact_directory, load_tree_policy_artifact
from amc_py.viper.dataset import ViperSample, read_viper_dataset, samples_to_xyw, write_viper_dataset
from amc_py.viper.metrics import compute_offline_tree_metrics, evaluate_tree_policy_once
from amc_py.viper.selection import SelectionConfig, select_best_tree
from amc_py.viper.teacher import collect_teacher_labeled_rollouts
from amc_py.viper.tree_policy import TreeBudgetPolicy
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash, fixed_point_config_to_dict
from amc_py.viper.schema import resolve_deployment_semantics_version
from amc_py.viper.dataset import upgrade_samples_to_fixed_point
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy


@dataclass(frozen=True, slots=True)
class TreeHyperParams:
    """单条树训练链共享的超参数。"""

    max_depth: int | None
    min_samples_leaf: int
    criterion: str
    weight_mode: str
    resample_size: int | None
    random_seed: int


def train_cart_tree(
    samples: Sequence[ViperSample],
    *,
    method: str,
    max_depth: int | None,
    min_samples_leaf: int,
    criterion: str,
    weight_mode: str,
    resample_size: int | None,
    random_seed: int,
    state_dim: int | None = None,
    action_dim: int | None = None,
    fixed_point_config: FixedPointConfig | None = None,
    allow_legacy_quantization: bool = False,
) -> tuple[DecisionTreeClassifier, dict[str, object]]:
    """训练一棵 CART 决策树。

    默认实现严格遵守计划：VIPER 主路径使用 weighted resampling，而不是直接依赖 sample_weight。
    """

    legacy_training = fixed_point_config is None
    fixed_point_config = fixed_point_config or FixedPointConfig()
    x, y, w = samples_to_xyw(samples, weight_mode=weight_mode, fixed_point_config=fixed_point_config, allow_legacy_quantization=(allow_legacy_quantization or legacy_training))
    if x.dtype != np.int32 or np.any(x > 2**24) or np.any(x < -(2**24)):
        raise ValueError("训练特征必须是 int32 且位于精确整数范围")
    if resample_size is None:
        resample_size = int(len(y))
    rng = np.random.default_rng(random_seed)
    if method in {"viper", "dagger"}:
        if method == "dagger":
            probabilities = np.full(len(w), 1.0 / len(w), dtype=np.float64)
        else:
            probabilities = w / w.sum()
        sampled_indices = rng.choice(len(y), size=int(resample_size), replace=True, p=probabilities)
        x_fit = x[sampled_indices]
        y_fit = y[sampled_indices]
    elif method == "bc":
        x_fit = x
        y_fit = y
    else:
        raise ValueError(f"不支持的 method: {method}")
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_seed,
    )
    classifier.fit(x_fit, y_fit)
    metadata = {
        "method": method,
        "state_dim": int(x.shape[1] if state_dim is None else state_dim),
        "action_dim": int((max(int(value) for value in y) + 1) if action_dim is None else action_dim),
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
        "weight_mode": weight_mode,
        "resampling_mode": ("weighted_resample" if method == "viper" else "uniform"),
        "train_sample_count": int(len(y_fit)),
        "valid_labeled_sample_count": int(len(y)),
        "sklearn_version": sklearn_version,
        "mask_aware_inference": True,
        "fallback_mode": "top1_or_noop",
        "training_feature_dtype": "int32",
        "fixed_point_config_hash": fixed_point_config_hash(fixed_point_config),
        "deployment_uses_sklearn": False,
        "tree_node_count": int(classifier.tree_.node_count),
        "tree_leaf_count": int(classifier.get_n_leaves()),
        "tree_depth": int(classifier.get_depth()),
    }
    return classifier, metadata


def _mean_metric(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _sum_metric(rows: Sequence[dict[str, object]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows))


def _copy_best_artifact(best_candidate: dict[str, object], output_dir: Path) -> Path:
    """复制最优树到稳定的 `best/` 目录，方便后续脚本直接引用。"""

    source_dir = Path(str(best_candidate["artifact_dir"]))
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("model.joblib", "metadata.json", "fixed_point_config.json", "integer_tree.json", "feature_names.json", "action_definitions.json", "rules.txt", "leaf_rules_int.json", "leaf_rules_int.csv", "artifact_manifest.json", "leaf_rules.json", "leaf_rules.csv", "selection_metrics.json"):
        stale_path = best_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    require_integer = metadata.get("artifact_schema_version") == "viper_integer_artifact_v1"
    validate_artifact_directory(source_dir, require_integer_tree=require_integer)
    for filename in required_artifact_files(metadata, require_integer_tree=require_integer):
        shutil.copy2(source_dir / filename, best_dir / filename)
    with (best_dir / "selection_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_candidate, handle, ensure_ascii=False, indent=2)
    validate_artifact_directory(best_dir, require_integer_tree=require_integer)
    load_tree_policy_artifact(best_dir, require_integer_tree=require_integer, allow_legacy_fallback=not require_integer)
    return best_dir


def run_viper_iterations(
    *,
    teacher: DqnBudgetAgent,
    initial_dataset: Path | None,
    experiment_config: ExperimentConfig,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    iterations: int,
    trajectories_per_iter: int | None,
    end_time: int,
    validation_end_time: int,
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
    tree_hyperparams: TreeHyperParams,
    output_dir: Path,
    method: str,
    workload_cli_config: dict[str, object] | None = None,
    workload_mismatch_warning: str | None = None,
    fixed_point_config: FixedPointConfig = FixedPointConfig(),
    allow_legacy_dataset_quantization: bool = False,
) -> dict[str, object]:
    """运行一条固定超参数的 BC/DAGGER/VIPER 训练链。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    if initial_dataset is None:
        aggregate_samples, aggregate_manifest = collect_teacher_labeled_rollouts(
            teacher=teacher,
            experiment_config=experiment_config,
            seeds=train_seeds,
            end_time=end_time,
            agent_period=agent_period,
            runtime_semantics=runtime_semantics,
            c_amc_sem_xf=c_amc_sem_xf,
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
            teacher_id=teacher_id,
            taskset_seed=None,
            scenario_split="train",
            fixed_point_config=fixed_point_config,
        )
    else:
        aggregate_samples, aggregate_manifest = read_viper_dataset(initial_dataset)
        aggregate_samples = upgrade_samples_to_fixed_point(aggregate_samples, fixed_point_config)
        aggregate_manifest = {
            **aggregate_manifest,
            "dataset_schema_version": "viper_fixed_v1",
            "student_observation_encoding": "fixed_point_int",
            "fixed_point_config": fixed_point_config_to_dict(fixed_point_config),
            "fixed_point_config_hash": fixed_point_config_hash(fixed_point_config),
            "teacher_observation_encoding": "float32",
            "tree_fallback_mode": "top1_or_noop",
        }
    candidates: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None
    current_policy: TreeBudgetPolicy | None = None
    train_rollout_seeds = list(train_seeds[: trajectories_per_iter or len(train_seeds)])
    for iteration in range(1, iterations + 1):
        feature_names = tuple(aggregate_manifest["feature_names"])
        action_definitions = list(aggregate_manifest["action_definitions"])
        classifier, metadata = train_cart_tree(
            aggregate_samples,
            method=("bc" if method == "bc" else method),
            max_depth=tree_hyperparams.max_depth,
            min_samples_leaf=tree_hyperparams.min_samples_leaf,
            criterion=tree_hyperparams.criterion,
            weight_mode=tree_hyperparams.weight_mode,
            resample_size=tree_hyperparams.resample_size,
            random_seed=tree_hyperparams.random_seed,
            state_dim=len(feature_names),
            action_dim=len(action_definitions),
            fixed_point_config=fixed_point_config,
            allow_legacy_quantization=allow_legacy_dataset_quantization,
        )
        metadata = {
            **metadata,
            "tree_id": f"{method}_iter_{iteration:03d}",
            "teacher_id": teacher_id,
            "teacher_model_path": "",
            "taskset_seed": aggregate_manifest.get("taskset_seed"),
            "observation_mode": feature_config.observation_mode,
            "iteration": iteration,
            "action_validation_mode": aggregate_manifest.get("action_validation_mode", experiment_config.action_validation_mode),
            "strict_candidate_deploy_cap": aggregate_manifest.get("strict_candidate_deploy_cap", experiment_config.strict_candidate_deploy_cap),
            "carry_over_aware_safety": aggregate_manifest.get("carry_over_aware_safety", experiment_config.carry_over_aware_safety),
            "lo_budget_overrun_guard_units": aggregate_manifest.get("lo_budget_overrun_guard_units", aggregate_manifest.get("budget_overrun_guard_units", experiment_config.lo_budget_overrun_guard_units)),
            "budget_overrun_semantics": aggregate_manifest.get("budget_overrun_semantics", experiment_config.budget_overrun_semantics),
            "tree_state_encoding": aggregate_manifest.get("student_observation_encoding", "fixed_point_int"),
            "deployment_semantics_version": aggregate_manifest.get("deployment_semantics_version", resolve_deployment_semantics_version(
                tree_state_encoding=aggregate_manifest.get("student_observation_encoding", "fixed_point_int"),
                tree_fallback_mode=aggregate_manifest.get("tree_fallback_mode", "top1_or_noop"),
                action_validation_mode=aggregate_manifest.get("action_validation_mode", experiment_config.action_validation_mode),
                strict_candidate_deploy_cap=aggregate_manifest.get("strict_candidate_deploy_cap", experiment_config.strict_candidate_deploy_cap),
                carry_over_aware_safety=aggregate_manifest.get("carry_over_aware_safety", experiment_config.carry_over_aware_safety),
                lo_budget_overrun_guard_units=aggregate_manifest.get("lo_budget_overrun_guard_units", aggregate_manifest.get("budget_overrun_guard_units", experiment_config.lo_budget_overrun_guard_units)),
            )),
        }
        if int(metadata["state_dim"]) != len(feature_names):
            raise ValueError("tree metadata.state_dim 与 feature_names 长度不一致")
        if int(metadata["action_dim"]) != len(action_definitions):
            raise ValueError("tree metadata.action_dim 与 action_definitions 长度不一致")
        artifact_dir = output_dir / f"iter_{iteration:03d}"
        save_tree_policy_artifact(
            artifact_dir,
            classifier=classifier,
            metadata=metadata,
            feature_names=feature_names,
            action_definitions=action_definitions,
            fixed_point_config=fixed_point_config,
        )
        from amc_py.viper.artifacts import load_tree_policy_artifact
        current_policy = load_tree_policy_artifact(artifact_dir, require_integer_tree=True, fixed_point_config=fixed_point_config)
        offline_metrics = compute_offline_tree_metrics(current_policy, aggregate_samples)
        validation_rows = [
            evaluate_tree_policy_once(
                tree_policy=current_policy,
                experiment_config=experiment_config,
                seed=seed,
                end_time=validation_end_time,
                agent_period=agent_period,
                runtime_semantics=runtime_semantics,
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
            )[0]
            for seed in validation_seeds
        ]
        validation_qos = _mean_metric(validation_rows, "lo_quality_qos")
        candidate_row = {
            "method": method,
            "max_depth": tree_hyperparams.max_depth,
            "min_samples_leaf": tree_hyperparams.min_samples_leaf,
            "iteration": iteration,
            "offline_accuracy": offline_metrics["offline_accuracy"],
            "weighted_fidelity": offline_metrics["weighted_fidelity"],
            "q_regret_mean": offline_metrics["q_regret_mean"],
            # 兼容旧产物的保留字段；正式选择逻辑与后续脚本必须使用无空格版本。
            "validation lo_quality_qos": validation_qos,
            "validation_lo_quality_qos_mean": validation_qos,
            "validation_lo_zero_service_ratio_mean": _mean_metric(validation_rows, "lo_zero_service_ratio"),
            "validation_lo_equiv_jne_mean": _mean_metric(validation_rows, "lo_equiv_jne"),
            "validation_tid_ratio_mean": _mean_metric(validation_rows, "tid_ratio"),
            "validation_deadline_misses_sum": _sum_metric(validation_rows, "deadline_misses"),
            "validation_hi_deadline_misses_sum": _sum_metric(validation_rows, "hi_deadline_misses"),
            "validation_lo_deadline_misses_sum": _sum_metric(validation_rows, "lo_deadline_misses"),
            "validation_tree_raw_top1_invalid_rate_mean": _mean_metric(
                validation_rows,
                "tree_raw_top1_invalid_rate",
            ),
            "validation_tree_fallback_rate_mean": _mean_metric(
                validation_rows,
                "tree_fallback_rate",
            ),
            "validation_tree_no_valid_action_rate_mean": _mean_metric(
                validation_rows,
                "tree_no_valid_action_rate",
            ),
            "validation_tree_q_regret_mean": _mean_metric(
                [row for row in validation_rows if row.get("tree_q_regret_mean") is not None],
                "tree_q_regret_mean",
            )
            if any(row.get("tree_q_regret_mean") is not None for row in validation_rows)
            else None,
            "validation_tree_q_regret_p95": _mean_metric(
                [row for row in validation_rows if row.get("tree_q_regret_p95") is not None],
                "tree_q_regret_p95",
            )
            if any(row.get("tree_q_regret_p95") is not None for row in validation_rows)
            else None,
            "deadline_misses": _sum_metric(validation_rows, "deadline_misses"),
            "hi_deadline_misses": _sum_metric(validation_rows, "hi_deadline_misses"),
            "lo_deadline_misses": _sum_metric(validation_rows, "lo_deadline_misses"),
            "tree_raw_top1_invalid_rate": _mean_metric(validation_rows, "tree_raw_top1_invalid_rate"),
            "tree_fallback_rate": _mean_metric(validation_rows, "tree_fallback_rate"),
            "tree_no_valid_action_rate": _mean_metric(validation_rows, "tree_no_valid_action_rate"),
            "lo_quality_qos": validation_qos,
            "parent_lo_quality_qos": float("-inf"),
            "qos_retention": None,
            "tree_depth": int(metadata["tree_depth"]),
            "tree_node_count": int(metadata["tree_node_count"]),
            "tree_leaf_count": int(metadata["tree_leaf_count"]),
            "artifact_dir": str(artifact_dir),
        }
        candidates.append(candidate_row)
        if method in {"dagger", "viper"} and iteration < iterations:
            new_samples, _ = collect_teacher_labeled_rollouts(
                teacher=teacher,
                experiment_config=experiment_config,
                seeds=train_rollout_seeds,
                end_time=end_time,
                agent_period=agent_period,
                runtime_semantics=runtime_semantics,
                c_amc_sem_xf=c_amc_sem_xf,
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
                teacher_id=teacher_id,
                taskset_seed=None,
                scenario_split="train",
                behavior_policy=current_policy,
                tree_iteration=iteration,
                fixed_point_config=fixed_point_config,
            )
            aggregate_samples.extend(new_samples)
    if not candidates:
        raise RuntimeError("训练结束后没有产生任何候选树")
    # 先把所有候选树指标落盘，再执行 selection。
    # 这样即使 select_best_tree() 因 gate 失败而抛错，外部仍然可以直接查看
    # candidates.csv，定位到底是 deadline miss / invalid rate / fallback rate / qos
    # 中的哪一项导致选择失败。
    with (output_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    selection_config = SelectionConfig()
    best_candidate = select_best_tree(candidates, config=selection_config)
    best_qos = max(float(row["validation_lo_quality_qos_mean"]) for row in candidates)
    selection_reason = (
        "safety_validity_gate_then_complexity_pareto_within_98pct_qos"
        if float(best_candidate["validation_lo_quality_qos_mean"]) >= best_qos * selection_config.complexity_qos_ratio
        else "best_validation_qos_after_gates"
    )
    best_dir = _copy_best_artifact({**best_candidate, "selection_reason": selection_reason}, output_dir)
    with (output_dir / "best_tree_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "artifact_dir",
                "best_artifact_dir",
                "selection_reason",
                "method",
                "max_depth",
                "min_samples_leaf",
                "iteration",
                "validation_lo_quality_qos_mean",
                "validation_tree_raw_top1_invalid_rate_mean",
                "validation_tree_fallback_rate_mean",
                "tree_depth",
                "tree_node_count",
                "tree_leaf_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "artifact_dir": best_candidate["artifact_dir"],
                "best_artifact_dir": str(best_dir),
                "selection_reason": selection_reason,
                "method": best_candidate["method"],
                "max_depth": best_candidate["max_depth"],
                "min_samples_leaf": best_candidate["min_samples_leaf"],
                "iteration": best_candidate["iteration"],
                "validation_lo_quality_qos_mean": best_candidate["validation_lo_quality_qos_mean"],
                "validation_tree_raw_top1_invalid_rate_mean": (
                    best_candidate["validation_tree_raw_top1_invalid_rate_mean"]
                ),
                "validation_tree_fallback_rate_mean": best_candidate["validation_tree_fallback_rate_mean"],
                "tree_depth": best_candidate["tree_depth"],
                "tree_node_count": best_candidate["tree_node_count"],
                "tree_leaf_count": best_candidate["tree_leaf_count"],
            }
        )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": method,
                "iterations": iterations,
                "teacher_id": teacher_id,
                "tree_hyperparams": asdict(tree_hyperparams),
                # 训练产物必须显式记录本次 tree 训练所使用的 workload 参数，
                # 这样后续检查时才能确认它与 teacher dataset / HOUT 评估完全同口径。
                "workload_cli_config": workload_cli_config or {},
                # 仅在用户显式允许 mismatch 时才记录 warning；默认情况下前面已经直接报错阻止继续运行。
                "workload_mismatch_warning": workload_mismatch_warning,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_viper_dataset(output_dir / "aggregate_dataset", aggregate_samples, aggregate_manifest)
    return {
        "best_candidate": {**best_candidate, "selection_reason": selection_reason, "best_artifact_dir": str(best_dir)},
        "candidate_count": len(candidates),
        "aggregate_sample_count": len(aggregate_samples),
    }
