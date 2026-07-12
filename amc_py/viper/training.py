"""BC / DAGGER / VIPER 树训练器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier

from amc_py.dqn import DqnBudgetAgent, ExperimentConfig
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.artifacts import save_tree_policy_artifact
from amc_py.viper.fixed_point import (
    FixedPointConfig,
    fixed_point_config_hash,
    fixed_point_config_to_dict,
    quantize_state_vector,
)
from amc_py.viper.dataset import ViperSample, read_viper_dataset, samples_to_xyw, write_viper_dataset
from amc_py.viper.metrics import compute_offline_tree_metrics, evaluate_tree_policy_once
from amc_py.viper.selection import SelectionConfig, select_best_tree
from amc_py.viper.teacher import collect_teacher_labeled_rollouts
from amc_py.viper.tree_policy import IntegerTreeBudgetPolicy, TreePolicyProtocol


@dataclass(frozen=True, slots=True)
class TreeHyperParams:
    """单条树训练链共享的超参数。"""

    max_depth: int | None
    min_samples_leaf: int
    criterion: str
    weight_mode: str
    resample_size: int | None
    random_seed: int


def _validate_fixed_point_training_inputs(x: np.ndarray, fixed_point_config: FixedPointConfig) -> None:
    """在 sklearn fit 前显式确认定点训练输入没有发生类型或范围漂移。"""

    if x.dtype != np.int32:
        raise ValueError("fixed_point_int 模式下训练输入必须是 np.int32")
    if not np.isfinite(x.astype(np.float64)).all():
        raise ValueError("fixed_point_int 模式下训练输入必须是有限值")
    if np.any(x < fixed_point_config.output_min) or np.any(x > fixed_point_config.output_max):
        raise ValueError("fixed_point_int 模式下训练输入超出 fixed-point 配置范围")
    if np.any(np.abs(x.astype(np.int64)) > 2**24):
        raise ValueError("fixed_point_int 模式下训练输入绝对值不得超过 2**24")
    round_trip = x.astype(np.float32).astype(np.int32)
    if not np.array_equal(round_trip, x):
        raise ValueError("fixed_point_int 模式下 float32 往返量化必须逐元素完全一致")


def _validate_int_vector(values: Sequence[object], *, field_name: str) -> tuple[int, ...]:
    """严格验证整数向量，拒绝 bool / float / 其他非 int 值。"""

    validated: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} 必须全部是 int")
        validated.append(int(value))
    return tuple(validated)


def _upgrade_legacy_fixed_point_dataset(
    samples: Sequence[ViperSample],
    manifest: dict[str, object],
    fixed_point_config: FixedPointConfig,
) -> tuple[list[ViperSample], dict[str, object]]:
    """把旧 teacher-only dataset 显式升级成完整 fixed-point schema。"""

    upgraded_samples: list[ViperSample] = []
    for sample in samples:
        upgraded_samples.append(
            replace(
                sample,
                student_state_vector_int=_validate_int_vector(
                    quantize_state_vector(sample.state_vector, fixed_point_config),
                    field_name="student_state_vector_int",
                ),
            )
        )
    upgraded_manifest = dict(manifest)
    upgraded_manifest["dataset_schema_version"] = "viper_dataset_fixed_int_v1"
    upgraded_manifest["teacher_state_encoding"] = "float32"
    upgraded_manifest["student_state_encoding"] = "fixed_point_int"
    upgraded_manifest["fixed_point_config"] = fixed_point_config_to_dict(fixed_point_config)
    upgraded_manifest["fixed_point_config_hash"] = fixed_point_config_hash(fixed_point_config)
    return upgraded_samples, upgraded_manifest


def _extract_integer_verification_states(
    samples: Sequence[ViperSample],
) -> list[tuple[int, ...]]:
    """收集可用于整数等价验证的 student_state_vector_int。"""

    states: list[tuple[int, ...]] = []
    for sample in samples:
        if sample.teacher_action_id is None:
            continue
        if sample.student_state_vector_int is None:
            raise ValueError("fixed-point verification 需要完整的 student_state_vector_int")
        states.append(_validate_int_vector(sample.student_state_vector_int, field_name="student_state_vector_int"))
    return states


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
    student_state_encoding: str = "legacy_float32",
    fixed_point_config: FixedPointConfig | None = None,
    allow_legacy_quantization: bool = False,
) -> tuple[DecisionTreeClassifier, dict[str, object]]:
    """训练一棵 CART 决策树。

    默认实现严格遵守计划：VIPER 主路径使用 weighted resampling，而不是直接依赖 sample_weight。
    """

    x, y, w = samples_to_xyw(
        samples,
        weight_mode=weight_mode,
        student_encoding=student_state_encoding, fixed_point_config=fixed_point_config,
        allow_legacy_quantization=allow_legacy_quantization,
    )
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
    if student_state_encoding == "fixed_point_int":
        if fixed_point_config is None:
            raise ValueError("fixed_point_int 模式必须提供 fixed_point_config")
        _validate_fixed_point_training_inputs(x_fit, fixed_point_config)
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
        "fallback_mode": "ranked_valid_or_none",
        "tree_node_count": int(classifier.tree_.node_count),
        "tree_leaf_count": int(classifier.get_n_leaves()),
        "tree_depth": int(classifier.get_depth()),
        "student_state_encoding": student_state_encoding,
        "fixed_point_config": (None if fixed_point_config is None else fixed_point_config_to_dict(fixed_point_config)),
        "fixed_point_config_hash": (None if fixed_point_config is None else fixed_point_config_hash(fixed_point_config)),
        "training_feature_dtype": str(x.dtype),
        "runtime_policy_type": "integer_tree_ranked_valid_or_none" if student_state_encoding == "fixed_point_int" else "legacy_sklearn_ranked_valid_or_none",
        "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none" if student_state_encoding == "fixed_point_int" else "legacy_sklearn_ranked_valid_or_none",
        "tree_state_encoding": student_state_encoding,
        "tree_fixed_point_scale": (None if fixed_point_config is None else int(fixed_point_config.scale)),
        "tree_fixed_point_config_hash": (None if fixed_point_config is None else fixed_point_config_hash(fixed_point_config)),
        "tree_artifact_schema_version": "viper_integer_ranked_artifact_v1" if student_state_encoding == "fixed_point_int" else "legacy_sklearn_ranked_valid_or_none",
        "integer_equivalence_verified": False,
        "integer_equivalence_state_count": 0,
        "integer_equivalence_verification_state_hash": None,
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
    for filename in (
        "model.joblib",
        "metadata.json",
        "feature_names.json",
        "action_definitions.json",
        "rules.txt",
        "integer_tree.json",
        "fixed_point_config.json",
        "artifact_manifest.json",
        "leaf_rules_int.json",
        "leaf_rules_int.csv",
        "leaf_rules.json",
        "leaf_rules.csv",
    ):
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, best_dir / filename)
    with (best_dir / "selection_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_candidate, handle, ensure_ascii=False, indent=2)
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
    student_state_encoding: str = "legacy_float32",
    fixed_point_config: FixedPointConfig | None = None,
    allow_legacy_quantization: bool = False,
) -> dict[str, object]:
    """运行一条固定超参数的 BC/DAGGER/VIPER 训练链。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    if student_state_encoding == "fixed_point_int" and fixed_point_config is None:
        fixed_point_config = FixedPointConfig()
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
            student_state_encoding=student_state_encoding,
            fixed_point_config=fixed_point_config,
        )
    else:
        aggregate_samples, aggregate_manifest = read_viper_dataset(initial_dataset)
        manifest_encoding = str(aggregate_manifest.get("student_state_encoding") or "legacy_float32")
        if student_state_encoding == "fixed_point_int":
            if fixed_point_config is None:
                raise ValueError("fixed_point_int 训练必须提供 fixed_point_config")
            manifest_hash = aggregate_manifest.get("fixed_point_config_hash")
            if manifest_encoding in {"legacy_float32", ""}:
                if not allow_legacy_quantization:
                    raise ValueError("旧 dataset 需要显式开启 allow_legacy_dataset_quantization 才能升级")
                aggregate_samples, aggregate_manifest = _upgrade_legacy_fixed_point_dataset(
                    aggregate_samples,
                    aggregate_manifest,
                    fixed_point_config,
                )
            elif manifest_encoding == "fixed_point_int":
                if manifest_hash != fixed_point_config_hash(fixed_point_config):
                    raise ValueError("initial dataset 的 fixed-point config hash 与训练配置不匹配")
            else:
                raise ValueError("initial dataset 的 student_state_encoding 与训练配置不匹配")
        elif manifest_encoding != student_state_encoding:
            raise ValueError("initial dataset 的 student_state_encoding 与训练配置不匹配")
    candidates: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None
    current_policy: TreePolicyProtocol | None = None
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
            student_state_encoding=student_state_encoding,
            fixed_point_config=fixed_point_config,
            allow_legacy_quantization=allow_legacy_quantization,
        )
        metadata = {
            **metadata,
            "tree_id": f"{method}_iter_{iteration:03d}",
            "teacher_id": teacher_id,
            "teacher_model_path": "",
            "taskset_seed": aggregate_manifest.get("taskset_seed"),
            "observation_mode": feature_config.observation_mode,
            "iteration": iteration,
            "student_state_encoding": student_state_encoding,
            "fixed_point_config": (None if fixed_point_config is None else fixed_point_config_to_dict(fixed_point_config)),
            "fixed_point_config_hash": (None if fixed_point_config is None else fixed_point_config_hash(fixed_point_config)),
            "tree_artifact_schema_version": "viper_integer_ranked_artifact_v1" if student_state_encoding == "fixed_point_int" else "legacy_sklearn_ranked_valid_or_none",
            "tree_runtime_policy_type": "integer_tree_ranked_valid_or_none" if student_state_encoding == "fixed_point_int" else "legacy_sklearn_ranked_valid_or_none",
            "tree_state_encoding": student_state_encoding,
            "tree_fixed_point_scale": (None if fixed_point_config is None else int(fixed_point_config.scale)),
            "tree_fixed_point_config_hash": (None if fixed_point_config is None else fixed_point_config_hash(fixed_point_config)),
        }
        if int(metadata["state_dim"]) != len(feature_names):
            raise ValueError("tree metadata.state_dim 与 feature_names 长度不一致")
        if int(metadata["action_dim"]) != len(action_definitions):
            raise ValueError("tree metadata.action_dim 与 action_definitions 长度不一致")
        verification_states: list[tuple[int, ...]] = []
        if student_state_encoding == "fixed_point_int":
            verification_states.extend(_extract_integer_verification_states(aggregate_samples))
            if validation_seeds:
                verification_samples, _ = collect_teacher_labeled_rollouts(
                    teacher=teacher,
                    experiment_config=experiment_config,
                    seeds=validation_seeds,
                    end_time=validation_end_time,
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
                    scenario_split="validation_verification",
                    student_state_encoding=student_state_encoding,
                    fixed_point_config=fixed_point_config,
                )
                verification_states.extend(_extract_integer_verification_states(verification_samples))
            if not verification_states:
                raise ValueError("fixed-point artifact 需要至少一个 verification state")
            verification_state_hash = hashlib.sha256(
                json.dumps([list(state) for state in verification_states], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            metadata = {
                **metadata,
                # 先在内存里保留未验证状态，等整数模型编译、验证并完成落盘后，
                # 再由保存函数在输出 artifact 中声明 verified=true。
                "integer_equivalence_verified": False,
                "integer_equivalence_state_count": len(verification_states),
                "integer_equivalence_verification_state_hash": verification_state_hash,
            }
        else:
            metadata = {
                **metadata,
                "integer_equivalence_verified": False,
                "integer_equivalence_state_count": 0,
                "integer_equivalence_verification_state_hash": None,
            }
        artifact_dir = output_dir / f"iter_{iteration:03d}"
        if student_state_encoding == "fixed_point_int":
            from amc_py.viper.integer_tree import compile_sklearn_tree_to_integer

            integer_model = compile_sklearn_tree_to_integer(
                classifier,
                state_dim=len(feature_names),
                action_dim=len(action_definitions),
                feature_names=feature_names,
                fixed_point_config=fixed_point_config,
                verification_states=tuple(verification_states),  # type: ignore[arg-type]
            )
            current_policy = IntegerTreeBudgetPolicy(
                model=integer_model,
                metadata=metadata,
                feature_names=feature_names,
                action_definitions=action_definitions,
                fixed_point_config=fixed_point_config,  # type: ignore[arg-type]
            )
        else:
            from amc_py.viper.tree_policy import TreeBudgetPolicy

            current_policy = TreeBudgetPolicy(
                classifier=classifier,
                metadata=metadata,
                feature_names=feature_names,
                action_definitions=action_definitions,
            )
        artifact_metadata = metadata
        if student_state_encoding == "fixed_point_int":
            artifact_metadata = {
                **metadata,
                "integer_equivalence_verified": True,
            }
        save_tree_policy_artifact(
            artifact_dir,
            classifier=classifier,
            metadata=artifact_metadata,
            feature_names=feature_names,
            action_definitions=action_definitions,
            verification_states=(
                tuple(
                    _validate_int_vector(state, field_name="verification_state")
                    for state in verification_states
                )
                if student_state_encoding == "fixed_point_int"
                else None
            ),
        )
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
                student_state_encoding=student_state_encoding,
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
