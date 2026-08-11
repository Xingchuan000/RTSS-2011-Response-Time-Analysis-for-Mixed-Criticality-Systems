"""训练 BC / DAGGER / VIPER tree 的 CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.dqn import DqnBudgetAgent
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.training import TreeHyperParams, run_viper_iterations
from amc_py.viper.fixed_point import FixedPointConfig

from scripts.collect_viper_teacher_data import (
    _build_experiment_config,
    _configure_qamc_experiment,
    _parse_seeds,
)
from scripts.common_mc_fairgen_cli import (
    add_mc_fairgen_args,
    assert_mc_fairgen_args_match_dataset,
    build_workload_cli_config,
)
from scripts.common_mc_stratified_dynamic_cli import (
    add_mc_stratified_dynamic_args,
    assert_mc_stratified_dynamic_args_match_dataset,
    build_mc_stratified_dynamic_workload_cli_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bc", "dagger", "viper"], required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--teacher-id", type=str, required=True)
    parser.add_argument("--initial-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", type=str, default="0")
    parser.add_argument("--validation-seeds", type=str, default="1")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--validation-end-time", type=int, default=None)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--max-depth-grid", type=str, default="2")
    parser.add_argument("--min-samples-leaf-grid", type=str, default="1")
    parser.add_argument("--criterion", choices=["gini", "entropy", "log_loss"], default="gini")
    parser.add_argument("--weight-mode", choices=["uniform", "viper_q_span", "q_margin_second"], default="viper_q_span")
    parser.add_argument("--resample-size", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--workload",
        choices=["small", "rtss11", "automotive", "mc_fairgen", "mc_stratified_dynamic"],
        default="small",
    )
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--dqn-runtime-semantics", choices=["AMC_PLUS", "AMC_RA", "AMC_RH", "C_AMC_SEM", "Q_AMC"], default="AMC_PLUS")
    parser.add_argument("--c-amc-sem-xf", type=float, default=0.5)
    parser.add_argument("--qamc-reference-config-path", type=Path, default=None)
    parser.add_argument("--qamc-profile-manifest-path", type=Path, default=None)
    parser.add_argument("--qamc-profile-spec-path", type=Path, default=None)
    parser.add_argument("--reward-mode", type=str, default="mendes")
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
        default="single",
    )
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--budget-floor-ratio", type=float, default=0.0)
    parser.add_argument("--forbid-decreasing-hi-budgets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--enable-deploy-cap-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deploy-cap-mask-ratio", type=float, default=4.0)
    parser.add_argument("--deploy-cap-mask-criticality", choices=["lo", "all"], default="lo")
    parser.add_argument("--observation-mode", type=str, default="v11_full_10d")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--fixed-taskset-seed", type=int, default=None)
    parser.add_argument("--automotive-num-runnables", type=int, default=150)
    parser.add_argument("--automotive-mode", type=str, default="paper_like")
    add_mc_fairgen_args(parser)
    add_mc_stratified_dynamic_args(parser)
    parser.add_argument("--allow-workload-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tree-state-encoding", choices=["legacy_float32", "fixed_point_int"], default="legacy_float32")
    parser.add_argument("--tree-fixed-point-scale", type=int, default=1_000_000)
    parser.add_argument("--tree-fixed-point-rounding", choices=["half_up_nonnegative"], default="half_up_nonnegative")
    parser.add_argument("--allow-legacy-dataset-quantization", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    teacher = DqnBudgetAgent.load(args.teacher_model)
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
    experiment_config, _qamc_metadata = _configure_qamc_experiment(
        args, _build_experiment_config(args)
    )
    workload_cli_config = (
        build_mc_stratified_dynamic_workload_cli_config(args)
        if args.workload == "mc_stratified_dynamic"
        else build_workload_cli_config(args)
    )
    fixed_point_config = FixedPointConfig(scale=args.tree_fixed_point_scale, output_max=args.tree_fixed_point_scale, rounding_mode=args.tree_fixed_point_rounding) if args.tree_state_encoding == "fixed_point_int" else None
    workload_mismatch_warning: str | None = None
    # 当 tree 训练复用已有 dataset 时，先做一次严格的参数一致性校验，
    # 防止 dataset 的 teacher 采样分布与当前训练/验证分布发生无声漂移。
    if args.workload == "mc_fairgen" and args.initial_dataset is not None:
        try:
            assert_mc_fairgen_args_match_dataset(args, args.initial_dataset)
        except ValueError as exc:
            if not args.allow_workload_mismatch:
                raise
            workload_mismatch_warning = str(exc)
    if args.workload == "mc_stratified_dynamic" and args.initial_dataset is not None:
        try:
            assert_mc_stratified_dynamic_args_match_dataset(args, args.initial_dataset)
        except ValueError as exc:
            if not args.allow_workload_mismatch:
                raise
            workload_mismatch_warning = str(exc)
    validation_end_time = args.validation_end_time if args.validation_end_time is not None else args.end_time
    max_depth_grid = [None if item.strip().lower() == "none" else int(item.strip()) for item in args.max_depth_grid.split(",") if item.strip()]
    min_leaf_grid = [int(item.strip()) for item in args.min_samples_leaf_grid.split(",") if item.strip()]
    for max_depth in max_depth_grid:
        for min_leaf in min_leaf_grid:
            run_dir = args.output_dir / f"depth_{'none' if max_depth is None else max_depth}" / f"leaf_{min_leaf}"
            run_viper_iterations(
                teacher=teacher,
                initial_dataset=args.initial_dataset,
                experiment_config=experiment_config,
                train_seeds=_parse_seeds(args.train_seeds),
                validation_seeds=_parse_seeds(args.validation_seeds),
                iterations=args.iterations,
                trajectories_per_iter=None,
                end_time=args.end_time,
                validation_end_time=validation_end_time,
                agent_period=args.agent_period,
                runtime_semantics=RuntimeSemantics(args.dqn_runtime_semantics),
                c_amc_sem_xf=args.c_amc_sem_xf,
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
                feature_config=feature_config,
                teacher_id=args.teacher_id,
                tree_hyperparams=TreeHyperParams(
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    criterion=args.criterion,
                    weight_mode=args.weight_mode,
                    resample_size=args.resample_size,
                    random_seed=args.random_seed,
                ),
                output_dir=run_dir,
                method=args.method,
                workload_cli_config=workload_cli_config,
                workload_mismatch_warning=workload_mismatch_warning,
                student_state_encoding=args.tree_state_encoding,
                fixed_point_config=fixed_point_config,
                allow_legacy_quantization=args.allow_legacy_dataset_quantization,
            )


if __name__ == "__main__":
    main()
