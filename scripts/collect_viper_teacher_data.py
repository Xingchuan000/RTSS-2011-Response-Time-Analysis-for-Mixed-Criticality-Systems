"""采集 VIPER teacher dataset 的 CLI。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.dqn import (
    DqnBudgetAgent,
    build_automotive_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from amc_py.viper.dataset import write_viper_dataset
from amc_py.viper.teacher import collect_teacher_labeled_rollouts
from amc_py.viper.fixed_point import FixedPointConfig
from scripts.common_mc_fairgen_cli import (
    add_mc_fairgen_args,
    build_mc_fairgen_config_from_args,
    build_workload_cli_config,
)


def _parse_seeds(raw_value: str) -> list[int]:
    seeds: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            seeds.extend(range(begin, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def _build_experiment_config(args: argparse.Namespace):
    if args.workload == "small":
        return build_small_nominal_experiment_config() if args.scenario == "nominal" else build_small_stress_experiment_config()
    if args.workload == "rtss11":
        return build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
        )
    if args.workload == "automotive":
        return build_automotive_experiment_config(
            num_runnables=args.automotive_num_runnables,
            mode=args.automotive_mode,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            budget_floor_ratio=args.budget_floor_ratio,
        )
    if args.workload == "mc_fairgen":
        return build_mc_fairgen_config_from_args(args)
    raise ValueError(f"unsupported workload: {args.workload}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive", "mc_fairgen"], default="small")
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--scenario-split", type=str, default="train")
    parser.add_argument("--dqn-runtime-semantics", choices=["AMC_PLUS", "AMC_RA", "AMC_RH", "C_AMC_SEM"], default="AMC_PLUS")
    parser.add_argument("--c-amc-sem-xf", type=float, default=0.5)
    parser.add_argument("--reward-mode", type=str, default="mendes")
    parser.add_argument("--action-space", choices=["triple", "pair", "single", "constraint_guided_pair", "constraint_guided_transfer"], default="single")
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
    parser.add_argument("--tree-state-encoding", choices=["legacy_float32", "fixed_point_int"], default="fixed_point_int")
    parser.add_argument("--tree-fixed-point-scale", type=int, default=1_000_000)
    parser.add_argument("--tree-fixed-point-rounding", choices=["half_up_nonnegative"], default="half_up_nonnegative")
    parser.add_argument("--tree-fallback-mode", choices=["ranked_valid_or_none", "top1_or_noop"], default="top1_or_noop")
    parser.add_argument("--action-validation-mode", choices=["legacy", "formal_v1"], default="legacy")
    parser.add_argument("--strict-candidate-deploy-cap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--carry-over-aware-safety", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lo-budget-overrun-guard-units", type=int, default=1)
    parser.add_argument("--allow-legacy-dataset-quantization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-integer-tree-artifact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formal-deployment-v1", action="store_true")
    parser.add_argument("--fixed-ranked-deployment-v1", action="store_true")
    add_mc_fairgen_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.formal_deployment_v1 and args.fixed_ranked_deployment_v1:
        raise ValueError("两个 deployment profile 不能同时使用")
    if args.fixed_ranked_deployment_v1:
        args.tree_state_encoding = "fixed_point_int"
        args.tree_fallback_mode = "ranked_valid_or_none"
        args.action_validation_mode = "formal_v1"
        args.strict_candidate_deploy_cap = True
        args.carry_over_aware_safety = True
        args.lo_budget_overrun_guard_units = 1
    if args.formal_deployment_v1:
        args.tree_state_encoding = "fixed_point_int"
        args.tree_fallback_mode = "top1_or_noop"
        args.action_validation_mode = "formal_v1"
        args.strict_candidate_deploy_cap = True
        args.carry_over_aware_safety = True
        args.lo_budget_overrun_guard_units = 1
        args.require_integer_tree_artifact = True
    teacher = DqnBudgetAgent.load(args.model)
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
    experiment_config = replace(_build_experiment_config(args), action_validation_mode=args.action_validation_mode, strict_candidate_deploy_cap=args.strict_candidate_deploy_cap, carry_over_aware_safety=args.carry_over_aware_safety, lo_budget_overrun_guard_units=args.lo_budget_overrun_guard_units)
    if args.action_validation_mode == "formal_v1" and args.action_space != "single":
        raise ValueError("formal_v1 只允许 single action space")
    samples, manifest = collect_teacher_labeled_rollouts(
        teacher=teacher,
        experiment_config=experiment_config,
        seeds=_parse_seeds(args.seeds),
        end_time=args.end_time,
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
        taskset_seed=args.fixed_taskset_seed,
        scenario_split=args.scenario_split,
        fixed_point_config=FixedPointConfig(scale=args.tree_fixed_point_scale, max_int=args.tree_fixed_point_scale, rounding_mode=args.tree_fixed_point_rounding),
        tree_fallback_mode=args.tree_fallback_mode,
    )
    manifest["teacher_model_path"] = str(args.model)
    # 无论后续数据集被谁消费，都把 workload CLI 口径完整落盘，便于检查
    # teacher 采样分布是否与 tree 训练/HOUT 评估保持一致。
    manifest["workload_cli_config"] = build_workload_cli_config(args)
    write_viper_dataset(args.output_dir, samples, manifest)
    with (args.output_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest["feature_names"], handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "action_definitions.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest["action_definitions"], handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
