"""按指定种子列表输出 task-level controllability 诊断结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from scripts.scan_taskset_headroom import (
    _apply_budget_scale_to_tasks,
    _build_scan_experiment_config,
    _load_manifest_rows,
    _parse_int_list_or_range,
    ScanConfig,
    _run_diagnostic_on_seed,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.dqn.experiment import resolve_experiment_bundle


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["automotive", "mc_fairgen"], default="mc_fairgen")
    parser.add_argument("--automotive-mode", type=str, default="paper_exact")
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)
    parser.add_argument("--automotive-num-runnables", type=int, default=150)
    parser.add_argument("--mc-fairgen-mode", type=str, default="paper_learnable_headroom")
    parser.add_argument("--mc-fairgen-num-tasks", type=int, default=12)
    parser.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    parser.add_argument("--mc-fairgen-period-source", type=str, default="controlled_medium")
    parser.add_argument("--mc-fairgen-period-scale", type=int, default=500)
    parser.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    parser.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    parser.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    parser.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.25)
    parser.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.45)
    parser.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    parser.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    parser.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.20)
    parser.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.40)
    parser.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    parser.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.12)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.02)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.25)
    parser.add_argument("--require-schedulable", action="store_true")
    parser.add_argument("--taskset-manifest", type=Path, default=None)
    parser.add_argument("--manifest-seed-column", type=str, default="candidate_seed")
    parser.add_argument("--fixed-taskset-seeds", type=str, default="")
    parser.add_argument("--seeds", type=str, default="200:210")
    parser.add_argument("--end-time", type=int, default=5_000_000)
    parser.add_argument("--agent-period", type=int, default=100_000)
    parser.add_argument("--reward-mode", type=str, default="interval_qos_v2")
    parser.add_argument("--action-space", type=str, default="single")
    parser.add_argument("--budget-increase-ratio", type=float, default=0.025)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.015)
    parser.add_argument("--include-explicit-noop", action="store_true")
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    parser.add_argument("--forbid-decreasing-hi-budgets", action="store_true")
    parser.add_argument("--observation-mode", choices=["v10_basic", "v11_full_10d"], default="v11_full_10d")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-scales", type=str, default="1.0")
    parser.add_argument("--eval-seeds", type=str, default="")
    parser.add_argument("--low-event-threshold", type=float, default=40.0)
    parser.add_argument("--high-event-threshold", type=float, default=120.0)
    parser.add_argument("--enable-ranked-pair-diagnostic", action="store_true")
    parser.add_argument("--ranked-pair-min-valid-count", type=float, default=2.0)
    parser.add_argument("--ranked-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--ranked-pair-top-k-surplus", type=int, default=3)
    parser.add_argument(
        "--ranked-pair-decrease-mode",
        type=str,
        default="top2_surplus",
        choices=["top1_surplus", "top2_surplus", "topk_surplus"],
    )
    parser.add_argument("--enable-constraint-guided-pair-diagnostic", action="store_true")
    parser.add_argument("--constraint-guided-pair-min-valid-count", type=float, default=1.0)
    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=4)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action="store_true")
    parser.add_argument("--enable-task-level-cancellation-diagnostic", action="store_true")
    parser.add_argument("--task-level-output-dir", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-task-details", type=Path, required=True)

    parser.add_argument(
        "--seed-limit",
        type=int,
        default=None,
        help="Optionally limit the number of manifest rows to diagnose.",
    )
    parser.add_argument(
        "--filter-recommended",
        action="store_true",
        help="Only keep rows marked as recommended_for_dqn/recommended_fast when such columns exist.",
    )
    return parser


def _controllability_group(valid_increase_cancel_coverage_mean: float, task_level_top2_cancel_share_mean: float) -> str:
    """第一版 controllability 分组。"""

    if valid_increase_cancel_coverage_mean >= 0.60 and task_level_top2_cancel_share_mean >= 0.50:
        return "high_controllable"
    if valid_increase_cancel_coverage_mean >= 0.35:
        return "medium_controllable"
    return "low_controllable"


def main() -> None:
    """执行诊断并输出 summary + task details。"""

    args = build_parser().parse_args()
    eval_seeds = _parse_int_list_or_range(args.seeds)
    if not eval_seeds:
        raise ValueError("--seeds 不能为空")
    fixed_seeds = _parse_int_list_or_range(args.fixed_taskset_seeds)
    if not fixed_seeds and args.taskset_manifest is not None:
        # 与 scan_taskset_headroom 一致：若提供 manifest 且未显式传 seed 列表，
        # 则直接从 manifest 的 seed 列读取任务集种子，避免重复手动维护。
        manifest_rows = _load_manifest_rows(
        args.taskset_manifest,
        seed_column=args.manifest_seed_column,
        seed_limit=args.seed_limit,
        filter_recommended=args.filter_recommended,
    )
        fixed_seeds = [int(row[args.manifest_seed_column]) for row in manifest_rows]
    if not fixed_seeds:
        raise ValueError("--fixed-taskset-seeds 不能为空；若想自动读取，请提供 --taskset-manifest")

    # 复用 scan 的配置对象，保证 workload/scenario 构造口径完全一致。
    config = ScanConfig.from_args(
        args=args,
        eval_seeds=eval_seeds,
        budget_scales=[1.0],
        manifest_rows_by_seed={},
    )
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

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for candidate_seed in fixed_seeds:
        experiment_config = _build_scan_experiment_config(config=config, fixed_taskset_seed=candidate_seed)
        per_seed_diag_rows: list[dict[str, Any]] = []
        baseline_mode_changes_values: list[float] = []
        baseline_lo_cancellations_values: list[float] = []
        baseline_deadline_misses_values: list[float] = []

        for eval_seed in eval_seeds:
            bundle_for_seed = resolve_experiment_bundle(experiment_config, eval_seed)
            scaled_tasks, _ = _apply_budget_scale_to_tasks(
                ordered_tasks=list(bundle_for_seed.ordered_tasks),
                budget_scale=1.0,
                budget_floor_ratio=args.budget_floor_ratio,
            )
            baseline_result = simulate_ordered_taskset_event_driven(
                ordered_tasks=scaled_tasks,
                scenario=bundle_for_seed.scenario,
                config=RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS),
            )
            baseline_mode_changes_values.append(float(baseline_result.mode_change_count()))
            baseline_lo_cancellations_values.append(float(baseline_result.lo_job_cancellation_count()))
            baseline_deadline_misses_values.append(float(len(baseline_result.deadline_misses)))

            diag = _run_diagnostic_on_seed(
                ordered_tasks=scaled_tasks,
                scenario=bundle_for_seed.scenario,
                feature_config=feature_config,
                eval_seed=eval_seed,
                end_time=args.end_time,
                agent_period=args.agent_period,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                enable_ranked_pair_diagnostic=False,
                ranked_pair_top_k_risk=3,
                ranked_pair_top_k_surplus=3,
                ranked_pair_decrease_mode="top2_surplus",
                enable_constraint_guided_pair_diagnostic=False,
                constraint_guided_pair_top_k_risk=3,
                constraint_guided_pair_top_k_decrease=4,
                constraint_guided_pair_prefer_lo=False,
                enable_task_level_cancellation_diagnostic=True,
            )
            per_seed_diag_rows.append(diag)

            for item in diag.get("task_level_per_task_rows", []):
                row = dict(item)
                row["candidate_seed"] = int(candidate_seed)
                row["eval_seed"] = int(eval_seed)
                detail_rows.append(row)

        baseline_mode_changes_mean = mean(baseline_mode_changes_values) if baseline_mode_changes_values else 0.0
        baseline_lo_cancellations_mean = mean(baseline_lo_cancellations_values) if baseline_lo_cancellations_values else 0.0
        scale_per_1m = float(args.end_time) / 1_000_000.0
        baseline_mode_changes_per_1m = baseline_mode_changes_mean / scale_per_1m
        baseline_lo_cancellations_per_1m = baseline_lo_cancellations_mean / scale_per_1m
        baseline_total_events_mean = baseline_mode_changes_mean + baseline_lo_cancellations_mean
        baseline_total_events_per_1m = baseline_total_events_mean / scale_per_1m
        baseline_lo_cancellation_ratio_total = baseline_lo_cancellations_mean / max(1.0, baseline_total_events_mean)

        valid_increase_cancel_coverage_mean = mean(
            item.get("valid_increase_cancel_coverage", 0.0) for item in per_seed_diag_rows
        )
        task_level_top2_cancel_share_mean = mean(item.get("task_level_top2_cancel_share", 0.0) for item in per_seed_diag_rows)
        controllability_score = (
            valid_increase_cancel_coverage_mean
            * task_level_top2_cancel_share_mean
            * baseline_lo_cancellations_per_1m
            / max(1.0, baseline_mode_changes_per_1m)
        )
        # decrease_source_score 仅用于诊断“decrease 对主要 cancellation source 的可覆盖性”，
        # 不能直接解释为 decrease 一定有效；真实收益仍需依赖 static probe 验证。
        valid_decrease_cancel_coverage_mean = mean(
            item.get("valid_decrease_cancel_coverage", 0.0) for item in per_seed_diag_rows
        )
        decrease_source_score = (
            valid_decrease_cancel_coverage_mean
            * task_level_top2_cancel_share_mean
            * baseline_lo_cancellations_per_1m
            / max(1.0, baseline_mode_changes_per_1m)
        )
        summary_rows.append(
            {
                "candidate_seed": int(candidate_seed),
                "num_eval_seeds": len(eval_seeds),
                "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
                "baseline_lo_cancellations_per_1m": baseline_lo_cancellations_per_1m,
                "baseline_mode_changes_mean": baseline_mode_changes_mean,
                "baseline_mode_changes_per_1m": baseline_mode_changes_per_1m,
                "baseline_total_events_per_1m": baseline_total_events_per_1m,
                "baseline_lo_cancellation_ratio_total": baseline_lo_cancellation_ratio_total,
                "valid_increase_count_mean": mean(item["valid_increase_count_mean"] for item in per_seed_diag_rows),
                "valid_decrease_count_mean": mean(item["valid_decrease_count_mean"] for item in per_seed_diag_rows),
                "task_level_top1_cancel_share_mean": mean(item.get("task_level_top1_cancel_share", 0.0) for item in per_seed_diag_rows),
                "task_level_top2_cancel_share_mean": task_level_top2_cancel_share_mean,
                "task_level_top3_cancel_share_mean": mean(item.get("task_level_top3_cancel_share", 0.0) for item in per_seed_diag_rows),
                "task_level_cancel_concentration_hhi_mean": mean(
                    item.get("task_level_cancel_concentration_hhi", 0.0) for item in per_seed_diag_rows
                ),
                # 兼容 selector 计划中的字段命名：保留一份别名列，避免后续脚本硬编码失败。
                "cancel_concentration_hhi_mean": mean(
                    item.get("task_level_cancel_concentration_hhi", 0.0) for item in per_seed_diag_rows
                ),
                "task_level_num_cancelled_lo_tasks_mean": mean(
                    item.get("task_level_num_cancelled_lo_tasks", 0.0) for item in per_seed_diag_rows
                ),
                "task_level_max_task_cancel_ratio_mean": mean(
                    item.get("task_level_max_task_cancel_ratio", 0.0) for item in per_seed_diag_rows
                ),
                "valid_increase_cancel_coverage_mean": valid_increase_cancel_coverage_mean,
                "valid_increase_top1_cancel_hit_rate": mean(
                    item.get("valid_increase_top1_cancel_hit", 0.0) for item in per_seed_diag_rows
                ),
                "valid_increase_top2_cancel_hit_count_mean": mean(
                    item.get("valid_increase_top2_cancel_hit_count", 0.0) for item in per_seed_diag_rows
                ),
                "valid_increase_top3_cancel_hit_count_mean": mean(
                    item.get("valid_increase_top3_cancel_hit_count", 0.0) for item in per_seed_diag_rows
                ),
                "valid_increase_cancelled_task_share_mean": mean(
                    item.get("valid_increase_cancelled_task_share", 0.0) for item in per_seed_diag_rows
                ),
                "valid_decrease_cancel_coverage_mean": valid_decrease_cancel_coverage_mean,
                "valid_decrease_top1_cancel_hit_rate": mean(
                    item.get("valid_decrease_top1_cancel_hit", 0.0) for item in per_seed_diag_rows
                ),
                "valid_decrease_top2_cancel_hit_count_mean": mean(
                    item.get("valid_decrease_top2_cancel_hit_count", 0.0) for item in per_seed_diag_rows
                ),
                "valid_decrease_top3_cancel_hit_count_mean": mean(
                    item.get("valid_decrease_top3_cancel_hit_count", 0.0) for item in per_seed_diag_rows
                ),
                "valid_decrease_cancelled_task_share_mean": mean(
                    item.get("valid_decrease_cancelled_task_share", 0.0) for item in per_seed_diag_rows
                ),
                "decrease_source_score": decrease_source_score,
                "controllability_score": controllability_score,
                "controllability_group": _controllability_group(
                    valid_increase_cancel_coverage_mean,
                    task_level_top2_cancel_share_mean,
                ),
                "baseline_deadline_misses_sum": sum(baseline_deadline_misses_values),
            }
        )

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_task_details.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with args.output_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    if detail_rows:
        with args.output_task_details.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)


if __name__ == "__main__":
    main()
