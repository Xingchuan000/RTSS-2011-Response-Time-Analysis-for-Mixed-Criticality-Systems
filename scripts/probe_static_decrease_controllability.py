#!/usr/bin/env python3
"""针对 decrease controllability 的静态探测脚本。

核心目标：
1. 区分两类 decrease 候选（top-cancelled 与 low-cancel-high-budget）；
2. 与 increase reference 做同口径对照；
3. 输出 detail + summary，供后续对比脚本汇总。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics, safe_relative_reduction, service_metrics_to_row
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.task_level_diagnostics import summarize_task_level_cancellations
from scripts.scan_qos_pressure_tasksets import (
    parse_int_list,
    parse_int_list_or_half_open_range,
    static_adjust_single_task,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["mc_fairgen"], default="mc_fairgen")
    parser.add_argument("--taskset-manifest", type=Path, default=None)
    parser.add_argument("--manifest-seed-column", type=str, default="candidate_seed")
    parser.add_argument("--candidate-seeds", type=str, default="")
    parser.add_argument("--seeds", type=str, default="200:210")
    parser.add_argument("--end-time", type=int, default=1_000_000)
    parser.add_argument("--budget-increase-ratio", type=float, default=0.025)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.015)
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    parser.add_argument("--repeat-counts", type=str, default="1,2,3")
    parser.add_argument("--top-k-cancelled", type=int, default=3)
    parser.add_argument("--top-k-low-cancel-high-budget", type=int, default=3)
    parser.add_argument("--top-k-increase-reference", type=int, default=3)
    parser.add_argument("--low-cancel-share-threshold", type=float, default=0.05)
    parser.add_argument("--stable-mode-delta", type=float, default=0.05)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-detail", type=Path, required=True)

    parser.add_argument("--mc-fairgen-mode", default="paper_learnable_headroom")
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
    return parser.parse_args()


def _is_hi(task: Task) -> bool:
    """统一判断 HI 任务。"""

    return task.criticality is Criticality.HI


def _parse_manifest(path: Path, seed_column: str) -> list[int]:
    """从 manifest 读取候选种子。"""

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen: set[int] = set()
    seeds: list[int] = []
    for row in rows:
        raw = str(row.get(seed_column, "")).strip()
        if not raw:
            continue
        seed = int(float(raw))
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _run_sim(tasks: list[Task], scenario, end_time: int):
    """统一运行 AMC_PLUS 事件仿真。"""

    return simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )


def _mean(values: list[float]) -> float:
    """求均值，空列表返回 0.0。"""

    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_task_aggregate(
    *,
    ordered_tasks: list[Task],
    per_seed_task_rows: list[list[dict[str, Any]]],
    budget_floor_ratio: float,
) -> list[dict[str, Any]]:
    """聚合所有 eval seed 的 per-task 统计，构建候选选择基础表。

    这里的聚合口径与计划文档一致：
    - cancelled_jobs_mean / cancel_share_mean 等全部跨 eval seed 求均值；
    - 预算相关字段来自固定 taskset 本身，不依赖 eval seed。
    """

    rows_by_index: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(ordered_tasks))}
    for rows in per_seed_task_rows:
        for row in rows:
            rows_by_index[int(row["task_index"])].append(row)

    merged: list[dict[str, Any]] = []
    for idx, task in enumerate(ordered_tasks):
        rows = rows_by_index.get(idx, [])
        released_jobs_mean = _mean([float(item.get("released_jobs", 0.0)) for item in rows])
        cancelled_jobs_mean = _mean([float(item.get("cancelled_jobs", 0.0)) for item in rows])
        cancel_share_mean = _mean([float(item.get("cancel_share_of_total", 0.0)) for item in rows])
        cancel_ratio_over_released_mean = _mean([float(item.get("cancel_ratio_over_released", 0.0)) for item in rows])
        cancelled_per_1m_mean = _mean([float(item.get("cancelled_per_1m", 0.0)) for item in rows])

        floor = int(math.floor(float(task.c_lo) * float(budget_floor_ratio)))
        floor = max(1, floor)
        budget_slack_to_floor = max(0, int(task.c_lo) - int(floor))
        increase_headroom = max(0, min(int(task.c_hi), int(task.deadline)) - int(task.c_lo))

        merged.append(
            {
                "task_index": int(idx),
                "task_name": str(task.name),
                "criticality": getattr(task.criticality, "name", str(task.criticality)),
                "period": int(task.period),
                "c_lo": int(task.c_lo),
                "c_hi": int(task.c_hi),
                "deadline": int(task.deadline),
                "released_jobs_mean": released_jobs_mean,
                "cancelled_jobs_mean": cancelled_jobs_mean,
                "cancel_share_mean": cancel_share_mean,
                "cancel_ratio_over_released_mean": cancel_ratio_over_released_mean,
                "cancelled_per_1m_mean": cancelled_per_1m_mean,
                "budget_c_lo": int(task.c_lo),
                "budget_c_hi": int(task.c_hi),
                "budget_slack_to_floor": int(budget_slack_to_floor),
                "increase_headroom": int(increase_headroom),
                "is_lo": not _is_hi(task),
            }
        )
    return merged


def _pick_candidates(
    *,
    task_agg: list[dict[str, Any]],
    top_k_cancelled: int,
    top_k_low_cancel_high_budget: int,
    top_k_increase_reference: int,
    low_cancel_share_threshold: float,
) -> list[dict[str, Any]]:
    """按三种 probe 类型选择候选任务。"""

    picks: list[dict[str, Any]] = []

    top_cancelled = [
        row
        for row in task_agg
        if bool(row["is_lo"]) and float(row["cancelled_jobs_mean"]) > 0.0 and int(row["budget_slack_to_floor"]) > 0
    ]
    top_cancelled = sorted(
        top_cancelled,
        key=lambda item: (float(item["cancelled_jobs_mean"]), float(item["cancel_share_mean"]), -int(item["task_index"])),
        reverse=True,
    )[: max(0, top_k_cancelled)]
    for row in top_cancelled:
        picks.append({"probe_type": "decrease_top_cancelled_lo", "probe_action": "decrease", "task": row})

    low_cancel_high_budget = [
        row
        for row in task_agg
        if int(row["budget_slack_to_floor"]) > 0
        and (
            float(row["cancel_share_mean"]) <= float(low_cancel_share_threshold)
            or float(row["cancelled_jobs_mean"]) <= 0.0
        )
    ]
    low_cancel_high_budget = sorted(
        low_cancel_high_budget,
        key=lambda item: (int(item["budget_slack_to_floor"]), int(item["c_lo"]), -int(item["task_index"])),
        reverse=True,
    )[: max(0, top_k_low_cancel_high_budget)]
    for row in low_cancel_high_budget:
        picks.append({"probe_type": "decrease_low_cancel_high_budget", "probe_action": "decrease", "task": row})

    increase_ref = [
        row
        for row in task_agg
        # increase 对照按计划允许先使用 headroom 判定候选，
        # 这里不强制 cancelled_jobs_mean > 0，避免短窗口下无对照候选。
        if bool(row["is_lo"]) and int(row["increase_headroom"]) > 0
    ]
    increase_ref = sorted(
        increase_ref,
        key=lambda item: (float(item["cancelled_jobs_mean"]), float(item["cancel_share_mean"]), -int(item["task_index"])),
        reverse=True,
    )
    if not increase_ref:
        # 若没有可 increase headroom 的任务，仍保留 increase 对照类型：
        # 这样 detail 会显式记录该类型在当前 taskset 下大多为 invalid 的事实，
        # 便于与 decrease 的有效性形成可解释对比。
        increase_ref = sorted(
            [row for row in task_agg if bool(row["is_lo"])],
            key=lambda item: (float(item["cancelled_jobs_mean"]), float(item["cancel_share_mean"]), -int(item["task_index"])),
            reverse=True,
        )
    increase_ref = increase_ref[: max(0, top_k_increase_reference)]
    for row in increase_ref:
        picks.append({"probe_type": "increase_valid_cancel_source", "probe_action": "increase", "task": row})

    return picks


def _select_best(rows: list[dict[str, Any]], require_stable: bool) -> dict[str, Any] | None:
    """按计划规则选择 best 行。"""

    candidates = [row for row in rows if (bool(row["stable_candidate"]) if require_stable else True)]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item["lo_cancellation_rel_reduction"]),
            float(item["lc_service_loss_abs_reduction"]),
            -float(item["mode_change_delta_ratio"]),
        ),
        reverse=True,
    )[0]


def _best_with_fallback(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """优先 stable best；若不存在则回退到 tradeoff best。"""

    stable = _select_best(rows, require_stable=True)
    if stable is not None:
        return stable
    return _select_best(rows, require_stable=False)


def _to_summary(best_row: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    """把 best 行映射到 summary 字段。"""

    if best_row is None:
        return {
            f"{prefix}_target_task_name": "",
            f"{prefix}_repeat_count": "",
            f"{prefix}_rel_lo_reduction": 0.0,
            f"{prefix}_lc_loss_reduction": 0.0,
            f"{prefix}_mode_delta_ratio": 0.0,
            f"{prefix}_stable_candidate": False,
        }
    return {
        f"{prefix}_target_task_name": str(best_row["target_task_name"]),
        f"{prefix}_repeat_count": int(best_row["repeat_count"]),
        f"{prefix}_rel_lo_reduction": float(best_row["lo_cancellation_rel_reduction"]),
        f"{prefix}_lc_loss_reduction": float(best_row["lc_service_loss_abs_reduction"]),
        f"{prefix}_mode_delta_ratio": float(best_row["mode_change_delta_ratio"]),
        f"{prefix}_stable_candidate": bool(best_row["stable_candidate"]),
    }


def main() -> None:
    """执行 static decrease probe。"""

    args = parse_args()
    eval_seeds = parse_int_list_or_half_open_range(args.seeds)
    if not eval_seeds:
        raise ValueError("--seeds 不能为空")
    repeat_counts = parse_int_list(args.repeat_counts)
    if not repeat_counts:
        raise ValueError("--repeat-counts 不能为空")

    # 严格按计划要求：
    # 1) candidate-seeds 非空时，直接使用，不要求出现在 manifest。
    # 2) candidate-seeds 为空时，必须从 manifest 读取。
    if args.candidate_seeds.strip():
        candidate_seeds = parse_int_list_or_half_open_range(args.candidate_seeds)
    else:
        if args.taskset_manifest is None:
            raise ValueError("当 --candidate-seeds 为空时，必须提供 --taskset-manifest")
        candidate_seeds = _parse_manifest(args.taskset_manifest, args.manifest_seed_column)

    if not candidate_seeds:
        raise ValueError("没有可用 candidate seeds")

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate_seed in candidate_seeds:
        cfg = build_mc_fairgen_experiment_config(
            mode=args.mc_fairgen_mode,
            num_tasks=args.mc_fairgen_num_tasks,
            hi_ratio=args.mc_fairgen_hi_ratio,
            period_source=args.mc_fairgen_period_source,
            period_scale=args.mc_fairgen_period_scale,
            fixed_taskset_seed=candidate_seed,
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

        bundles_by_seed: dict[int, Any] = {}
        baseline_rows: list[dict[str, float]] = []
        per_seed_task_rows: list[list[dict[str, Any]]] = []

        for eval_seed in eval_seeds:
            bundle = resolve_experiment_bundle(cfg, eval_seed)
            bundles_by_seed[eval_seed] = bundle
            base_result = _run_sim(list(bundle.ordered_tasks), bundle.scenario, args.end_time)
            base_metrics = compute_service_quality_metrics(base_result)
            base_row = {
                "mode_changes": float(base_result.mode_change_count()),
                "lo_cancellations": float(base_result.lo_job_cancellation_count()),
                **service_metrics_to_row(base_metrics),
            }
            baseline_rows.append(base_row)

            diag = summarize_task_level_cancellations(ordered_tasks=list(bundle.ordered_tasks), result=base_result)
            per_seed_task_rows.append(list(diag.get("per_task_rows", [])))

        baseline_lo_mean = mean(float(row["lo_cancellations"]) for row in baseline_rows)
        baseline_lc_loss_mean = mean(float(row["lc_service_loss"]) for row in baseline_rows)
        baseline_mode_mean = mean(float(row["mode_changes"]) for row in baseline_rows)
        baseline_hi_miss_sum = float(sum(float(row["hi_deadline_misses"]) for row in baseline_rows))

        ordered_tasks = list(next(iter(bundles_by_seed.values())).ordered_tasks)
        task_agg = _build_task_aggregate(
            ordered_tasks=ordered_tasks,
            per_seed_task_rows=per_seed_task_rows,
            budget_floor_ratio=args.budget_floor_ratio,
        )
        picks = _pick_candidates(
            task_agg=task_agg,
            top_k_cancelled=args.top_k_cancelled,
            top_k_low_cancel_high_budget=args.top_k_low_cancel_high_budget,
            top_k_increase_reference=args.top_k_increase_reference,
            low_cancel_share_threshold=args.low_cancel_share_threshold,
        )

        seed_rows: list[dict[str, Any]] = []
        for pick in picks:
            task_row = dict(pick["task"])
            task_idx = int(task_row["task_index"])
            action = str(pick["probe_action"])
            probe_type = str(pick["probe_type"])
            for repeat_count in repeat_counts:
                invalid_count = 0
                probe_rows: list[dict[str, float]] = []
                for eval_seed in eval_seeds:
                    bundle = bundles_by_seed[eval_seed]
                    adjusted_tasks, changed, invalid, _ = static_adjust_single_task(
                        tasks=list(bundle.ordered_tasks),
                        target_idx=task_idx,
                        action=action,
                        repeat_count=int(repeat_count),
                        inc_ratio=args.budget_increase_ratio,
                        dec_ratio=args.budget_decrease_ratio,
                        budget_floor_ratio=args.budget_floor_ratio,
                    )
                    if invalid:
                        invalid_count += 1
                    result = _run_sim(
                        adjusted_tasks if changed else list(bundle.ordered_tasks),
                        bundle.scenario,
                        args.end_time,
                    )
                    metrics = compute_service_quality_metrics(result)
                    probe_rows.append(
                        {
                            "mode_changes": float(result.mode_change_count()),
                            "lo_cancellations": float(result.lo_job_cancellation_count()),
                            **service_metrics_to_row(metrics),
                        }
                    )

                probe_lo_mean = mean(float(row["lo_cancellations"]) for row in probe_rows)
                probe_lc_loss_mean = mean(float(row["lc_service_loss"]) for row in probe_rows)
                probe_mode_mean = mean(float(row["mode_changes"]) for row in probe_rows)
                probe_hi_miss_sum = float(sum(float(row["hi_deadline_misses"]) for row in probe_rows))

                lo_abs = baseline_lo_mean - probe_lo_mean
                lo_rel = safe_relative_reduction(baseline_lo_mean, probe_lo_mean)
                lc_abs = baseline_lc_loss_mean - probe_lc_loss_mean
                lc_rel = safe_relative_reduction(baseline_lc_loss_mean, probe_lc_loss_mean)
                mode_delta = probe_mode_mean - baseline_mode_mean
                mode_delta_ratio = mode_delta / baseline_mode_mean if baseline_mode_mean > 0 else 0.0

                stable_candidate = (
                    probe_hi_miss_sum == 0.0
                    and mode_delta_ratio <= float(args.stable_mode_delta)
                    and probe_lc_loss_mean <= baseline_lc_loss_mean
                )
                qos_candidate = probe_hi_miss_sum == 0.0 and probe_lc_loss_mean <= baseline_lc_loss_mean
                tradeoff_risk = lo_rel > 0.03 and mode_delta_ratio > float(args.stable_mode_delta)

                row = {
                    "candidate_seed": int(candidate_seed),
                    "probe_type": probe_type,
                    "probe_action": action,
                    "target_task_index": task_idx,
                    "target_task_name": str(task_row["task_name"]),
                    "target_criticality": str(task_row["criticality"]),
                    "target_period": int(task_row["period"]),
                    "target_c_lo": int(task_row["c_lo"]),
                    "target_c_hi": int(task_row["c_hi"]),
                    "target_deadline": int(task_row["deadline"]),
                    "target_baseline_cancelled_jobs_mean": float(task_row["cancelled_jobs_mean"]),
                    "target_baseline_cancel_share_mean": float(task_row["cancel_share_mean"]),
                    "target_baseline_cancel_ratio_mean": float(task_row["cancel_ratio_over_released_mean"]),
                    "target_budget_slack_to_floor": int(task_row["budget_slack_to_floor"]),
                    "target_increase_headroom": int(task_row["increase_headroom"]),
                    "repeat_count": int(repeat_count),
                    "invalid_count": int(invalid_count),
                    "baseline_lo_cancellations_mean": float(baseline_lo_mean),
                    "probe_lo_cancellations_mean": float(probe_lo_mean),
                    "lo_cancellation_abs_reduction": float(lo_abs),
                    "lo_cancellation_rel_reduction": float(lo_rel),
                    "baseline_lc_service_loss_mean": float(baseline_lc_loss_mean),
                    "probe_lc_service_loss_mean": float(probe_lc_loss_mean),
                    "lc_service_loss_abs_reduction": float(lc_abs),
                    "lc_service_loss_rel_reduction": float(lc_rel),
                    "baseline_mode_changes_mean": float(baseline_mode_mean),
                    "probe_mode_changes_mean": float(probe_mode_mean),
                    "mode_change_delta": float(mode_delta),
                    "mode_change_delta_ratio": float(mode_delta_ratio),
                    "baseline_hi_deadline_misses_sum": float(baseline_hi_miss_sum),
                    "probe_hi_deadline_misses_sum": float(probe_hi_miss_sum),
                    "stable_candidate": bool(stable_candidate),
                    "qos_candidate": bool(qos_candidate),
                    "tradeoff_risk": bool(tradeoff_risk),
                }
                detail_rows.append(row)
                seed_rows.append(row)

        best_overall = _best_with_fallback(seed_rows)
        best_decrease = _best_with_fallback([row for row in seed_rows if str(row["probe_action"]) == "decrease"])
        best_dec_top = _best_with_fallback([row for row in seed_rows if str(row["probe_type"]) == "decrease_top_cancelled_lo"])
        best_dec_low = _best_with_fallback([row for row in seed_rows if str(row["probe_type"]) == "decrease_low_cancel_high_budget"])
        best_inc_ref = _best_with_fallback([row for row in seed_rows if str(row["probe_type"]) == "increase_valid_cancel_source"])

        summary = {
            "candidate_seed": int(candidate_seed),
            "num_eval_seeds": len(eval_seeds),
            "probe_end_time": int(args.end_time),
            "baseline_lo_cancellations_mean": float(baseline_lo_mean),
            "baseline_lc_service_loss_mean": float(baseline_lc_loss_mean),
            "baseline_mode_changes_mean": float(baseline_mode_mean),
            "baseline_hi_deadline_misses_sum": float(baseline_hi_miss_sum),
            "best_overall_probe_type": str(best_overall["probe_type"]) if best_overall else "",
            "best_overall_action": str(best_overall["probe_action"]) if best_overall else "",
            "best_overall_target_task_name": str(best_overall["target_task_name"]) if best_overall else "",
            "best_overall_repeat_count": int(best_overall["repeat_count"]) if best_overall else "",
            "best_overall_rel_lo_reduction": float(best_overall["lo_cancellation_rel_reduction"]) if best_overall else 0.0,
            "best_overall_lc_loss_reduction": float(best_overall["lc_service_loss_abs_reduction"]) if best_overall else 0.0,
            "best_overall_mode_delta_ratio": float(best_overall["mode_change_delta_ratio"]) if best_overall else 0.0,
            "best_decrease_probe_type": str(best_decrease["probe_type"]) if best_decrease else "",
        }
        summary.update(_to_summary(best_decrease, "best_decrease"))
        summary.update(_to_summary(best_dec_top, "best_decrease_top_cancelled_lo"))
        summary.update(_to_summary(best_dec_low, "best_decrease_low_cancel_high_budget"))
        summary.update(_to_summary(best_inc_ref, "best_increase_reference"))

        summary["decrease_probe_has_stable_positive"] = bool(
            best_decrease is not None
            and bool(best_decrease["stable_candidate"])
            and float(best_decrease["lo_cancellation_rel_reduction"]) > 0.0
        )
        summary["decrease_probe_best_beats_increase_reference"] = bool(
            best_decrease is not None
            and best_inc_ref is not None
            and float(best_decrease["lo_cancellation_rel_reduction"])
            > float(best_inc_ref["lo_cancellation_rel_reduction"])
        )
        summary_rows.append(summary)

    args.output_detail.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    if detail_rows:
        with args.output_detail.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
    else:
        args.output_detail.write_text("", encoding="utf-8")

    if summary_rows:
        with args.output_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        args.output_summary.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
