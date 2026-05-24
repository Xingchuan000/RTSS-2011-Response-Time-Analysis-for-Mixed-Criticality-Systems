#!/usr/bin/env python3
"""对候选 taskset 做 stable-improvement 低成本探测。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics, safe_relative_reduction, service_metrics_to_row
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from scripts.scan_qos_pressure_tasksets import (
    choose_single_task_sweep_indices,
    parse_int_list,
    parse_int_list_or_half_open_range,
    parse_str_list,
    select_sequence_static_best,
    select_single_task_static_best,
    static_adjust_single_task,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", default="mc_fairgen")
    p.add_argument("--taskset-manifest", type=Path, required=True)
    p.add_argument("--manifest-seed-column", default="candidate_seed")
    p.add_argument("--candidate-seeds", default="")
    p.add_argument("--seeds", default="200:206")
    p.add_argument("--end-time", type=int, default=3_000_000)
    p.add_argument("--agent-period", type=int, default=100000)
    p.add_argument("--budget-increase-ratio", type=float, default=0.025)
    p.add_argument("--budget-decrease-ratio", type=float, default=0.015)
    p.add_argument("--budget-floor-ratio", type=float, default=0.9)
    p.add_argument("--top-k-tasks", type=int, default=6)
    p.add_argument("--repeat-counts", default="1,2,3")
    p.add_argument("--actions", default="increase,decrease")
    p.add_argument("--enable-pair-probe", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pair-top-k", type=int, default=4)
    p.add_argument("--stable-mode-delta", type=float, default=0.05)
    p.add_argument("--output-summary", type=Path, required=True)
    p.add_argument("--output-detail", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--mc-fairgen-mode", default="paper_learnable_headroom")
    p.add_argument("--mc-fairgen-num-tasks", type=int, default=12)
    p.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    p.add_argument("--mc-fairgen-period-source", type=str, default="controlled_medium")
    p.add_argument("--mc-fairgen-period-scale", type=int, default=100)
    p.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    p.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    p.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    p.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    p.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.25)
    p.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.45)
    p.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    p.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    p.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.20)
    p.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.40)
    p.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    p.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.12)
    p.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    p.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    p.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.02)
    p.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.25)
    return p.parse_args()


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return sum(values) / max(1, len(values))


def _sum(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(float(r[key]) for r in rows))


def _run_sim(tasks, scenario, end_time: int):
    return simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )


def _parse_manifest(path: Path, seed_column: str) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seeds: list[int] = []
    seen: set[int] = set()
    for row in rows:
        text = str(row.get(seed_column, "")).strip()
        if not text:
            continue
        value = int(float(text))
        if value in seen:
            continue
        seen.add(value)
        seeds.append(value)
    return seeds


def main() -> None:
    args = parse_args()
    if args.end_time <= 0:
        raise ValueError("end_time must be positive for per-1M metrics")

    # candidate seed 与 eval seed 必须解耦：
    # - candidate seed 决定“扫描哪些 taskset”；
    # - eval seed 决定“每个 taskset 用哪些场景种子做评估统计”。
    # 若未显式传入 --candidate-seeds，则默认扫描 manifest 中全部候选。
    manifest_candidates = _parse_manifest(args.taskset_manifest, args.manifest_seed_column)
    if args.candidate_seeds.strip():
        selected_candidates = set(parse_int_list_or_half_open_range(args.candidate_seeds))
        candidates = [seed for seed in manifest_candidates if seed in selected_candidates]
    else:
        candidates = manifest_candidates

    if not candidates:
        raise ValueError("no candidate seeds selected; check --candidate-seeds and manifest content")

    # eval seed 仍由 --seeds 单独控制，用于 baseline/probe 统计。
    eval_seeds = parse_int_list_or_half_open_range(args.seeds)
    repeat_counts = parse_int_list(args.repeat_counts)
    actions = parse_str_list(args.actions)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for candidate_seed in candidates:
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
        base_rows: list[dict[str, float]] = []
        bundles = {}
        for seed in eval_seeds:
            bundle = resolve_experiment_bundle(cfg, seed)
            bundles[seed] = bundle
            result = _run_sim(list(bundle.ordered_tasks), bundle.scenario, args.end_time)
            metrics = compute_service_quality_metrics(result)
            base_rows.append({"mode_changes": result.mode_change_count(), "lo_cancellations": result.lo_job_cancellation_count(), **service_metrics_to_row(metrics)})

        baseline_mode = _mean(base_rows, "mode_changes")
        baseline_lo_cancel = _mean(base_rows, "lo_cancellations")
        baseline_lc_loss = _mean(base_rows, "lc_service_loss")
        baseline_hi_miss = _sum(base_rows, "hi_deadline_misses")
        top_indices = choose_single_task_sweep_indices(list(next(iter(bundles.values())).ordered_tasks), args.top_k_tasks, args.budget_floor_ratio)

        for task_idx in top_indices:
            for action in actions:
                for repeat in repeat_counts:
                    probe_rows: list[dict[str, float]] = []
                    invalid_count = 0
                    task_name = list(next(iter(bundles.values())).ordered_tasks)[task_idx].name
                    for seed in eval_seeds:
                        bundle = bundles[seed]
                        adjusted, changed, invalid, _ = static_adjust_single_task(
                            tasks=list(bundle.ordered_tasks),
                            target_idx=task_idx,
                            action=action,
                            repeat_count=repeat,
                            inc_ratio=args.budget_increase_ratio,
                            dec_ratio=args.budget_decrease_ratio,
                            budget_floor_ratio=args.budget_floor_ratio,
                        )
                        if invalid:
                            invalid_count += 1
                        result = _run_sim(adjusted if changed else list(bundle.ordered_tasks), bundle.scenario, args.end_time)
                        metrics = compute_service_quality_metrics(result)
                        probe_rows.append({"mode_changes": result.mode_change_count(), "lo_cancellations": result.lo_job_cancellation_count(), **service_metrics_to_row(metrics)})
                    probe_mode = _mean(probe_rows, "mode_changes")
                    probe_lo = _mean(probe_rows, "lo_cancellations")
                    probe_loss = _mean(probe_rows, "lc_service_loss")
                    probe_hi = _sum(probe_rows, "hi_deadline_misses")
                    row = {
                        "candidate_seed": candidate_seed,
                        "target_task_index": task_idx,
                        "target_task_name": task_name,
                        "single_action": action,
                        "repeat_count": repeat,
                        "invalid_count": invalid_count,
                        "single_task_lo_cancellations_mean": probe_lo,
                        "single_task_mode_changes_mean": probe_mode,
                        "single_task_lc_service_loss_mean": probe_loss,
                        "single_task_hi_deadline_misses_sum": probe_hi,
                        "single_task_relative_lo_cancellation_reduction": safe_relative_reduction(baseline_lo_cancel, probe_lo),
                        "single_task_relative_lc_loss_reduction": safe_relative_reduction(baseline_lc_loss, probe_loss),
                        "single_task_mode_change_delta_ratio": (probe_mode - baseline_mode) / baseline_mode if baseline_mode > 0 else 0.0,
                    }
                    detail_rows.append(row)

        if args.enable_pair_probe:
            pair_indices = top_indices[: max(1, args.pair_top_k)]
            for inc_idx in pair_indices:
                for dec_idx in pair_indices:
                    if inc_idx == dec_idx:
                        continue
                    inc_name = list(next(iter(bundles.values())).ordered_tasks)[inc_idx].name
                    dec_name = list(next(iter(bundles.values())).ordered_tasks)[dec_idx].name
                    for seq_len in (2, 4):
                        probe_rows: list[dict[str, float]] = []
                        for seed in eval_seeds:
                            bundle = bundles[seed]
                            adjusted, _, _, _ = static_adjust_single_task(
                                tasks=list(bundle.ordered_tasks),
                                target_idx=inc_idx,
                                action="increase",
                                repeat_count=seq_len // 2,
                                inc_ratio=args.budget_increase_ratio,
                                dec_ratio=args.budget_decrease_ratio,
                                budget_floor_ratio=args.budget_floor_ratio,
                            )
                            adjusted, _, _, _ = static_adjust_single_task(
                                tasks=adjusted,
                                target_idx=dec_idx,
                                action="decrease",
                                repeat_count=seq_len // 2,
                                inc_ratio=args.budget_increase_ratio,
                                dec_ratio=args.budget_decrease_ratio,
                                budget_floor_ratio=args.budget_floor_ratio,
                            )
                            result = _run_sim(adjusted, bundle.scenario, args.end_time)
                            metrics = compute_service_quality_metrics(result)
                            probe_rows.append({"mode_changes": result.mode_change_count(), "lo_cancellations": result.lo_job_cancellation_count(), **service_metrics_to_row(metrics)})
                        probe_mode = _mean(probe_rows, "mode_changes")
                        probe_lo = _mean(probe_rows, "lo_cancellations")
                        probe_loss = _mean(probe_rows, "lc_service_loss")
                        probe_hi = _sum(probe_rows, "hi_deadline_misses")
                        detail_rows.append(
                            {
                                "candidate_seed": candidate_seed,
                                "target_task_index": inc_idx,
                                "target_task_name": inc_name,
                                "single_action": "pair_alternate",
                                "repeat_count": seq_len,
                                "pair_task_name": dec_name,
                                "invalid_count": 0,
                                "single_task_lo_cancellations_mean": probe_lo,
                                "single_task_mode_changes_mean": probe_mode,
                                "single_task_lc_service_loss_mean": probe_loss,
                                "single_task_hi_deadline_misses_sum": probe_hi,
                                "single_task_relative_lo_cancellation_reduction": safe_relative_reduction(baseline_lo_cancel, probe_lo),
                                "single_task_relative_lc_loss_reduction": safe_relative_reduction(baseline_lc_loss, probe_loss),
                                "single_task_mode_change_delta_ratio": (probe_mode - baseline_mode) / baseline_mode if baseline_mode > 0 else 0.0,
                            }
                        )

        seed_rows = [r for r in detail_rows if int(r["candidate_seed"]) == candidate_seed]
        best_stable = select_single_task_static_best(
            rows=seed_rows,
            baseline_loss=baseline_lc_loss,
            baseline_mode_changes=baseline_mode,
            stable_delta=args.stable_mode_delta,
        )
        best_qos = select_single_task_static_best(rows=seed_rows, baseline_loss=baseline_lc_loss, baseline_mode_changes=baseline_mode)
        stable_found = best_stable is not None and float(best_stable["single_task_lo_cancellations_mean"]) < baseline_lo_cancel
        qos_found = best_qos is not None
        qos_reduction = float(best_qos["single_task_relative_lo_cancellation_reduction"]) if best_qos else 0.0
        summary_rows.append(
            {
                "candidate_seed": candidate_seed,
                "probe_eval_seed_count": len(eval_seeds),
                "probe_end_time": args.end_time,
                "probe_top_k_tasks": args.top_k_tasks,
                "probe_total_candidates": len(seed_rows),
                "probe_valid_candidates": sum(1 for r in seed_rows if float(r["single_task_hi_deadline_misses_sum"]) == 0.0),
                "baseline_lo_cancellations_mean": baseline_lo_cancel,
                "baseline_lc_service_loss_mean": baseline_lc_loss,
                "baseline_mode_changes_mean": baseline_mode,
                "baseline_hi_deadline_misses_sum": baseline_hi_miss,
                "probe_stable_best_found": bool(stable_found),
                "probe_stable_best_action_type": (best_stable or {}).get("single_action", ""),
                "probe_stable_best_target_task_name": (best_stable or {}).get("target_task_name", ""),
                "probe_stable_best_repeat_count": (best_stable or {}).get("repeat_count", ""),
                "probe_stable_best_pair_task_name": (best_stable or {}).get("pair_task_name", ""),
                "probe_stable_best_lo_cancellations_mean": (best_stable or {}).get("single_task_lo_cancellations_mean", ""),
                "probe_stable_best_lc_service_loss_mean": (best_stable or {}).get("single_task_lc_service_loss_mean", ""),
                "probe_stable_best_mode_changes_mean": (best_stable or {}).get("single_task_mode_changes_mean", ""),
                "probe_stable_best_relative_lo_cancellation_reduction": (best_stable or {}).get("single_task_relative_lo_cancellation_reduction", ""),
                "probe_stable_best_relative_lc_loss_reduction": (best_stable or {}).get("single_task_relative_lc_loss_reduction", ""),
                "probe_stable_best_mode_change_delta_ratio": (best_stable or {}).get("single_task_mode_change_delta_ratio", ""),
                "probe_qos_best_found": bool(qos_found),
                "probe_qos_best_relative_lo_cancellation_reduction": qos_reduction,
                "probe_qos_best_mode_change_delta_ratio": (best_qos or {}).get("single_task_mode_change_delta_ratio", ""),
                "probe_tradeoff_risk": bool(qos_found and qos_reduction > 0.05 and not stable_found),
            }
        )

    args.output_detail.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        with args.output_detail.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)
    else:
        args.output_detail.write_text("", encoding="utf-8")

    if summary_rows:
        with args.output_summary.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    else:
        args.output_summary.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
