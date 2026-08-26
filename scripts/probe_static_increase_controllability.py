#!/usr/bin/env python3
"""Static increase controllability / oracle probe.

This script is intentionally independent from DQN training. It asks:

    If we statically increase the LO tasks that are valid increase candidates
    and have the largest baseline LO-cancellation contribution, how much LO
    cancellation / LC service loss can be reduced?

It is designed as the formal counterpart of `probe_static_decrease_controllability.py`.
Compared with the increase reference embedded in the decrease probe, this script:

1. distinguishes rank-1/rank-2/rank-3 valid-increase cancellation sources;
2. supports multi-target oracle probes: top2 and top3 valid-increase sources;
3. records valid-eval fraction instead of silently treating invalid probes as
   normal zero-improvement rows;
4. selects best probes only from rows satisfying a minimum valid-eval fraction;
5. outputs detail + summary CSVs suitable for action-representation diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics, safe_relative_reduction, service_metrics_to_row
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.task_level_diagnostics import summarize_task_level_cancellations
from scripts.scan_qos_pressure_tasksets import parse_int_list, parse_int_list_or_half_open_range


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--workload", choices=["mc_fairgen"], default="mc_fairgen")
    parser.add_argument("--taskset-manifest", type=Path, default=None)
    parser.add_argument("--manifest-seed-column", type=str, default="candidate_seed")
    parser.add_argument("--candidate-seeds", type=str, default="")
    parser.add_argument("--seeds", type=str, default="200:220")
    parser.add_argument("--end-time", type=int, default=1_000_000)

    parser.add_argument("--budget-increase-ratio", type=float, default=0.025)
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    parser.add_argument("--repeat-counts", type=str, default="1,2,3")
    parser.add_argument("--top-k-valid-increase", type=int, default=3)
    parser.add_argument("--min-valid-eval-fraction", type=float, default=0.8)
    parser.add_argument("--stable-mode-delta", type=float, default=0.05)
    parser.add_argument("--meaningful-threshold-small", type=float, default=0.003)
    parser.add_argument("--meaningful-threshold-large", type=float, default=0.005)
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


def _parse_manifest(path: Path, seed_column: str) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[int] = []
    seen: set[int] = set()
    for row in rows:
        raw = str(row.get(seed_column, "")).strip()
        if not raw:
            continue
        seed = int(float(raw))
        if seed not in seen:
            seen.add(seed)
            out.append(seed)
    return out


def _is_hi(task: Task) -> bool:
    """Robustly detect HI criticality.

    Do not rely on identity comparison only, because task.criticality may come
    from an equivalent enum/value representation in different modules.
    """
    crit = getattr(task, "criticality", None)
    name = str(getattr(crit, "name", crit)).strip().lower()
    value = str(getattr(crit, "value", "")).strip().lower()
    text = str(crit).strip().lower()
    return (
        name in {"hi", "high"}
        or value in {"hi", "high"}
        or text.endswith(".hi")
        or text in {"hi", "high"}
    )



def _run_sim(tasks: list[Task], scenario, end_time: int):
    return simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _increase_bounds(task: Task) -> int:
    """Return the static increase upper bound used by the oracle probe.

    HI tasks are bounded by min(C_HI, deadline).  For LO tasks, C_HI is only a
    placeholder in this codebase and is initially equal to C_LO, but the Task
    dataclass still enforces c_hi >= c_lo.  The oracle therefore uses deadline
    as the LO upper bound and, when applying the change, raises the placeholder
    c_hi together with c_lo to keep the dataclass invariant valid.
    """
    if _is_hi(task):
        return min(int(task.c_hi), int(task.deadline))
    return int(task.deadline)

def _increase_headroom(task: Task) -> int:
    return max(0, _increase_bounds(task) - int(task.c_lo))


def _apply_static_increase_to_tasks(
    *,
    tasks: list[Task],
    target_indices: list[int],
    repeat_count: int,
    inc_ratio: float,
) -> tuple[list[Task], bool, bool, str, dict[int, int], dict[int, int]]:
    """Apply DQN-style repeated increase to one or more target tasks.

    Returns:
    - adjusted tasks;
    - changed: at least one task budget changed;
    - invalid: at least one requested target could not complete all repeat steps;
    - invalid_reason;
    - before budgets by task index;
    - after budgets by task index.

    A row may be both changed and invalid when repeat_count is too large and
    clips at the upper bound after one or more successful steps. The detail CSV
    records valid_eval_fraction so downstream analysis can filter these cases.
    """

    adjusted = list(tasks)
    changed = False
    invalid = False
    reasons: list[str] = []
    before: dict[int, int] = {}
    after: dict[int, int] = {}

    for target_idx in target_indices:
        task = adjusted[int(target_idx)]
        current = int(task.c_lo)
        before[int(target_idx)] = current
        upper_bound = _increase_bounds(task)

        for _ in range(int(repeat_count)):
            if current >= upper_bound:
                invalid = True
                reasons.append(f"task{target_idx}:c_lo_at_or_above_upper_bound")
                break
            new_value = int(math.ceil(current * (1.0 + float(inc_ratio))))
            current = min(new_value, upper_bound)
            current = max(1, int(current))

        if current != int(task.c_lo):
            changed = True
        after[int(target_idx)] = current
        if _is_hi(task):
            adjusted[int(target_idx)] = replace(task, c_lo=int(current))
        else:
            # c_hi is not semantically used for LO tasks, but Task requires
            # c_hi >= c_lo. Keep the placeholder consistent after an increase.
            adjusted[int(target_idx)] = replace(task, c_lo=int(current), c_hi=max(int(task.c_hi), int(current)))

    return adjusted, changed, invalid, ";".join(reasons), before, after


def _build_task_aggregate(
    *,
    ordered_tasks: list[Task],
    per_seed_task_rows: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows_by_index: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(ordered_tasks))}
    for rows in per_seed_task_rows:
        for row in rows:
            rows_by_index[int(row["task_index"])].append(row)


    merged: list[dict[str, Any]] = []
    for idx, task in enumerate(ordered_tasks):
        rows = rows_by_index.get(idx, [])
        released_jobs_mean = _mean([_to_float(item.get("released_jobs", 0.0)) for item in rows])
        cancelled_jobs_mean = _mean([_to_float(item.get("cancelled_jobs", 0.0)) for item in rows])
        cancel_share_mean = _mean([_to_float(item.get("cancel_share_of_total", 0.0)) for item in rows])
        cancel_ratio_over_released_mean = _mean([_to_float(item.get("cancel_ratio_over_released", 0.0)) for item in rows])
        cancelled_per_1m_mean = _mean([_to_float(item.get("cancelled_per_1m", 0.0)) for item in rows])

        is_valid_increase_union = any(
            _to_bool(item.get("is_valid_increase_union", False))
            or _to_bool(item.get("is_valid_increase", False))
            for item in rows
        )
        valid_increase_seen_fraction = _mean([
            _to_float(item.get("valid_increase_seen_fraction", 0.0))
            for item in rows
        ])
        is_valid_decrease_union = any(
            _to_bool(item.get("is_valid_decrease_union", False))
            or _to_bool(item.get("is_valid_decrease", False))
            for item in rows
        )
        valid_decrease_seen_fraction = _mean([
            _to_float(item.get("valid_decrease_seen_fraction", 0.0))
            for item in rows
        ])

        increase_headroom = _increase_headroom(task)

        merged.append(
            {
                "task_index": int(idx),
                "task_name": str(task.name),
                "criticality": getattr(task.criticality, "name", str(task.criticality)),
                "period": int(task.period),
                "deadline": int(task.deadline),
                "c_lo": int(task.c_lo),
                "c_hi": int(task.c_hi),
                "released_jobs_mean": released_jobs_mean,
                "cancelled_jobs_mean": cancelled_jobs_mean,
                "cancel_share_mean": cancel_share_mean,
                "cancel_ratio_over_released_mean": cancel_ratio_over_released_mean,
                "cancelled_per_1m_mean": cancelled_per_1m_mean,
                "increase_headroom": int(increase_headroom),
                "increase_upper_bound": int(_increase_bounds(task)),
                "is_lo": not _is_hi(task),
                "is_valid_increase_union": bool(is_valid_increase_union),
                "valid_increase_seen_fraction": float(valid_increase_seen_fraction),
                "is_valid_decrease_union": bool(is_valid_decrease_union),
                "valid_decrease_seen_fraction": float(valid_decrease_seen_fraction),
            }
        )
    return merged


def _valid_increase_sources(task_agg: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Pick LO increase cancellation sources.

    Important fix:
    this standalone static probe builds per-task rows from
    summarize_task_level_cancellations(), which does not run the RL action-mask
    diagnostic and therefore cannot truly populate valid-increase fields.
    The previous implementation required those fields to be true, so
    candidates were always empty and detail_rows was never written.

    Selection is now two-stage:
    1. use strict valid-increase flags if any task actually has them;
    2. otherwise fall back to the static oracle criterion: LO task + positive
       cancellation pressure + positive increase headroom.
    """


    def _base_ok(row: dict[str, Any]) -> bool:
        is_lo = bool(row.get("is_lo", False)) or "lo" in str(row.get("task_name", "")).lower()
        has_cancellation = (
            _to_float(row.get("cancelled_jobs_mean", 0.0)) > 0.0
            or _to_float(row.get("cancel_share_mean", 0.0)) > 0.0
            or _to_float(row.get("cancel_ratio_over_released_mean", 0.0)) > 0.0
        )
        has_headroom = int(row.get("increase_headroom", 0)) > 0
        return is_lo and has_cancellation and has_headroom

    def _strict_valid(row: dict[str, Any]) -> bool:
        return (
            _to_bool(row.get("is_valid_increase_union", False))
            or _to_float(row.get("valid_increase_seen_fraction", 0.0)) > 0.0
            or _to_bool(row.get("is_valid_increase", False))
        )

    strict_candidates = [row for row in task_agg if _base_ok(row) and _strict_valid(row)]
    fallback_candidates = [row for row in task_agg if _base_ok(row)]
    candidates = strict_candidates if strict_candidates else fallback_candidates

    candidates = sorted(
        candidates,
        key=lambda item: (
            _to_float(item.get("cancel_share_mean", 0.0)),
            _to_float(item.get("cancelled_jobs_mean", 0.0)),
            _to_float(item.get("valid_increase_seen_fraction", 0.0)),
            int(item.get("increase_headroom", 0)),
            -int(item.get("task_index", 0)),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(top_k))]

def _build_probe_specs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    if len(sources) >= 1:
        specs.append({"probe_type": "increase_top_valid_cancel_source", "targets": [sources[0]]})
    if len(sources) >= 2:
        specs.append({"probe_type": "increase_second_valid_cancel_source", "targets": [sources[1]]})
    if len(sources) >= 3:
        specs.append({"probe_type": "increase_third_valid_cancel_source", "targets": [sources[2]]})
    if len(sources) >= 2:
        specs.append({"probe_type": "increase_top2_valid_cancel_sources", "targets": sources[:2]})
    if len(sources) >= 3:
        specs.append({"probe_type": "increase_top3_valid_cancel_sources", "targets": sources[:3]})

    return specs


def _select_best(rows: list[dict[str, Any]], *, require_stable: bool, min_valid_eval_fraction: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if float(row.get("valid_eval_fraction", 0.0)) >= float(min_valid_eval_fraction)
        and (bool(row.get("stable_candidate", False)) if require_stable else True)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item["lo_cancellation_rel_reduction"]),
            float(item["lc_service_loss_abs_reduction"]),
            -float(item["mode_change_delta_ratio"]),
            float(item["valid_eval_fraction"]),
        ),
        reverse=True,
    )[0]


def _best_with_fallback(rows: list[dict[str, Any]], *, min_valid_eval_fraction: float) -> dict[str, Any] | None:
    stable = _select_best(rows, require_stable=True, min_valid_eval_fraction=min_valid_eval_fraction)
    if stable is not None:
        return stable
    return _select_best(rows, require_stable=False, min_valid_eval_fraction=min_valid_eval_fraction)


def _summary_from_best(best: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if best is None:
        return {
            f"{prefix}_probe_type": "",
            f"{prefix}_target_task_names": "",
            f"{prefix}_repeat_count": "",
            f"{prefix}_rel_lo_reduction": 0.0,
            f"{prefix}_lc_loss_reduction": 0.0,
            f"{prefix}_mode_delta_ratio": 0.0,
            f"{prefix}_stable_candidate": False,
            f"{prefix}_valid_eval_fraction": 0.0,
        }
    return {
        f"{prefix}_probe_type": str(best["probe_type"]),
        f"{prefix}_target_task_names": str(best["target_task_names"]),
        f"{prefix}_repeat_count": int(best["repeat_count"]),
        f"{prefix}_rel_lo_reduction": float(best["lo_cancellation_rel_reduction"]),
        f"{prefix}_lc_loss_reduction": float(best["lc_service_loss_abs_reduction"]),
        f"{prefix}_mode_delta_ratio": float(best["mode_change_delta_ratio"]),
        f"{prefix}_stable_candidate": bool(best["stable_candidate"]),
        f"{prefix}_valid_eval_fraction": float(best["valid_eval_fraction"]),
    }


def _build_mc_config(args: argparse.Namespace, candidate_seed: int):
    return build_mc_fairgen_experiment_config(
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


def main() -> None:
    args = parse_args()
    eval_seeds = parse_int_list_or_half_open_range(args.seeds)
    if not eval_seeds:
        raise ValueError("--seeds 不能为空")
    repeat_counts = parse_int_list(args.repeat_counts)
    if not repeat_counts:
        raise ValueError("--repeat-counts 不能为空")

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
        cfg = _build_mc_config(args, int(candidate_seed))

        bundles_by_seed: dict[int, Any] = {}
        baseline_rows: list[dict[str, float]] = []
        per_seed_task_rows: list[list[dict[str, Any]]] = []

        for eval_seed in eval_seeds:
            bundle = resolve_experiment_bundle(cfg, int(eval_seed))
            bundles_by_seed[int(eval_seed)] = bundle
            base_result = _run_sim(list(bundle.ordered_tasks), bundle.scenario, int(args.end_time))
            base_metrics = compute_service_quality_metrics(base_result)
            baseline_rows.append(
                {
                    "mode_changes": float(base_result.mode_change_count()),
                    "lo_cancellations": float(base_result.lo_job_cancellation_count()),
                    **service_metrics_to_row(base_metrics),
                }
            )
            diag = summarize_task_level_cancellations(ordered_tasks=list(bundle.ordered_tasks), result=base_result)
            per_seed_task_rows.append(list(diag.get("per_task_rows", [])))

        baseline_lo_mean = mean(float(row["lo_cancellations"]) for row in baseline_rows)
        baseline_lc_loss_mean = mean(float(row["lc_service_loss"]) for row in baseline_rows)
        baseline_mode_mean = mean(float(row["mode_changes"]) for row in baseline_rows)
        baseline_hi_miss_sum = float(sum(float(row["hi_deadline_misses"]) for row in baseline_rows))

        ordered_tasks = list(next(iter(bundles_by_seed.values())).ordered_tasks)
        task_agg = _build_task_aggregate(ordered_tasks=ordered_tasks, per_seed_task_rows=per_seed_task_rows)
        valid_sources = _valid_increase_sources(task_agg, int(args.top_k_valid_increase))
        probe_specs = _build_probe_specs(valid_sources)

        seed_rows: list[dict[str, Any]] = []

        for spec in probe_specs:
            probe_type = str(spec["probe_type"])
            targets: list[dict[str, Any]] = list(spec["targets"])
            target_indices = [int(row["task_index"]) for row in targets]
            target_task_names = ";".join(str(row["task_name"]) for row in targets)
            target_criticalities = ";".join(str(row["criticality"]) for row in targets)
            target_periods = ";".join(str(int(row["period"])) for row in targets)
            target_cancel_share_sum = float(sum(float(row["cancel_share_mean"]) for row in targets))
            target_cancelled_jobs_sum = float(sum(float(row["cancelled_jobs_mean"]) for row in targets))
            target_increase_headroom_sum = int(sum(int(row["increase_headroom"]) for row in targets))

            for repeat_count in repeat_counts:
                invalid_count = 0
                changed_count = 0
                invalid_reasons: list[str] = []
                probe_rows: list[dict[str, float]] = []

                for eval_seed in eval_seeds:
                    bundle = bundles_by_seed[int(eval_seed)]
                    adjusted_tasks, changed, invalid, invalid_reason, _, _ = _apply_static_increase_to_tasks(
                        tasks=list(bundle.ordered_tasks),
                        target_indices=target_indices,
                        repeat_count=int(repeat_count),
                        inc_ratio=float(args.budget_increase_ratio),
                    )
                    if invalid:
                        invalid_count += 1
                        if invalid_reason:
                            invalid_reasons.append(invalid_reason)
                    if changed:
                        changed_count += 1

                    result = _run_sim(
                        adjusted_tasks if changed else list(bundle.ordered_tasks),
                        bundle.scenario,
                        int(args.end_time),
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
                valid_eval_fraction = 1.0 - float(invalid_count) / float(max(1, len(eval_seeds)))
                changed_eval_fraction = float(changed_count) / float(max(1, len(eval_seeds)))

                stable_candidate = (
                    probe_hi_miss_sum == 0.0
                    and mode_delta_ratio <= float(args.stable_mode_delta)
                    and probe_lc_loss_mean <= baseline_lc_loss_mean
                    and valid_eval_fraction >= float(args.min_valid_eval_fraction)
                )
                meaningful_positive_small = stable_candidate and lo_rel >= float(args.meaningful_threshold_small)
                meaningful_positive_large = stable_candidate and lo_rel >= float(args.meaningful_threshold_large)

                row = {
                    "candidate_seed": int(candidate_seed),
                    "probe_type": probe_type,
                    "probe_action": "increase",
                    "target_task_indices": ";".join(str(idx) for idx in target_indices),
                    "target_task_names": target_task_names,
                    "target_criticalities": target_criticalities,
                    "target_periods": target_periods,
                    "target_baseline_cancel_share_sum": float(target_cancel_share_sum),
                    "target_baseline_cancelled_jobs_sum": float(target_cancelled_jobs_sum),
                    "target_increase_headroom_sum": int(target_increase_headroom_sum),
                    "repeat_count": int(repeat_count),
                    "invalid_count": int(invalid_count),
                    "valid_eval_fraction": float(valid_eval_fraction),
                    "changed_count": int(changed_count),
                    "changed_eval_fraction": float(changed_eval_fraction),
                    "invalid_reasons_sample": "|".join(invalid_reasons[:5]),
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
                    "meaningful_positive_003": bool(meaningful_positive_small),
                    "meaningful_positive_005": bool(meaningful_positive_large),
                }
                detail_rows.append(row)
                seed_rows.append(row)

        best_overall = _best_with_fallback(seed_rows, min_valid_eval_fraction=float(args.min_valid_eval_fraction))
        best_single = _best_with_fallback(
            [row for row in seed_rows if str(row["probe_type"]) in {
                "increase_top_valid_cancel_source",
                "increase_second_valid_cancel_source",
                "increase_third_valid_cancel_source",
            }],
            min_valid_eval_fraction=float(args.min_valid_eval_fraction),
        )
        best_top2 = _best_with_fallback(
            [row for row in seed_rows if str(row["probe_type"]) == "increase_top2_valid_cancel_sources"],
            min_valid_eval_fraction=float(args.min_valid_eval_fraction),
        )
        best_top3 = _best_with_fallback(
            [row for row in seed_rows if str(row["probe_type"]) == "increase_top3_valid_cancel_sources"],
            min_valid_eval_fraction=float(args.min_valid_eval_fraction),
        )

        summary = {
            "candidate_seed": int(candidate_seed),
            "num_eval_seeds": len(eval_seeds),
            "probe_end_time": int(args.end_time),
            "num_valid_increase_sources": int(len(valid_sources)),
            "valid_increase_source_task_names": ";".join(str(row["task_name"]) for row in valid_sources),
            "valid_increase_source_cancel_share_sum": float(sum(float(row["cancel_share_mean"]) for row in valid_sources)),
            "valid_increase_source_cancelled_jobs_sum": float(sum(float(row["cancelled_jobs_mean"]) for row in valid_sources)),
            "baseline_lo_cancellations_mean": float(baseline_lo_mean),
            "baseline_lc_service_loss_mean": float(baseline_lc_loss_mean),
            "baseline_mode_changes_mean": float(baseline_mode_mean),
            "baseline_hi_deadline_misses_sum": float(baseline_hi_miss_sum),
        }
        summary.update(_summary_from_best(best_overall, "best_increase"))
        summary.update(_summary_from_best(best_single, "best_single_increase"))
        summary.update(_summary_from_best(best_top2, "best_top2_increase"))
        summary.update(_summary_from_best(best_top3, "best_top3_increase"))
        summary["increase_probe_has_stable_positive"] = bool(
            best_overall is not None
            and bool(best_overall["stable_candidate"])
            and float(best_overall["lo_cancellation_rel_reduction"]) > 0.0
        )
        summary["increase_probe_has_meaningful_positive_003"] = bool(
            best_overall is not None and bool(best_overall.get("meaningful_positive_003", False))
        )
        summary["increase_probe_has_meaningful_positive_005"] = bool(
            best_overall is not None and bool(best_overall.get("meaningful_positive_005", False))
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
        # Keep the file parseable even when no probe specs are generated.
        # This makes the failure mode visible to downstream checks.
        empty_detail_fields = [
            "candidate_seed", "probe_type", "probe_action", "target_task_indices",
            "target_task_names", "target_criticalities", "target_periods",
            "target_baseline_cancel_share_sum", "target_baseline_cancelled_jobs_sum",
            "target_increase_headroom_sum", "repeat_count", "invalid_count",
            "valid_eval_fraction", "changed_count", "changed_eval_fraction",
            "invalid_reasons_sample", "baseline_lo_cancellations_mean",
            "probe_lo_cancellations_mean", "lo_cancellation_abs_reduction",
            "lo_cancellation_rel_reduction", "baseline_lc_service_loss_mean",
            "probe_lc_service_loss_mean", "lc_service_loss_abs_reduction",
            "lc_service_loss_rel_reduction", "baseline_mode_changes_mean",
            "probe_mode_changes_mean", "mode_change_delta", "mode_change_delta_ratio",
            "baseline_hi_deadline_misses_sum", "probe_hi_deadline_misses_sum",
            "stable_candidate", "meaningful_positive_003", "meaningful_positive_005",
        ]
        with args.output_detail.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=empty_detail_fields)
            writer.writeheader()

    if summary_rows:
        with args.output_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        args.output_summary.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
