"""Generate standalone MC-Stratified-Dynamic candidate manifests.

The CLI only constructs the new workload provider and runs the public
schedulability analysis for reporting.  It does not invoke a controller,
training loop, legacy selector, or another taskset generator.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from amc_py.experiments import evaluate_taskset
from amc_py.workloads.mc_stratified_dynamic import (
    MCStratifiedDynamicWorkloadConfig,
    MCStratifiedDynamicWorkloadProvider,
)


MANIFEST_SCHEMA_VERSION = "mc_stratified_dynamic_manifest_v1"

MANIFEST_FIELDS = [
    "schema_version",
    "candidate_seed",
    "taskset_seed",
    "scenario_seed",
    "period_family",
    "num_tasks",
    "num_hi",
    "num_lo",
    "total_util_target",
    "total_util_actual",
    "criticality_factor_mean",
    "criticality_factor_min",
    "criticality_factor_max",
    "initial_budget_util_total",
    "initial_budget_util_hi",
    "initial_budget_util_lo",
    "amc_rtb_schedulable",
    "amc_rtb_min_slack",
    "attempts",
    "generator_config_hash",
]

REJECTION_FIELDS = [
    "schema_version",
    "candidate_seed",
    "rejection_reason",
]


def _config_hash(config: MCStratifiedDynamicWorkloadConfig) -> str:
    payload = asdict(config)
    payload.pop("seed", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    defaults = MCStratifiedDynamicWorkloadConfig()
    parser = argparse.ArgumentParser(
        description="Generate mc_stratified_dynamic candidate tasksets and a reproducible CSV manifest."
    )
    parser.add_argument("--candidate-seed-start", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=defaults.num_tasks)
    parser.add_argument("--hi-ratio", type=float, default=defaults.hi_ratio)
    parser.add_argument(
        "--period-family",
        choices=("semi_harmonic", "log_uniform", "seed_paired"),
        default=defaults.period_family,
    )
    parser.add_argument("--period-scale", type=int, default=defaults.period_scale)
    parser.add_argument("--tick-ns", type=int, default=defaults.tick_ns)

    parser.add_argument("--total-util-min", type=float, default=defaults.total_util_min)
    parser.add_argument("--total-util-max", type=float, default=defaults.total_util_max)
    parser.add_argument(
        "--criticality-factor-min", type=float, default=defaults.criticality_factor_min
    )
    parser.add_argument(
        "--criticality-factor-max", type=float, default=defaults.criticality_factor_max
    )
    parser.add_argument("--max-task-util", type=float, default=defaults.max_task_util)

    parser.add_argument(
        "--lo-budget-quantile-min", type=float, default=defaults.lo_budget_quantile_min
    )
    parser.add_argument(
        "--lo-budget-quantile-max", type=float, default=defaults.lo_budget_quantile_max
    )
    parser.add_argument(
        "--hi-budget-quantile-min", type=float, default=defaults.hi_budget_quantile_min
    )
    parser.add_argument(
        "--hi-budget-quantile-max", type=float, default=defaults.hi_budget_quantile_max
    )

    parser.add_argument(
        "--normal-cost-ratio-min", type=float, default=defaults.normal_cost_ratio_min
    )
    parser.add_argument(
        "--normal-cost-ratio-max", type=float, default=defaults.normal_cost_ratio_max
    )
    parser.add_argument(
        "--lo-stress-cost-ratio-min", type=float, default=defaults.lo_stress_cost_ratio_min
    )
    parser.add_argument(
        "--lo-stress-cost-ratio-max", type=float, default=defaults.lo_stress_cost_ratio_max
    )
    parser.add_argument(
        "--hi-stress-cost-ratio-min", type=float, default=defaults.hi_stress_cost_ratio_min
    )
    parser.add_argument(
        "--hi-stress-cost-ratio-max", type=float, default=defaults.hi_stress_cost_ratio_max
    )

    parser.add_argument("--lo-stress-duty-min", type=float, default=defaults.lo_stress_duty_min)
    parser.add_argument("--lo-stress-duty-max", type=float, default=defaults.lo_stress_duty_max)
    parser.add_argument("--lo-stress-dwell-min", type=float, default=defaults.lo_stress_dwell_min)
    parser.add_argument("--lo-stress-dwell-max", type=float, default=defaults.lo_stress_dwell_max)
    parser.add_argument("--hi-stress-duty-min", type=float, default=defaults.hi_stress_duty_min)
    parser.add_argument("--hi-stress-duty-max", type=float, default=defaults.hi_stress_duty_max)
    parser.add_argument("--hi-stress-dwell-min", type=float, default=defaults.hi_stress_dwell_min)
    parser.add_argument("--hi-stress-dwell-max", type=float, default=defaults.hi_stress_dwell_max)

    parser.add_argument(
        "--log-uniform-period-min-ms",
        type=int,
        default=defaults.log_uniform_period_min_ms,
    )
    parser.add_argument(
        "--log-uniform-period-max-ms",
        type=int,
        default=defaults.log_uniform_period_max_ms,
    )
    parser.add_argument("--require-schedulable", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=defaults.max_attempts)
    parser.add_argument("--sched-method", default=defaults.sched_method)
    parser.add_argument("--priority-policy", default=defaults.priority_policy)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-rejections", type=Path, required=True)
    return parser


def _config_from_args(args: argparse.Namespace, seed: int) -> MCStratifiedDynamicWorkloadConfig:
    return MCStratifiedDynamicWorkloadConfig(
        seed=seed,
        num_tasks=args.num_tasks,
        hi_ratio=args.hi_ratio,
        period_family=args.period_family,
        period_scale=args.period_scale,
        tick_ns=args.tick_ns,
        total_util_min=args.total_util_min,
        total_util_max=args.total_util_max,
        criticality_factor_min=args.criticality_factor_min,
        criticality_factor_max=args.criticality_factor_max,
        max_task_util=args.max_task_util,
        lo_budget_quantile_min=args.lo_budget_quantile_min,
        lo_budget_quantile_max=args.lo_budget_quantile_max,
        hi_budget_quantile_min=args.hi_budget_quantile_min,
        hi_budget_quantile_max=args.hi_budget_quantile_max,
        normal_cost_ratio_min=args.normal_cost_ratio_min,
        normal_cost_ratio_max=args.normal_cost_ratio_max,
        lo_stress_cost_ratio_min=args.lo_stress_cost_ratio_min,
        lo_stress_cost_ratio_max=args.lo_stress_cost_ratio_max,
        hi_stress_cost_ratio_min=args.hi_stress_cost_ratio_min,
        hi_stress_cost_ratio_max=args.hi_stress_cost_ratio_max,
        lo_stress_duty_min=args.lo_stress_duty_min,
        lo_stress_duty_max=args.lo_stress_duty_max,
        lo_stress_dwell_min=args.lo_stress_dwell_min,
        lo_stress_dwell_max=args.lo_stress_dwell_max,
        hi_stress_duty_min=args.hi_stress_duty_min,
        hi_stress_duty_max=args.hi_stress_duty_max,
        hi_stress_dwell_min=args.hi_stress_dwell_min,
        hi_stress_dwell_max=args.hi_stress_dwell_max,
        log_uniform_period_min_ms=args.log_uniform_period_min_ms,
        log_uniform_period_max_ms=args.log_uniform_period_max_ms,
        require_schedulable=args.require_schedulable,
        max_attempts=args.max_attempts,
        sched_method=args.sched_method,
        priority_policy=args.priority_policy,
    )


def _schedulability_fields(tasks: tuple[Any, ...]) -> tuple[bool, float]:
    result = evaluate_taskset(
        list(tasks),
        method="amc_rtb",
        priority_policy="dm",
    )
    if not result.schedulable:
        return False, -1.0
    slacks = [
        task.deadline - result.response_times[task.name]
        for task in tasks
    ]
    return True, float(min(slacks)) if slacks else 0.0


def _row_from_bundle(
    *,
    candidate_seed: int,
    config_hash: str,
    bundle: Any,
    schedulable: bool,
    min_slack: float,
) -> dict[str, Any]:
    metadata = bundle.metadata or {}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "candidate_seed": candidate_seed,
        "taskset_seed": bundle.taskset_seed,
        "scenario_seed": bundle.scenario_seed,
        "period_family": metadata["period_family"],
        "num_tasks": metadata["num_tasks"],
        "num_hi": metadata["num_hi"],
        "num_lo": metadata["num_lo"],
        "total_util_target": metadata["total_util_target"],
        "total_util_actual": metadata["total_util_actual"],
        "criticality_factor_mean": metadata["criticality_factor_mean"],
        "criticality_factor_min": metadata["criticality_factor_min"],
        "criticality_factor_max": metadata["criticality_factor_max"],
        "initial_budget_util_total": metadata["initial_budget_util_total"],
        "initial_budget_util_hi": metadata["initial_budget_util_hi"],
        "initial_budget_util_lo": metadata["initial_budget_util_lo"],
        "amc_rtb_schedulable": schedulable,
        "amc_rtb_min_slack": min_slack,
        "attempts": bundle.attempts,
        "generator_config_hash": config_hash,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.num_candidates <= 0:
        raise ValueError("num-candidates must be > 0")

    manifest_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for candidate_index in range(args.num_candidates):
        candidate_seed = args.candidate_seed_start + candidate_index
        try:
            config = _config_from_args(args, candidate_seed)
            config_hash = _config_hash(config)
            provider = MCStratifiedDynamicWorkloadProvider(config)
            bundle = provider.build(candidate_seed)
            schedulable, min_slack = _schedulability_fields(bundle.tasks)
            manifest_rows.append(
                _row_from_bundle(
                    candidate_seed=candidate_seed,
                    config_hash=config_hash,
                    bundle=bundle,
                    schedulable=schedulable,
                    min_slack=min_slack,
                )
            )
        except (RuntimeError, ValueError) as exc:
            rejection_rows.append(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "candidate_seed": candidate_seed,
                    "rejection_reason": str(exc),
                }
            )

    _write_csv(args.output_manifest, MANIFEST_FIELDS, manifest_rows)
    _write_csv(args.output_rejections, REJECTION_FIELDS, rejection_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
