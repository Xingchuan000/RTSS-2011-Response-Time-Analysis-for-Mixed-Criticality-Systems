"""Select a validation-tuned static budget for the formal10 C-AMC-sem workload."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.dqn.experiment import (
    build_mc_fairgen_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.evaluation.csem_static_baseline import (
    StaticBudgetSelection,
    build_static_lo_budgets,
    check_static_budget_feasibility,
    run_static_budget_baseline,
    save_static_budget_selection,
)
from amc_py.metrics import compute_lo_quality_weighted_metrics
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


def build_parser() -> argparse.ArgumentParser:
    """Build the formal10-only selection CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset-seed", type=int, required=True)
    parser.add_argument("--validation-seeds", default="1400:1419")
    parser.add_argument("--end-time", type=int, default=20_000_000)
    parser.add_argument(
        "--alpha-grid",
        default="1.00,1.25,1.50,1.75,2.00,2.50,3.00,3.50,4.00",
    )
    parser.add_argument("--c-amc-sem-xf", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _parse_range(text: str) -> list[int]:
    """Parse the fixed ``begin:end`` validation-seed format."""

    begin, end = (int(x) for x in text.split(":"))
    if end < begin:
        raise ValueError("validation seed range must satisfy begin <= end")
    return list(range(begin, end + 1))


def _parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated alpha grid."""

    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("alpha grid must not be empty")
    return values


def _build_formal10_config(taskset_seed: int):
    """Build the fixed formal10 mc_fairgen workload configuration."""

    return build_mc_fairgen_experiment_config(
        mode="paper_learnable_headroom",
        num_tasks=12,
        hi_ratio=0.5,
        period_source="controlled_medium",
        period_scale=500,
        require_schedulable=True,
        scenario_seed_offset=100000,
        fixed_taskset_seed=taskset_seed,
        u_hi_lo_min=0.20,
        u_hi_lo_max=0.35,
        u_hi_hi_min=0.45,
        u_hi_hi_max=0.70,
        u_lo_lo_min=0.25,
        u_lo_lo_max=0.45,
        hi_budget_rho_min=0.55,
        hi_budget_rho_max=0.75,
        lo_budget_rho_min=0.20,
        lo_budget_rho_max=0.40,
        hi_overrun_prob=0.08,
        lo_overrun_prob=0.12,
        hi_overrun_factor_min=1.02,
        hi_overrun_factor_max=1.25,
        lo_overrun_factor_min=1.02,
        lo_overrun_factor_max=1.25,
    )


def _runtime_config(end_time: int, xf: float) -> RuntimeConfig:
    """Build the formal10 C-AMC-sem runtime configuration."""

    return RuntimeConfig(
        end_time=end_time,
        semantics=RuntimeSemantics.C_AMC_SEM,
        capture_trace=False,
        capture_debug_events=False,
        record_dropped_lo_releases=True,
        drop_lo_jobs_on_hi_switch=False,
        c_amc_sem_lo_degradation_ratio=xf,
        c_amc_sem_primary_on_switch_time=True,
    )


def _evaluate_alpha(
    *,
    experiment_config,
    taskset_seed: int,
    validation_seeds: list[int],
    end_time: int,
    xf: float,
    alpha: float,
) -> dict[str, object]:
    """Evaluate one alpha, skipping validation when safety rejects it."""

    probe_bundle = resolve_experiment_bundle(experiment_config, validation_seeds[0])
    budgets = build_static_lo_budgets(probe_bundle.ordered_tasks, alpha=alpha)
    report = check_static_budget_feasibility(probe_bundle.ordered_tasks, budgets)
    if not report.accepted:
        return {
            "taskset_seed": taskset_seed,
            "alpha": alpha,
            "safety_feasible": False,
            "safety_reason": report.reason,
            "validation_count": 0,
            "lo_quality_qos_mean": "",
            "lo_zero_service_ratio_mean": "",
            "lo_equiv_jne_mean": "",
            "deadline_misses_sum": "",
        }

    qos_values: list[float] = []
    zero_values: list[float] = []
    jne_values: list[float] = []
    miss_sum = 0

    for seed in validation_seeds:
        bundle = resolve_experiment_bundle(experiment_config, seed)
        run = run_static_budget_baseline(
            ordered_tasks=bundle.ordered_tasks,
            scenario=bundle.scenario,
            runtime_config=_runtime_config(end_time, xf),
            alpha=alpha,
        )
        metrics = compute_lo_quality_weighted_metrics(run.runtime_result)
        qos_values.append(float(metrics.lo_quality_qos))
        zero_values.append(float(metrics.lo_zero_service_ratio))
        jne_values.append(float(metrics.lo_equiv_jne))
        miss_sum += len(run.runtime_result.deadline_misses)

    return {
        "taskset_seed": taskset_seed,
        "alpha": alpha,
        "safety_feasible": True,
        "safety_reason": "accepted",
        "validation_count": len(validation_seeds),
        "lo_quality_qos_mean": mean(qos_values),
        "lo_zero_service_ratio_mean": mean(zero_values),
        "lo_equiv_jne_mean": mean(jne_values),
        "deadline_misses_sum": miss_sum,
    }


def _selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    """Return the lexicographic key specified by Plan 01."""

    return (
        float(row["lo_quality_qos_mean"]),
        -float(row["lo_zero_service_ratio_mean"]),
        -float(row["lo_equiv_jne_mean"]),
        -float(row["alpha"]),
    )


def main() -> None:
    """Run static alpha selection for one fixed formal10 taskset."""

    args = build_parser().parse_args()
    validation_seeds = _parse_range(args.validation_seeds)
    alpha_grid = _parse_float_list(args.alpha_grid)
    if not validation_seeds:
        raise ValueError("validation seeds must not be empty")
    if args.end_time <= 0:
        raise ValueError("end time must be positive")
    if not 0.0 < args.c_amc_sem_xf <= 1.0:
        raise ValueError("c-amc-sem xf must be in (0, 1]")

    experiment_config = _build_formal10_config(args.taskset_seed)
    rows = [
        _evaluate_alpha(
            experiment_config=experiment_config,
            taskset_seed=args.taskset_seed,
            validation_seeds=validation_seeds,
            end_time=args.end_time,
            xf=args.c_amc_sem_xf,
            alpha=alpha,
        )
        for alpha in alpha_grid
    ]

    valid_rows = [
        row
        for row in rows
        if bool(row["safety_feasible"])
        and int(row["deadline_misses_sum"]) == 0
    ]
    if not valid_rows:
        raise RuntimeError("NO_FEASIBLE_STATIC_BASELINE_CANDIDATE")
    selected = max(valid_rows, key=_selection_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "static_candidates.csv"
    fieldnames = [
        "taskset_seed",
        "alpha",
        "safety_feasible",
        "safety_reason",
        "validation_count",
        "lo_quality_qos_mean",
        "lo_zero_service_ratio_mean",
        "lo_equiv_jne_mean",
        "deadline_misses_sum",
    ]
    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bundle = resolve_experiment_bundle(experiment_config, validation_seeds[0])
    selected_budgets = build_static_lo_budgets(
        bundle.ordered_tasks,
        alpha=float(selected["alpha"]),
    )
    selection = StaticBudgetSelection(
        taskset_seed=args.taskset_seed,
        selection_seeds=args.validation_seeds,
        selection_end_time=args.end_time,
        alpha=float(selected["alpha"]),
        selection_metric="lo_quality_qos_mean",
        selection_metric_value=float(selected["lo_quality_qos_mean"]),
        validation_lo_zero_service_ratio=float(selected["lo_zero_service_ratio_mean"]),
        validation_lo_equiv_jne=float(selected["lo_equiv_jne_mean"]),
        budgets=selected_budgets,
    )
    save_static_budget_selection(
        selection,
        args.output_dir / "selected_static_budget.json",
    )


if __name__ == "__main__":
    main()
