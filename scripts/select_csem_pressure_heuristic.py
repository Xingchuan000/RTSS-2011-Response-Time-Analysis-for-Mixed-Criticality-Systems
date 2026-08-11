"""Select formal10 C-AMC-sem pressure-threshold parameters on validation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.dqn import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.evaluation.csem_dynamic_baselines import run_pressure_threshold_baseline
from amc_py.metrics import compute_lo_quality_weighted_metrics
from amc_py.rl.feature_config import FeatureConfig


REWARD_MODE = "interval_qos_v2_single_recovery_full_C5_overinc016_abs005"
AGENT_PERIOD = 25_000
ACTION_SPACE = "single"
BUDGET_INCREASE_RATIO = 0.02
BUDGET_DECREASE_RATIO = 0.02
BUDGET_FLOOR_RATIO = 0.9
DEPLOY_CAP_MASK_RATIO = 4.0
DEPLOY_CAP_MASK_CRITICALITY = "lo"
SELECTION_END_TIME_DEFAULT = 20_000_000


def _parse_seeds(raw_value: str) -> list[int]:
    seeds: list[int] = []
    for part in (item.strip() for item in raw_value.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"validation seed range must satisfy begin<=end: {part}")
            seeds.extend(range(begin, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("validation seeds must not be empty")
    return seeds


def _parse_grid(raw_value: str) -> list[float]:
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold grid must not be empty")
    return values


def _build_formal10_config(taskset_seed: int):
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


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        observation_mode="v11_full_10d",
        ema_alpha=0.2,
        overrun_ema_alpha=0.1,
        history_k=8,
        event_window=10,
        max_cost_weight=0.7,
        risk_max_scale=3.0,
        include_safety_margin=True,
    )


def _run_candidate(
    *,
    experiment_config,
    taskset_seed: int,
    validation_seeds: list[int],
    end_time: int,
    u_low: float,
    u_high: float,
) -> dict[str, int | float]:
    rows: list[dict[str, int | float]] = []
    for validation_seed in validation_seeds:
        bundle = resolve_experiment_bundle(experiment_config, validation_seed)
        result = run_pressure_threshold_baseline(
            ordered_tasks=bundle.ordered_tasks,
            u_low=u_low,
            u_high=u_high,
            experiment_config=experiment_config,
            seed=validation_seed,
            end_time=end_time,
            agent_period=AGENT_PERIOD,
            reward_mode=REWARD_MODE,
            action_space=ACTION_SPACE,
            budget_increase_ratio=BUDGET_INCREASE_RATIO,
            budget_decrease_ratio=BUDGET_DECREASE_RATIO,
            include_explicit_noop=False,
            budget_floor_ratio=BUDGET_FLOOR_RATIO,
            forbid_decreasing_hi_budgets=True,
            mask_detail_mode="full",
            enable_deploy_cap_mask=True,
            deploy_cap_mask_ratio=DEPLOY_CAP_MASK_RATIO,
            deploy_cap_mask_criticality=DEPLOY_CAP_MASK_CRITICALITY,
            feature_config=_feature_config(),
            c_amc_sem_xf=0.5,
        )
        quality = compute_lo_quality_weighted_metrics(result.runtime_result)
        rows.append(
            {
                "lo_quality_qos": float(quality.lo_quality_qos),
                "lo_zero_service_ratio": float(quality.lo_zero_service_ratio),
                "lo_equiv_jne": float(quality.lo_equiv_jne),
                "deadline_misses": len(result.runtime_result.deadline_misses),
                "selected_invalid_mask_actions": int(
                    result.debug_statistics["selected_invalid_mask_actions"]
                ),
                "action_rate": (
                    float(result.selected_action_count) / float(result.step_count)
                    if result.step_count > 0
                    else 0.0
                ),
            }
        )

    count = len(rows)
    return {
        "taskset_seed": int(taskset_seed),
        "u_low": float(u_low),
        "u_high": float(u_high),
        "validation_count": count,
        "lo_quality_qos_mean": sum(row["lo_quality_qos"] for row in rows) / count,
        "lo_zero_service_ratio_mean": sum(row["lo_zero_service_ratio"] for row in rows) / count,
        "lo_equiv_jne_mean": sum(row["lo_equiv_jne"] for row in rows) / count,
        "deadline_misses_sum": sum(row["deadline_misses"] for row in rows),
        "selected_invalid_mask_actions_sum": sum(
            row["selected_invalid_mask_actions"] for row in rows
        ),
        "action_rate_mean": sum(row["action_rate"] for row in rows) / count,
    }


def _select_candidate(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    valid = [
        row
        for row in rows
        if int(row["deadline_misses_sum"]) == 0
        and int(row["selected_invalid_mask_actions_sum"]) == 0
    ]
    if not valid:
        raise RuntimeError("no pressure-threshold candidate passed validation filters")
    # Stable final tie-breakers: lower action rate, then wider noop band.
    return max(
        valid,
        key=lambda row: (
            float(row["lo_quality_qos_mean"]),
            -float(row["lo_zero_service_ratio_mean"]),
            -float(row["lo_equiv_jne_mean"]),
            -float(row["action_rate_mean"]),
            float(row["u_high"]) - float(row["u_low"]),
            -float(row["u_low"]),
            float(row["u_high"]),
        ),
    )


def _write_candidates(path: Path, rows: list[dict[str, int | float]]) -> None:
    fieldnames = [
        "taskset_seed",
        "u_low",
        "u_high",
        "validation_count",
        "lo_quality_qos_mean",
        "lo_zero_service_ratio_mean",
        "lo_equiv_jne_mean",
        "deadline_misses_sum",
        "selected_invalid_mask_actions_sum",
        "action_rate_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset-seed", type=int, required=True)
    parser.add_argument("--validation-seeds", default="1400:1419")
    parser.add_argument("--end-time", type=int, default=SELECTION_END_TIME_DEFAULT)
    parser.add_argument("--u-low-grid", default="0.80,0.90,1.00")
    parser.add_argument("--u-high-grid", default="1.00,1.10,1.20")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    validation_seeds = _parse_seeds(args.validation_seeds)
    u_low_grid = _parse_grid(args.u_low_grid)
    u_high_grid = _parse_grid(args.u_high_grid)
    candidates: list[dict[str, int | float]] = []
    experiment_config = _build_formal10_config(args.taskset_seed)
    for u_low in u_low_grid:
        for u_high in u_high_grid:
            if u_low >= u_high:
                continue
            candidates.append(
                _run_candidate(
                    experiment_config=experiment_config,
                    taskset_seed=args.taskset_seed,
                    validation_seeds=validation_seeds,
                    end_time=args.end_time,
                    u_low=u_low,
                    u_high=u_high,
                )
            )
    if not candidates:
        raise ValueError("u-low-grid/u-high-grid contain no pair with u_low < u_high")

    selected = _select_candidate(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_candidates(args.output_dir / "heuristic_candidates.csv", candidates)
    selected_payload = {
        "taskset_seed": int(args.taskset_seed),
        "selection_seeds": str(args.validation_seeds),
        "selection_end_time": int(args.end_time),
        "u_low": float(selected["u_low"]),
        "u_high": float(selected["u_high"]),
        "selection_metric": "lo_quality_qos",
        "selection_metric_value": float(selected["lo_quality_qos_mean"]),
        "validation_lo_zero_service_ratio": float(selected["lo_zero_service_ratio_mean"]),
        "validation_lo_equiv_jne": float(selected["lo_equiv_jne_mean"]),
        "action_rate": float(selected["action_rate_mean"]),
    }
    (args.output_dir / "selected_pressure_heuristic.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
