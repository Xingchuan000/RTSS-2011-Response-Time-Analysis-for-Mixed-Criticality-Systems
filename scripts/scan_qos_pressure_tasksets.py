"""扫描候选 taskset seed 的 baseline QoS pressure，并可选执行 static sweep。

说明：
- baseline 运行时语义固定为 AMC_PLUS；
- static sweep 是 static budget scaling sweep：仅在每次仿真前对任务预算做一次静态缩放，
  不会在运行中按 agent period 动态执行动作；
- 该 sweep 仅用于低成本可学习性代理信号（cheap learnability proxy）。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median

from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics, safe_relative_reduction, service_metrics_to_row
from amc_py.models import Criticality, Task
from amc_py.qos_pressure import classify_qos_pressure_bucket, recommend_for_qos_dqn
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics


def parse_int_list_or_half_open_range(raw: str) -> list[int]:
    """解析整数列表或半开区间，例如 `1,2,3` 或 `200:229`。"""

    text = raw.strip()
    if not text:
        return []
    if ":" in text:
        begin_text, end_text = (token.strip() for token in text.split(":", maxsplit=1))
        begin = int(begin_text)
        end = int(end_text)
        if end <= begin:
            raise ValueError("range 必须满足 end > begin")
        return list(range(begin, end))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    """解析浮点列表，例如 `0,0.015,0.025`。"""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def to_csv_value(value: object) -> str | int | float:
    """统一 CSV 输出：缺失值写空字符串，不写 `None` 文本。"""

    if value is None:
        return ""
    return value


def mean_of(rows: list[dict[str, object]], key: str) -> float | None:
    """对可空数字字段求均值，自动跳过空值。"""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / float(len(values))


def median_of(rows: list[dict[str, object]], key: str) -> float | None:
    """对可空数字字段求中位数，自动跳过空值。"""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    if not values:
        return None
    return float(median(values))


def sum_of(rows: list[dict[str, object]], key: str) -> float | None:
    """对可空数字字段求和，自动跳过空值。"""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values))


def static_adjust_budget_scale(task: Task, inc_ratio: float, dec_ratio: float, budget_floor_ratio: float) -> Task:
    """按 `budget_scale` 策略静态重写预算，保持任务其余属性不变。"""

    if task.criticality is Criticality.HI:
        new_c_lo = min(task.c_hi, round(task.c_lo * (1.0 + inc_ratio)))
    else:
        floor = round(task.c_lo * budget_floor_ratio)
        new_c_lo = max(floor, round(task.c_lo * (1.0 - dec_ratio)))
    new_c_lo = max(1, int(new_c_lo))
    return replace(task, c_lo=new_c_lo)


def run_single_simulation(ordered_tasks: list[Task], scenario, end_time: int):
    """执行一次 AMC_PLUS baseline 仿真并返回结果对象。"""

    return simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )


def scan_candidate_seed(args: argparse.Namespace, candidate_seed: int, eval_seeds: list[int], static_eval_seeds: list[int]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """扫描单个 candidate seed，返回主汇总行和 static detail 行。"""

    main_row: dict[str, object] = {
        "candidate_seed": candidate_seed,
        "amcrtb_schedulable": True,
        "scan_error": "",
        "num_eval_seeds": len(eval_seeds),
        "baseline_runtime_semantics": "AMC_PLUS",
        "end_time": args.end_time,
        "agent_period": args.agent_period,
        "workload": args.workload,
        "mc_fairgen_mode": args.mc_fairgen_mode,
        "mc_fairgen_num_tasks": args.mc_fairgen_num_tasks,
        "mc_fairgen_hi_ratio": args.mc_fairgen_hi_ratio,
        "mc_fairgen_period_source": args.mc_fairgen_period_source,
        "mc_fairgen_period_scale": args.mc_fairgen_period_scale,
        "mc_fairgen_u_hi_lo_min": args.mc_fairgen_u_hi_lo_min,
        "mc_fairgen_u_hi_lo_max": args.mc_fairgen_u_hi_lo_max,
        "mc_fairgen_u_hi_hi_min": args.mc_fairgen_u_hi_hi_min,
        "mc_fairgen_u_hi_hi_max": args.mc_fairgen_u_hi_hi_max,
        "mc_fairgen_u_lo_lo_min": args.mc_fairgen_u_lo_lo_min,
        "mc_fairgen_u_lo_lo_max": args.mc_fairgen_u_lo_lo_max,
        "mc_fairgen_hi_budget_rho_min": args.mc_fairgen_hi_budget_rho_min,
        "mc_fairgen_hi_budget_rho_max": args.mc_fairgen_hi_budget_rho_max,
        "mc_fairgen_lo_budget_rho_min": args.mc_fairgen_lo_budget_rho_min,
        "mc_fairgen_lo_budget_rho_max": args.mc_fairgen_lo_budget_rho_max,
        "mc_fairgen_hi_overrun_prob": args.mc_fairgen_hi_overrun_prob,
        "mc_fairgen_lo_overrun_prob": args.mc_fairgen_lo_overrun_prob,
        "mc_fairgen_hi_overrun_factor_min": args.mc_fairgen_hi_overrun_factor_min,
        "mc_fairgen_hi_overrun_factor_max": args.mc_fairgen_hi_overrun_factor_max,
        "mc_fairgen_lo_overrun_factor_min": args.mc_fairgen_lo_overrun_factor_min,
        "mc_fairgen_lo_overrun_factor_max": args.mc_fairgen_lo_overrun_factor_max,
    }
    detail_rows: list[dict[str, object]] = []

    try:
        config = build_mc_fairgen_experiment_config(
            mode=args.mc_fairgen_mode,
            num_tasks=args.mc_fairgen_num_tasks,
            hi_ratio=args.mc_fairgen_hi_ratio,
            period_source=args.mc_fairgen_period_source,
            period_scale=args.mc_fairgen_period_scale,
            require_schedulable=args.require_schedulable,
            max_attempts=args.max_attempts,
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

        baseline_rows: list[dict[str, object]] = []
        for eval_seed in eval_seeds:
            bundle = resolve_experiment_bundle(config, seed=eval_seed)
            result = run_single_simulation(list(bundle.ordered_tasks), bundle.scenario, args.end_time)
            metrics = compute_service_quality_metrics(result)
            baseline_rows.append(
                {
                    "eval_seed": eval_seed,
                    "mode_changes": result.mode_change_count(),
                    "lo_cancellations": result.lo_job_cancellation_count(),
                    **service_metrics_to_row(metrics),
                }
            )

    except Exception as exc:  # noqa: BLE001
        main_row["amcrtb_schedulable"] = False
        main_row["scan_error"] = str(exc)
        baseline_rows = []

    main_row["baseline_mode_changes_mean"] = mean_of(baseline_rows, "mode_changes")
    main_row["baseline_mode_changes_median"] = median_of(baseline_rows, "mode_changes")
    main_row["baseline_lo_cancellations_mean"] = mean_of(baseline_rows, "lo_cancellations")
    main_row["baseline_lo_cancellations_median"] = median_of(baseline_rows, "lo_cancellations")
    main_row["baseline_released_lo_jobs_mean"] = mean_of(baseline_rows, "released_lo_jobs")
    main_row["baseline_cancelled_lo_jobs_mean"] = mean_of(baseline_rows, "cancelled_lo_jobs")
    main_row["baseline_completed_lo_jobs_mean"] = mean_of(baseline_rows, "completed_lo_jobs")
    main_row["baseline_lo_deadline_misses_mean"] = mean_of(baseline_rows, "lo_deadline_misses")
    main_row["baseline_hi_deadline_misses_sum"] = sum_of(baseline_rows, "hi_deadline_misses")
    main_row["baseline_hi_deadline_misses_mean"] = mean_of(baseline_rows, "hi_deadline_misses")
    main_row["baseline_lc_service_loss_mean"] = mean_of(baseline_rows, "lc_service_loss")
    main_row["baseline_lc_service_loss_median"] = median_of(baseline_rows, "lc_service_loss")
    main_row["baseline_lc_qos_mean"] = mean_of(baseline_rows, "lc_qos")
    main_row["baseline_lc_qos_median"] = median_of(baseline_rows, "lc_qos")
    main_row["baseline_min_lc_service_mean"] = mean_of(baseline_rows, "min_lc_service")
    main_row["baseline_budget_adjust_count_mean"] = mean_of(baseline_rows, "budget_adjust_count")
    main_row["baseline_mean_abs_budget_change_mean"] = mean_of(baseline_rows, "mean_abs_budget_change")

    bucket = classify_qos_pressure_bucket(main_row["baseline_lc_service_loss_mean"] if baseline_rows else None)
    main_row["qos_pressure_bucket"] = bucket

    # static sweep 默认字段
    main_row["static_sweep_found_valid"] = ""
    main_row["static_sweep_best_inc_ratio"] = ""
    main_row["static_sweep_best_dec_ratio"] = ""
    main_row["static_sweep_best_lc_service_loss_mean"] = ""
    main_row["static_sweep_best_lc_qos_mean"] = ""
    main_row["static_sweep_relative_lc_loss_reduction"] = ""
    main_row["static_sweep_best_mode_changes_mean"] = ""
    main_row["static_sweep_best_mode_change_delta_ratio"] = ""
    main_row["static_sweep_best_hi_deadline_misses_sum"] = ""
    main_row["static_sweep_policy"] = ""

    if args.enable_static_sweep and baseline_rows:
        baseline_loss = main_row["baseline_lc_service_loss_mean"]
        baseline_mode = main_row["baseline_mode_changes_mean"]
        static_candidates: list[dict[str, object]] = []

        for inc_ratio in parse_float_list(args.sweep_inc_ratios):
            for dec_ratio in parse_float_list(args.sweep_dec_ratios):
                static_rows: list[dict[str, object]] = []
                for eval_seed in static_eval_seeds:
                    bundle = resolve_experiment_bundle(config, seed=eval_seed)
                    adjusted_tasks = [
                        static_adjust_budget_scale(task, inc_ratio, dec_ratio, args.budget_floor_ratio)
                        for task in list(bundle.ordered_tasks)
                    ]
                    result = run_single_simulation(
                        adjusted_tasks,
                        bundle.scenario,
                        args.static_sweep_end_time,
                    )
                    metrics = compute_service_quality_metrics(result)
                    static_rows.append(
                        {
                            "mode_changes": result.mode_change_count(),
                            **service_metrics_to_row(metrics),
                        }
                    )

                static_loss = mean_of(static_rows, "lc_service_loss")
                static_qos = mean_of(static_rows, "lc_qos")
                static_mode = mean_of(static_rows, "mode_changes")
                static_hi_sum = sum_of(static_rows, "hi_deadline_misses")
                reduction = (
                    safe_relative_reduction(float(baseline_loss), float(static_loss))
                    if baseline_loss not in (None, "") and static_loss is not None
                    else None
                )
                detail_rows.append(
                    {
                        "candidate_seed": candidate_seed,
                        "inc_ratio": inc_ratio,
                        "dec_ratio": dec_ratio,
                        "static_lc_service_loss_mean": static_loss,
                        "static_lc_qos_mean": static_qos,
                        "static_mode_changes_mean": static_mode,
                        "static_hi_deadline_misses_sum": static_hi_sum,
                        "static_relative_lc_loss_reduction": reduction,
                    }
                )

                if static_hi_sum == 0 and baseline_loss is not None and static_loss is not None and float(static_loss) <= float(baseline_loss):
                    static_candidates.append(
                        {
                            "inc_ratio": inc_ratio,
                            "dec_ratio": dec_ratio,
                            "static_lc_service_loss_mean": static_loss,
                            "static_lc_qos_mean": static_qos,
                            "static_mode_changes_mean": static_mode,
                            "static_hi_deadline_misses_sum": static_hi_sum,
                            "static_relative_lc_loss_reduction": reduction,
                        }
                    )

        if static_candidates:
            best = sorted(
                static_candidates,
                key=lambda row: (
                    float(row["static_lc_service_loss_mean"]),
                    float(row["static_mode_changes_mean"] if row["static_mode_changes_mean"] is not None else 10**9),
                    float(row["inc_ratio"]),
                    float(row["dec_ratio"]),
                ),
            )[0]
            main_row["static_sweep_found_valid"] = True
            main_row["static_sweep_best_inc_ratio"] = best["inc_ratio"]
            main_row["static_sweep_best_dec_ratio"] = best["dec_ratio"]
            main_row["static_sweep_best_lc_service_loss_mean"] = best["static_lc_service_loss_mean"]
            main_row["static_sweep_best_lc_qos_mean"] = best["static_lc_qos_mean"]
            main_row["static_sweep_relative_lc_loss_reduction"] = best["static_relative_lc_loss_reduction"]
            main_row["static_sweep_best_mode_changes_mean"] = best["static_mode_changes_mean"]
            if baseline_mode in (None, "") or float(baseline_mode) == 0.0:
                main_row["static_sweep_best_mode_change_delta_ratio"] = ""
            else:
                main_row["static_sweep_best_mode_change_delta_ratio"] = (
                    float(best["static_mode_changes_mean"]) - float(baseline_mode)
                ) / float(baseline_mode)
            main_row["static_sweep_best_hi_deadline_misses_sum"] = best["static_hi_deadline_misses_sum"]
            main_row["static_sweep_policy"] = args.static_sweep_policy
        else:
            main_row["static_sweep_found_valid"] = False
            main_row["static_sweep_policy"] = args.static_sweep_policy

    min_static = args.min_static_sweep_reduction if args.enable_static_sweep else None
    rec, reason = recommend_for_qos_dqn(
        amcrtb_schedulable=bool(main_row["amcrtb_schedulable"]),
        baseline_hi_deadline_misses_sum=main_row["baseline_hi_deadline_misses_sum"],
        baseline_lc_service_loss_mean=main_row["baseline_lc_service_loss_mean"],
        baseline_released_lo_jobs_mean=main_row["baseline_released_lo_jobs_mean"],
        baseline_cancelled_lo_jobs_mean=main_row["baseline_cancelled_lo_jobs_mean"],
        baseline_mode_changes_mean=main_row["baseline_mode_changes_mean"],
        static_sweep_relative_lc_loss_reduction=(
            None
            if main_row["static_sweep_relative_lc_loss_reduction"] in (None, "")
            else float(main_row["static_sweep_relative_lc_loss_reduction"])
        ),
        min_lc_service_loss=0.10,
        max_lc_service_loss=0.30,
        min_released_lo_jobs=100.0,
        min_cancelled_lo_jobs=10.0,
        min_mode_changes=1.0,
        min_static_sweep_reduction=min_static,
    )
    main_row["recommended_for_qos_dqn"] = rec
    main_row["qos_reject_reason"] = reason

    return main_row, detail_rows


def build_parser() -> argparse.ArgumentParser:
    """构建 QoS pressure 扫描脚本参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=500)
    parser.add_argument("--candidate-seeds", type=str, default="")
    parser.add_argument("--eval-seeds", type=str, default="200:229")
    parser.add_argument("--end-time", type=int, default=1000000)
    parser.add_argument("--agent-period", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--workload", choices=["mc_fairgen"], default="mc_fairgen")
    parser.add_argument("--mc-fairgen-mode", type=str, default="paper_learnable_headroom")
    parser.add_argument("--mc-fairgen-num-tasks", type=int, default=12)
    parser.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    parser.add_argument("--mc-fairgen-period-source", type=str, default="automotive")
    parser.add_argument("--mc-fairgen-period-scale", type=int, default=100)
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
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-attempts", type=int, default=100)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output-selected-seeds", type=str, default="")

    parser.add_argument(
        "--enable-static-sweep",
        action="store_true",
        help=(
            "启用 static budget scaling sweep：每次仿真前静态缩放一次任务预算，"
            "用于低成本可学习性代理评估。"
        ),
    )
    parser.add_argument("--sweep-inc-ratios", type=str, default="0,0.015,0.025,0.035")
    parser.add_argument("--sweep-dec-ratios", type=str, default="0,0.010,0.015")
    parser.add_argument("--static-sweep-end-time", type=int, default=None)
    parser.add_argument("--static-sweep-eval-seeds", type=str, default="")
    parser.add_argument("--static-sweep-mode-delta", type=float, default=0.20)
    parser.add_argument("--min-static-sweep-reduction", type=float, default=0.05)
    parser.add_argument("--static-sweep-detail-output", type=str, default="")
    parser.add_argument("--static-sweep-policy", choices=["budget_scale"], default="budget_scale")
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    return parser


def main() -> None:
    """执行 QoS pressure baseline/static 扫描并输出 CSV。"""

    args = build_parser().parse_args()
    if args.workers != 1:
        raise ValueError("第一版仅支持 --workers 1")

    candidate_seeds = parse_int_list_or_half_open_range(args.candidate_seeds) if args.candidate_seeds.strip() else list(range(args.seed_start, args.seed_end))
    eval_seeds = parse_int_list_or_half_open_range(args.eval_seeds)
    if not eval_seeds:
        raise ValueError("--eval-seeds 不能为空")

    if args.static_sweep_end_time is None:
        args.static_sweep_end_time = args.end_time
    static_eval_seeds = parse_int_list_or_half_open_range(args.static_sweep_eval_seeds) if args.static_sweep_eval_seeds.strip() else list(eval_seeds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, object]] = []
    all_detail_rows: list[dict[str, object]] = []
    bucket_counter: Counter[str] = Counter()
    error_count = 0

    for idx, candidate_seed in enumerate(candidate_seeds, start=1):
        row, detail_rows = scan_candidate_seed(args, candidate_seed, eval_seeds, static_eval_seeds)
        main_rows.append(row)
        all_detail_rows.extend(detail_rows)
        bucket_counter[str(row.get("qos_pressure_bucket", "unknown"))] += 1
        if row.get("scan_error"):
            error_count += 1
        if idx % 20 == 0 or idx == len(candidate_seeds):
            print(
                f"[QoS Scan] processed={idx}/{len(candidate_seeds)} medium={bucket_counter.get('medium', 0)} "
                f"easy={bucket_counter.get('easy', 0)} hard={bucket_counter.get('hard', 0)} "
                f"overloaded={bucket_counter.get('overloaded', 0)} errors={error_count}",
                flush=True,
            )

    fieldnames = list(main_rows[0].keys()) if main_rows else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in main_rows:
            writer.writerow({key: to_csv_value(value) for key, value in row.items()})

    if args.output_selected_seeds:
        selected_path = Path(args.output_selected_seeds)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        with selected_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_seed"])
            writer.writeheader()
            for row in main_rows:
                if str(row.get("recommended_for_qos_dqn", "")).lower() == "true":
                    writer.writerow({"candidate_seed": int(row["candidate_seed"])})

    if args.enable_static_sweep and args.static_sweep_detail_output:
        detail_path = Path(args.static_sweep_detail_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_fieldnames = [
            "candidate_seed",
            "inc_ratio",
            "dec_ratio",
            "static_lc_service_loss_mean",
            "static_lc_qos_mean",
            "static_mode_changes_mean",
            "static_hi_deadline_misses_sum",
            "static_relative_lc_loss_reduction",
        ]
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
            writer.writeheader()
            for row in all_detail_rows:
                writer.writerow({key: to_csv_value(row.get(key)) for key in detail_fieldnames})

    recommended_count = sum(1 for row in main_rows if str(row.get("recommended_for_qos_dqn", "")).lower() == "true")
    print("[QoS Scan Summary]", flush=True)
    print(f"easy={bucket_counter.get('easy', 0)}", flush=True)
    print(f"medium={bucket_counter.get('medium', 0)}", flush=True)
    print(f"hard={bucket_counter.get('hard', 0)}", flush=True)
    print(f"overloaded={bucket_counter.get('overloaded', 0)}", flush=True)
    print(f"unknown={bucket_counter.get('unknown', 0)}", flush=True)
    print(f"recommended_for_qos_dqn={recommended_count}", flush=True)


if __name__ == "__main__":
    main()
