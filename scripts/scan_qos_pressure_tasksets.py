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
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median

from amc_py.dqn.experiment import build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_service_quality_metrics, safe_relative_reduction, service_metrics_to_row
from amc_py.models import Criticality, Task
from amc_py.qos_pressure import (
    classify_improvement_type,
    classify_qos_pressure_bucket,
    classify_single_improvement_type,
    recommend_for_qos_dqn,
)
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


def parse_str_list(raw: str) -> list[str]:
    """解析逗号分隔字符串列表。"""

    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    """解析逗号分隔整数列表。"""

    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def stable_prefix(delta: float) -> str:
    """把稳定约束 delta 转为主 CSV 字段前缀。"""

    if abs(delta - 0.05) < 1e-12:
        return "stable005_static"
    if abs(delta - 0.10) < 1e-12:
        return "stable010_static"
    return f"stable{int(round(delta * 100)):03d}_static"


def select_static_best(
    rows: list[dict[str, object]],
    baseline_loss: float,
    baseline_mode_changes: float,
    stable_delta: float | None = None,
    abs_tolerance: float = 0.0,
) -> dict[str, object] | None:
    """按计划文档规则，从 static sweep 明细中选择最优组合。"""

    candidates: list[dict[str, object]] = []
    for row in rows:
        if row.get("static_hi_deadline_misses_sum") != 0:
            continue
        static_loss = row.get("static_lc_service_loss_mean")
        if static_loss is None or float(static_loss) > float(baseline_loss):
            continue
        if stable_delta is not None:
            mode_limit = float(baseline_mode_changes) * (1.0 + float(stable_delta)) + float(abs_tolerance)
            static_mode = row.get("static_mode_changes_mean")
            if static_mode is None or float(static_mode) > mode_limit:
                continue
        candidates.append(row)

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda item: (
            float(item["static_lc_service_loss_mean"]),
            float(item["static_mode_changes_mean"]),
            abs(float(item["inc_ratio"])) + abs(float(item["dec_ratio"])),
            float(item["inc_ratio"]),
            float(item["dec_ratio"]),
        ),
    )[0]


def add_static_best_fields(prefix: str, row: dict[str, object], best: dict[str, object] | None, baseline_mode: float | None) -> None:
    """把 best 结果写入主行，统一字段命名，避免散落重复逻辑。"""

    if best is None:
        row[f"{prefix}_found_valid"] = False
        row[f"{prefix}_inc_ratio"] = None
        row[f"{prefix}_dec_ratio"] = None
        row[f"{prefix}_lc_service_loss_mean"] = None
        row[f"{prefix}_lc_qos_mean"] = None
        row[f"{prefix}_relative_lc_loss_reduction"] = None
        row[f"{prefix}_mode_changes_mean"] = None
        row[f"{prefix}_mode_change_delta_ratio"] = None
        row[f"{prefix}_hi_deadline_misses_sum"] = None
        return

    row[f"{prefix}_found_valid"] = True
    row[f"{prefix}_inc_ratio"] = best["inc_ratio"]
    row[f"{prefix}_dec_ratio"] = best["dec_ratio"]
    row[f"{prefix}_lc_service_loss_mean"] = best["static_lc_service_loss_mean"]
    row[f"{prefix}_lc_qos_mean"] = best["static_lc_qos_mean"]
    row[f"{prefix}_relative_lc_loss_reduction"] = best["static_relative_lc_loss_reduction"]
    row[f"{prefix}_mode_changes_mean"] = best["static_mode_changes_mean"]
    row[f"{prefix}_hi_deadline_misses_sum"] = best["static_hi_deadline_misses_sum"]
    if baseline_mode in (None, "") or float(baseline_mode) == 0.0:
        row[f"{prefix}_mode_change_delta_ratio"] = None
    else:
        row[f"{prefix}_mode_change_delta_ratio"] = (
            float(best["static_mode_changes_mean"]) - float(baseline_mode)
        ) / float(baseline_mode)


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


def static_adjust_single_task(
    tasks: list[Task],
    target_idx: int,
    action: str,
    repeat_count: int,
    inc_ratio: float,
    dec_ratio: float,
    budget_floor_ratio: float,
) -> tuple[list[Task], bool, bool, str]:
    """只对单个任务应用 DQN 风格预算扰动。

    返回值含义：
    1. 调整后的任务列表；
    2. 是否发生了预算变化；
    3. 是否为无效候选（invalid candidate）；
    4. 无效原因字符串（有效候选时为空字符串）。
    """

    adjusted = list(tasks)
    task = adjusted[target_idx]
    original_budget = int(task.c_lo)
    current = int(task.c_lo)
    changed = False
    invalid = False
    invalid_reason = ""

    for _ in range(repeat_count):
        if action == "increase":
            new_value = math.ceil(current * (1.0 + inc_ratio))
            # single-task sweep 只改 c_lo，因此无论 HI/LO 都必须同时满足：
            # 1) c_lo <= c_hi（Task dataclass 约束）；
            # 2) c_lo <= deadline（预算不可超过任务时限）。
            # 故上界统一为 min(c_hi, deadline)，避免出现“只对 HI 限 c_hi”的不一致。
            upper_bound = min(int(task.c_hi), int(task.deadline))
            # 记录“无效候选”而非抛异常：若 increase 前已到上界，说明该动作不可再推进。
            if current >= int(upper_bound):
                invalid = True
                invalid_reason = "c_lo_would_exceed_c_hi_or_deadline"
            current = min(new_value, upper_bound)
        elif action == "decrease":
            new_value = math.floor(current * (1.0 - dec_ratio))
            # decrease 必须基于 original_c_lo 的 floor，避免 repeat decrease 把预算持续压到 floor 以下。
            floor = math.floor(original_budget * budget_floor_ratio)
            if current <= int(max(1, floor)):
                invalid = True
                invalid_reason = "c_lo_would_below_floor"
            current = max(floor, new_value)
        else:
            # 无效 action 不让脚本崩溃，按 invalid 候选记录并保持预算不变。
            invalid = True
            invalid_reason = "unsupported_single_task_action"
            current = int(task.c_lo)
        current = max(1, int(current))

    # 防御性收口：即使未来上面的分支被改坏，这里也保证不会写出非法 Task。
    current = min(int(current), int(task.c_hi), int(task.deadline))
    current = max(1, int(current))

    if current != int(task.c_lo):
        changed = True
    adjusted[target_idx] = replace(task, c_lo=current)
    return adjusted, changed, invalid, invalid_reason


def choose_single_task_sweep_indices(tasks: list[Task], top_k: int, budget_floor_ratio: float) -> list[int]:
    """按 headroom 分数选择 single-task 扫描候选任务下标。"""

    indices = list(range(len(tasks)))
    if top_k <= 0:
        return indices
    scored: list[tuple[float, int]] = []
    for idx, task in enumerate(tasks):
        # headroom 计算与扰动逻辑保持同口径：increase 上界统一为 min(c_hi, deadline)。
        upper_bound = min(int(task.c_hi), int(task.deadline))
        increase_headroom = max(0, int(upper_bound) - int(task.c_lo))
        floor = max(1, math.floor(float(task.c_lo) * float(budget_floor_ratio)))
        decrease_headroom = max(0, int(task.c_lo) - int(floor))
        score = float(increase_headroom + decrease_headroom)
        scored.append((score, idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in scored[:top_k]]


def select_single_task_static_best(
    rows: list[dict[str, object]],
    baseline_loss: float,
    baseline_mode_changes: float,
    stable_delta: float | None = None,
    abs_tolerance: float = 0.0,
) -> dict[str, object] | None:
    """从 single-task sweep 明细中选择最优项。"""

    candidates: list[dict[str, object]] = []
    for row in rows:
        if row.get("single_task_hi_deadline_misses_sum") != 0:
            continue
        static_loss = row.get("single_task_lc_service_loss_mean")
        if static_loss is None or float(static_loss) > float(baseline_loss):
            continue
        if stable_delta is not None:
            mode_limit = float(baseline_mode_changes) * (1.0 + float(stable_delta)) + float(abs_tolerance)
            static_mode = row.get("single_task_mode_changes_mean")
            if static_mode is None or float(static_mode) > mode_limit:
                continue
        candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item["single_task_lc_service_loss_mean"]),
            float(item["single_task_mode_changes_mean"]),
            int(item["repeat_count"]),
            int(item["target_task_index"]),
        ),
    )[0]


def select_sequence_static_best(
    rows: list[dict[str, object]],
    baseline_loss: float,
    baseline_mode_changes: float,
    stable_delta: float | None = None,
    abs_tolerance: float = 0.0,
) -> dict[str, object] | None:
    """从 sequence sweep 明细中选择最优项。"""

    candidates: list[dict[str, object]] = []
    for row in rows:
        if row.get("sequence_hi_deadline_misses_sum") != 0:
            continue
        static_loss = row.get("sequence_lc_service_loss_mean")
        if static_loss is None or float(static_loss) > float(baseline_loss):
            continue
        if stable_delta is not None:
            mode_limit = float(baseline_mode_changes) * (1.0 + float(stable_delta)) + float(abs_tolerance)
            static_mode = row.get("sequence_mode_changes_mean")
            if static_mode is None or float(static_mode) > mode_limit:
                continue
        candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item["sequence_lc_service_loss_mean"]),
            float(item["sequence_mode_changes_mean"]),
            int(item["sequence_length"]),
            str(item["sequence_pattern"]),
        ),
    )[0]


def add_single_task_best_fields(prefix: str, row: dict[str, object], best: dict[str, object] | None, baseline_mode: float | None) -> None:
    """把 single-task best 写回主 CSV 行。"""

    if best is None:
        row[f"{prefix}_found_valid"] = False
        row[f"{prefix}_task_index"] = None
        row[f"{prefix}_task_name"] = None
        row[f"{prefix}_action"] = None
        row[f"{prefix}_repeat_count"] = None
        row[f"{prefix}_lc_service_loss_mean"] = None
        row[f"{prefix}_lc_qos_mean"] = None
        row[f"{prefix}_relative_lc_loss_reduction"] = None
        row[f"{prefix}_mode_changes_mean"] = None
        row[f"{prefix}_mode_change_delta_ratio"] = None
        row[f"{prefix}_hi_deadline_misses_sum"] = None
        return
    row[f"{prefix}_found_valid"] = True
    row[f"{prefix}_task_index"] = best["target_task_index"]
    row[f"{prefix}_task_name"] = best["target_task_name"]
    row[f"{prefix}_action"] = best["single_action"]
    row[f"{prefix}_repeat_count"] = best["repeat_count"]
    row[f"{prefix}_lc_service_loss_mean"] = best["single_task_lc_service_loss_mean"]
    row[f"{prefix}_lc_qos_mean"] = best["single_task_lc_qos_mean"]
    row[f"{prefix}_relative_lc_loss_reduction"] = best["single_task_relative_lc_loss_reduction"]
    row[f"{prefix}_mode_changes_mean"] = best["single_task_mode_changes_mean"]
    row[f"{prefix}_hi_deadline_misses_sum"] = best["single_task_hi_deadline_misses_sum"]
    if baseline_mode in (None, "") or float(baseline_mode) == 0.0:
        row[f"{prefix}_mode_change_delta_ratio"] = None
    else:
        row[f"{prefix}_mode_change_delta_ratio"] = (float(best["single_task_mode_changes_mean"]) - float(baseline_mode)) / float(baseline_mode)


def add_sequence_best_fields(prefix: str, row: dict[str, object], best: dict[str, object] | None, baseline_mode: float | None) -> None:
    """把 sequence best 写回主 CSV 行。"""

    if best is None:
        row[f"{prefix}_found_valid"] = False
        row[f"{prefix}_pattern"] = None
        row[f"{prefix}_sequence_length"] = None
        row[f"{prefix}_inc_task_index"] = None
        row[f"{prefix}_inc_task_name"] = None
        row[f"{prefix}_dec_task_index"] = None
        row[f"{prefix}_dec_task_name"] = None
        row[f"{prefix}_relative_lc_loss_reduction"] = None
        row[f"{prefix}_lc_service_loss_mean"] = None
        row[f"{prefix}_lc_qos_mean"] = None
        row[f"{prefix}_mode_changes_mean"] = None
        row[f"{prefix}_mode_change_delta_ratio"] = None
        row[f"{prefix}_hi_deadline_misses_sum"] = None
        return
    row[f"{prefix}_found_valid"] = True
    row[f"{prefix}_pattern"] = best["sequence_pattern"]
    row[f"{prefix}_sequence_length"] = best["sequence_length"]
    row[f"{prefix}_inc_task_index"] = best["inc_task_index"]
    row[f"{prefix}_inc_task_name"] = best["inc_task_name"]
    row[f"{prefix}_dec_task_index"] = best["dec_task_index"]
    row[f"{prefix}_dec_task_name"] = best["dec_task_name"]
    row[f"{prefix}_relative_lc_loss_reduction"] = best["sequence_relative_lc_loss_reduction"]
    row[f"{prefix}_lc_service_loss_mean"] = best["sequence_lc_service_loss_mean"]
    row[f"{prefix}_lc_qos_mean"] = best["sequence_lc_qos_mean"]
    row[f"{prefix}_mode_changes_mean"] = best["sequence_mode_changes_mean"]
    row[f"{prefix}_hi_deadline_misses_sum"] = best["sequence_hi_deadline_misses_sum"]
    if baseline_mode in (None, "") or float(baseline_mode) == 0.0:
        row[f"{prefix}_mode_change_delta_ratio"] = None
    else:
        row[f"{prefix}_mode_change_delta_ratio"] = (float(best["sequence_mode_changes_mean"]) - float(baseline_mode)) / float(baseline_mode)


def run_single_simulation(ordered_tasks: list[Task], scenario, end_time: int):
    """执行一次 AMC_PLUS baseline 仿真并返回结果对象。"""

    return simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=RuntimeConfig(end_time=end_time, semantics=RuntimeSemantics.AMC_PLUS),
    )


def scan_candidate_seed(
    args: argparse.Namespace,
    candidate_seed: int,
    eval_seeds: list[int],
    static_eval_seeds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """扫描单个 candidate seed，返回主汇总行及各类 sweep 明细。"""

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
    single_task_detail_rows: list[dict[str, object]] = []
    sequence_detail_rows: list[dict[str, object]] = []

    try:
        if args.end_time <= 0:
            # per-1M 指标必须建立在正时间长度上，避免除零或错误归一化。
            raise ValueError("end_time must be positive for per-1M metrics")
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

        bundle_cache: dict[int, object] = {}

        def get_bundle(eval_seed: int):
            if eval_seed not in bundle_cache:
                bundle_cache[eval_seed] = resolve_experiment_bundle(config, seed=eval_seed)
            return bundle_cache[eval_seed]

        baseline_rows: list[dict[str, object]] = []
        for eval_seed in eval_seeds:
            bundle = get_bundle(eval_seed)
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
    main_row["baseline_total_events_mean"] = (
        (main_row["baseline_mode_changes_mean"] or 0.0) + (main_row["baseline_lo_cancellations_mean"] or 0.0)
    )
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
    scale_per_1m = float(args.end_time) / 1_000_000.0
    main_row["baseline_total_events_per_1m"] = (
        (main_row["baseline_total_events_mean"] or 0.0) / scale_per_1m
    )
    main_row["baseline_mode_changes_per_1m"] = (
        (main_row["baseline_mode_changes_mean"] or 0.0) / scale_per_1m
    )
    main_row["baseline_lo_cancellations_per_1m"] = (
        (main_row["baseline_cancelled_lo_jobs_mean"] or 0.0) / scale_per_1m
    )

    bucket = classify_qos_pressure_bucket(main_row["baseline_lc_service_loss_mean"] if baseline_rows else None)
    main_row["qos_pressure_bucket"] = bucket

    # static sweep 结果字段（包含新语义字段和旧兼容字段），先初始化为空。
    add_static_best_fields("static_qos_best", main_row, None, main_row["baseline_mode_changes_mean"])
    add_static_best_fields("stable005_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_static_best_fields("stable010_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_static_best_fields("global_static_qos_best", main_row, None, main_row["baseline_mode_changes_mean"])
    add_static_best_fields("global_stable005_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_static_best_fields("global_stable010_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_single_task_best_fields("single_static_qos_best", main_row, None, main_row["baseline_mode_changes_mean"])
    add_single_task_best_fields("single_stable005_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_single_task_best_fields("single_stable010_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_sequence_best_fields("sequence_static_qos_best", main_row, None, main_row["baseline_mode_changes_mean"])
    add_sequence_best_fields("sequence_stable005_static", main_row, None, main_row["baseline_mode_changes_mean"])
    add_sequence_best_fields("sequence_stable010_static", main_row, None, main_row["baseline_mode_changes_mean"])
    main_row["tradeoff_gap_005"] = None
    main_row["tradeoff_gap_010"] = None
    main_row["tradeoff_only_flag_005"] = False
    main_row["tradeoff_only_flag_010"] = False
    main_row["improvement_type"] = "no_static_improvement"
    main_row["single_improvement_type"] = "single_no_improvement"
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
        static_rows_all: list[dict[str, object]] = []
        stable_deltas = parse_float_list(args.stable_static_mode_deltas)
        if args.static_sweep_stage == "quick":
            # quick 档位用于低成本预筛，固定使用小网格比例集合。
            inc_ratios = parse_float_list(args.quick_sweep_inc_ratios)
            dec_ratios = parse_float_list(args.quick_sweep_dec_ratios)
        else:
            # full 档位直接使用用户传入 sweep 网格，适合少量候选做正式扫描。
            inc_ratios = parse_float_list(args.sweep_inc_ratios)
            dec_ratios = parse_float_list(args.sweep_dec_ratios)

        for inc_ratio in inc_ratios:
            for dec_ratio in dec_ratios:
                static_rows: list[dict[str, object]] = []
                for eval_seed in static_eval_seeds:
                    bundle = get_bundle(eval_seed)
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
                static_rows_all.append(
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

        if baseline_loss is not None and baseline_mode is not None:
            # 1) static_qos_best：只约束“HI 不 miss + LC 不劣化”，不约束 mode changes。
            qos_best = select_static_best(
                static_rows_all,
                baseline_loss=float(baseline_loss),
                baseline_mode_changes=float(baseline_mode),
                stable_delta=None,
                abs_tolerance=float(args.stable_static_mode_abs_tolerance),
            )
            add_static_best_fields("static_qos_best", main_row, qos_best, baseline_mode)
            add_static_best_fields("global_static_qos_best", main_row, qos_best, baseline_mode)
            for delta in stable_deltas:
                prefix = stable_prefix(float(delta))
                # 2) stableXXX_static：在 static_qos_best 约束基础上再加 mode changes 上限。
                stable_best = select_static_best(
                    static_rows_all,
                    baseline_loss=float(baseline_loss),
                    baseline_mode_changes=float(baseline_mode),
                    stable_delta=float(delta),
                    abs_tolerance=float(args.stable_static_mode_abs_tolerance),
                )
                add_static_best_fields(prefix, main_row, stable_best, baseline_mode)
                add_static_best_fields(f"global_{prefix}", main_row, stable_best, baseline_mode)
            main_row["static_sweep_found_valid"] = bool(main_row["static_qos_best_found_valid"])
            main_row["static_sweep_best_inc_ratio"] = main_row["static_qos_best_inc_ratio"]
            main_row["static_sweep_best_dec_ratio"] = main_row["static_qos_best_dec_ratio"]
            main_row["static_sweep_best_lc_service_loss_mean"] = main_row["static_qos_best_lc_service_loss_mean"]
            main_row["static_sweep_best_lc_qos_mean"] = main_row["static_qos_best_lc_qos_mean"]
            main_row["static_sweep_relative_lc_loss_reduction"] = main_row["static_qos_best_relative_lc_loss_reduction"]
            main_row["static_sweep_best_mode_changes_mean"] = main_row["static_qos_best_mode_changes_mean"]
            main_row["static_sweep_best_mode_change_delta_ratio"] = main_row["static_qos_best_mode_change_delta_ratio"]
            main_row["static_sweep_best_hi_deadline_misses_sum"] = main_row["static_qos_best_hi_deadline_misses_sum"]
            main_row["static_sweep_policy"] = args.static_sweep_policy

            # 3) tradeoff 诊断：static 最优改进很高但 stable 改进很低时，标记为 trade-off-only。
            qos_red = 0.0 if main_row["static_qos_best_relative_lc_loss_reduction"] is None else float(main_row["static_qos_best_relative_lc_loss_reduction"])
            stable005_red = 0.0 if main_row["stable005_static_relative_lc_loss_reduction"] is None else float(main_row["stable005_static_relative_lc_loss_reduction"])
            stable010_red = 0.0 if main_row["stable010_static_relative_lc_loss_reduction"] is None else float(main_row["stable010_static_relative_lc_loss_reduction"])
            main_row["tradeoff_gap_005"] = qos_red - stable005_red
            main_row["tradeoff_gap_010"] = qos_red - stable010_red
            main_row["tradeoff_only_flag_005"] = qos_red >= 0.05 and stable005_red < 0.01
            main_row["tradeoff_only_flag_010"] = qos_red >= 0.05 and stable010_red < 0.01
            # 4) improvement_type 用于后续汇总和可视化分组。
            main_row["improvement_type"] = classify_improvement_type(
                static_qos_best_found_valid=bool(main_row["static_qos_best_found_valid"]),
                static_qos_best_relative_lc_loss_reduction=main_row["static_qos_best_relative_lc_loss_reduction"],
                stable005_static_relative_lc_loss_reduction=main_row["stable005_static_relative_lc_loss_reduction"],
                stable010_static_relative_lc_loss_reduction=main_row["stable010_static_relative_lc_loss_reduction"],
            )
        else:
            main_row["static_sweep_found_valid"] = False
            main_row["static_sweep_policy"] = args.static_sweep_policy

    if args.enable_single_task_sweep and baseline_rows:
        baseline_loss = main_row["baseline_lc_service_loss_mean"]
        baseline_mode = main_row["baseline_mode_changes_mean"]
        single_rows_all: list[dict[str, object]] = []
        stable_deltas = parse_float_list(args.stable_static_mode_deltas)
        actions = parse_str_list(args.single_task_sweep_actions)
        repeat_counts = parse_int_list(args.single_task_repeat_counts)
        if args.single_task_sweep_stage == "quick":
            inc_ratios = parse_float_list(args.quick_sweep_inc_ratios)
            dec_ratios = parse_float_list(args.quick_sweep_dec_ratios)
        else:
            inc_ratios = parse_float_list(args.sweep_inc_ratios)
            dec_ratios = parse_float_list(args.sweep_dec_ratios)

        base_bundle = get_bundle(static_eval_seeds[0])
        target_indices = choose_single_task_sweep_indices(
            list(base_bundle.ordered_tasks),
            int(args.single_task_top_k_by_headroom),
            float(args.budget_floor_ratio),
        )

        for task_idx in target_indices:
            for action in actions:
                for repeat_count in repeat_counts:
                    for inc_ratio in inc_ratios:
                        for dec_ratio in dec_ratios:
                            static_rows: list[dict[str, object]] = []
                            changed_any = False
                            invalid_any = False
                            invalid_reason_last = ""
                            for eval_seed in static_eval_seeds:
                                bundle = get_bundle(eval_seed)
                                adjusted_tasks, changed, invalid_candidate, invalid_reason = static_adjust_single_task(
                                    list(bundle.ordered_tasks),
                                    task_idx,
                                    action,
                                    int(repeat_count),
                                    float(inc_ratio),
                                    float(dec_ratio),
                                    float(args.budget_floor_ratio),
                                )
                                changed_any = changed_any or changed
                                invalid_any = invalid_any or invalid_candidate
                                if invalid_candidate and invalid_reason:
                                    invalid_reason_last = invalid_reason
                                if not changed:
                                    result = run_single_simulation(list(bundle.ordered_tasks), bundle.scenario, args.static_sweep_end_time)
                                else:
                                    result = run_single_simulation(adjusted_tasks, bundle.scenario, args.static_sweep_end_time)
                                metrics = compute_service_quality_metrics(result)
                                static_rows.append({"mode_changes": result.mode_change_count(), **service_metrics_to_row(metrics)})
                            # 无效/无变化候选不再导致崩溃；仍写明细，便于离线定位“动作不可达”的原因。
                            static_loss = mean_of(static_rows, "lc_service_loss")
                            static_qos = mean_of(static_rows, "lc_qos")
                            static_mode = mean_of(static_rows, "mode_changes")
                            static_hi_sum = sum_of(static_rows, "hi_deadline_misses")
                            reduction = (
                                safe_relative_reduction(float(baseline_loss), float(static_loss))
                                if baseline_loss not in (None, "") and static_loss is not None
                                else None
                            )
                            task = base_bundle.ordered_tasks[task_idx]
                            row = {
                                "candidate_seed": candidate_seed,
                                "target_task_index": task_idx,
                                "target_task_name": task.name,
                                "target_task_criticality": task.criticality.name,
                                "single_action": action,
                                "repeat_count": repeat_count,
                                "inc_ratio": inc_ratio,
                                "dec_ratio": dec_ratio,
                                "single_task_lc_service_loss_mean": static_loss,
                                "single_task_lc_qos_mean": static_qos,
                                "single_task_mode_changes_mean": static_mode,
                                "single_task_hi_deadline_misses_sum": static_hi_sum,
                                "single_task_relative_lc_loss_reduction": reduction,
                                "single_task_mode_change_delta_ratio": (
                                    None if baseline_mode in (None, "") or float(baseline_mode) == 0.0 else (float(static_mode) - float(baseline_mode)) / float(baseline_mode)
                                ),
                                "invalid_single_task_candidate": bool(invalid_any),
                                "invalid_reason": invalid_reason_last,
                            }
                            single_task_detail_rows.append(row)
                            if changed_any:
                                single_rows_all.append(row)

        if baseline_loss is not None and baseline_mode is not None:
            single_qos_best = select_single_task_static_best(
                single_rows_all,
                baseline_loss=float(baseline_loss),
                baseline_mode_changes=float(baseline_mode),
                stable_delta=None,
                abs_tolerance=float(args.stable_static_mode_abs_tolerance),
            )
            add_single_task_best_fields("single_static_qos_best", main_row, single_qos_best, baseline_mode)
            for delta in stable_deltas:
                prefix = "single_stable005_static" if abs(float(delta) - 0.05) < 1e-12 else "single_stable010_static"
                single_stable_best = select_single_task_static_best(
                    single_rows_all,
                    baseline_loss=float(baseline_loss),
                    baseline_mode_changes=float(baseline_mode),
                    stable_delta=float(delta),
                    abs_tolerance=float(args.stable_static_mode_abs_tolerance),
                )
                add_single_task_best_fields(prefix, main_row, single_stable_best, baseline_mode)

    if args.enable_single_sequence_sweep and baseline_rows:
        baseline_loss = main_row["baseline_lc_service_loss_mean"]
        baseline_mode = main_row["baseline_mode_changes_mean"]
        sequence_rows_all: list[dict[str, object]] = []
        stable_deltas = parse_float_list(args.stable_static_mode_deltas)
        patterns = parse_str_list(args.single_sequence_patterns)
        lengths = parse_int_list(args.single_sequence_lengths)

        base_bundle = get_bundle(static_eval_seeds[0])
        candidate_indices = choose_single_task_sweep_indices(
            list(base_bundle.ordered_tasks),
            int(args.single_sequence_top_k_tasks),
            float(args.budget_floor_ratio),
        )

        for pattern in patterns:
            for seq_len in lengths:
                if pattern in {"inc_repeat", "dec_repeat"}:
                    for i in candidate_indices:
                        static_rows: list[dict[str, object]] = []
                        changed_any = False
                        for eval_seed in static_eval_seeds:
                            bundle = get_bundle(eval_seed)
                            adjusted_tasks = list(bundle.ordered_tasks)
                            for _ in range(seq_len):
                                adjusted_tasks, changed, _, _ = static_adjust_single_task(
                                    adjusted_tasks,
                                    i,
                                    "increase" if pattern == "inc_repeat" else "decrease",
                                    1,
                                    float(parse_float_list(args.sweep_inc_ratios)[-1]),
                                    float(parse_float_list(args.sweep_dec_ratios)[-1]),
                                    float(args.budget_floor_ratio),
                                )
                                changed_any = changed_any or changed
                            if not changed_any:
                                result = run_single_simulation(list(bundle.ordered_tasks), bundle.scenario, args.static_sweep_end_time)
                            else:
                                result = run_single_simulation(adjusted_tasks, bundle.scenario, args.static_sweep_end_time)
                            metrics = compute_service_quality_metrics(result)
                            static_rows.append({"mode_changes": result.mode_change_count(), **service_metrics_to_row(metrics)})
                        if not changed_any:
                            continue
                        static_loss = mean_of(static_rows, "lc_service_loss")
                        static_qos = mean_of(static_rows, "lc_qos")
                        static_mode = mean_of(static_rows, "mode_changes")
                        static_hi_sum = sum_of(static_rows, "hi_deadline_misses")
                        reduction = safe_relative_reduction(float(baseline_loss), float(static_loss)) if baseline_loss not in (None, "") and static_loss is not None else None
                        row = {
                            "candidate_seed": candidate_seed,
                            "sequence_pattern": pattern,
                            "sequence_length": seq_len,
                            "inc_task_index": i if pattern == "inc_repeat" else None,
                            "inc_task_name": base_bundle.ordered_tasks[i].name if pattern == "inc_repeat" else None,
                            "dec_task_index": i if pattern == "dec_repeat" else None,
                            "dec_task_name": base_bundle.ordered_tasks[i].name if pattern == "dec_repeat" else None,
                            "sequence_lc_service_loss_mean": static_loss,
                            "sequence_lc_qos_mean": static_qos,
                            "sequence_mode_changes_mean": static_mode,
                            "sequence_hi_deadline_misses_sum": static_hi_sum,
                            "sequence_relative_lc_loss_reduction": reduction,
                            "sequence_mode_change_delta_ratio": None if baseline_mode in (None, "") or float(baseline_mode) == 0.0 else (float(static_mode) - float(baseline_mode)) / float(baseline_mode),
                        }
                        sequence_detail_rows.append(row)
                        sequence_rows_all.append(row)
                elif pattern in {"inc_dec_pair", "inc_dec_alternate"}:
                    for i in candidate_indices:
                        for j in candidate_indices:
                            if i == j:
                                continue
                            static_rows: list[dict[str, object]] = []
                            changed_any = False
                            for eval_seed in static_eval_seeds:
                                bundle = get_bundle(eval_seed)
                                adjusted_tasks = list(bundle.ordered_tasks)
                                if pattern == "inc_dec_pair":
                                    adjusted_tasks, changed_i, _, _ = static_adjust_single_task(
                                        adjusted_tasks, i, "increase", 1, float(parse_float_list(args.sweep_inc_ratios)[-1]), float(parse_float_list(args.sweep_dec_ratios)[-1]), float(args.budget_floor_ratio)
                                    )
                                    adjusted_tasks, changed_j, _, _ = static_adjust_single_task(
                                        adjusted_tasks, j, "decrease", 1, float(parse_float_list(args.sweep_inc_ratios)[-1]), float(parse_float_list(args.sweep_dec_ratios)[-1]), float(args.budget_floor_ratio)
                                    )
                                    changed_any = changed_any or changed_i or changed_j
                                else:
                                    for step_idx in range(seq_len):
                                        if step_idx % 2 == 0:
                                            adjusted_tasks, changed, _, _ = static_adjust_single_task(
                                                adjusted_tasks, i, "increase", 1, float(parse_float_list(args.sweep_inc_ratios)[-1]), float(parse_float_list(args.sweep_dec_ratios)[-1]), float(args.budget_floor_ratio)
                                            )
                                        else:
                                            adjusted_tasks, changed, _, _ = static_adjust_single_task(
                                                adjusted_tasks, j, "decrease", 1, float(parse_float_list(args.sweep_inc_ratios)[-1]), float(parse_float_list(args.sweep_dec_ratios)[-1]), float(args.budget_floor_ratio)
                                            )
                                        changed_any = changed_any or changed
                                if not changed_any:
                                    result = run_single_simulation(list(bundle.ordered_tasks), bundle.scenario, args.static_sweep_end_time)
                                else:
                                    result = run_single_simulation(adjusted_tasks, bundle.scenario, args.static_sweep_end_time)
                                metrics = compute_service_quality_metrics(result)
                                static_rows.append({"mode_changes": result.mode_change_count(), **service_metrics_to_row(metrics)})
                            if not changed_any:
                                continue
                            static_loss = mean_of(static_rows, "lc_service_loss")
                            static_qos = mean_of(static_rows, "lc_qos")
                            static_mode = mean_of(static_rows, "mode_changes")
                            static_hi_sum = sum_of(static_rows, "hi_deadline_misses")
                            reduction = safe_relative_reduction(float(baseline_loss), float(static_loss)) if baseline_loss not in (None, "") and static_loss is not None else None
                            row = {
                                "candidate_seed": candidate_seed,
                                "sequence_pattern": pattern,
                                "sequence_length": seq_len,
                                "inc_task_index": i,
                                "inc_task_name": base_bundle.ordered_tasks[i].name,
                                "dec_task_index": j,
                                "dec_task_name": base_bundle.ordered_tasks[j].name,
                                "sequence_lc_service_loss_mean": static_loss,
                                "sequence_lc_qos_mean": static_qos,
                                "sequence_mode_changes_mean": static_mode,
                                "sequence_hi_deadline_misses_sum": static_hi_sum,
                                "sequence_relative_lc_loss_reduction": reduction,
                                "sequence_mode_change_delta_ratio": None if baseline_mode in (None, "") or float(baseline_mode) == 0.0 else (float(static_mode) - float(baseline_mode)) / float(baseline_mode),
                            }
                            sequence_detail_rows.append(row)
                            sequence_rows_all.append(row)

        if baseline_loss is not None and baseline_mode is not None:
            sequence_qos_best = select_sequence_static_best(
                sequence_rows_all,
                baseline_loss=float(baseline_loss),
                baseline_mode_changes=float(baseline_mode),
                stable_delta=None,
                abs_tolerance=float(args.stable_static_mode_abs_tolerance),
            )
            add_sequence_best_fields("sequence_static_qos_best", main_row, sequence_qos_best, baseline_mode)
            for delta in stable_deltas:
                prefix = "sequence_stable005_static" if abs(float(delta) - 0.05) < 1e-12 else "sequence_stable010_static"
                sequence_stable_best = select_sequence_static_best(
                    sequence_rows_all,
                    baseline_loss=float(baseline_loss),
                    baseline_mode_changes=float(baseline_mode),
                    stable_delta=float(delta),
                    abs_tolerance=float(args.stable_static_mode_abs_tolerance),
                )
                add_sequence_best_fields(prefix, main_row, sequence_stable_best, baseline_mode)

    single005 = main_row["single_stable005_static_relative_lc_loss_reduction"]
    single010 = main_row["single_stable010_static_relative_lc_loss_reduction"]
    seq005 = main_row["sequence_stable005_static_relative_lc_loss_reduction"]
    seq010 = main_row["sequence_stable010_static_relative_lc_loss_reduction"]
    main_row["dqn_proxy_stable005_relative_lc_loss_reduction"] = max(
        0.0 if single005 is None else float(single005),
        0.0 if seq005 is None else float(seq005),
    )
    main_row["dqn_proxy_stable010_relative_lc_loss_reduction"] = max(
        0.0 if single010 is None else float(single010),
        0.0 if seq010 is None else float(seq010),
    )

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
        min_stable_static_sweep_reduction=float(args.min_stable_static_sweep_reduction),
        stable_static_delta=float(args.stable_static_delta),
        allow_relaxed_stable=bool(args.allow_relaxed_stable_static),
        exclude_tradeoff_only=bool(args.exclude_tradeoff_only),
        stable005_static_relative_lc_loss_reduction=main_row["stable005_static_relative_lc_loss_reduction"],
        stable010_static_relative_lc_loss_reduction=main_row["stable010_static_relative_lc_loss_reduction"],
        tradeoff_only_flag_005=bool(main_row["tradeoff_only_flag_005"]),
        tradeoff_only_flag_010=bool(main_row["tradeoff_only_flag_010"]),
        single_stable005_static_relative_lc_loss_reduction=main_row["single_stable005_static_relative_lc_loss_reduction"],
        single_stable010_static_relative_lc_loss_reduction=main_row["single_stable010_static_relative_lc_loss_reduction"],
        min_single_stable_static_sweep_reduction=float(args.min_single_stable_sweep_reduction),
        require_single_action_improvement=bool(args.require_single_action_improvement),
    )
    main_row["recommended_for_qos_dqn"] = rec
    main_row["qos_reject_reason"] = reason
    main_row["single_improvement_type"] = classify_single_improvement_type(
        single_static_qos_best_found_valid=bool(main_row["single_static_qos_best_found_valid"]),
        single_static_qos_best_relative_lc_loss_reduction=main_row["single_static_qos_best_relative_lc_loss_reduction"],
        single_stable005_static_relative_lc_loss_reduction=main_row["single_stable005_static_relative_lc_loss_reduction"],
        single_stable010_static_relative_lc_loss_reduction=main_row["single_stable010_static_relative_lc_loss_reduction"],
    )

    return main_row, detail_rows, single_task_detail_rows, sequence_detail_rows


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
    parser.add_argument(
        "--stable-static-mode-deltas",
        type=str,
        default="0.05,0.10",
        help="稳定 static sweep 的 mode change 约束增量，逗号分隔，例如 0.05,0.10。",
    )
    parser.add_argument(
        "--stable-static-mode-abs-tolerance",
        type=float,
        default=0.0,
        help="稳定 static sweep 的绝对 mode change 容忍项，会加到 baseline*(1+delta) 上。",
    )
    parser.add_argument(
        "--static-sweep-stage",
        choices=["none", "quick", "full"],
        default="full",
        help="static sweep 成本档位：quick 使用小网格，full 使用 sweep-* 比例参数。",
    )
    parser.add_argument("--quick-sweep-inc-ratios", type=str, default="0,0.015,0.025,0.035")
    parser.add_argument("--quick-sweep-dec-ratios", type=str, default="0,0.015")
    parser.add_argument("--static-sweep-end-time", type=int, default=None)
    parser.add_argument("--static-sweep-eval-seeds", type=str, default="")
    parser.add_argument("--min-static-sweep-reduction", type=float, default=0.05)
    parser.add_argument("--min-stable-static-sweep-reduction", type=float, default=0.0)
    parser.add_argument("--stable-static-delta", type=float, default=0.05)
    parser.add_argument("--allow-relaxed-stable-static", action="store_true")
    parser.add_argument("--exclude-tradeoff-only", action="store_true")
    parser.add_argument("--static-sweep-detail-output", type=str, default="")
    parser.add_argument("--static-sweep-policy", choices=["budget_scale"], default="budget_scale")
    parser.add_argument("--budget-floor-ratio", type=float, default=0.9)
    parser.add_argument("--enable-single-task-sweep", action="store_true", help="启用 single-action 对齐的单任务静态扰动 sweep。")
    parser.add_argument("--single-task-sweep-stage", choices=["quick", "full"], default="quick", help="single-task sweep 成本档位。")
    parser.add_argument("--single-task-sweep-actions", type=str, default="increase,decrease", help="逗号分隔动作集合。")
    parser.add_argument("--single-task-repeat-counts", type=str, default="1", help="单任务扰动重复次数，例如 1 或 1,2。")
    parser.add_argument("--single-task-top-k-by-headroom", type=int, default=0, help="quick 模式下只扫描 headroom 最大的前 K 个任务。")
    parser.add_argument("--single-task-sweep-detail-output", type=str, default="", help="输出 single-task sweep 明细 CSV。")
    parser.add_argument("--min-single-stable-sweep-reduction", type=float, default=0.0, help="single stable improvement 最小阈值。")
    parser.add_argument("--enable-single-sequence-sweep", action="store_true", help="启用固定 single-action 序列 proxy sweep。")
    parser.add_argument("--single-sequence-patterns", type=str, default="inc_repeat,dec_repeat,inc_dec_alternate", help="固定序列模式。")
    parser.add_argument("--single-sequence-lengths", type=str, default="2,4", help="固定动作序列长度。")
    parser.add_argument("--single-sequence-top-k-tasks", type=int, default=4, help="序列 sweep 候选任务 top K。")
    parser.add_argument("--single-sequence-sweep-detail-output", type=str, default="", help="输出 sequence sweep 明细 CSV。")
    parser.add_argument("--require-single-action-improvement", action="store_true", help="推荐阶段要求 single-action stable 改进。")
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
    if args.static_sweep_stage == "none":
        args.enable_static_sweep = False
    static_eval_seeds = parse_int_list_or_half_open_range(args.static_sweep_eval_seeds) if args.static_sweep_eval_seeds.strip() else list(eval_seeds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, object]] = []
    all_detail_rows: list[dict[str, object]] = []
    all_single_task_detail_rows: list[dict[str, object]] = []
    all_sequence_detail_rows: list[dict[str, object]] = []
    bucket_counter: Counter[str] = Counter()
    error_count = 0

    for idx, candidate_seed in enumerate(candidate_seeds, start=1):
        row, detail_rows, single_detail_rows, sequence_detail_rows = scan_candidate_seed(args, candidate_seed, eval_seeds, static_eval_seeds)
        main_rows.append(row)
        all_detail_rows.extend(detail_rows)
        all_single_task_detail_rows.extend(single_detail_rows)
        all_sequence_detail_rows.extend(sequence_detail_rows)
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

    if args.enable_single_task_sweep and args.single_task_sweep_detail_output:
        detail_path = Path(args.single_task_sweep_detail_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_fieldnames = [
            "candidate_seed",
            "target_task_index",
            "target_task_name",
            "target_task_criticality",
            "single_action",
            "repeat_count",
            "inc_ratio",
            "dec_ratio",
            "single_task_lc_service_loss_mean",
            "single_task_lc_qos_mean",
            "single_task_mode_changes_mean",
            "single_task_hi_deadline_misses_sum",
            "single_task_relative_lc_loss_reduction",
            "single_task_mode_change_delta_ratio",
            "invalid_single_task_candidate",
            "invalid_reason",
        ]
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
            writer.writeheader()
            for row in all_single_task_detail_rows:
                writer.writerow({key: to_csv_value(row.get(key)) for key in detail_fieldnames})

    if args.enable_single_sequence_sweep and args.single_sequence_sweep_detail_output:
        detail_path = Path(args.single_sequence_sweep_detail_output)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_fieldnames = [
            "candidate_seed",
            "sequence_pattern",
            "sequence_length",
            "inc_task_index",
            "inc_task_name",
            "dec_task_index",
            "dec_task_name",
            "sequence_lc_service_loss_mean",
            "sequence_lc_qos_mean",
            "sequence_mode_changes_mean",
            "sequence_hi_deadline_misses_sum",
            "sequence_relative_lc_loss_reduction",
            "sequence_mode_change_delta_ratio",
        ]
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
            writer.writeheader()
            for row in all_sequence_detail_rows:
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
