"""Task-level cancellation source 与 controllability 诊断工具。

本模块只做统计，不改运行时语义，输入统一来自已完成仿真的 `SimulationResult`。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from amc_py.models import Criticality, Task
from amc_py.runtime_models import SimulationResult


def _is_hi_task(task: Task) -> bool:
    """统一判断任务是否为 HI 关键级。"""

    criticality = getattr(task, "criticality", None)
    if criticality is Criticality.HI:
        return True
    name = getattr(criticality, "name", str(criticality)).upper()
    return name.endswith("HI") or name == "HI"


def summarize_task_level_cancellations(
    *,
    ordered_tasks: list[Task],
    result: SimulationResult,
) -> dict[str, Any]:
    """汇总 task-level cancellation 统计。

    设计约束：
    1. 统计以 runtime `result` 为准，不从外部推断；
    2. HI 任务理论上不会产生 LO cancellation，但仍保留输出以便做 sanity check；
    3. aggregate 只聚焦 LO cancellation source，避免被 HI 任务零值干扰。
    """

    released_counter: Counter[str] = Counter(job.task.name for job in result.jobs)
    completed_counter: Counter[str] = Counter(
        job.task.name for job in result.jobs if (getattr(job, "completion_time", None) is not None and not bool(job.dropped))
    )
    dropped_counter: Counter[str] = Counter(job.task.name for job in result.jobs if bool(job.dropped))
    cancelled_counter: Counter[str] = Counter(event.task for event in result.job_cancellations)

    total_cancelled_jobs = int(sum(cancelled_counter.values()))
    end_time = int(getattr(result, "end_time", 0))
    per_1m_scale = float(max(1, end_time))

    per_task_rows: list[dict[str, Any]] = []
    lo_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(ordered_tasks):
        released_jobs = int(released_counter.get(task.name, 0))
        completed_jobs = int(completed_counter.get(task.name, 0))
        dropped_jobs = int(dropped_counter.get(task.name, 0))
        cancelled_jobs = int(cancelled_counter.get(task.name, 0))
        cancel_ratio_over_released = float(cancelled_jobs) / float(max(1, released_jobs))
        cancel_share_of_total = float(cancelled_jobs) / float(max(1, total_cancelled_jobs))
        cancelled_per_1m = float(cancelled_jobs) / per_1m_scale * 1_000_000.0

        row = {
            "task_index": int(task_index),
            "task_name": str(task.name),
            "criticality": getattr(task.criticality, "name", str(task.criticality)),
            "period": int(task.period),
            "deadline": int(task.deadline),
            "c_lo": int(task.c_lo),
            "c_hi": int(task.c_hi),
            "released_jobs": released_jobs,
            "completed_jobs": completed_jobs,
            "dropped_jobs": dropped_jobs,
            "cancelled_jobs": cancelled_jobs,
            "cancel_ratio_over_released": cancel_ratio_over_released,
            "cancel_share_of_total": cancel_share_of_total,
            "cancelled_per_1m": cancelled_per_1m,
        }
        per_task_rows.append(row)
        if not _is_hi_task(task):
            lo_rows.append(row)

    total_lo_released_jobs = int(sum(int(row["released_jobs"]) for row in lo_rows))
    total_lo_cancelled_jobs = int(sum(int(row["cancelled_jobs"]) for row in lo_rows))
    lo_rows_by_cancel = sorted(
        lo_rows,
        key=lambda item: (int(item["cancelled_jobs"]), -int(item["task_index"])),
        reverse=True,
    )
    if total_lo_cancelled_jobs <= 0:
        lo_shares: list[float] = []
    else:
        lo_shares = [
            float(row["cancelled_jobs"]) / float(total_lo_cancelled_jobs)
            for row in lo_rows_by_cancel
        ]

    def _top_row(index: int) -> dict[str, Any] | None:
        if index < len(lo_rows_by_cancel):
            return lo_rows_by_cancel[index]
        return None

    top1 = _top_row(0)
    top2 = _top_row(1)
    top3 = _top_row(2)
    top1_share = float(lo_shares[0]) if len(lo_shares) >= 1 else 0.0
    top2_share = float(sum(lo_shares[:2])) if len(lo_shares) >= 2 else top1_share
    top3_share = float(sum(lo_shares[:3])) if len(lo_shares) >= 3 else top2_share

    aggregate = {
        "task_level_total_lo_released_jobs": total_lo_released_jobs,
        "task_level_total_lo_cancelled_jobs": total_lo_cancelled_jobs,
        "task_level_top1_cancel_share": top1_share,
        "task_level_top2_cancel_share": top2_share,
        "task_level_top3_cancel_share": top3_share,
        "task_level_cancel_concentration_hhi": float(sum(share * share for share in lo_shares)),
        "task_level_num_cancelled_lo_tasks": int(sum(1 for row in lo_rows if int(row["cancelled_jobs"]) > 0)),
        "task_level_num_lo_tasks_with_cancel_ratio_ge_10pct": int(
            sum(1 for row in lo_rows if float(row["cancel_ratio_over_released"]) >= 0.10)
        ),
        "task_level_num_lo_tasks_with_cancel_ratio_ge_20pct": int(
            sum(1 for row in lo_rows if float(row["cancel_ratio_over_released"]) >= 0.20)
        ),
        "task_level_max_task_cancel_ratio": float(
            max((float(row["cancel_ratio_over_released"]) for row in lo_rows), default=0.0)
        ),
        "task_level_top1_cancel_task_name": str(top1["task_name"]) if top1 else "",
        "task_level_top2_cancel_task_name": str(top2["task_name"]) if top2 else "",
        "task_level_top3_cancel_task_name": str(top3["task_name"]) if top3 else "",
        "task_level_top1_cancel_task_index": int(top1["task_index"]) if top1 else -1,
        "task_level_top2_cancel_task_index": int(top2["task_index"]) if top2 else -1,
        "task_level_top3_cancel_task_index": int(top3["task_index"]) if top3 else -1,
    }
    return {"per_task_rows": per_task_rows, "aggregate": aggregate}


def extract_valid_action_task_indices(env, mask: tuple[bool, ...]) -> dict[str, set[int]]:
    """从 action mask 提取可 increase / decrease 的任务索引集合。"""

    valid_increase_indices: set[int] = set()
    valid_decrease_indices: set[int] = set()
    for action, is_valid in zip(env._actions, mask, strict=True):  # noqa: SLF001
        if not is_valid:
            continue
        if bool(getattr(action, "is_noop", False)):
            continue
        increase_idx = getattr(action, "increase_idx", None)
        if increase_idx is not None:
            valid_increase_indices.add(int(increase_idx))
        for idx in getattr(action, "decrease_indices", ()):
            valid_decrease_indices.add(int(idx))
    return {
        "valid_increase_indices": valid_increase_indices,
        "valid_decrease_indices": valid_decrease_indices,
    }


def compute_valid_action_cancel_coverage(
    *,
    ordered_tasks: list[Task],
    per_task_rows: list[dict[str, Any]],
    valid_increase_indices: set[int],
    valid_decrease_indices: set[int],
) -> dict[str, Any]:
    """计算 valid action 对 cancellation source 的覆盖度指标。

    指标设计说明：
    1. coverage 统计“被 valid action 覆盖到的取消作业量占比”；
    2. top-k hit 统计“主要 cancellation source 是否命中 valid action 候选”；
    3. increase / decrease 两侧完全对称，便于后续做可控性对照分析。
    """

    task_by_index: dict[int, Task] = {idx: task for idx, task in enumerate(ordered_tasks)}
    lo_rows = [row for row in per_task_rows if not _is_hi_task(task_by_index[int(row["task_index"])])]
    lo_rows_cancelled = [row for row in lo_rows if int(row["cancelled_jobs"]) > 0]
    lo_rows_sorted = sorted(lo_rows_cancelled, key=lambda item: (int(item["cancelled_jobs"]), -int(item["task_index"])), reverse=True)

    total_lo_cancelled = int(sum(int(row["cancelled_jobs"]) for row in lo_rows))
    covered_increase_cancelled = int(
        sum(
            int(row["cancelled_jobs"])
            for row in lo_rows
            if int(row["task_index"]) in valid_increase_indices
        )
    )
    covered_decrease_cancelled = int(
        sum(
            int(row["cancelled_jobs"])
            for row in lo_rows
            if int(row["task_index"]) in valid_decrease_indices
        )
    )
    top1_idx = int(lo_rows_sorted[0]["task_index"]) if len(lo_rows_sorted) >= 1 else -1
    top2_indices = [int(row["task_index"]) for row in lo_rows_sorted[:2]]
    top3_indices = [int(row["task_index"]) for row in lo_rows_sorted[:3]]
    increase_cancelled_rows = [row for row in lo_rows_cancelled if int(row["task_index"]) in valid_increase_indices]
    increase_top_row = max(increase_cancelled_rows, key=lambda item: int(item["cancelled_jobs"])) if increase_cancelled_rows else None
    decrease_cancelled_rows = [row for row in lo_rows_cancelled if int(row["task_index"]) in valid_decrease_indices]
    decrease_top_row = max(decrease_cancelled_rows, key=lambda item: int(item["cancelled_jobs"])) if decrease_cancelled_rows else None

    return {
        "valid_increase_cancel_coverage": float(covered_increase_cancelled) / float(max(1, total_lo_cancelled)),
        "valid_decrease_cancel_coverage": float(covered_decrease_cancelled) / float(max(1, total_lo_cancelled)),
        "valid_increase_lo_cancel_coverage": float(covered_increase_cancelled) / float(max(1, total_lo_cancelled)),
        "valid_increase_top1_cancel_hit": 1 if top1_idx in valid_increase_indices else 0,
        "valid_increase_top2_cancel_hit_count": int(sum(1 for idx in top2_indices if idx in valid_increase_indices)),
        "valid_increase_top3_cancel_hit_count": int(sum(1 for idx in top3_indices if idx in valid_increase_indices)),
        "valid_increase_cancelled_task_count": int(
            sum(1 for row in lo_rows_cancelled if int(row["task_index"]) in valid_increase_indices)
        ),
        "valid_increase_cancelled_task_share": float(
            sum(1 for row in lo_rows_cancelled if int(row["task_index"]) in valid_increase_indices)
        )
        / float(max(1, len(lo_rows_cancelled))),
        "valid_increase_top_cancel_task_name": (
            str(increase_top_row["task_name"]) if increase_top_row is not None else ""
        ),
        "valid_increase_top_cancel_task_index": (
            int(increase_top_row["task_index"]) if increase_top_row is not None else -1
        ),
        # decrease 侧新增指标：含义仅为“主要 cancellation source 是否可被 decrease 覆盖”，
        # 不代表 decrease 动作一定带来 QoS 改善。
        "valid_decrease_top1_cancel_hit": 1 if top1_idx in valid_decrease_indices else 0,
        "valid_decrease_top2_cancel_hit_count": int(sum(1 for idx in top2_indices if idx in valid_decrease_indices)),
        "valid_decrease_top3_cancel_hit_count": int(sum(1 for idx in top3_indices if idx in valid_decrease_indices)),
        "valid_decrease_cancelled_task_count": int(
            sum(1 for row in lo_rows_cancelled if int(row["task_index"]) in valid_decrease_indices)
        ),
        "valid_decrease_cancelled_task_share": float(
            sum(1 for row in lo_rows_cancelled if int(row["task_index"]) in valid_decrease_indices)
        )
        / float(max(1, len(lo_rows_cancelled))),
        "valid_decrease_top_cancel_task_name": (
            str(decrease_top_row["task_name"]) if decrease_top_row is not None else ""
        ),
        "valid_decrease_top_cancel_task_index": (
            int(decrease_top_row["task_index"]) if decrease_top_row is not None else -1
        ),
    }
