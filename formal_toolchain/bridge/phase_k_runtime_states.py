"""Phase K 的 PreClosed(0) runtime/reference 状态输入。

该模块只负责把当前 target 的真实 runtime 初始化状态投影为 Phase K
需要的 concrete/reference state。它不生成任何 PASS certificate；证书仍由
``compile_phase_k`` 生成 candidate，再交给 fresh verifier 重新检查。
"""

from __future__ import annotations

from typing import Any, Mapping

from amc_py.event_runtime import EventRuntimeEngine
from amc_py.runtime_models import RuntimeConfig

from formal_toolchain.adapters.runtime_config_copy import copy_runtime_config
from formal_toolchain.adapters.formal_scenario_factory import build_formal_scenario
from .state_relation import P0ReferenceState, p0_state_from_runtime_engine


def build_preclosed_runtime_states(target: Any,
                                   reference_taskset: Mapping[str, Any]):
    """从 target 的真实配置构造 time-0 PreClosed concrete/reference 状态。"""

    cfg = target.runtime_config
    runtime_config = copy_runtime_config(cfg)
    if not hasattr(target, "scenario") or not hasattr(target.scenario, "actual_cost_for"):
        raise TypeError("target.scenario 必须实现 actual_cost_for(task, release_index)")
    scenario = build_formal_scenario(
        base_scenario=target.scenario,
        ordered_tasks=target.ordered_tasks,
    )
    engine = EventRuntimeEngine.build(
        ordered_tasks=target.ordered_tasks, scenario=scenario, config=runtime_config,
    )
    engine.run_until(0, include_boundary=True)
    concrete = p0_state_from_runtime_engine(engine)

    task_rows = {str(item["name"]): item for item in reference_taskset.get("tasks", [])}
    target_names = {str(task.name) for task in target.ordered_tasks}
    if set(task_rows) != target_names:
        raise ValueError("PreClosed reference task mapping 不完整")

    reference_jobs = []
    for job in concrete.active_jobs:
        task = task_rows[job.job_key[0]]
        raw = int(job.raw_actual_cost if job.raw_actual_cost is not None else job.demand)
        if job.is_degraded:
            degraded = task.get("degraded_cost")
            if not isinstance(degraded, int):
                raise ValueError("degraded LO 缺少 reference degraded_cost")
            demand = min(raw, degraded)
        elif job.criticality == "LO" and job.release_budget is not None:
            demand = min(raw, int(job.release_budget) + 1)
        else:
            demand = raw
        reference_jobs.append(type(job)(
            job_key=job.job_key, priority_index=int(task["priority_index"]),
            release_time=job.release_time, deadline=job.deadline,
            release_category=job.release_category, release_budget=job.release_budget,
            demand=demand, service=job.service, state=job.state, mode=job.mode,
            hi_completed=job.hi_completed, hi_deadline_miss=job.hi_deadline_miss,
            criticality=job.criticality, released_mode=job.released_mode,
            is_degraded=job.is_degraded, raw_actual_cost=raw,
            removal_demand=demand))

    demand_by_key = {job.job_key: job.remaining for job in reference_jobs}
    projected_queue = []
    for item in concrete.queue_projection:
        event_time, kind, task_name, release_index, token = item
        key = ((str(task_name), int(release_index))
               if task_name is not None and release_index is not None else None)
        if kind in {"BUDGET_UPDATE", "CONTROLLER", "OBSERVATION", "TREE", "MASK"}:
            continue
        projected_kind = ("JOB_COMPLETION"
                          if kind in {"BUDGET_OVERRUN", "PRIMARY_LO_CANCELLATION"}
                          else kind)
        projected_time = int(event_time)
        if projected_kind == "JOB_COMPLETION" and key in demand_by_key:
            projected_time = concrete.time + int(demand_by_key[key])
        projected_queue.append((projected_time, projected_kind, task_name,
                                release_index, token))
    reference = P0ReferenceState(
        time=concrete.time, mode=concrete.mode, active_jobs=tuple(reference_jobs),
        ready_jobs=tuple(job.job_key for job in reference_jobs if job.state == "active"),
        running_job=concrete.running_job,
        global_future_budgets=concrete.global_future_budgets,
        miss_flags=concrete.miss_flags, queue_projection=tuple(sorted(projected_queue)),
        next_controller_boundary=concrete.next_controller_boundary,
        next_timing_boundary=min((int(item[0]) for item in projected_queue
                                  if int(item[0]) >= concrete.time), default=None),
        released_ledger=concrete.released_ledger,
        terminal_ledger=concrete.terminal_ledger,
        miss_ledger=concrete.miss_ledger,
        effective_event_frontier=concrete.effective_event_frontier,
    )
    return concrete, reference


__all__ = ["build_preclosed_runtime_states"]
