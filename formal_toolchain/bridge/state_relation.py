"""Phase K02/K03：纯 P0 timing state IR 与状态关系。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.model_bounds import P0ModelBounds


@dataclass(frozen=True, slots=True)
class P0Job:
    """P0 job：只保留 timing-relevant 字段。"""
    job_key: tuple[str, int]
    priority_index: int
    release_time: int
    deadline: int
    release_category: str
    release_budget: int | None
    demand: int
    service: int = 0
    state: str = "active"
    mode: str = "LO"
    hi_completed: bool = False
    hi_deadline_miss: bool = False
    criticality: str = "LO"
    released_mode: str = "LO"
    is_degraded: bool = False
    # raw_actual_cost 保留 runtime 的原始执行需求；removal_demand 是按
    # release-fixed 规则计算、供 concrete/reference remaining 关系使用的需求。
    raw_actual_cost: int | None = None
    removal_demand: int | None = None

    @property
    def remaining(self) -> int:
        demand = self.demand if self.removal_demand is None else self.removal_demand
        return max(0, demand - self.service)


@dataclass(frozen=True, slots=True)
class P0ConcreteState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()
    queue_projection: tuple[tuple[Any, ...], ...] = ()
    next_controller_boundary: int | None = None
    next_timing_boundary: int | None = None


@dataclass(frozen=True, slots=True)
class P0ReferenceState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()
    queue_projection: tuple[tuple[Any, ...], ...] = ()
    next_controller_boundary: int | None = None
    next_timing_boundary: int | None = None


@dataclass(frozen=True, slots=True)
class P0Event:
    time: int
    kind: str
    job_key: tuple[str, int] | None = None
    payload: tuple[tuple[str, Any], ...] = ()


def p0_state_relation_schema() -> tuple[str, ...]:
    """状态关系实际比较的字段清单，供证明对象做版本绑定。"""
    return (
        "time", "mode", "active_jobs.job_key", "active_jobs.active", "active_jobs.ready",
        "active_jobs.running", "active_jobs.priority_index", "active_jobs.release_time",
        "active_jobs.deadline", "active_jobs.release_category", "active_jobs.release_budget",
        "active_jobs.demand", "active_jobs.service", "active_jobs.state", "active_jobs.mode",
        "active_jobs.hi_completed", "active_jobs.hi_deadline_miss", "ready_jobs",
        "running_job", "global_future_budgets", "miss_flags",
        "affected_job_key", "affected_job_active", "affected_job_ready",
        "affected_job_running", "affected_job_priority", "affected_job_release",
        "affected_job_deadline", "affected_job_category", "affected_job_budget",
        "affected_job_demand", "affected_job_service", "affected_job_hi_complete",
        "affected_job_hi_miss", "affected_task_budget", "frame.other_jobs",
        "frame.other_task_budgets", "queue_projection", "next_controller_boundary",
        "event_job_key", "running_job_key", "selected_job_key",
        "job_slots.pointwise", "task_budget_slots.pointwise", "queue_slots.pointwise",
        "queue_slots.minimum_future_time", "queue_slots.event_identity", "next_timing_boundary",
    )


def p0_smt_relation_fields(bounds: P0ModelBounds) -> tuple[str, ...]:
    """按 bounds 生成每条 transition query 实际声明的字段。"""
    scalar_fields = (
        "time", "service", "remaining", "budget", "miss", "mode", "active",
        "ready", "running", "priority", "release", "deadline", "category",
        "job_key", "hi_complete", "future_budget", "affected_job_key",
        "affected_job_active", "affected_job_ready", "affected_job_running",
        "affected_job_priority", "affected_job_release", "affected_job_deadline",
        "affected_job_category", "affected_job_budget", "affected_job_demand",
        "affected_job_service", "affected_job_hi_complete", "affected_job_hi_miss",
        "affected_task_budget",
        "next_controller_boundary", "event_job_key", "running_job_key",
        "selected_job_key",
        "ready_empty",
        # queue 只保留决定下一步 timing transition 的摘要，不展开 heap slot。
        "queue_min_time", "queue_min_kind", "queue_min_job_key", "queue_min_token",
        "queue_next_release_time", "queue_next_deadline_time", "queue_event_count",
        "queue_token_epoch",
    )
    job_fields = tuple(
        f"job_{slot}_{field}" for slot in range(bounds.job_slots)
        for field in ("present", "key", "active", "ready", "running", "priority",
                      "release", "deadline", "category", "criticality", "mode",
                      "released_mode", "is_degraded", "budget", "demand", "service",
                      "completion_token", "overrun_token", "hi_complete", "hi_miss")
    )
    task_fields = tuple(
        f"task_{slot}_{field}" for slot in range(bounds.task_slots)
        for field in ("present", "key", "criticality", "future_budget")
    )
    return scalar_fields + job_fields + task_fields


def p0_state_relation_schema_hash(bounds: P0ModelBounds) -> str:
    # This list is also consumed by case_templates when it builds the actual SMT
    # declarations.  Keeping the hash here prevents a descriptive Python-only
    # schema from being mistaken for the proved relation.
    return sha256_object({"schema": "p0_state_relation_v5_dynamic",
                          "model_bounds": bounds.to_dict(),
                          "smt_fields": p0_smt_relation_fields(bounds),
                          "python_relation_fields": p0_state_relation_schema()})


def p0_state_from_runtime_engine(engine: Any) -> P0ConcreteState:
    """从真实 ``EventRuntimeEngine`` 当前状态构造 PreClosed P0 state。

    该函数只读 engine，不创建或补写任何 synthetic job；调用方必须先把
    engine 推进到明确的 time-0 closure 边界。
    """
    jobs = []
    for job in tuple(engine.state.active_jobs):
        key = (str(job.task.name), int(job.release_index))
        raw_actual_cost = int(getattr(job, "original_actual_cost", job.actual_cost))
        if getattr(job, "removal_demand", None) is not None:
            removal_demand = int(job.removal_demand)
        elif bool(job.is_degraded):
            removal_demand = raw_actual_cost
        elif str(getattr(job.task.criticality, "value", job.task.criticality)) == "LO" and job.runtime_budget_at_release is not None:
            removal_demand = min(raw_actual_cost, int(job.runtime_budget_at_release) + 1)
        else:
            removal_demand = raw_actual_cost
        jobs.append(P0Job(
            job_key=key, priority_index=int(engine.priority_map[job.task.name]),
            release_time=int(job.release_time), deadline=int(job.absolute_deadline),
            release_category=("degraded" if bool(job.is_degraded) else "normal"),
            release_budget=None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            demand=raw_actual_cost, service=int(job.executed_time),
            state="dropped" if bool(job.dropped) else ("finished" if job.finished() else "active"),
            mode=engine.state.mode.name, hi_completed=bool(job.task.criticality.value == "HI" and job.finished()),
            hi_deadline_miss=any(m.task == job.task.name and m.release_index == job.release_index
                                 for m in engine.result.deadline_misses),
            criticality=str(getattr(job.task.criticality, "value", job.task.criticality)),
            released_mode=str(getattr(job.released_in_mode, "name", job.released_in_mode)),
            is_degraded=bool(job.is_degraded), raw_actual_cost=raw_actual_cost,
            removal_demand=removal_demand))
    active_keys = tuple(job.job_key for job in jobs if job.state not in {"dropped", "finished"})
    running = engine.state.running_job
    running_key = None if running is None else (str(running.task.name), int(running.release_index))
    queue_projection = []
    queue_snapshot = getattr(engine.queue, "snapshot", None)
    if callable(queue_snapshot):
        for item in queue_snapshot():
            queue_projection.append((int(item.time), str(item.event_type), item.task_name,
                                     item.release_index, item.token))
    else:
        for item in getattr(engine.queue, "_heap", ()):
            event = item[3]
            queue_projection.append((int(event.time), str(event.event_type.value), event.task_name,
                                     event.release_index, event.token))
    budgets = tuple(sorted((str(name), int(value)) for name, value in engine.runtime_budgets.budgets.items()))
    queue_projection = tuple(sorted(queue_projection))
    next_boundary = min((int(item[0]) for item in queue_projection
                         if int(item[0]) >= int(engine.current_time)), default=None)
    return P0ConcreteState(time=int(engine.current_time), mode=str(engine.state.mode.name),
                           active_jobs=tuple(jobs), ready_jobs=active_keys,
                           running_job=running_key, global_future_budgets=budgets,
                           miss_flags=tuple((str(m.task), int(m.release_index)) for m in engine.result.deadline_misses),
                           queue_projection=queue_projection,
                           next_controller_boundary=None,
                           next_timing_boundary=next_boundary)


def remaining_remove(job: P0Job) -> int:
    """按 release-fixed demand 计算关系中的 concrete remaining。"""
    demand = job.removal_demand if job.removal_demand is not None else job.demand
    return max(0, demand - job.service)


def relation_holds(concrete: P0ConcreteState, reference: P0ReferenceState) -> bool:
    """检查 K03 的 job、时序字段、miss 与 future-budget state 关系。

    budget-update 的事件标签会被投影掉，但它产生的未来预算状态仍是
    timing-relevant state，不能在关系检查中静默忽略。
    """
    if (concrete.time, concrete.mode, concrete.ready_jobs, concrete.running_job,
            concrete.global_future_budgets, concrete.miss_flags,
            concrete.queue_projection, concrete.next_controller_boundary,
            concrete.next_timing_boundary) != (
            reference.time, reference.mode, reference.ready_jobs,
            reference.running_job, reference.global_future_budgets, reference.miss_flags,
            reference.queue_projection, reference.next_controller_boundary,
            reference.next_timing_boundary):
        return False
    c_jobs = {job.job_key: job for job in concrete.active_jobs}
    r_jobs = {job.job_key: job for job in reference.active_jobs}
    if set(c_jobs) != set(r_jobs):
        return False
    for key, c_job in c_jobs.items():
        r_job = r_jobs[key]
        if (c_job.priority_index, c_job.release_time, c_job.deadline,
                c_job.release_category, c_job.release_budget, c_job.mode, c_job.state,
                c_job.hi_completed, c_job.hi_deadline_miss, c_job.criticality,
                c_job.released_mode, c_job.is_degraded, c_job.raw_actual_cost) != (
                r_job.priority_index, r_job.release_time, r_job.deadline,
                r_job.release_category, r_job.release_budget, r_job.mode, r_job.state,
                r_job.hi_completed, r_job.hi_deadline_miss, r_job.criticality,
                r_job.released_mode, r_job.is_degraded, r_job.raw_actual_cost):
            return False
        if c_job.remaining != r_job.remaining:
            return False
    return True
