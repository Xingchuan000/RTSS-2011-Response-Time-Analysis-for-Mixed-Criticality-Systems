"""Phase K02/K03：纯 P0 timing state IR 与状态关系。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from formal_toolchain.core.hashing import sha256_object


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

    @property
    def remaining(self) -> int:
        return max(0, self.demand - self.service)


@dataclass(frozen=True, slots=True)
class P0ConcreteState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class P0ReferenceState:
    time: int
    mode: str
    active_jobs: tuple[P0Job, ...] = ()
    ready_jobs: tuple[tuple[str, int], ...] = ()
    running_job: tuple[str, int] | None = None
    global_future_budgets: tuple[tuple[str, int], ...] = ()
    miss_flags: tuple[tuple[str, int], ...] = ()


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
        "frame.other_task_budgets",
    )


def p0_smt_relation_fields() -> tuple[str, ...]:
    """Exact flat affected-object model declared in every transition query."""
    return (
        "time", "service", "remaining", "budget", "miss", "mode", "active",
        "ready", "running", "priority", "release", "deadline", "category",
        "job_key", "hi_complete", "future_budget", "affected_job_key",
        "affected_job_active", "affected_job_ready", "affected_job_running",
        "affected_job_priority", "affected_job_release", "affected_job_deadline",
        "affected_job_category", "affected_job_budget", "affected_job_demand",
        "affected_job_service", "affected_job_hi_complete", "affected_job_hi_miss",
        "affected_task_budget", "other_jobs_frame_unchanged",
        "other_task_budgets_frame_unchanged",
    )


def p0_state_relation_schema_hash() -> str:
    # This list is also consumed by case_templates when it builds the actual SMT
    # declarations.  Keeping the hash here prevents a descriptive Python-only
    # schema from being mistaken for the proved relation.
    return sha256_object({"schema": "P0_state_relation_v3_affected_job_frame",
                          "smt_fields": p0_smt_relation_fields(),
                          "python_relation_fields": p0_state_relation_schema()})


def remaining_remove(job: P0Job) -> int:
    """按 release-fixed demand 计算关系中的 concrete remaining。"""
    return max(0, job.demand - job.service)


def relation_holds(concrete: P0ConcreteState, reference: P0ReferenceState) -> bool:
    """检查 K03 的 job、时序字段、miss 与 future-budget state 关系。

    budget-update 的事件标签会被投影掉，但它产生的未来预算状态仍是
    timing-relevant state，不能在关系检查中静默忽略。
    """
    if (concrete.time, concrete.mode, concrete.ready_jobs, concrete.running_job,
            concrete.global_future_budgets, concrete.miss_flags) != (
            reference.time, reference.mode, reference.ready_jobs,
            reference.running_job, reference.global_future_budgets, reference.miss_flags):
        return False
    c_jobs = {job.job_key: job for job in concrete.active_jobs}
    r_jobs = {job.job_key: job for job in reference.active_jobs}
    if set(c_jobs) != set(r_jobs):
        return False
    for key, c_job in c_jobs.items():
        r_job = r_jobs[key]
        if (c_job.priority_index, c_job.release_time, c_job.deadline,
                c_job.release_category, c_job.release_budget, c_job.mode, c_job.state,
                c_job.hi_completed, c_job.hi_deadline_miss) != (
                r_job.priority_index, r_job.release_time, r_job.deadline,
                r_job.release_category, r_job.release_budget, r_job.mode, r_job.state,
                r_job.hi_completed, r_job.hi_deadline_miss):
            return False
        if remaining_remove(c_job) != r_job.remaining:
            return False
    return True
