from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.bridge.state_relation import p0_smt_relation_fields
from formal_toolchain.reference.p0_transition_contract import (
    ReferenceP0Environment,
    build_reference_p0_identity_query,
)
from formal_toolchain.reference.reference_state import ReferenceState
from formal_toolchain.bridge.logical_events import LogicalEventKind
from formal_toolchain.core.z3_resources import new_context, new_solver


def _encode_job_key(key: tuple[str, int] | int | None) -> int:
    if key is None:
        return 0
    if isinstance(key, int):
        return key
    name, idx = key
    h = 0
    for c in name:
        h = h * 31 + ord(c)
    return h * 100000 + idx


def _mode_int(mode: str) -> int:
    return 1 if mode == "HI" else 0


def _criticality_int(crit: str) -> int:
    return 1 if crit == "HI" else 0


def _category_int(release_class: str) -> int:
    mapping = {
        "LO_PRIMARY": 0,
        "LO_DEGRADED_HI_MODE": 1,
        "HI_ABNORMAL": 2,
        "HI_ABNORMAL_SWITCH_TRIGGER": 3,
        "LO_PRIMARY_SAME_BATCH_SWITCH_TIME": 4,
    }
    return mapping.get(release_class, 0)



def project_executable_reference_state(
    *,
    state: ReferenceState,
    taskset: Any,
    bounds: P0ModelBounds,
    job_slot_by_key: Mapping[tuple[str, int], int],
    task_slot_by_name: Mapping[str, int],
    affected_job_key: tuple[str, int] | None = None,
    affected_task_name: str | None = None,
) -> dict[str, int]:
    result: dict[str, int] = {}
    a_jk = affected_job_key if affected_job_key is not None else state.running
    a_job = state.jobs.get(a_jk) if a_jk is not None else None
    affected_record = (
        state.released.get(a_jk)
        if a_jk is not None
        else None
    )
    total_service = sum(job.executed for job in state.jobs.values())
    total_remaining = sum(max(0, job.budget - job.executed) for job in state.jobs.values())
    running_job = state.jobs.get(state.running) if state.running is not None else None
    focus_job = a_job if affected_job_key is not None else None
    mode_val = _mode_int(state.mode)
    _ = taskset

    hi_miss = int(any(miss.criticality == "HI" for miss in state.misses))
    hi_complete = 1 if any(
        record.terminal_kind == "COMPLETED" and record.job_key in state.released
        and state.released[record.job_key].criticality == "HI"
        for record in state.terminal.values()
    ) else 0

    result["time"] = state.time
    result["service"] = a_job.executed if a_job is not None else total_service
    result["remaining"] = max(0, a_job.budget - a_job.executed) if a_job is not None else total_remaining
    result["budget"] = a_job.budget if a_job is not None else 0
    result["miss"] = hi_miss
    result["mode"] = mode_val
    result["active"] = len(state.jobs)
    result["ready"] = len(state.ready_order)
    result["running"] = 1 if state.running is not None else 0
    running_record = (
        state.released.get(state.running)
        if state.running is not None
        else None
    )
    result["priority"] = (
        running_record.priority_index
        if running_record is not None
        else 0
    )
    result["release"] = focus_job.release_time if focus_job is not None else 0
    result["deadline"] = focus_job.absolute_deadline if focus_job is not None else 0
    result["category"] = _category_int(focus_job.release_class) if focus_job is not None else 0
    result["job_key"] = _encode_job_key(a_jk) if focus_job is not None else 0
    result["hi_complete"] = hi_complete
    result["future_budget"] = 0
    result["affected_job_key"] = _encode_job_key(a_jk)
    result["affected_job_active"] = 1 if a_jk is not None and a_jk in state.jobs else 0
    result["affected_job_ready"] = 1 if a_jk is not None and a_jk in state.ready_order else 0
    result["affected_job_running"] = 1 if a_jk is not None and state.running == a_jk else 0
    result["affected_job_priority"] = (
        affected_record.priority_index
        if affected_record is not None
        else 0
    )
    result["affected_job_release"] = a_job.release_time if a_job is not None else 0
    result["affected_job_deadline"] = a_job.absolute_deadline if a_job is not None else 0
    result["affected_job_category"] = _category_int(a_job.release_class) if a_job is not None else 0
    result["affected_job_budget"] = (
        affected_record.release_budget
        if affected_record is not None
        else 0
    )
    result["affected_job_demand"] = (
        a_job.removal_demand
        if a_job is not None
        else 0
    )
    result["affected_job_service"] = a_job.executed if a_job is not None else 0
    result["affected_job_hi_complete"] = 0
    result["affected_job_hi_miss"] = 1 if a_jk is not None and any(
        miss.job_key == a_jk for miss in state.misses
    ) else 0
    result["affected_task_budget"] = 0
    result["next_controller_boundary"] = 0
    result["event_job_key"] = _encode_job_key(a_jk)
    result["running_job_key"] = _encode_job_key(state.running) if state.running is not None else 0
    result["selected_job_key"] = _encode_job_key(state.running) if state.running is not None else 0
    result["ready_empty"] = 1 if len(state.ready_order) == 0 else 0
    future_events = [e for e in state.frontier if e.time > state.time]
    result["queue_min_time"] = min(e.time for e in future_events) if future_events else 2147483647
    result["queue_min_kind"] = 0
    result["queue_min_job_key"] = 0
    result["queue_min_token"] = 0
    release_times = [
        e.time for e in state.frontier
        if e.kind in (LogicalEventKind.REL, LogicalEventKind.ARR_BATCH)
    ]
    deadline_times = [
        e.time for e in state.frontier
        if e.kind == LogicalEventKind.DDL
    ]
    result["queue_next_release_time"] = min(release_times) if release_times else 2147483647
    result["queue_next_deadline_time"] = min(deadline_times) if deadline_times else 2147483647
    result["queue_event_count"] = len(state.frontier)
    result["queue_token_epoch"] = 0

    for slot in range(bounds.job_slots):
        result[f"job_{slot}_present"] = 0
        result[f"job_{slot}_key"] = 0
        result[f"job_{slot}_active"] = 0
        result[f"job_{slot}_ready"] = 0
        result[f"job_{slot}_running"] = 0
        result[f"job_{slot}_priority"] = 0
        result[f"job_{slot}_release"] = 0
        result[f"job_{slot}_deadline"] = 0
        result[f"job_{slot}_category"] = 0
        result[f"job_{slot}_criticality"] = 0
        result[f"job_{slot}_mode"] = 0
        result[f"job_{slot}_released_mode"] = 0
        result[f"job_{slot}_is_degraded"] = 0
        result[f"job_{slot}_budget"] = 0
        result[f"job_{slot}_demand"] = 0
        result[f"job_{slot}_service"] = 0
        result[f"job_{slot}_completion_token"] = 0
        result[f"job_{slot}_overrun_token"] = 0
        result[f"job_{slot}_hi_complete"] = 0
        result[f"job_{slot}_hi_miss"] = 0

    for jk, slot in job_slot_by_key.items():
        record = state.released.get(jk)
        job = state.jobs.get(jk)
        terminal = state.terminal.get(jk)
        if record is None:
            continue
        executed_service = (
            job.executed
            if job is not None
            else terminal.executed_service if terminal is not None else 0
        )
        result[f"job_{slot}_present"] = int(job is not None)
        result[f"job_{slot}_key"] = _encode_job_key(jk)
        result[f"job_{slot}_active"] = int(job is not None)
        result[f"job_{slot}_ready"] = int(job is not None and jk in state.ready_order)
        result[f"job_{slot}_running"] = int(job is not None and state.running == jk)
        result[f"job_{slot}_priority"] = int(record.priority_index)
        result[f"job_{slot}_release"] = int(record.release_time)
        result[f"job_{slot}_deadline"] = int(record.absolute_deadline)
        result[f"job_{slot}_category"] = _category_int(record.release_class)
        result[f"job_{slot}_criticality"] = _criticality_int(record.criticality)
        result[f"job_{slot}_mode"] = _mode_int(record.released_mode)
        result[f"job_{slot}_released_mode"] = _mode_int(record.released_mode)
        result[f"job_{slot}_is_degraded"] = int(record.release_class == "LO_DEGRADED_HI_MODE")
        result[f"job_{slot}_budget"] = int(record.release_budget)
        result[f"job_{slot}_demand"] = int(record.removal_demand)
        result[f"job_{slot}_service"] = int(executed_service)
        result[f"job_{slot}_completion_token"] = 0
        result[f"job_{slot}_overrun_token"] = 0
        result[f"job_{slot}_hi_complete"] = 0
        result[f"job_{slot}_hi_miss"] = int(any(
            miss.job_key == jk and miss.criticality == "HI"
            for miss in state.misses
        ))

    for slot in range(bounds.task_slots):
        result[f"task_{slot}_present"] = 0
        result[f"task_{slot}_key"] = 0
        result[f"task_{slot}_criticality"] = 0
        result[f"task_{slot}_future_budget"] = 0
    for name, slot in task_slot_by_name.items():
        if slot >= bounds.task_slots:
            continue
        task = next(
            t for t in (taskset.tasks if hasattr(taskset, "tasks") else taskset["tasks"])
            if (t["name"] if isinstance(t, Mapping) else t.name) == name
        )
        result[f"task_{slot}_present"] = 1
        result[f"task_{slot}_key"] = slot + 1
        result[f"task_{slot}_criticality"] = _criticality_int(
            task["criticality"] if isinstance(task, Mapping) else task.criticality
        )
        result[f"task_{slot}_future_budget"] = state.ghost_future_budgets.get(
            name, 0
        )

    return result


def validate_executable_reference_p0_step(
    *,
    step: Any,
    taskset: Any,
    bounds: P0ModelBounds,
    environment: ReferenceP0Environment,
    job_slot_by_key: Mapping[tuple[str, int], int],
    task_slot_by_name: Mapping[str, int],
    affected_job_key: tuple[str, int] | None,
    affected_task_name: str | None,
) -> dict[str, Any]:
    before_projection = project_executable_reference_state(
        state=step.before,
        taskset=taskset,
        bounds=bounds,
        job_slot_by_key=job_slot_by_key,
        task_slot_by_name=task_slot_by_name,
        affected_job_key=affected_job_key,
        affected_task_name=affected_task_name,
    )

    after_projection = project_executable_reference_state(
        state=step.after,
        taskset=taskset,
        bounds=bounds,
        job_slot_by_key=job_slot_by_key,
        task_slot_by_name=task_slot_by_name,
        affected_job_key=affected_job_key,
        affected_task_name=affected_task_name,
    )
    smt2 = build_reference_p0_identity_query(
        case_id=step.case_id,
        before=before_projection,
        actual_after=after_projection,
        environment=environment,
        bounds=bounds,
    )
    try:
        import z3
    except ImportError as exc:
        raise ValueError("REFERENCE_P0_Z3_NOT_AVAILABLE") from exc
    context = new_context(z3)
    solver = new_solver(z3, context=context)
    try:
        solver.add(z3.parse_smt2_string(smt2, ctx=context))
    except z3.Z3Exception as exc:
        raise ValueError(
            "REFERENCE_P0_CANONICAL_DELTA_PARSE_FAILED:"
            f"{step.case_id}:{exc}"
        ) from exc
    result = solver.check()
    if result != z3.sat:
        raise ValueError(
            "REFERENCE_EXECUTABLE_P0_TRANSITION_MISMATCH:"
            f"{step.case_id}:{result}"
        )
    return {
        "status": "PASS",
        "case_id": step.case_id,
        "validator_backend": "z3-direct-canonical-delta-v1",
    }
