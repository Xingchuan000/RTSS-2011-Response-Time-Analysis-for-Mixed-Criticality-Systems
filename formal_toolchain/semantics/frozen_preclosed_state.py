"""Frozen C-AMC-sem/P0 PreClosed(0) semantics.

This module is the authoritative boot projection used by N4/N5 verification.
It deliberately does not import or execute ``amc_py.event_runtime``.  Mutable
experiment runtimes (including q-AMC) may evolve without changing the already
certified C-AMC-sem transition system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.bridge.state_relation import P0ConcreteState, P0Job, P0ReferenceState
from formal_toolchain.adapters.formal_runtime_snapshot import (
    FormalRuntimeSnapshot,
    ReleasedJobRecord,
    TokenState,
)
from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.semantics.frozen_runtime_contract import (
    CONTRACT_VERSION,
    frozen_contract_manifest,
)


@dataclass(frozen=True, slots=True)
class FrozenRawEvent:
    """Minimal raw event record consumed by ``effective_frontier``."""

    time: int
    event_type: str
    task_name: str | None = None
    release_index: int | None = None
    token: int | None = None
    fifo_rank: int = 0


def _enum_name(value: Any) -> str:
    return str(getattr(value, "value", getattr(value, "name", value)))


def _task_row_map(reference_taskset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = reference_taskset.get("tasks", ())
    result = {str(row["name"]): row for row in rows}
    if len(result) != len(tuple(rows)):
        raise ValueError("FROZEN_PRECLOSED_REFERENCE_TASK_NAMES_NOT_UNIQUE")
    return result


def _runtime_budget(task: Any, row: Mapping[str, Any]) -> int:
    for attr in ("initial_runtime_budget", "runtime_budget", "budget"):
        value = getattr(task, attr, None)
        if value is not None:
            return int(value)
    for key in ("initial_runtime_budget", "c_lo", "code_c_lo", "C_LO"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return int(getattr(task, "c_lo"))


def _config_bool(cfg: Any, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


def _end_time(cfg: Any) -> int | None:
    value = getattr(cfg, "end_time", None)
    return None if value is None else int(value)


def _scenario_cost(target: Any, task: Any) -> int:
    scenario = getattr(target, "scenario", None)
    method = getattr(scenario, "actual_cost_for", None)
    if not callable(method):
        raise TypeError("FROZEN_PRECLOSED_SCENARIO_ACTUAL_COST_INTERFACE_MISSING")
    value = int(method(task, 0))
    if value <= 0:
        raise ValueError("FROZEN_PRECLOSED_NONPOSITIVE_ACTUAL_COST")
    return value


def _reference_degraded_cost(row: Mapping[str, Any], task: Any, cfg: Any) -> int:
    value = row.get("degraded_cost")
    if value is not None:
        return int(value)
    # Fallback only for old reference-taskset schemas.  Python round implements
    # ties-to-even, matching the frozen runtime contract.
    ratio = float(getattr(cfg, "c_amc_sem_lo_degradation_ratio", 0.5))
    c_lo = int(getattr(task, "c_lo", row.get("c_lo", row.get("code_c_lo", 1))))
    return max(1, min(c_lo, int(round(ratio * c_lo))))


def _criticality(task: Any) -> str:
    value = _enum_name(getattr(task, "criticality", "LO"))
    if value not in {"LO", "HI"}:
        raise ValueError(f"FROZEN_PRECLOSED_CRITICALITY_INVALID:{value}")
    return value


def _c_lo(task: Any, row: Mapping[str, Any]) -> int:
    return int(getattr(task, "c_lo", row.get("c_lo", row.get("code_c_lo"))))


def build_frozen_preclosed_bundle(
    target: Any,
    reference_taskset: Mapping[str, Any],
) -> tuple[P0ConcreteState, P0ReferenceState, FormalRuntimeSnapshot]:
    """Build the authoritative C-AMC-sem ``PreClosed(0)`` pair.

    The input oracle is queried exactly once for release index zero.  The
    resulting jobs, release classes and event frontier are derived from the
    frozen contract rather than from mutable runtime control flow.
    """

    tasks = tuple(getattr(target, "ordered_tasks", ()))
    if not tasks:
        raise ValueError("FROZEN_PRECLOSED_TASKSET_EMPTY")
    rows = _task_row_map(reference_taskset)
    names = tuple(str(task.name) for task in tasks)
    if set(rows) != set(names):
        raise ValueError("FROZEN_PRECLOSED_REFERENCE_TASK_MAPPING_INCOMPLETE")

    cfg = getattr(target, "runtime_config", None)
    semantics = _enum_name(getattr(cfg, "semantics", "C_AMC_SEM"))
    if semantics != "C_AMC_SEM":
        raise ValueError(f"FROZEN_PRECLOSED_ROUTE_REQUIRES_C_AMC_SEM:{semantics}")

    actual = {str(task.name): _scenario_cost(target, task) for task in tasks}
    abnormal_hi = [
        task for task in tasks
        if _criticality(task) == "HI" and actual[str(task.name)] > _c_lo(task, rows[str(task.name)])
    ]
    mode = "HI" if abnormal_hi else "LO"
    trigger_name = None if not abnormal_hi else str(min(
        abnormal_hi, key=lambda task: names.index(str(task.name))
    ).name)
    primary_on_switch = _config_bool(cfg, "c_amc_sem_primary_on_switch_time", True)

    concrete_jobs: list[P0Job] = []
    reference_jobs: list[P0Job] = []
    released: list[ReleasedJobRecord] = []
    budgets: list[tuple[str, int]] = []

    for priority_index, task in enumerate(tasks):
        name = str(task.name)
        row = rows[name]
        crit = _criticality(task)
        raw = actual[name]
        budget = _runtime_budget(task, row)
        budgets.append((name, budget))

        degraded = bool(crit == "LO" and mode == "HI" and not primary_on_switch)
        released_mode = "HI" if degraded or crit == "HI" and mode == "HI" else "LO"
        release_budget = _reference_degraded_cost(row, task, cfg) if degraded else budget
        if degraded:
            removal = min(raw, release_budget)
            release_class = "LO_DEGRADED_HI_MODE"
            category = "degraded"
        elif crit == "LO":
            removal = min(raw, budget + 1)
            release_class = (
                "LO_PRIMARY_SAME_BATCH_SWITCH_TIME" if mode == "HI" else "LO_PRIMARY_NORMAL"
            )
            category = "normal"
        else:
            removal = raw
            if name == trigger_name:
                release_class = "HI_ABNORMAL_SWITCH_TRIGGER"
            elif raw > _c_lo(task, row):
                release_class = "HI_ABNORMAL"
            else:
                release_class = "HI_NORMAL"
            category = "normal"

        deadline = int(getattr(task, "deadline"))
        key = (name, 0)
        concrete_jobs.append(P0Job(
            job_key=key,
            priority_index=priority_index,
            release_time=0,
            deadline=deadline,
            release_category=category,
            release_budget=release_budget,
            demand=raw,
            service=0,
            state="active",
            mode=mode,
            criticality=crit,
            released_mode=released_mode,
            is_degraded=degraded,
            raw_actual_cost=raw,
            removal_demand=removal,
        ))
        reference_jobs.append(P0Job(
            job_key=key,
            priority_index=int(row.get("priority_index", priority_index)),
            release_time=0,
            deadline=deadline,
            release_category=category,
            release_budget=release_budget,
            demand=removal,
            service=0,
            state="active",
            mode=mode,
            criticality=crit,
            released_mode=released_mode,
            is_degraded=degraded,
            raw_actual_cost=raw,
            removal_demand=removal,
        ))
        released.append(ReleasedJobRecord(
            job_key=key,
            release_time=0,
            absolute_deadline=deadline,
            criticality=crit,
            released_mode=released_mode,
            release_class=release_class,
            release_budget=release_budget,
            raw_actual_cost=raw,
            removal_demand=removal,
            priority_index=priority_index,
            provenance=f"{CONTRACT_VERSION}:release_fixed",
        ))

    ready = tuple(job.job_key for job in concrete_jobs)
    running = ready[0]

    # Frozen queue after the complete time-zero arrival closure.  It contains
    # only timing-relevant events; controller and mutable implementation events
    # are intentionally outside the N4/N5 semantic boundary.
    raw_events: list[FrozenRawEvent] = []
    fifo = 0
    limit = _end_time(cfg)
    for task in tasks:
        period = int(getattr(task, "period"))
        if limit is None or period < limit:
            raw_events.append(FrozenRawEvent(
                time=period,
                event_type="JOB_ARRIVAL",
                task_name=str(task.name),
                release_index=1,
                fifo_rank=fifo,
            ))
            fifo += 1
        raw_events.append(FrozenRawEvent(
            time=int(getattr(task, "deadline")),
            event_type="DEADLINE_CHECK",
            task_name=str(task.name),
            release_index=0,
            fifo_rank=fifo,
        ))
        fifo += 1

    running_job = concrete_jobs[0]
    completion_token = 1
    raw_events.append(FrozenRawEvent(
        time=int(running_job.removal_demand or running_job.demand),
        event_type="JOB_COMPLETION",
        task_name=running_job.job_key[0],
        release_index=0,
        token=completion_token,
        fifo_rank=fifo,
    ))

    token_state = TokenState(
        completion_tokens=((running, completion_token),),
        overrun_tokens=(),
        response_tokens=(),
    )
    active_keys = ready
    view = type("_FrozenSnapshotView", (), {
        "completion_token": lambda self, key: completion_token if key == running else None,
        "overrun_token": lambda self, key: None,
        "response_token": lambda self, key: None,
        "active_job_keys": active_keys,
    })()
    frontier = effective_frontier(tuple(raw_events), view)
    queue_projection = tuple(sorted(
        (event.time, event.event_type, event.task_name, event.release_index, event.token)
        for event in raw_events
    ))
    next_boundary = min(event.time for event in raw_events)
    released_ledger = tuple(released)

    concrete = P0ConcreteState(
        time=0,
        mode=mode,
        active_jobs=tuple(concrete_jobs),
        ready_jobs=ready,
        running_job=running,
        global_future_budgets=tuple(sorted(budgets)),
        miss_flags=(),
        queue_projection=queue_projection,
        next_controller_boundary=None,
        next_timing_boundary=next_boundary,
        released_ledger=released_ledger,
        terminal_ledger=(),
        miss_ledger=(),
        effective_event_frontier=frontier,
    )
    reference = P0ReferenceState(
        time=0,
        mode=mode,
        active_jobs=tuple(reference_jobs),
        ready_jobs=ready,
        running_job=running,
        global_future_budgets=tuple(sorted(budgets)),
        miss_flags=(),
        queue_projection=queue_projection,
        next_controller_boundary=None,
        next_timing_boundary=next_boundary,
        released_ledger=released_ledger,
        terminal_ledger=(),
        miss_ledger=(),
        effective_event_frontier=frontier,
    )
    snapshot = FormalRuntimeSnapshot(
        released_ledger=released_ledger,
        terminal_ledger=(),
        miss_ledger=(),
        token_state=token_state,
        queue_snapshot=tuple(raw_events),
        effective_event_frontier=frontier,
        active_job_keys=active_keys,
        mode=mode,
        time=0,
    )
    return concrete, reference, snapshot


def frozen_preclosed_contract_witness(source_root: Any) -> dict[str, Any]:
    return {
        "schema_version": "frozen_preclosed_contract_v1",
        "formal_semantics_contract": CONTRACT_VERSION,
        "formal_semantics_contract_hash": frozen_contract_manifest(source_root)["semantic_hash"],
        "runtime_dependency": "NONE",
        "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
    }


__all__ = [
    "FrozenRawEvent",
    "build_frozen_preclosed_bundle",
    "frozen_preclosed_contract_witness",
]
