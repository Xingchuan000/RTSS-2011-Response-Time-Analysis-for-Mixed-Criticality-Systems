from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from formal_toolchain.bridge.effective_event_frontier import effective_frontier


class ReleaseClass(str, Enum):
    HI_NORMAL = "HI_NORMAL"
    HI_ABNORMAL_SWITCH_TRIGGER = "HI_ABNORMAL_SWITCH_TRIGGER"
    LO_PRIMARY_NORMAL = "LO_PRIMARY_NORMAL"
    LO_PRIMARY_SAME_BATCH_SWITCH_TIME = "LO_PRIMARY_SAME_BATCH_SWITCH_TIME"
    LO_DEGRADED_HI_MODE = "LO_DEGRADED_HI_MODE"
    LO_DROPPED_HI_MODE = "LO_DROPPED_HI_MODE"


class TerminalKind(str, Enum):
    COMPLETED = "COMPLETED"
    PRIMARY_LO_BUDGET_CANCELLATION = "PRIMARY_LO_BUDGET_CANCELLATION"
    ACTIVE_LO_DROPPED_ON_MODE_SWITCH = "ACTIVE_LO_DROPPED_ON_MODE_SWITCH"
    LO_RELEASE_DROPPED_IN_HI_MODE = "LO_RELEASE_DROPPED_IN_HI_MODE"


@dataclass(frozen=True, slots=True)
class ReleasedJobRecord:
    job_key: tuple[str, int]
    release_time: int
    absolute_deadline: int
    criticality: str
    released_mode: str
    release_class: str
    release_budget: int | None
    raw_actual_cost: int
    removal_demand: int
    priority_index: int
    provenance: str


@dataclass(frozen=True, slots=True)
class TerminalRecord:
    job_key: tuple[str, int]
    terminal_kind: str
    terminal_time: int
    executed_service: int


@dataclass(frozen=True, slots=True)
class MissRecord:
    job_key: tuple[str, int]
    criticality: str
    release_time: int
    release_class: str
    mode_at_miss: str
    miss_time: int
    absolute_deadline: int
    executed_at_miss: int
    priority_index: int
    reference_job_key: tuple[str, int] | None = None


@dataclass(frozen=True, slots=True)
class TokenState:
    completion_tokens: tuple[tuple[tuple[str, int], int], ...]
    overrun_tokens: tuple[tuple[tuple[str, int], int], ...]
    response_tokens: tuple[tuple[tuple[str, int], int], ...]


@dataclass(frozen=True, slots=True)
class FormalRuntimeSnapshot:
    released_ledger: tuple[ReleasedJobRecord, ...]
    terminal_ledger: tuple[TerminalRecord, ...]
    miss_ledger: tuple[MissRecord, ...]
    token_state: TokenState
    queue_snapshot: tuple[Any, ...]
    effective_event_frontier: tuple[Any, ...]
    active_job_keys: tuple[tuple[str, int], ...]
    mode: str
    time: int

    def completion_token(self, key: tuple[str, int]) -> int | None:
        for item_key, token in self.token_state.completion_tokens:
            if item_key == key:
                return token
        return None

    def overrun_token(self, key: tuple[str, int]) -> int | None:
        for item_key, token in self.token_state.overrun_tokens:
            if item_key == key:
                return token
        return None

    def response_token(self, key: tuple[str, int]) -> int | None:
        for item_key, token in self.token_state.response_tokens:
            if item_key == key:
                return token
        return None


def _release_fixed_removal_demand(job: Any) -> int:
    raw = int(getattr(job, "original_actual_cost", job.actual_cost))
    criticality = str(getattr(getattr(job.task, "criticality", job.task), "value", getattr(job.task, "criticality", "LO")))
    if bool(getattr(job, "is_degraded", False)):
        return int(job.actual_cost)
    if criticality == "LO":
        budget = getattr(job, "runtime_budget_at_release", None)
        if budget is not None:
            return min(raw, int(budget) + 1)
    return raw


def _release_class(job: Any, mode: str) -> str:
    criticality = str(getattr(getattr(job.task, "criticality", job.task), "value", getattr(job.task, "criticality", "LO")))
    is_degraded = bool(getattr(job, "is_degraded", False))
    released_in_mode = str(getattr(job, "released_in_mode", "LO"))
    if criticality == "HI":
        if mode == "HI" and released_in_mode == "LO":
            return ReleaseClass.HI_ABNORMAL_SWITCH_TRIGGER
        return ReleaseClass.HI_NORMAL
    if mode == "HI":
        if released_in_mode == "HI":
            return ReleaseClass.LO_DEGRADED_HI_MODE
        return ReleaseClass.LO_DROPPED_HI_MODE
    if is_degraded:
        return ReleaseClass.LO_DEGRADED_HI_MODE
    if released_in_mode == "HI":
        return ReleaseClass.LO_PRIMARY_SAME_BATCH_SWITCH_TIME
    return ReleaseClass.LO_PRIMARY_NORMAL


def _terminal_kind(job: Any, cancellations: dict, losses: dict) -> str:
    key = (str(job.task.name), int(job.release_index))
    if job.completion_time is not None and not bool(getattr(job, "dropped", False)):
        return TerminalKind.COMPLETED
    cancel = cancellations.get(key)
    if cancel is not None:
        reason = str(getattr(cancel, "reason", "BUDGET_CANCELLATION")).upper()
        if reason.startswith("PRIMARY"):
            return TerminalKind.PRIMARY_LO_BUDGET_CANCELLATION
        return TerminalKind.ACTIVE_LO_DROPPED_ON_MODE_SWITCH
    loss = losses.get(key)
    if loss is not None:
        return TerminalKind.LO_RELEASE_DROPPED_IN_HI_MODE
    if bool(getattr(job, "dropped", False)):
        return TerminalKind.ACTIVE_LO_DROPPED_ON_MODE_SWITCH
    raise ValueError(f"TERMINAL_JOB_REASON_MISSING: job {key}")


def _priority_index(job: Any, priority_map: dict | None = None) -> int:
    if priority_map is not None:
        name = str(job.task.name)
        return priority_map.get(name, 0)
    return int(getattr(job.task, "priority_index", 0))


def build_formal_runtime_snapshot(
    engine: Any,
    priority_map: dict[str, int] | None = None,
) -> FormalRuntimeSnapshot:
    mode = str(getattr(engine.state.mode, "name", engine.state.mode))
    cancellations = {
        (str(e.task), int(e.release_index)): e
        for e in getattr(getattr(engine, "result", engine), "job_cancellations", ())
    }
    losses = {
        (str(e.task), int(e.release_index)): e
        for e in getattr(getattr(engine, "result", engine), "lo_job_losses", ())
    }
    raw_released = []
    for job in tuple(getattr(engine, "jobs_by_key", {}).values()):
        pidx = _priority_index(job, priority_map)
        raw_released.append(ReleasedJobRecord(
            job_key=(str(job.task.name), int(job.release_index)),
            release_time=int(job.release_time),
            absolute_deadline=int(job.absolute_deadline),
            criticality=str(getattr(getattr(job.task, "criticality", job.task), "value", getattr(job.task, "criticality", "LO"))),
            released_mode=str(getattr(job, "released_in_mode", "LO")),
            release_class=_release_class(job, mode),
            release_budget=None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            raw_actual_cost=int(getattr(job, "original_actual_cost", job.actual_cost)),
            removal_demand=_release_fixed_removal_demand(job),
            priority_index=pidx,
            provenance="release_fixed",
        ))
    released_ledger = tuple(sorted(raw_released, key=lambda x: (
        x.release_time, x.priority_index, x.job_key[0], x.job_key[1]
    )))
    active_keys = tuple((str(job.task.name), int(job.release_index))
                        for job in tuple(engine.state.active_jobs))

    raw_terminal = []
    for job in tuple(getattr(engine, "jobs_by_key", {}).values()):
        if not (bool(job.dropped) or job.finished()):
            continue
        ttime = int(getattr(job, "drop_time", None) or getattr(job, "completion_time", None) or 0)
        if ttime <= 0 and hasattr(job, "completion_time") and job.completion_time is None:
            continue
        raw_terminal.append(TerminalRecord(
            job_key=(str(job.task.name), int(job.release_index)),
            terminal_kind=_terminal_kind(job, cancellations, losses),
            terminal_time=ttime,
            executed_service=int(job.executed_time),
        ))
    terminal_ledger = tuple(sorted(raw_terminal, key=lambda x: (
        x.terminal_time, x.job_key[0], x.job_key[1]
    )))

    raw_misses = []
    for m in tuple(getattr(engine.result, "deadline_misses", ())):
        pidx = _priority_index(m, priority_map) if hasattr(m, "task") else 0
        raw_misses.append(MissRecord(
            job_key=(str(m.task), int(m.release_index)),
            mode_at_miss=str(getattr(m, "mode_at_miss", mode)),
            miss_time=int(getattr(m, "absolute_deadline", getattr(m, "time", 0))),
            absolute_deadline=int(getattr(m, "absolute_deadline", getattr(m, "time", 0))),
            executed_at_miss=int(getattr(m, "executed_at_miss", getattr(m, "executed_time", 0))),
            priority_index=pidx,
        ))
    miss_ledger = tuple(sorted(raw_misses, key=lambda x: (
        x.miss_time, x.priority_index, x.job_key[0], x.job_key[1]
    )))

    def _normalize_key(key: Any) -> tuple[str, int]:
        if isinstance(key, tuple) and len(key) == 2:
            return str(key[0]), int(key[1])
        if hasattr(key, "task") and hasattr(key, "release_index"):
            return str(getattr(key, "task")), int(getattr(key, "release_index"))
        raise TypeError(f"invalid token key: {key!r}")

    token_state = TokenState(
        completion_tokens=tuple(sorted((_normalize_key(key), int(value))
                                       for key, value in getattr(engine.state, "valid_completion_tokens", {}).items())),
        overrun_tokens=tuple(sorted((_normalize_key(key), int(value))
                                    for key, value in getattr(engine.state, "valid_overrun_tokens", {}).items())),
        response_tokens=tuple(sorted((_normalize_key(key), int(value))
                                     for key, value in getattr(engine.state, "valid_response_expiry_tokens", {}).items())),
    )
    queue_snapshot = tuple(engine.queue.snapshot()) if callable(getattr(engine.queue, "snapshot", None)) else tuple()
    snapshot_view = type(
        "_FormalSnapshotView",
        (),
        {
            "completion_token": lambda self, key: next((token for item_key, token in token_state.completion_tokens if item_key == key), None),
            "overrun_token": lambda self, key: next((token for item_key, token in token_state.overrun_tokens if item_key == key), None),
            "response_token": lambda self, key: next((token for item_key, token in token_state.response_tokens if item_key == key), None),
            "active_job_keys": active_keys,
        },
    )()
    frontier = effective_frontier(queue_snapshot, snapshot_view)
    return FormalRuntimeSnapshot(
        released_ledger=released_ledger,
        terminal_ledger=terminal_ledger,
        miss_ledger=miss_ledger,
        token_state=token_state,
        queue_snapshot=queue_snapshot,
        effective_event_frontier=frontier,
        active_job_keys=active_keys,
        mode=mode,
        time=int(engine.current_time),
    )


def as_dict(snapshot: FormalRuntimeSnapshot) -> dict[str, Any]:
    return asdict(snapshot)
