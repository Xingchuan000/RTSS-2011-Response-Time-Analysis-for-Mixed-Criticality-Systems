from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LogicalEventKind(str, Enum):
    REM = "REM"
    REC = "REC"
    DDL = "DDL"
    ARR_BATCH = "ARR_BATCH"
    SW = "SW"
    REL = "REL"
    DSP = "DSP"
    SVC = "SVC"


PHASE_RANK = {
    LogicalEventKind.REM: 0,
    LogicalEventKind.REC: 1,
    LogicalEventKind.DDL: 2,
    LogicalEventKind.ARR_BATCH: 3,
    LogicalEventKind.SW: 4,
    LogicalEventKind.REL: 5,
    LogicalEventKind.DSP: 6,
    LogicalEventKind.SVC: 7,
}


@dataclass(frozen=True, slots=True, order=True)
class LogicalEvent:
    time: int
    phase_rank: int
    kind: LogicalEventKind
    job_key: tuple[str, int] | None = None
    batch_jobs: tuple[tuple[str, int], ...] = ()
    fifo_rank: int = 0

    def logical_key(self) -> tuple:
        return (self.time, self.phase_rank, self.kind.value, self.job_key, self.batch_jobs, self.fifo_rank)


def raw_event_kind(event) -> str:
    et = getattr(event, "event_type", None)
    if et is not None:
        return str(getattr(et, "value", et))
    return str(getattr(event, "kind", ""))


def job_key_of(event) -> tuple[str, int] | None:
    tn = getattr(event, "task_name", None)
    ri = getattr(event, "release_index", None)
    if tn is not None and ri is not None:
        return (str(tn), int(ri))
    return None


def valid_completion_token(event, snapshot) -> bool:
    key = job_key_of(event)
    if key is None:
        return False
    token = getattr(event, "token", None)
    if token is None:
        return False
    return snapshot.completion_token(key) == token


def valid_overrun_token(event, snapshot) -> bool:
    key = job_key_of(event)
    if key is None:
        return False
    token = getattr(event, "token", None)
    if token is None:
        return False
    return snapshot.overrun_token(key) == token


def stale_response_event(event, snapshot) -> bool:
    key = job_key_of(event)
    if key is None:
        return True
    token = getattr(event, "token", None)
    if token is None:
        return True
    return snapshot.response_token(key) != token


def deadline_event_effective(event, snapshot) -> bool:
    key = job_key_of(event)
    if key is None:
        return False
    return key in getattr(snapshot, "active_job_keys", ())


def project_raw_event(event, snapshot) -> tuple[LogicalEvent, ...]:
    raw = raw_event_kind(event)
    key = job_key_of(event)
    fifo = getattr(event, "fifo_rank", 0)
    if raw == "RECOVERY":
        return (LogicalEvent(
            time=int(event.time), phase_rank=PHASE_RANK[LogicalEventKind.REC],
            kind=LogicalEventKind.REC, fifo_rank=fifo,
        ),)
    if raw == "DEADLINE_CHECK":
        if not deadline_event_effective(event, snapshot):
            return ()
        return (LogicalEvent(
            time=int(event.time), phase_rank=PHASE_RANK[LogicalEventKind.DDL],
            kind=LogicalEventKind.DDL, job_key=key, fifo_rank=fifo,
        ),)
    if raw == "JOB_COMPLETION":
        if not valid_completion_token(event, snapshot):
            return ()
        return (LogicalEvent(
            time=int(event.time), phase_rank=PHASE_RANK[LogicalEventKind.REM],
            kind=LogicalEventKind.REM, job_key=key, fifo_rank=fifo,
        ),)
    if raw == "BUDGET_OVERRUN":
        if not valid_overrun_token(event, snapshot):
            return ()
        return (LogicalEvent(
            time=int(event.time), phase_rank=PHASE_RANK[LogicalEventKind.SW],
            kind=LogicalEventKind.SW, job_key=key, fifo_rank=fifo,
        ),)
    if raw in ("BUDGET_UPDATE", "CONTROLLER"):
        return ()
    if raw == "RESPONSE_TIME_EXPIRY":
        if stale_response_event(event, snapshot):
            return ()
        return (LogicalEvent(
            time=int(event.time), phase_rank=PHASE_RANK[LogicalEventKind.DSP],
            kind=LogicalEventKind.DSP, job_key=key, fifo_rank=fifo,
        ),)
    if raw == "JOB_ARRIVAL":
        raise RuntimeError("arrivals must be grouped by project_arrival_batch")
    raise ValueError(f"UNKNOWN_RAW_EVENT:{raw}")


def project_arrival_batch(events, *, time: int) -> LogicalEvent:
    jobs = sorted(
        (jk for e in events
         if raw_event_kind(e) == "JOB_ARRIVAL" and int(e.time) == time
         for jk in [job_key_of(e)] if jk is not None),
    )
    fifo_ranks = [int(getattr(e, "fifo_rank", 0)) for e in events
                  if raw_event_kind(e) == "JOB_ARRIVAL" and int(e.time) == time]
    return LogicalEvent(
        time=time,
        phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
        kind=LogicalEventKind.ARR_BATCH,
        batch_jobs=tuple(jobs),
        fifo_rank=min(fifo_ranks) if fifo_ranks else 0,
    )
