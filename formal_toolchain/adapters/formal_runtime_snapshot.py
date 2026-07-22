from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from formal_toolchain.bridge.effective_event_frontier import effective_frontier


@dataclass(frozen=True, slots=True)
class ReleasedJobRecord:
    job_key: tuple[str, int]
    release_time: int
    absolute_deadline: int
    criticality: str
    released_mode: str
    release_budget: int | None
    raw_actual_cost: int
    removal_demand: int


@dataclass(frozen=True, slots=True)
class TerminalRecord:
    job_key: tuple[str, int]
    terminal_kind: str
    terminal_time: int | None
    executed_service: int


@dataclass(frozen=True, slots=True)
class MissRecord:
    job_key: tuple[str, int]
    mode_at_miss: str
    miss_time: int
    absolute_deadline: int
    executed_at_miss: int


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


def build_formal_runtime_snapshot(engine: Any) -> FormalRuntimeSnapshot:
    released_ledger = tuple(
        ReleasedJobRecord(
            job_key=(str(job.task.name), int(job.release_index)),
            release_time=int(job.release_time),
            absolute_deadline=int(job.absolute_deadline),
            criticality=str(getattr(job.task.criticality, "value", job.task.criticality)),
            released_mode=str(getattr(job.released_in_mode, "name", job.released_in_mode)),
            release_budget=None if job.runtime_budget_at_release is None else int(job.runtime_budget_at_release),
            raw_actual_cost=int(getattr(job, "original_actual_cost", job.actual_cost)),
            removal_demand=int(
                getattr(
                    job,
                    "removal_demand",
                    getattr(
                        job,
                        "removed_demand",
                        job.actual_cost if not bool(getattr(job, "is_degraded", False)) else getattr(job, "original_actual_cost", job.actual_cost),
                    ),
                )
            ),
        )
        for job in tuple(getattr(engine, "jobs_by_key", {}).values())
    )
    active_keys = tuple((str(job.task.name), int(job.release_index)) for job in tuple(engine.state.active_jobs))
    terminal_ledger = tuple(
        TerminalRecord(
            job_key=(str(job.task.name), int(job.release_index)),
            terminal_kind="DROPPED" if bool(job.dropped) else "FINISHED",
            terminal_time=(
                int(getattr(job, "drop_time"))
                if getattr(job, "drop_time", None) is not None
                else int(getattr(job, "completion_time"))
                if getattr(job, "completion_time", None) is not None
                else None
            ),
            executed_service=int(job.executed_time),
        )
        for job in tuple(getattr(engine, "jobs_by_key", {}).values())
        if bool(job.dropped) or job.finished()
    )
    miss_ledger = tuple(
        MissRecord(
            job_key=(str(m.task), int(m.release_index)),
            mode_at_miss=str(getattr(m, "mode_at_miss", getattr(engine.state.mode, "name", engine.state.mode))),
            miss_time=int(getattr(m, "time", engine.current_time)),
            absolute_deadline=int(getattr(m, "absolute_deadline", engine.current_time)),
            executed_at_miss=int(getattr(m, "executed_at_miss", getattr(m, "executed_time", 0))),
        )
        for m in tuple(engine.result.deadline_misses)
    )
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
        mode=str(engine.state.mode.name),
        time=int(engine.current_time),
    )


def as_dict(snapshot: FormalRuntimeSnapshot) -> dict[str, Any]:
    return asdict(snapshot)
