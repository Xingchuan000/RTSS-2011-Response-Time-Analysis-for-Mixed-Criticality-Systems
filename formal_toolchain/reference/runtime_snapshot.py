from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from formal_toolchain.reference.reference_state import ReferenceState, ReferenceJob


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    time: int
    mode: str
    released: tuple[dict[str, Any], ...]
    terminal: tuple[dict[str, Any], ...]
    misses: tuple[dict[str, Any], ...]
    ready_order: tuple[tuple[str, int], ...]
    running: tuple[str, int] | None
    frontier: tuple[Any, ...]


def build_reference_runtime_snapshot(state: ReferenceState) -> ReferenceSnapshot:
    released = tuple(
        {
            "job_key": jk,
            "release_time": job.release_time,
            "absolute_deadline": job.absolute_deadline,
            "criticality": job.criticality,
            "priority_index": job.priority_index if hasattr(job, "priority_index") else 0,
            "executed": job.executed,
            "demand": job.budget if hasattr(job, "budget") else 0,
        }
        for jk, job in state.jobs.items()
    )
    terminal = tuple(
        {
            "job_key": jk,
            "terminal_time": rec.terminal_time if hasattr(rec, "terminal_time") else 0,
            "executed_service": rec.executed_service if hasattr(rec, "executed_service") else 0,
        }
        for jk, rec in state.terminal.items()
    ) if hasattr(state, "terminal") else ()
    misses = tuple(
        {
            "job_key": m.job_key if hasattr(m, "job_key") else m,
            "miss_time": getattr(m, "miss_time", 0),
            "criticality": getattr(m, "criticality", "LO"),
        }
        for m in state.misses
    ) if hasattr(state, "misses") else ()
    return ReferenceSnapshot(
        time=state.time,
        mode=state.mode,
        released=released,
        terminal=terminal,
        misses=misses,
        ready_order=tuple(state.ready_order) if hasattr(state, "ready_order") else (),
        running=state.running,
        frontier=tuple(state.frontier) if hasattr(state, "frontier") else (),
    )
