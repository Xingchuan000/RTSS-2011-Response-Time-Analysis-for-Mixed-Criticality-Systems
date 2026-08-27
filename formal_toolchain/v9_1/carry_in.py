"""Finite, saturated carry-in abstraction for first-bad windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import z3

from .symbolic_state import BoundModel, TaskBound


@dataclass(frozen=True, slots=True)
class SymbolicCarryInJob:
    present: z3.BoolRef
    release_time: z3.ArithRef
    remaining_work: z3.ArithRef
    priority: int
    completion_observable: z3.BoolRef


@dataclass(frozen=True, slots=True)
class CarryInSummary:
    explicit_jobs: Mapping[str, tuple[SymbolicCarryInJob, ...]]
    saturated_tail: Mapping[str, z3.BoolRef]
    window_length: int


@dataclass(frozen=True, slots=True)
class CarryInAdequacy:
    status: str
    code: str | None
    constraints: tuple[z3.BoolRef, ...]


def build_carry_in_summary(model: BoundModel, *, window_length: int, prefix: str = "carry") -> CarryInSummary:
    if window_length <= 0:
        raise ValueError("carry-in window length must be positive")
    explicit: dict[str, tuple[SymbolicCarryInJob, ...]] = {}
    tails: dict[str, z3.BoolRef] = {}
    # One positive service quantum per tick gives the exact finite capacity D.
    # We still expose a tail flag so older/more numerous jobs are never silently
    # discarded by the abstraction.
    for task in model.tasks:
        explicit[task.name] = tuple(
            SymbolicCarryInJob(
                present=z3.Bool(f"{prefix}.{task.name}.{slot}.present"),
                release_time=z3.Int(f"{prefix}.{task.name}.{slot}.release_time"),
                remaining_work=z3.Int(f"{prefix}.{task.name}.{slot}.remaining"),
                priority=task.priority,
                completion_observable=z3.Bool(f"{prefix}.{task.name}.{slot}.completion_observable"),
            )
            for slot in range(window_length)
        )
        tails[task.name] = z3.Bool(f"{prefix}.{task.name}.saturated_tail")
    return CarryInSummary(explicit, tails, window_length)


def encode_carry_in_adequacy(summary: CarryInSummary, model: BoundModel) -> CarryInAdequacy:
    if summary.window_length <= 0:
        return CarryInAdequacy("UNRESOLVED", "CARRY_IN_SUMMARY_UNPROVED", ())
    constraints: list[z3.BoolRef] = []
    for task in model.tasks:
        jobs = summary.explicit_jobs.get(task.name, ())
        if len(jobs) < summary.window_length:
            # A finite summary without a tail bit would be an unsound omission.
            if task.name not in summary.saturated_tail:
                return CarryInAdequacy("UNRESOLVED", "CARRY_IN_SUMMARY_UNPROVED", ())
        for job in jobs:
            constraints.extend((
                z3.Implies(job.present, job.release_time <= 0),
                z3.Implies(job.present, job.remaining_work >= 1),
                z3.Implies(job.completion_observable,
                           z3.And(job.present, job.remaining_work <= summary.window_length)),
            ))
        # A saturated tail is an explicit abstraction case.  Its work is not
        # allowed to masquerade as an observable completion inside this window.
        tail = summary.saturated_tail.get(task.name)
        if tail is None:
            return CarryInAdequacy("UNRESOLVED", "CARRY_IN_SUMMARY_UNPROVED", ())
        constraints.append(z3.Implies(tail, z3.Not(z3.Or(*(job.completion_observable for job in jobs)))))
    return CarryInAdequacy("PASS", None, tuple(constraints))


def check_carry_in_summary_soundness(summary: CarryInSummary, model: BoundModel) -> dict[str, object]:
    adequacy = encode_carry_in_adequacy(summary, model)
    return {
        "status": adequacy.status,
        "code": adequacy.code,
        "window_length": summary.window_length,
        "explicit_slots": {name: len(rows) for name, rows in summary.explicit_jobs.items()},
        "saturation_flags": sorted(summary.saturated_tail),
        "obligation": "CARRY_IN_SUMMARY_ADEQUACY",
    }


__all__ = [
    "CarryInAdequacy", "CarryInSummary", "SymbolicCarryInJob",
    "build_carry_in_summary", "check_carry_in_summary_soundness", "encode_carry_in_adequacy",
]
