"""Reachable carry-in refinement for the V10.1 PCSSC proof route.

This module derives the contiguous fixed-priority prefix whose jobs are
guaranteed to finish no later than their next release under the universal
per-job execution envelope already bound by the verifier.  V10.1 uses that one
dominance theorem both to tighten SafePrefix carry-in and, when R_i^U <= D_i,
to prove the corresponding target FirstBadEventWindow set empty before Event
formula allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .symbolic_state import BoundModel, SymbolicKernelState, TaskBound


@dataclass(frozen=True, slots=True)
class ResponseEnvelope:
    task_name: str
    priority: int
    service_upper: int
    response_bound: int


@dataclass(frozen=True, slots=True)
class ProtectedPriorityPrefix:
    members: tuple[ResponseEnvelope, ...]

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(row.task_name for row in self.members)

    @property
    def response_by_task(self) -> dict[str, int]:
        return {row.task_name: row.response_bound for row in self.members}

    def contains(self, task_name: str) -> bool:
        return task_name in self.response_by_task


def universal_service_upper(task: TaskBound) -> int:
    """Maximum processor service one concrete job can request.

    HI effective demand is the raw actual demand.  LO effective demand is a
    minimum of raw demand and a budget/degraded cap, hence raw actual demand is
    also a valid universal upper bound for LO.
    """

    return int(task.actual_demand_upper)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def derive_protected_priority_prefix(model: BoundModel) -> ProtectedPriorityPrefix:
    """Compute the maximal contiguous prefix with ``R_i^U <= T_i``.

    The recurrence is the ordinary preemptive fixed-priority response envelope

        R_i = C_i^U + sum_{j<i} ceil(R_i/T_j) C_j^U.

    As soon as one task exceeds its period the protected prefix ends.  Later
    tasks are deliberately not re-admitted: their response equation would need
    a carry-in theorem for the already-unprotected higher-priority task.
    """

    members: list[ResponseEnvelope] = []
    for index, task in enumerate(model.tasks):
        c_i = universal_service_upper(task)
        response = c_i
        while True:
            updated = c_i + sum(
                _ceil_div(response, hp.period) * universal_service_upper(hp)
                for hp in model.tasks[:index]
            )
            if updated == response:
                break
            response = updated
            # Monotonicity makes recovery impossible once T_i is exceeded.
            if response > task.period:
                return ProtectedPriorityPrefix(tuple(members))
        if response > task.period:
            break
        members.append(ResponseEnvelope(
            task_name=task.name,
            priority=int(task.priority),
            service_upper=c_i,
            response_bound=int(response),
        ))
    return ProtectedPriorityPrefix(tuple(members))


def reachable_carry_in_consistency(
    state: SymbolicKernelState,
    model: BoundModel,
) -> z3.BoolRef:
    """V10.1 SafePrefix carry-in refinement.

    For a protected LO task the historical aggregate slot is unreachable: the
    previous exact job has settled before the next release, so P3 never folds
    unfinished work into slot 0.  A protected exact job that is still
    incomplete at a P0/P1/... state must also be younger than its certified
    response envelope.  Completed jobs are allowed at P0 exactly on their
    completion timestamp because P0 settlement happens immediately afterwards.

    Tasks outside the protected prefix retain the base two-slot representation.
    This is intentional: V10.1 does not invent a finite backlog bound where none
    has been proved.  Target-specific PCSSC response analysis only depends on protected
    higher-priority tasks; lower-priority unconstrained aggregates cannot delay
    the target under strict fixed-priority dispatch.
    """

    prefix = derive_protected_priority_prefix(model)
    response_by_task = prefix.response_by_task
    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        if task.deadline > task.period:
            clauses.append(z3.BoolVal(False))
            continue
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            clauses.append(z3.Implies(job.present, job.release_time <= state.t))
        response = response_by_task.get(task.name)
        if response is None:
            continue
        if task.criticality == "LO":
            aggregate = state.jobs[(task.name, 0)]
            clauses.extend((z3.Not(aggregate.present), z3.Not(aggregate.ready)))
            exact_slot = 1
        else:
            exact_slot = 0
        exact = state.jobs[(task.name, exact_slot)]
        incomplete = z3.And(
            exact.present,
            exact.executed_service < exact.effective_demand,
        )
        clauses.append(z3.Implies(
            incomplete,
            exact.release_time + int(response) > state.t,
        ))
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


__all__ = [
    "ProtectedPriorityPrefix",
    "ResponseEnvelope",
    "derive_protected_priority_prefix",
    "reachable_carry_in_consistency",
    "universal_service_upper",
]
