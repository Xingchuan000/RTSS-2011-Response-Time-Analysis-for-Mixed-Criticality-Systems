"""Target-local scheduling projection for V9.3 FirstBadEventWindow search.

A first-bad window for target tau_i starts at one release of tau_i and ends at
that job's deadline.  On every candidate bad path that target job stays pending;
strict fixed-priority dispatch therefore never executes a lower-priority job.
Those lower-priority jobs cannot consume processor service needed by the target.

The Event proof still retains the *full* controller budget/history state because
the deployed tree may inspect lower-priority history or global budget features.
Only scheduling state (jobs, eta, release/deadline/completion sources) is
projected to the target priority prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import z3

from .carry_in import reachable_carry_in_consistency
from .invariant_templates import (
    budget_bounds,
    carry_in_consistency,
    history_bounds,
    job_field_consistency,
    no_prior_hi_miss_consistency,
    state_well_formedness,
)
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState


@dataclass(frozen=True, slots=True)
class TargetSchedulingProjection:
    target_task: str
    active_task_names: tuple[str, ...]
    dropped_lower_task_names: tuple[str, ...]
    active_model: BoundModel

    def contains(self, task_name: str) -> bool:
        return task_name in self.active_task_names


def derive_target_scheduling_projection(
    model: BoundModel, target_task: str
) -> TargetSchedulingProjection:
    target = model.task_by_name.get(target_task)
    if target is None or target.criticality != "HI":
        raise ValueError("TARGET_LOCAL_PROJECTION_REQUIRES_HI_TARGET")
    active = tuple(task for task in model.tasks if task.priority <= target.priority)
    dropped = tuple(task.name for task in model.tasks if task.priority > target.priority)
    active_model = replace(model, tasks=active)
    return TargetSchedulingProjection(
        target_task=target_task,
        active_task_names=tuple(task.name for task in active),
        dropped_lower_task_names=dropped,
        active_model=active_model,
    )


def event_root_linear_phase_formula(
    state: SymbolicKernelState,
    model: BoundModel,
    target_task: str,
) -> z3.BoolRef:
    """Exact phase-zero periodic root without modulo arithmetic.

    At the P0 root, ``eta_i`` is the positive release age used by the Full
    kernel: ``T_i`` exactly on a release timestamp and otherwise the positive
    residue in ``[1,T_i-1]``.  Express the same relation with one Euclidean
    quotient per active task, using only linear integer arithmetic.  All task
    phases remain correlated through the shared absolute ``state.t``.
    """

    projection = derive_target_scheduling_projection(model, target_task)
    clauses: list[z3.BoolRef] = []
    for task in projection.active_model.tasks:
        q = z3.Int(f"event.root.phase_q.{target_task}.{task.name}")
        eta = state.eta[task.name]
        period = int(task.period)
        clauses.extend((
            q >= 0,
            z3.Or(
                z3.And(eta == period, state.t == q * period),
                z3.And(eta >= 1, eta < period, state.t == q * period + eta),
            ),
        ))
    return z3.And(*clauses)


def _active_settled_job_consistency(
    state: SymbolicKernelState, projection: TargetSchedulingProjection
) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    active = set(projection.active_task_names)
    for (task_name, _), job in state.jobs.items():
        if task_name not in active:
            continue
        clauses.append(z3.Implies(
            z3.And(state.p >= 1, job.present),
            job.executed_service < job.effective_demand,
        ))
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


def event_root_safe_prefix_formula(
    state: SymbolicKernelState,
    model: BoundModel,
    invariant: SafePrefixInvariant,
    target_task: str,
) -> z3.BoolRef:
    """Project Psi to fields that can influence this target before its deadline.

    Full budget/history bounds remain because exact P5 reads them.  Job/eta and
    no-prior-miss clauses are retained only for the target priority prefix.  The
    omitted lower-priority scheduling fields are never read by the projected
    Event kernel, so this is existential elimination of dead state, not a fixed
    lower-priority scenario.
    """

    del invariant  # V9.3 Event route uses the canonical base-Psi components below.
    projection = derive_target_scheduling_projection(model, target_task)
    active_model = projection.active_model
    return z3.And(
        state_well_formedness(state, active_model),
        budget_bounds(state, model),
        event_root_linear_phase_formula(state, model, target_task),
        job_field_consistency(state, active_model),
        no_prior_hi_miss_consistency(state, active_model),
        _active_settled_job_consistency(state, projection),
        history_bounds(state, model),
        carry_in_consistency(state, active_model),
        reachable_carry_in_consistency(state, active_model),
    )


def target_pending_after_origin(
    state: SymbolicKernelState,
    model: BoundModel,
    target_task: str,
) -> z3.BoolRef:
    """The original target job is still live and incomplete at an Event node."""

    task = model.task_by_name[target_task]
    if task.criticality != "HI":
        raise ValueError("TARGET_PENDING_REQUIRES_HI_TARGET")
    job = state.jobs[(target_task, 0)]
    return z3.And(
        job.present,
        z3.Not(job.removed),
        job.ready,
        job.executed_service < job.effective_demand,
    )


__all__ = [
    "TargetSchedulingProjection",
    "derive_target_scheduling_projection",
    "event_root_linear_phase_formula",
    "event_root_safe_prefix_formula",
    "target_pending_after_origin",
]
