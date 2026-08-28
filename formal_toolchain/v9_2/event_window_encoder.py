"""V9.2 exact event-driven first-HI-bad-window encoding.

The terminal proof route no longer allocates D*8 Full-kernel states.  It
allocates a mathematically bounded number of event boundaries and, for each
active boundary, one exact P0--P6 closure plus one algebraic silent interval.
No Event-layer abstraction is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Mapping

import z3

from .environment_encoder import (
    declare_environment,
    target_release_constraints,
)
from .event_kernel import (
    EVENT_KERNEL_VERSION,
    EventStepEncoding,
    ExactControllerPool,
    build_exact_controller_pool,
    encode_event_step,
    event_step_or_terminal_stutter,
)
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import (
    encode_p0_settle,
    encode_p1_idle_recovery,
    encode_p2_deadline_observe,
)


ENCODER_VERSION = "V9_2_EVENT_FIRST_BAD_WINDOW_V4_P5_SUPPORT_STUTTER_FACTORED"
ENCODER_COMPLETE = True
ENCODER_READINESS_GAPS: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventBound:
    target_task: str
    deadline: int
    release_bound_by_task: Mapping[str, int]
    total_release_bound: int
    initial_job_representative_bound: int
    hi_deadline_bound: int
    completion_bound: int
    controller_bound: int
    horizon_boundary_bound: int
    finite_event_bound: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_task": self.target_task,
            "deadline": self.deadline,
            "release_bound_by_task": dict(self.release_bound_by_task),
            "total_release_bound": self.total_release_bound,
            "initial_job_representative_bound": self.initial_job_representative_bound,
            "hi_deadline_bound": self.hi_deadline_bound,
            "completion_bound": self.completion_bound,
            "controller_bound": self.controller_bound,
            "horizon_boundary_bound": self.horizon_boundary_bound,
            "finite_event_bound": self.finite_event_bound,
        }


@dataclass(frozen=True, slots=True)
class EventWindowEncoding:
    target_task: str
    deadline: int
    formula: z3.BoolRef
    event_states: tuple[SymbolicKernelState, ...]
    event_steps: tuple[EventStepEncoding, ...]
    terminal_phase_states: tuple[SymbolicKernelState, ...]
    environment: Any
    event_bound: EventBound
    controller_pool: ExactControllerPool
    source_obligations: tuple[str, ...]
    event_layer_added_abstractions: tuple[str, ...] = ()
    exact_p5_in_event_window: bool = True
    microstep_terminal_fallback_used: bool = False

    @property
    def start_state(self) -> SymbolicKernelState:
        return self.event_states[0]

    @property
    def terminal_state(self) -> SymbolicKernelState:
        return self.event_states[-1]

    def smt2(self) -> str:
        solver = z3.Solver()
        solver.add(self.formula)
        return solver.sexpr()


def derive_finite_event_bound(model: BoundModel, target_task: str) -> EventBound:
    """Derive a structural upper bound; never choose slots for memory reasons.

    The count deliberately over-counts simultaneous event classes.  Every
    active event macro advances to a distinct later timestamp, so an upper
    bound on the sum of possible release/deadline/completion/controller
    timestamps is also an upper bound on the number of active macros.
    """

    target = model.task_by_name.get(target_task)
    if target is None or target.criticality != "HI":
        raise ValueError("EVENT_BOUND_TARGET_MUST_BE_HI")
    deadline = int(target.deadline)

    # In an interval of length D, a phase-zero periodic stream has at most
    # floor(D/T)+1 future timestamps.  The +1 is intentionally conservative
    # with respect to the arbitrary absolute origin phase.
    release_by_task = {
        task.name: deadline // int(task.period) + 1
        for task in model.tasks
    }
    total_release = sum(release_by_task.values())
    # Only semantically usable two-slot representatives contribute terminal
    # events: HI uses one exact slot; LO uses aggregate+exact.  Unused HI slot1
    # is structurally absent and must not inflate the finite event bound.
    initial_jobs = sum(1 if row.criticality == "HI" else 2 for row in model.tasks)

    # Every present HI carry-in and every future HI release has at most one
    # deadline event in the window.
    hi_deadline_bound = len(model.hi_tasks) + sum(
        release_by_task[task.name] for task in model.hi_tasks
    )

    # Each HI release can contribute at most one exact-job terminal event.
    # Each LO release can contribute at most two: the new exact job and a
    # revived aggregate after folding unfinished prior exact work.  This is the
    # same structural argument as the previous 2*all-releases bound, but avoids
    # counting a nonexistent second HI completion.  It changes only allocated
    # Event slots, never the Event transition relation.
    future_terminal_bound = sum(
        release_by_task[row.name] * (1 if row.criticality == "HI" else 2)
        for row in model.tasks
    )
    completion_bound = initial_jobs + future_terminal_bound

    controller_bound = deadline // int(model.agent_period) + 1
    horizon_bound = 1

    # +2 is proof-structural slack for the start boundary and any simultaneous
    # class over-count.  It is independent of memory availability.
    bound = (
        total_release
        + hi_deadline_bound
        + completion_bound
        + controller_bound
        + horizon_bound
        + 2
    )
    return EventBound(
        target_task=target.name,
        deadline=deadline,
        release_bound_by_task=release_by_task,
        total_release_bound=total_release,
        initial_job_representative_bound=initial_jobs,
        hi_deadline_bound=hi_deadline_bound,
        completion_bound=completion_bound,
        controller_bound=controller_bound,
        horizon_boundary_bound=horizon_bound,
        finite_event_bound=bound,
    )


def _allowed_environment_ticks(model: BoundModel, target_task: str, deadline: int) -> dict[str, tuple[int, ...]]:
    target = model.task_by_name[target_task]
    return {
        task.name: tuple(
            tick for tick in range(deadline + 1)
            if tick % gcd(target.period, task.period) == 0
        )
        for task in model.tasks
    }


def build_event_first_bad_window(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    target_task: str,
) -> EventWindowEncoding:
    task = model.task_by_name.get(target_task)
    if task is None or task.criticality != "HI":
        raise ValueError("FIRST_BAD_EVENT_WINDOW_TARGET_MUST_BE_HI")
    if any(row.deadline > row.period for row in model.tasks):
        raise ValueError("V9_2_TWO_SLOT_CARRY_IN_REQUIRES_D_LE_T")
    if model.max_jobs_per_task < (2 if any(row.criticality == "LO" for row in model.tasks) else 1):
        raise ValueError("V9_2_TWO_SLOT_CARRY_IN_CAPACITY_INVALID")

    deadline = int(task.deadline)
    event_bound = derive_finite_event_bound(model, target_task)
    allowed_ticks = _allowed_environment_ticks(model, target_task, deadline)
    env = declare_environment(
        "event.window.env",
        model,
        release_count=deadline + 1,
        allowed_ticks_by_task=allowed_ticks,
    )

    horizon_time = env.phase.origin_time + deadline
    # ``finite_event_bound`` counts active macros.  One extra boundary state is
    # required for the destination of the final macro.
    event_states = tuple(
        declare_state(f"event.window.y.{index}", model)
        for index in range(event_bound.finite_event_bound + 1)
    )
    start = event_states[0]
    controller_pool = build_exact_controller_pool(
        model,
        origin_time=env.phase.origin_time,
        horizon_time=horizon_time,
        controller_bound=event_bound.controller_bound,
        prefix="event.window.controller",
    )

    clauses: list[z3.BoolRef] = [
        controller_pool.formula,
        *env.constraints,
        env.phase.origin_time == start.t,
        invariant.formula(start),
        start.hi_miss_ledger == 0,
        start.p == 0,
        start.frontier.selected_slot == -1,
        z3.Not(start.frontier.running),
        start.eta[target_task] == task.period,
        *target_release_constraints(env, task),
    ]

    event_steps: list[EventStepEncoding] = []
    for index, (source, destination) in enumerate(zip(event_states, event_states[1:])):
        step = encode_event_step(
            source,
            destination,
            model,
            env,
            horizon_time=horizon_time,
            prefix=f"event.window.step.{index}",
            controller_pool=controller_pool,
        )
        event_steps.append(step)
        clauses.append(event_step_or_terminal_stutter(
            step, model, horizon_time=horizon_time
        ))
        clauses.extend((
            source.t <= horizon_time,
            source.p == 0,
            source.hi_miss_ledger == 0,
        ))

    terminal = event_states[-1]
    clauses.extend((
        terminal.t == horizon_time,
        terminal.p == 0,
        terminal.hi_miss_ledger == 0,
    ))

    # Process the target deadline timestamp exactly through P2.  If the target
    # completed exactly at the horizon, P0 settles it before P2 and the bad
    # predicate is false, matching the Full kernel.
    terminal_p1 = declare_state("event.window.terminal.p1", model)
    terminal_p2 = declare_state("event.window.terminal.p2", model)
    terminal_p3 = declare_state("event.window.terminal.p3", model)
    clauses.extend((
        encode_p0_settle(terminal, terminal_p1, model),
        encode_p1_idle_recovery(terminal_p1, terminal_p2, model),
        encode_p2_deadline_observe(terminal_p2, terminal_p3, model),
    ))
    target_slots = [
        terminal_p2.jobs[(target_task, slot)]
        for slot in range(model.max_jobs_per_task)
    ]
    target_at_deadline = z3.Or(*(
        job.present
        & (job.absolute_deadline == terminal_p2.t)
        & (job.executed_service < job.effective_demand)
        for job in target_slots
    ))
    clauses.extend((
        terminal_p2.t == horizon_time,
        terminal_p2.p == 2,
        terminal_p2.hi_miss_ledger == 0,
        target_at_deadline,
        terminal_p3.p == 3,
        terminal_p3.hi_miss_ledger >= 1,
    ))

    return EventWindowEncoding(
        target_task=target_task,
        deadline=deadline,
        formula=z3.And(*clauses),
        event_states=event_states,
        event_steps=tuple(event_steps),
        terminal_phase_states=(terminal_p1, terminal_p2, terminal_p3),
        environment=env,
        event_bound=event_bound,
        controller_pool=controller_pool,
        source_obligations=(
            "event_start_is_exact_full_p0_safe_prefix_projection",
            "event_boundary_retains_all_full_safety_relevant_fields",
            "next_event_is_exact_minimum_of_bound_event_sources",
            "no_release_deadline_completion_controller_event_skipped",
            "silent_interval_service_and_eta_equal_repeated_p7",
            "event_macro_composes_exact_p0_through_p6",
            "first_bad_event_window_uses_window_global_exact_p5_pool",
            "controller_pool_count_derived_from_agent_period_bound",
            "indexed_demand_lookup_is_exact_finite_formula_factoring",
            "phase_ssa_shares_only_canonical_frame_fields",
            "terminal_stutter_is_factored_as_one_fieldwise_ite_destination_update",
            "event_layer_added_abstractions_empty",
            "target_deadline_processed_exactly_through_p2",
            "finite_event_bound_is_structural_not_memory_selected",
        ),
    )


def write_event_first_bad_window(encoding: EventWindowEncoding, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = encoding.smt2()
    path.write_text(text, encoding="utf-8")
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "ENCODER_COMPLETE",
    "ENCODER_READINESS_GAPS",
    "ENCODER_VERSION",
    "EVENT_KERNEL_VERSION",
    "EventBound",
    "EventWindowEncoding",
    "build_event_first_bad_window",
    "derive_finite_event_bound",
    "write_event_first_bad_window",
]
