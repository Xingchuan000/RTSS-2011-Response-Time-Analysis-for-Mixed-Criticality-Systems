"""V9.3 exact event-driven first-HI-bad-window encoding.

The terminal proof route no longer allocates D*8 Full-kernel states.  It
allocates a mathematically bounded number of event boundaries and, for each
active boundary, one exact P0--P6 closure plus one algebraic silent interval.
No Event-layer abstraction is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import z3

from .solver_runtime import make_solver

from .environment_encoder import (
    declare_event_graph_environment,
    target_release_constraints,
)
from .carry_in import derive_protected_priority_prefix
from .event_kernel import EVENT_KERNEL_VERSION, EventStepEncoding
from .safe_prefix_invariant import SafePrefixInvariant
from .target_projection import event_root_safe_prefix_formula
from .target_projection import TargetSchedulingProjection, derive_target_scheduling_projection
from .symbolic_state import (
    BoundModel, SymbolicKernelState, declare_sparse_successor, declare_state,
)
from .transition_encoder import (
    encode_p0_settle,
    encode_p1_idle_recovery,
    encode_p2_deadline_observe,
)


ENCODER_VERSION = "V9_3_EVENT_FIRST_BAD_WINDOW_EXPLICIT_EVENT_GRAPH"
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
    source_obligations: tuple[str, ...]
    event_layer_added_abstractions: tuple[str, ...] = ()
    exact_p5_in_event_window: bool = True

    @property
    def start_state(self) -> SymbolicKernelState:
        return self.event_states[0]

    @property
    def terminal_state(self) -> SymbolicKernelState:
        return self.event_states[-1]

    def smt2(self) -> str:
        solver = make_solver()
        solver.add(self.formula)
        return solver.sexpr()


@dataclass(frozen=True, slots=True)
class TerminalBadEncoding:
    """Exact target-deadline P0->P2 bad observation at one Event depth."""

    depth: int
    terminal_state: SymbolicKernelState
    phase_states: tuple[SymbolicKernelState, ...]
    formula: z3.BoolRef


@dataclass(frozen=True, slots=True)
class EventGraphProblem:
    """Shared symbolic root for explicit exact Event-graph exploration."""

    target_task: str
    deadline: int
    model: BoundModel
    environment: Any
    horizon_time: z3.ArithRef
    event_bound: EventBound
    base_formula: z3.BoolRef
    start_state: SymbolicKernelState
    projection: TargetSchedulingProjection
    source_obligations: tuple[str, ...]
    event_layer_added_abstractions: tuple[str, ...] = ()
    exact_p5_in_event_window: bool = True

    def build_terminal_bad_query(
        self, state: SymbolicKernelState, *, depth: int
    ) -> TerminalBadEncoding:
        return _build_terminal_bad_query(
            state, self.model, self.target_task, self.horizon_time, depth=depth,
            active_task_names=self.projection.active_task_names,
        )

    def materialize_sat_path(
        self,
        *,
        root_case: z3.BoolRef,
        event_states: tuple[SymbolicKernelState, ...],
        event_steps: tuple[EventStepEncoding, ...],
        path_formulas: tuple[z3.BoolRef, ...],
        terminal: TerminalBadEncoding,
    ) -> EventWindowEncoding:
        return EventWindowEncoding(
            target_task=self.target_task,
            deadline=self.deadline,
            formula=z3.And(
                self.base_formula, root_case, *path_formulas, terminal.formula
            ),
            event_states=event_states,
            event_steps=event_steps,
            terminal_phase_states=terminal.phase_states,
            environment=self.environment,
            event_bound=self.event_bound,
            source_obligations=self.source_obligations,
            event_layer_added_abstractions=self.event_layer_added_abstractions,
            exact_p5_in_event_window=self.exact_p5_in_event_window,
        )



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
    active_tasks = tuple(
        task for task in model.tasks if task.priority <= target.priority
    )

    # In an interval of length D, a phase-zero periodic stream has at most
    # floor(D/T)+1 future timestamps.  The +1 is intentionally conservative
    # with respect to the arbitrary absolute origin phase.
    release_by_task = {
        task.name: deadline // int(task.period) + 1
        for task in active_tasks
    }
    total_release = sum(release_by_task.values())
    # V9.3 reachable carry-in refinement removes the historical aggregate for
    # protected LO tasks.  Only unprotected LO tasks retain aggregate+exact
    # representatives in the structural completion bound.
    protected = set(derive_protected_priority_prefix(model).task_names)
    initial_jobs = sum(
        1 if row.criticality == "HI" or row.name in protected else 2
        for row in active_tasks
    )

    # Every present HI carry-in and every future HI release has at most one
    # deadline event in the window.
    active_hi = tuple(task for task in active_tasks if task.criticality == "HI")
    hi_deadline_bound = len(active_hi) + sum(
        release_by_task[task.name] for task in active_hi
    )

    # Protected LO releases cannot revive an aggregate: their preceding exact
    # job has settled before the next release.  Unprotected LO remains at two
    # possible completion representatives.
    future_terminal_bound = sum(
        release_by_task[row.name]
        * (1 if row.criticality == "HI" or row.name in protected else 2)
        for row in active_tasks
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


def _validate_event_window_target(model: BoundModel, target_task: str) -> Any:
    task = model.task_by_name.get(target_task)
    if task is None or task.criticality != "HI":
        raise ValueError("FIRST_BAD_EVENT_WINDOW_TARGET_MUST_BE_HI")
    if any(row.deadline > row.period for row in model.tasks):
        raise ValueError("V9_2_TWO_SLOT_CARRY_IN_REQUIRES_D_LE_T")
    if model.max_jobs_per_task < (2 if any(row.criticality == "LO" for row in model.tasks) else 1):
        raise ValueError("V9_2_TWO_SLOT_CARRY_IN_CAPACITY_INVALID")
    return task


def _build_terminal_bad_query(
    terminal: SymbolicKernelState,
    model: BoundModel,
    target_task: str,
    horizon_time: z3.ArithRef,
    *,
    depth: int,
    active_task_names: tuple[str, ...] | None = None,
) -> TerminalBadEncoding:
    """Exact final deadline observation shared by monolithic/incremental BMC."""

    # Reuse the already-certified exact SSA frame elimination for P0--P2 so
    # every depth query allocates only the fields those phases may mutate.
    terminal_p1 = declare_sparse_successor(
        f"event.window.depth.{depth}.terminal.p1", terminal, model, phase=1,
        mutable=frozenset({"jobs.present", "jobs.removed", "jobs.ready"}),
    )
    terminal_p2 = declare_sparse_successor(
        f"event.window.depth.{depth}.terminal.p2", terminal_p1, model, phase=2,
        mutable=frozenset({"mode"}),
    )
    terminal_p3 = declare_sparse_successor(
        f"event.window.depth.{depth}.terminal.p3", terminal_p2, model, phase=3,
        mutable=frozenset({"hi_miss_ledger"}),
    )
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
    formula = z3.And(
        terminal.t == horizon_time,
        terminal.p == 0,
        terminal.hi_miss_ledger == 0,
        encode_p0_settle(
            terminal, terminal_p1, model, active_task_names=active_task_names
        ),
        encode_p1_idle_recovery(
            terminal_p1, terminal_p2, model, active_task_names=active_task_names
        ),
        encode_p2_deadline_observe(
            terminal_p2, terminal_p3, model, active_task_names=active_task_names
        ),
        terminal_p2.t == horizon_time,
        terminal_p2.p == 2,
        terminal_p2.hi_miss_ledger == 0,
        target_at_deadline,
        terminal_p3.p == 3,
        terminal_p3.hi_miss_ledger >= 1,
    )
    return TerminalBadEncoding(
        depth=int(depth),
        terminal_state=terminal,
        phase_states=(terminal_p1, terminal_p2, terminal_p3),
        formula=formula,
    )


def _window_source_obligations() -> tuple[str, ...]:
    rows = [
        "event_start_is_target_local_safe_prefix_plus_reachable_carry_in",
        "target_priority_prefix_jobs_and_eta_are_retained",
        "full_policy_budget_and_history_state_is_retained",
        "lower_priority_job_scheduling_is_removed_by_fixed_priority_hi_target_dominance",
        "omitted_lower_priority_hi_release_mode_switch_can_only_reduce_target_interference",
        "next_event_source_partition_is_exact_disjoint_first_minimum_owner",
        "no_target_or_higher_priority_release_deadline_completion_controller_event_skipped",
        "silent_interval_service_and_eta_equal_repeated_p7",
        "event_macro_composes_exact_p0_through_p6",
        "controller_timestamp_is_owned_by_canonical_event_source",
        "controller_p5_branch_is_specialized_by_graph_node",
        "release_demand_is_fresh_bounded_per_explicit_release_node",
        "event_candidate_minima_are_exact_scalar_ssa_definitions",
        "event_root_periodic_phase_uses_exact_linear_quotients_without_modulo",
        "periodic_event_candidates_use_exact_quotient_free_scalar_successors",
        "p6_dispatch_is_compiled_per_exact_winner_case",
        "source_time_checks_reuse_the_solved_incremental_parent_context",
        "phase_ssa_shares_only_canonical_frame_fields",
    ]
    rows.append("explicit_event_graph_explores_each_reachable_source_prefix_once")
    rows.extend((
        "target_deadline_processed_exactly_through_p2",
        "finite_event_bound_is_structural_not_memory_selected",
    ))
    return tuple(rows)


def build_event_graph_problem(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    target_task: str,
) -> EventGraphProblem:
    """Build the exact symbolic root for source-specialized Event-graph DFS."""

    task = _validate_event_window_target(model, target_task)
    deadline = int(task.deadline)
    projection = derive_target_scheduling_projection(model, target_task)
    event_bound = derive_finite_event_bound(model, target_task)
    env = declare_event_graph_environment(
        "event.window.env", model, horizon=deadline + 1,
    )
    start = declare_state("event.window.y.root", model)
    horizon_time = env.phase.origin_time + deadline
    base_formula = z3.And(
        *env.constraints,
        env.phase.origin_time == start.t,
        event_root_safe_prefix_formula(start, model, invariant, target_task),
        start.hi_miss_ledger == 0,
        start.p == 0,
        start.frontier.selected_slot == -1,
        z3.Not(start.frontier.running),
        start.eta[target_task] == task.period,
        *target_release_constraints(env, task),
    )
    return EventGraphProblem(
        target_task=target_task, deadline=deadline, model=model,
        environment=env, horizon_time=horizon_time, event_bound=event_bound,
        base_formula=base_formula, start_state=start, projection=projection,
        source_obligations=_window_source_obligations(),
        event_layer_added_abstractions=(
            "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE",
        ),
    )



__all__ = [
    "ENCODER_COMPLETE",
    "ENCODER_READINESS_GAPS",
    "ENCODER_VERSION",
    "EVENT_KERNEL_VERSION",
    "EventBound",
    "EventGraphProblem",
    "EventWindowEncoding",
    "TerminalBadEncoding",
    "build_event_graph_problem",
    "derive_finite_event_bound",
]
