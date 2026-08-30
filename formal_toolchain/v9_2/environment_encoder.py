"""Exact-periodic admissible environment encoding for V9.2.

A first-bad window is relative to an arbitrary absolute periodic phase.  Demand
variables are therefore indexed by *relative integer tick* rather than by a
finite job slot: slot recycling cannot accidentally reuse one release's demand
for another release.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from collections.abc import Collection
from typing import Mapping

import z3

from .symbolic_state import BoundModel, TaskBound


@dataclass(frozen=True, slots=True)
class PeriodicPhaseContext:
    absolute_time_residue: z3.ArithRef
    origin_time: z3.ArithRef
    modulus: int


@dataclass(frozen=True, slots=True)
class SymbolicEnvironment:
    actual_demands: Mapping[tuple[str, int], z3.ArithRef]
    demand_lookup: Mapping[str, z3.FuncDeclRef]
    allowed_ticks_by_task: Mapping[str, tuple[int, ...]]
    regular_tick_step_by_task: Mapping[str, int | None]
    constraints: tuple[z3.BoolRef, ...]
    release_times: Mapping[tuple[str, int], int]
    phase: PeriodicPhaseContext
    horizon: int
    lazy_release_demands: bool = False


def periodic_phase_constraints(
    ctx: PeriodicPhaseContext,
    taskset: BoundModel | tuple[TaskBound, ...],
    agent_period: int,
) -> tuple[z3.BoolRef, ...]:
    tasks = taskset.tasks if isinstance(taskset, BoundModel) else tuple(taskset)
    modulus = lcm(agent_period, *(task.period for task in tasks))
    if modulus != ctx.modulus:
        raise ValueError("periodic phase modulus does not match taskset LCM")
    return (
        ctx.absolute_time_residue >= 0,
        ctx.absolute_time_residue < modulus,
        ctx.origin_time >= 0,
        ctx.origin_time % modulus == ctx.absolute_time_residue,
    )


def declare_environment(
    prefix: str,
    model: BoundModel,
    *,
    release_count: int,
    allowed_ticks_by_task: Mapping[str, Collection[int]] | None = None,
) -> SymbolicEnvironment:
    """Declare independent actual demand for each task at each relative tick.

    ``release_count`` is retained as the public argument name for callers from
    earlier V9.2 development, but its meaning is now the finite relative-time
    horizon size.  A task consumes ``A[task,k]`` only when a periodic release is
    due at ``origin_time + k``.
    """

    if release_count <= 0:
        raise ValueError("release_count must be positive")
    modulus = lcm(model.agent_period, *(task.period for task in model.tasks))
    residue = z3.Int(f"{prefix}.absolute_time_residue")
    origin = z3.Int(f"{prefix}.origin_time")
    phase = PeriodicPhaseContext(residue, origin, modulus)
    demands: dict[tuple[str, int], z3.ArithRef] = {}
    lookups: dict[str, z3.FuncDeclRef] = {}
    allowed_ticks: dict[str, tuple[int, ...]] = {}
    regular_steps: dict[str, int | None] = {}
    release_times: dict[tuple[str, int], int] = {}
    constraints: list[z3.BoolRef] = list(periodic_phase_constraints(phase, model, model.agent_period))
    for task in model.tasks:
        lower = task.actual_demand_min
        upper = task.actual_demand_upper
        if allowed_ticks_by_task is None:
            ticks = range(release_count)
        else:
            raw_ticks = allowed_ticks_by_task.get(task.name, ())
            ticks = tuple(sorted({int(tick) for tick in raw_ticks if 0 <= int(tick) < release_count}))
            if not ticks:
                raise ValueError(f"no admissible relative release ticks declared for {task.name}")
        ticks = tuple(ticks)
        allowed_ticks[task.name] = ticks
        step: int | None = None
        if ticks and ticks[0] == 0:
            if len(ticks) == 1:
                step = release_count
            else:
                candidate = ticks[1] - ticks[0]
                if candidate > 0 and ticks == tuple(range(0, release_count, candidate)):
                    step = candidate
        regular_steps[task.name] = step
        lookup = z3.Function(f"{prefix}.A_lookup.{task.name}", z3.IntSort(), z3.IntSort())
        lookups[task.name] = lookup
        for tick in ticks:
            key = (task.name, tick)
            value = z3.Int(f"{prefix}.A.{task.name}.{tick}")
            demands[key] = value
            release_times[key] = tick
            constraints.extend((
                z3.And(value >= lower, value <= upper),
                lookup(z3.IntVal(tick)) == value,
            ))
    return SymbolicEnvironment(
        demands, lookups, allowed_ticks, regular_steps, tuple(constraints),
        release_times, phase, int(release_count), False
    )


def declare_event_graph_environment(
    prefix: str,
    model: BoundModel,
    *,
    horizon: int,
) -> SymbolicEnvironment:
    """Declare only the absolute periodic phase for explicit Event-graph search.

    The old finite-window encoder predeclared one demand scalar for every
    relative tick at which a task *might* release under some admissible origin
    phase, then tied them to per-task lookup UFs.  That table was useful when a
    monolithic BMC could revisit/recycle a job slot, but it is unnecessary in
    the explicit Event graph: every P3 occurrence belongs to one unique event
    node and every concrete release gets one fresh demand scalar at that node.

    This environment therefore keeps only the exact common periodic phase.
    ``demand_for_time`` allocates a bounded release demand lazily when P3 is
    actually encoded.  Different releases remain independent, exactly as in
    the original environment semantics.
    """

    if horizon <= 0:
        raise ValueError("event-graph horizon must be positive")
    # The explicit Event graph never consumes the old window-global LCM
    # residue.  Per-task phase is already represented exactly by ``eta`` in
    # the root SafePrefix and controller phase is carried separately by the
    # graph solver.  Keeping ``origin % lcm(all periods)`` here only injects a
    # huge, semantically dead Presburger term into every graph query.
    origin = z3.Int(f"{prefix}.origin_time")
    phase = PeriodicPhaseContext(z3.IntVal(0), origin, 1)
    constraints = (origin >= 0,)
    return SymbolicEnvironment(
        actual_demands={},
        demand_lookup={},
        allowed_ticks_by_task={},
        regular_tick_step_by_task={},
        constraints=constraints,
        release_times={},
        phase=phase,
        horizon=int(horizon),
        lazy_release_demands=True,
    )


def demand_for_time(
    env: SymbolicEnvironment,
    task: TaskBound,
    absolute_time: z3.ArithRef,
) -> tuple[z3.ArithRef, z3.BoolRef]:
    """Select the fresh release demand at ``absolute_time`` in O(1) AST size.

    Every admissible finite release tick still owns an independent explicit
    demand variable.  A window-global uninterpreted lookup function is tied to
    those variables once in :func:`declare_environment`; P3 then indexes that
    exact table by ``absolute_time-origin`` instead of rebuilding a long nested
    If chain in every Event slot.  This is formula factoring only: no demand
    value, release timestamp, or environment behavior is added or removed.
    """

    if env.lazy_release_demands:
        # ``absolute_time`` is a structurally shared event-state timestamp and
        # therefore gives a stable unique name per graph node.  Reusing the
        # same scalar across alternative outgoing edges from that node is also
        # exact: P3 at the current timestamp precedes the choice of the *next*
        # event and must observe one common demand for the same release.
        demand = z3.Int(f"{absolute_time}.A.{task.name}")
        covered = z3.And(
            demand >= int(task.actual_demand_min),
            demand <= int(task.actual_demand_upper),
        )
        return demand, covered

    ticks = env.allowed_ticks_by_task.get(task.name, ())
    if not ticks:
        raise ValueError(f"environment has no demand variables for task {task.name}")
    relative = absolute_time - env.phase.origin_time
    lookup = env.demand_lookup[task.name]
    step = env.regular_tick_step_by_task.get(task.name)
    if step is not None:
        if len(ticks) == 1:
            covered = relative == ticks[0]
        else:
            covered = z3.And(
                relative >= 0,
                relative < env.horizon,
                relative % int(step) == 0,
            )
    else:
        covered = z3.Or(*(relative == tick for tick in ticks))
    return lookup(relative), covered


def target_release_constraints(
    env: SymbolicEnvironment, task: TaskBound, *, release_index: int = 0
) -> tuple[z3.BoolRef, ...]:
    if release_index != 0:
        raise ValueError("V9.2 relative first-bad windows pin the target at relative release 0")
    if not env.lazy_release_demands and (task.name, 0) not in env.actual_demands:
        raise KeyError("target release was not declared")
    return (env.phase.origin_time % task.period == 0,)


def classify_from_actual_demand(actual_demand: z3.ArithRef, task: TaskBound) -> z3.BoolRef:
    if task.criticality != "HI":
        return z3.BoolVal(False)
    return actual_demand > task.c_lo


__all__ = [
    "PeriodicPhaseContext", "SymbolicEnvironment", "classify_from_actual_demand",
    "declare_environment", "declare_event_graph_environment", "demand_for_time", "periodic_phase_constraints",
    "target_release_constraints",
]
