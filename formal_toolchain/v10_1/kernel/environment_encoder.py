"""Exact-periodic admissible environment encoding for V10.1.

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
    phase: PeriodicPhaseContext
    horizon: int


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
    earlier V10.1 development, but its meaning is now the finite relative-time
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
            constraints.extend((
                z3.And(value >= lower, value <= upper),
                lookup(z3.IntVal(tick)) == value,
            ))
    return SymbolicEnvironment(
        demands, lookups, allowed_ticks, regular_steps, tuple(constraints),
        phase, int(release_count)
    )


def demand_for_time(
    env: SymbolicEnvironment,
    task: TaskBound,
    absolute_time: z3.ArithRef,
) -> tuple[z3.ArithRef, z3.BoolRef]:
    """Select the independently bound demand for ``absolute_time``.

    Every finite relative release tick owns one explicit demand scalar.  P3
    indexes the shared lookup by ``absolute_time-origin``; no scheduler-event
    search or lazy event-node demand allocation is part of the V10.1 kernel.
    """

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


def classify_from_actual_demand(actual_demand: z3.ArithRef, task: TaskBound) -> z3.BoolRef:
    if task.criticality != "HI":
        return z3.BoolVal(False)
    return actual_demand > task.c_lo


__all__ = [
    "PeriodicPhaseContext", "SymbolicEnvironment", "classify_from_actual_demand",
    "declare_environment", "demand_for_time", "periodic_phase_constraints",
]
