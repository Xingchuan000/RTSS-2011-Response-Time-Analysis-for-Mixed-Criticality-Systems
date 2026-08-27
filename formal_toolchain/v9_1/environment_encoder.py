"""Exact-periodic admissible environment encoding for V9.1.

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
    constraints: tuple[z3.BoolRef, ...]
    release_times: Mapping[tuple[str, int], int]
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
    earlier V9.1 development, but its meaning is now the finite relative-time
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
        for tick in ticks:
            key = (task.name, tick)
            value = z3.Int(f"{prefix}.A.{task.name}.{tick}")
            demands[key] = value
            release_times[key] = tick
            constraints.append(z3.And(value >= lower, value <= upper))
    return SymbolicEnvironment(demands, tuple(constraints), release_times, phase, int(release_count))


def demand_for_time(
    env: SymbolicEnvironment,
    task: TaskBound,
    absolute_time: z3.ArithRef,
) -> tuple[z3.ArithRef, z3.BoolRef]:
    """Select the fresh demand associated with ``absolute_time``.

    The second result proves that the timestamp is covered by the finite
    environment prefix.  P3 requires it whenever a release is due, so no
    release can silently fall outside the quantified demand domain.
    """

    rows = sorted(
        (tick, demand)
        for (name, tick), demand in env.actual_demands.items()
        if name == task.name
    )
    if not rows:
        raise ValueError(f"environment has no demand variables for task {task.name}")
    covered = z3.Or(*(absolute_time == env.phase.origin_time + tick for tick, _ in rows))
    value: z3.ArithRef = rows[-1][1]
    for tick, demand in reversed(rows[:-1]):
        value = z3.If(absolute_time == env.phase.origin_time + tick, demand, value)
    return value, covered


def target_release_constraints(
    env: SymbolicEnvironment, task: TaskBound, *, release_index: int = 0
) -> tuple[z3.BoolRef, ...]:
    if release_index != 0:
        raise ValueError("V9.1 relative first-bad windows pin the target at relative release 0")
    if (task.name, 0) not in env.actual_demands:
        raise KeyError("target release was not declared")
    return (env.phase.origin_time % task.period == 0,)


def classify_from_actual_demand(actual_demand: z3.ArithRef, task: TaskBound) -> z3.BoolRef:
    if task.criticality != "HI":
        return z3.BoolVal(False)
    return actual_demand > task.c_lo


__all__ = [
    "PeriodicPhaseContext", "SymbolicEnvironment", "classify_from_actual_demand",
    "declare_environment", "demand_for_time", "periodic_phase_constraints",
    "target_release_constraints",
]
