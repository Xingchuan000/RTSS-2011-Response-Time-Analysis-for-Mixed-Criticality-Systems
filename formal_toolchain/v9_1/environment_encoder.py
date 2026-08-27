"""Exact-periodic admissible environment encoding for V9.1."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Mapping

import z3

from .symbolic_state import BoundModel, TaskBound


@dataclass(frozen=True, slots=True)
class PeriodicPhaseContext:
    absolute_time_residue: z3.ArithRef
    modulus: int


@dataclass(frozen=True, slots=True)
class SymbolicEnvironment:
    actual_demands: Mapping[tuple[str, int], z3.ArithRef]
    constraints: tuple[z3.BoolRef, ...]
    release_times: Mapping[tuple[str, int], int]
    phase: PeriodicPhaseContext


def periodic_phase_constraints(
    ctx: PeriodicPhaseContext,
    taskset: BoundModel | tuple[TaskBound, ...],
    agent_period: int,
) -> tuple[z3.BoolRef, ...]:
    """Constrain the absolute residue without widening periodic releases.

    The residue is retained because controller triggers depend on absolute time.
    A caller additionally pins the target task residue to zero modulo its period
    when constructing a target pre-release window.
    """

    tasks = taskset.tasks if isinstance(taskset, BoundModel) else tuple(taskset)
    modulus = lcm(agent_period, *(task.period for task in tasks))
    if modulus != ctx.modulus:
        raise ValueError("periodic phase modulus does not match taskset LCM")
    return (ctx.absolute_time_residue >= 0, ctx.absolute_time_residue < modulus)


def declare_environment(prefix: str, model: BoundModel, *, release_count: int) -> SymbolicEnvironment:
    """Declare independent fresh demand integers for every finite release."""

    if release_count <= 0:
        raise ValueError("release_count must be positive")
    modulus = lcm(model.agent_period, *(task.period for task in model.tasks))
    residue = z3.Int(f"{prefix}.absolute_time_residue")
    phase = PeriodicPhaseContext(residue, modulus)
    demands: dict[tuple[str, int], z3.ArithRef] = {}
    release_times: dict[tuple[str, int], int] = {}
    constraints: list[z3.BoolRef] = list(periodic_phase_constraints(phase, model, model.agent_period))
    for task in model.tasks:
        for release_index in range(release_count):
            key = (task.name, release_index)
            value = z3.Int(f"{prefix}.A.{task.name}.{release_index}")
            demands[key] = value
            release_times[key] = release_index * task.period
            upper = task.c_hi if task.criticality == "HI" else task.c_lo
            constraints.append(z3.And(value >= 1, value <= upper))
    return SymbolicEnvironment(demands, tuple(constraints), release_times, phase)


def target_release_constraints(
    env: SymbolicEnvironment, task: TaskBound, *, release_index: int = 0
) -> tuple[z3.BoolRef, ...]:
    """Pin a target pre-release origin to an exact periodic release phase."""

    if (task.name, release_index) not in env.actual_demands:
        raise KeyError("target release was not declared")
    return (env.phase.absolute_time_residue % task.period == 0,)


def classify_from_actual_demand(actual_demand: z3.ArithRef, task: TaskBound) -> z3.BoolRef:
    """HI abnormality is derived from A, never an independent symbolic Bool."""

    if task.criticality != "HI":
        return z3.BoolVal(False)
    return actual_demand > task.c_lo


__all__ = [
    "PeriodicPhaseContext", "SymbolicEnvironment", "classify_from_actual_demand",
    "declare_environment", "periodic_phase_constraints", "target_release_constraints",
]
