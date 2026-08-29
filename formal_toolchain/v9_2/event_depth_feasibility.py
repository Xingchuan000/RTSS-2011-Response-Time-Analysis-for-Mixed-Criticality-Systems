"""Cheap exact event-depth feasibility facts for the V9.3 Event route.

These checks never approximate scheduler state.  They use only periodic phase
arithmetic plus the already-derived protected-priority response envelopes to
rule out Event depths/leaves that cannot possibly jump directly to the target
horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from .carry_in import derive_protected_priority_prefix
from .symbolic_state import BoundModel


@dataclass(frozen=True, slots=True)
class EventDepthFloor:
    target_task: str
    minimum_depth: int
    witness_task: str | None
    witness_response_bound: int | None


def possible_relative_phase_offsets(target_period: int, stream_period: int) -> tuple[int, ...]:
    """All possible first phase-zero stream ticks relative to a target release."""

    step = gcd(int(target_period), int(stream_period))
    return tuple(range(0, int(stream_period), step))


def _stream_minimum_depth(
    *,
    deadline: int,
    stream_period: int,
    response_bound: int,
    offsets: tuple[int, ...],
) -> int:
    """Minimum active Event macros forced by one protected periodic stream.

    A release strictly after the window origin is an Event boundary.  If its
    protected response envelope ends strictly before the horizon, the job must
    also create a later completion boundary.  For R<T these release/completion
    timestamps alternate and are distinct for the same stream.  The horizon is
    one final active macro.
    """

    if response_bound >= stream_period:
        return 1
    floors: list[int] = []
    for offset in offsets:
        ticks = tuple(range(int(offset), int(deadline), int(stream_period)))
        release_boundaries = sum(1 for tick in ticks if tick > 0)
        forced_completions = sum(
            1 for tick in ticks if tick + int(response_bound) < int(deadline)
        )
        floors.append(1 + release_boundaries + forced_completions)
    return min(floors) if floors else 1


def derive_minimum_event_depth(model: BoundModel, target_task: str) -> EventDepthFloor:
    """Return a sound host-side lower bound on a target bad-window depth.

    Only *higher-priority* protected streams are used.  The target task's own
    response envelope is deliberately excluded so this remains an Event-shape
    pruning theorem rather than a second RTA terminal safety route.
    """

    target = model.task_by_name[target_task]
    prefix = derive_protected_priority_prefix(model)
    response = prefix.response_by_task
    best_depth = 1
    witness: str | None = None
    witness_r: int | None = None
    for task in model.tasks:
        if task.priority >= target.priority:
            break
        r_i = response.get(task.name)
        if r_i is None:
            continue
        floor = _stream_minimum_depth(
            deadline=int(target.deadline),
            stream_period=int(task.period),
            response_bound=int(r_i),
            offsets=possible_relative_phase_offsets(target.period, task.period),
        )
        if floor > best_depth:
            best_depth = floor
            witness = task.name
            witness_r = int(r_i)
    return EventDepthFloor(target_task, best_depth, witness, witness_r)


def penultimate_release_is_impossible(
    model: BoundModel,
    target_task: str,
    release_task: str,
    release_tick: int,
) -> bool:
    """Whether a penultimate release must be followed by another pre-horizon event.

    If the released task is a protected higher-priority task and its response
    envelope ends strictly before the target horizon, that job must settle at a
    later timestamp before the horizon.  Hence the release boundary cannot be
    the penultimate boundary of an exact Event prefix.
    """

    target = model.task_by_name[target_task]
    task = model.task_by_name[release_task]
    if task.priority >= target.priority:
        return False
    r_i = derive_protected_priority_prefix(model).response_by_task.get(task.name)
    if r_i is None:
        return False
    return int(release_tick) + int(r_i) < int(target.deadline)


__all__ = [
    "EventDepthFloor",
    "derive_minimum_event_depth",
    "penultimate_release_is_impossible",
    "possible_relative_phase_offsets",
]
