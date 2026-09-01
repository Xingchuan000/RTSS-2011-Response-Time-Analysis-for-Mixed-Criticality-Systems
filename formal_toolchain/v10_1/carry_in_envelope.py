"""Reachable aggregate carry-in envelopes for V10.1 PCSSC.

The response window starts at a first-bad HI release.  Carry-in is therefore
work already pending strictly before relative time zero.  This module bounds
that work as one *aggregate uniprocessor backlog*, rather than summing mutually
independent per-task carry maxima.

Two facts make the construction finite without an Event Graph:

* releases are frozen exact-periodic phase-zero releases; and
* both the pre-switch and post-switch aggregate arrival curves must have
  utilization below one for an unprotected backlog certificate to exist.

For PRE_HI, at most one LO->HI switch is allowed in the busy interval.  LO jobs
released exactly at the switch retain primary service; HI jobs released exactly
at the switch use HI service, matching the V10.1 endpoint semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import ceil, gcd, lcm
from typing import Iterable

from .periodic_release import compatible_release_phases


@dataclass(frozen=True, slots=True)
class CarryTaskSpec:
    name: str
    criticality: str
    period: int
    pre_switch_cap: int
    post_switch_cap: int

    def __post_init__(self) -> None:
        if self.criticality not in {"LO", "HI"}:
            raise ValueError("CARRY_TASK_CRITICALITY_INVALID")
        if self.period <= 0 or self.pre_switch_cap < 0 or self.post_switch_cap < 0:
            raise ValueError("CARRY_TASK_BOUND_INVALID")


class CarryInEnvelopeUnresolved(RuntimeError):
    pass


def _ceil_div(value: int, divisor: int) -> int:
    return -((-int(value)) // int(divisor))


def _count_releases(phase: int, period: int, lower: int, upper: int) -> int:
    """Count ``phase + k*period`` in the half-open interval [lower, upper)."""
    if upper <= lower:
        return 0
    return _ceil_div(int(upper) - int(phase), int(period)) - _ceil_div(
        int(lower) - int(phase), int(period)
    )


def _utilization(specs: tuple[CarryTaskSpec, ...], *, post: bool) -> Fraction:
    return sum(
        (Fraction(spec.post_switch_cap if post else spec.pre_switch_cap, spec.period)
         for spec in specs),
        Fraction(0, 1),
    )


@lru_cache(maxsize=4096)
def _busy_period_upper(specs: tuple[CarryTaskSpec, ...], *, post: bool) -> int:
    """Synchronous-release busy-period upper bound.

    Synchronous releases dominate the length of an aggregate sporadic busy
    period.  We need only a finite upper bound, not an exact response theorem.
    """
    if not specs:
        return 0
    if _utilization(specs, post=post) >= 1:
        raise CarryInEnvelopeUnresolved(
            "AGGREGATE_CARRY_IN_UTILIZATION_NOT_STABLE:"
            + ("POST_SWITCH" if post else "PRE_SWITCH")
        )
    caps = tuple(
        int(spec.post_switch_cap if post else spec.pre_switch_cap) for spec in specs
    )
    value = sum(caps)
    while True:
        following = sum(
            ceil(int(value) / int(spec.period)) * cap
            for spec, cap in zip(specs, caps)
        )
        if following <= value:
            return int(value)
        value = int(following)


def _arbitrary_phase_backlog_upper(
    specs: tuple[CarryTaskSpec, ...], *, post: bool, busy_upper: int
) -> int:
    """Arrival-curve backlog upper for an arbitrary phase at one instant."""
    if busy_upper <= 0 or not specs:
        return 0
    caps = tuple(
        int(spec.post_switch_cap if post else spec.pre_switch_cap) for spec in specs
    )
    candidates = {1}
    # ceil(L/T) changes at 1+kT.  Between changes demand is constant while
    # service increases, so no other integer L can maximize the backlog.
    for spec in specs:
        length = 1
        while length <= int(busy_upper):
            candidates.add(int(length))
            length += int(spec.period)
    best = 0
    for length in candidates:
        demand = sum(
            ceil(int(length) / int(spec.period)) * cap
            for spec, cap in zip(specs, caps)
        )
        best = max(best, int(demand) - int(length))
    return max(0, int(best))


def _transition_busy_upper(specs: tuple[CarryTaskSpec, ...]) -> tuple[int, int, int]:
    """Return (pre_busy, pre_backlog, total_single_switch_busy_upper)."""
    pre_busy = _busy_period_upper(specs, post=False)
    pre_backlog = _arbitrary_phase_backlog_upper(
        specs, post=False, busy_upper=pre_busy
    )
    post_busy = _busy_period_upper(specs, post=True)
    if not specs:
        return pre_busy, pre_backlog, 0

    # Starting a post-switch segment with at most pre_backlog units of old work,
    # a post-fixpoint of B0 + sum ceil(Q/T) C_HI bounds the continuously-busy
    # duration after the switch.  Start with the initial burst; reaching a
    # post-fixpoint is sufficient and needs no artificial iteration limit.
    caps = tuple(int(spec.post_switch_cap) for spec in specs)
    value = int(pre_backlog + sum(caps))
    while True:
        following = int(pre_backlog) + sum(
            ceil(int(value) / int(spec.period)) * cap
            for spec, cap in zip(specs, caps)
        )
        if following <= value:
            post_with_initial = int(value)
            break
        value = int(following)
    return int(pre_busy), int(pre_backlog), int(pre_busy + max(post_busy, post_with_initial))


def _phase_event_ages(
    specs: tuple[CarryTaskSpec, ...], phases: tuple[int, ...], horizon: int
) -> dict[int, tuple[int, int, int]]:
    """age -> (post demand, total pre-post delta, LO endpoint delta)."""
    groups: dict[int, list[int]] = {}
    for spec, phase in zip(specs, phases):
        period = int(spec.period)
        age = period if int(phase) == 0 else period - int(phase)
        while age <= int(horizon):
            row = groups.setdefault(int(age), [0, 0, 0])
            row[0] += int(spec.post_switch_cap)
            delta = int(spec.pre_switch_cap) - int(spec.post_switch_cap)
            row[1] += delta
            if spec.criticality == "LO":
                # LO release at the switch is primary; HI release at the switch
                # is already post-switch/high.
                row[2] += delta
            age += period
    return {age: (row[0], row[1], row[2]) for age, row in groups.items()}


def fixed_phase_single_switch_backlog(
    specs: tuple[CarryTaskSpec, ...], phases: tuple[int, ...]
) -> tuple[int, dict[str, int]]:
    """Aggregate carry bound for one exact joint phase vector.

    For a busy interval of length L ending at zero, begin with the hypothetical
    all-post-switch demand.  Moving the single switch through the interval adds
    the pre/post delta of an oldest prefix of release groups.  Processing groups
    from newest to oldest allows the maximum prefix delta to be maintained in
    O(number_of_release_groups), while the service term is exactly ``-L``.
    """
    if len(specs) != len(phases):
        raise ValueError("CARRY_PHASE_DIMENSION_MISMATCH")
    if not specs:
        return 0, {"busy_horizon": 0, "witness_length": 0}
    for spec, phase in zip(specs, phases):
        if not (0 <= int(phase) < int(spec.period)):
            raise ValueError("CARRY_PHASE_OUT_OF_RANGE")

    pre_busy, pre_backlog, horizon = _transition_busy_upper(specs)
    groups = _phase_event_ages(specs, phases, horizon)
    post_demand = 0
    max_switch_delta = 0
    best = 0
    witness = 0
    witness_delta = 0

    # Ages grow from recent to old.  Adding an older release group prepends it
    # to the existing release sequence.  A switch may be before the group,
    # exactly at it (LO endpoint delta only), just after it (full delta), or in
    # the already accumulated newer suffix.  The max recurrence below includes
    # all four possibilities; admitting a switch in an empty integer gap can
    # only enlarge the set, never exclude a real history.
    for age in sorted(groups):
        group_post, group_delta, group_lo_endpoint = groups[age]
        post_demand += int(group_post)
        max_switch_delta = max(
            0,
            int(group_lo_endpoint),
            int(group_delta),
            int(group_delta) + int(max_switch_delta),
        )
        backlog = int(post_demand) + int(max_switch_delta) - int(age)
        if backlog > best:
            best = int(backlog)
            witness = int(age)
            witness_delta = int(max_switch_delta)

    return max(0, int(best)), {
        "pre_switch_busy_upper": int(pre_busy),
        "pre_switch_backlog_upper": int(pre_backlog),
        "busy_horizon": int(horizon),
        "witness_length": int(witness),
        "witness_switch_delta": int(witness_delta),
    }


def _target_release_progression(
    target_period: int, controller_period: int, theta: int
) -> tuple[int, int]:
    ti = int(target_period)
    tc = int(controller_period)
    th = int(theta)
    g = gcd(ti, tc)
    if th < 0 or th >= tc or th % g != 0:
        raise CarryInEnvelopeUnresolved(
            f"EXACT_PERIODIC_CONTROLLER_PHASE_INCOMPATIBLE:{theta}"
        )
    modulus = tc // g
    if modulus == 1:
        n0 = 0
    else:
        a = (ti // g) % modulus
        rhs = (-th // g) % modulus
        n0 = (rhs * pow(a, -1, modulus)) % modulus
    return int(n0), int(modulus)


def _joint_q_cycle(
    target_period: int, q_step: int, specs: tuple[CarryTaskSpec, ...]
) -> int:
    cycle = 1
    for spec in specs:
        delta = (int(q_step) * int(target_period)) % int(spec.period)
        orbit = 1 if delta == 0 else int(spec.period) // gcd(int(spec.period), delta)
        cycle = lcm(int(cycle), int(orbit))
    return int(cycle)


def _independent_completion_carry(
    specs: tuple[CarryTaskSpec, ...],
    phases: tuple[int, ...],
    completion_bounds: tuple[int, ...],
) -> int:
    total = 0
    for spec, phase, completion in zip(specs, phases, completion_bounds):
        if int(completion) <= 0 or int(completion) > int(spec.period):
            raise CarryInEnvelopeUnresolved("COMPLETION_ENVELOPE_NOT_SINGLE_JOB")
        if int(phase) == 0:
            continue
        age = int(spec.period) - int(phase)
        residual_time = int(completion) - int(age)
        if residual_time > 0:
            # A pre-switch LO carry may still be primary; a HI carry may be high.
            raw = max(int(spec.pre_switch_cap), int(spec.post_switch_cap))
            total += min(raw, residual_time)
    return int(total)


def _future_post_switch_work(
    specs: tuple[CarryTaskSpec, ...], phases: tuple[int, ...], horizon: int
) -> int:
    total = 0
    for spec, phase in zip(specs, phases):
        if int(phase) >= int(horizon):
            continue
        jobs = ((int(horizon) - 1 - int(phase)) // int(spec.period)) + 1
        total += int(jobs) * int(spec.post_switch_cap)
    return int(total)


@lru_cache(maxsize=4096)
def joint_phase_pre_hi_interference(
    target_period: int,
    controller_period: int,
    theta: int,
    horizon: int,
    specs: tuple[CarryTaskSpec, ...],
    completion_bounds: tuple[int, ...],
) -> tuple[int, dict[str, int]]:
    """Joint exact-periodic PRE_HI interference for a completion-protected prefix.

    A single target-release index determines *all* higher-priority phases.  We
    enumerate its finite modular orbit, not the Cartesian product of per-task
    phases.  For each joint phase, aggregate single-switch backlog is intersected
    with the already-proved per-task completion carry envelope; taking the
    minimum is sound because both independently dominate the same real carry-in.
    """
    if len(specs) != len(completion_bounds):
        raise ValueError("COMPLETION_BOUND_DIMENSION_MISMATCH")
    if not specs:
        return 0, {"joint_phase_cycle": 1, "witness_q": 0, "carry_in": 0, "future": 0}
    if any(int(bound) <= 0 or int(bound) > int(spec.period)
           for spec, bound in zip(specs, completion_bounds)):
        raise CarryInEnvelopeUnresolved("COMPLETION_ENVELOPE_NOT_SINGLE_JOB")

    n0, q_step = _target_release_progression(
        int(target_period), int(controller_period), int(theta)
    )
    cycle = _joint_q_cycle(int(target_period), int(q_step), specs)
    maximum = -1
    witness_q = 0
    witness_carry = 0
    witness_future = 0
    witness_aggregate = 0
    witness_completion = 0

    for q in range(int(cycle)):
        n = int(n0) + int(q) * int(q_step)
        phases = tuple(
            (-int(n) * int(target_period)) % int(spec.period) for spec in specs
        )
        aggregate_carry, _ = fixed_phase_single_switch_backlog(specs, phases)
        completion_carry = _independent_completion_carry(
            specs, phases, completion_bounds
        )
        carry = min(int(aggregate_carry), int(completion_carry))
        future = _future_post_switch_work(specs, phases, int(horizon))
        value = int(carry) + int(future)
        if value > maximum:
            maximum = value
            witness_q = int(q)
            witness_carry = int(carry)
            witness_future = int(future)
            witness_aggregate = int(aggregate_carry)
            witness_completion = int(completion_carry)

    if maximum < 0:
        raise CarryInEnvelopeUnresolved("JOINT_PERIODIC_CARRY_ORBIT_EMPTY")
    return int(maximum), {
        "joint_phase_cycle": int(cycle),
        "witness_q": int(witness_q),
        "carry_in": int(witness_carry),
        "future": int(witness_future),
        "aggregate_carry_bound": int(witness_aggregate),
        "completion_carry_bound": int(witness_completion),
    }


def _candidate_ages_from_phase_sets(
    specs: tuple[CarryTaskSpec, ...],
    phase_sets: tuple[tuple[int, ...], ...],
    horizon: int,
) -> tuple[int, ...]:
    ages = {1, int(horizon)}
    for spec, phases in zip(specs, phase_sets):
        for phase in phases:
            age = int(spec.period) if int(phase) == 0 else int(spec.period) - int(phase)
            while age <= int(horizon):
                for value in (age - 1, age, age + 1):
                    if 1 <= value <= int(horizon):
                        ages.add(int(value))
                age += int(spec.period)
    return tuple(sorted(ages))


@lru_cache(maxsize=8192)
def phase_relaxed_lo_entry_carry(
    target_period: int,
    controller_period: int,
    theta: int,
    specs: tuple[CarryTaskSpec, ...],
) -> tuple[int, dict[str, int]]:
    """LO-entry aggregate backlog with per-task compatible phases relaxed apart."""
    if not specs:
        return 0, {"candidate_lengths": 0, "busy_horizon": 0}
    busy = _busy_period_upper(specs, post=False)
    phase_sets = tuple(
        compatible_release_phases(
            int(target_period), int(spec.period), int(controller_period), int(theta)
        )
        for spec in specs
    )
    ages = _candidate_ages_from_phase_sets(specs, phase_sets, busy)
    best = 0
    witness = 0
    for length in ages:
        demand = 0
        for spec, phases in zip(specs, phase_sets):
            count = max(
                _count_releases(int(phase), int(spec.period), -int(length), 0)
                for phase in phases
            )
            demand += int(count) * int(spec.pre_switch_cap)
        value = int(demand) - int(length)
        if value > best:
            best = int(value)
            witness = int(length)
    return max(0, int(best)), {
        "candidate_lengths": len(ages),
        "busy_horizon": int(busy),
        "witness_length": int(witness),
    }


@lru_cache(maxsize=131072)
def fixed_phase_lo_entry_backlog(
    specs: tuple[CarryTaskSpec, ...], phases: tuple[int, ...]
) -> tuple[int, dict[str, int]]:
    """Exact-joint-phase aggregate carry at a LO-entry target release.

    ``phases`` is the single phase vector induced by one concrete target-release
    index.  Carry is work pending strictly before relative time zero, so a
    release exactly at zero is excluded.  For any continuously-busy suffix
    ending at zero, unfinished work is bounded by arrival demand minus processor
    supply.  With fixed periodic phases, that expression can change its demand
    only when the left endpoint crosses a release; evaluating those finite
    boundary ages is therefore complete for the aggregate backlog maximum.
    """
    if len(specs) != len(phases):
        raise ValueError("CARRY_PHASE_DIMENSION_MISMATCH")
    if not specs:
        return 0, {"busy_horizon": 0, "candidate_lengths": 0, "witness_length": 0}
    for spec, phase in zip(specs, phases):
        if not (0 <= int(phase) < int(spec.period)):
            raise ValueError("CARRY_PHASE_OUT_OF_RANGE")

    busy = _busy_period_upper(specs, post=False)
    ages = {1}
    for spec, phase in zip(specs, phases):
        age = int(spec.period) if int(phase) == 0 else int(spec.period) - int(phase)
        while age <= int(busy):
            ages.add(int(age))
            age += int(spec.period)

    best = 0
    witness = 0
    for length in sorted(ages):
        demand = sum(
            _count_releases(int(phase), int(spec.period), -int(length), 0)
            * int(spec.pre_switch_cap)
            for spec, phase in zip(specs, phases)
        )
        backlog = int(demand) - int(length)
        if backlog > best:
            best = int(backlog)
            witness = int(length)
    return max(0, int(best)), {
        "busy_horizon": int(busy),
        "candidate_lengths": len(ages),
        "witness_length": int(witness),
    }


@lru_cache(maxsize=4096)
def target_release_joint_phase_orbit(
    target_period: int,
    controller_period: int,
    theta: int,
    specs: tuple[CarryTaskSpec, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Finite exact-periodic joint phase orbit for one target/controller theta.

    Each row is ``(q, phases)`` where all higher-priority phases are generated
    from the *same* target-release index.  This is the correlation that V10.13
    preserves when combining carry-in with future interference.
    """
    if not specs:
        return ((0, ()),)
    n0, q_step = _target_release_progression(
        int(target_period), int(controller_period), int(theta)
    )
    cycle = _joint_q_cycle(int(target_period), int(q_step), specs)
    rows: list[tuple[int, tuple[int, ...]]] = []
    for q in range(int(cycle)):
        n = int(n0) + int(q) * int(q_step)
        phases = tuple(
            (-int(n) * int(target_period)) % int(spec.period) for spec in specs
        )
        rows.append((int(q), phases))
    return tuple(rows)


@lru_cache(maxsize=8192)
def phase_relaxed_single_switch_carry(
    target_period: int,
    controller_period: int,
    theta: int,
    specs: tuple[CarryTaskSpec, ...],
) -> tuple[int, dict[str, int]]:
    """Single-switch aggregate backlog with cross-task phase relaxation.

    For each task the same compatible phase is used on both sides of the switch;
    different tasks may choose different compatible phases.  This only adds
    cross-task combinations, hence remains a sound upper bound when exact joint
    phase enumeration is unavailable because some task lacks a completion
    envelope.
    """
    if not specs:
        return 0, {"candidate_lengths": 0, "candidate_switch_ages": 0, "busy_horizon": 0}
    _, _, horizon = _transition_busy_upper(specs)
    phase_sets = tuple(
        compatible_release_phases(
            int(target_period), int(spec.period), int(controller_period), int(theta)
        )
        for spec in specs
    )
    ages = _candidate_ages_from_phase_sets(specs, phase_sets, horizon)
    best = 0
    witness_length = 0
    witness_switch = 0

    # Counts change only when an interval endpoint crosses a compatible release.
    # Between candidate ages demand is constant while service increases, so the
    # finite boundary set is complete for an integer-time maximum.
    for length in ages:
        for switch_age in ages:
            if switch_age > int(horizon):
                continue
            demand = 0
            for spec, phases in zip(specs, phase_sets):
                task_best = 0
                for phase in phases:
                    if int(length) <= int(switch_age):
                        post = _count_releases(
                            int(phase), int(spec.period), -int(length), 0
                        )
                        value = int(post) * int(spec.post_switch_cap)
                        if (
                            int(length) == int(switch_age)
                            and spec.criticality == "LO"
                            and ((-int(switch_age) - int(phase)) % int(spec.period) == 0)
                        ):
                            value += int(spec.pre_switch_cap) - int(spec.post_switch_cap)
                    else:
                        pre = _count_releases(
                            int(phase), int(spec.period), -int(length), -int(switch_age)
                        )
                        post = _count_releases(
                            int(phase), int(spec.period), -int(switch_age), 0
                        )
                        value = (
                            int(pre) * int(spec.pre_switch_cap)
                            + int(post) * int(spec.post_switch_cap)
                        )
                        if (
                            spec.criticality == "LO"
                            and ((-int(switch_age) - int(phase)) % int(spec.period) == 0)
                        ):
                            value += int(spec.pre_switch_cap) - int(spec.post_switch_cap)
                    task_best = max(task_best, int(value))
                demand += int(task_best)
            value = int(demand) - int(length)
            if value > best:
                best = int(value)
                witness_length = int(length)
                witness_switch = int(switch_age)

    return max(0, int(best)), {
        "candidate_lengths": len(ages),
        "candidate_switch_ages": len(ages),
        "busy_horizon": int(horizon),
        "witness_length": int(witness_length),
        "witness_switch_age": int(witness_switch),
    }


__all__ = [
    "CarryInEnvelopeUnresolved",
    "CarryTaskSpec",
    "fixed_phase_single_switch_backlog",
    "fixed_phase_lo_entry_backlog",
    "target_release_joint_phase_orbit",
    "joint_phase_pre_hi_interference",
    "phase_relaxed_lo_entry_carry",
    "phase_relaxed_single_switch_carry",
]
