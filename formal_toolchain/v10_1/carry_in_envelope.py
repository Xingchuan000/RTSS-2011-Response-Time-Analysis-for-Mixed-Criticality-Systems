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


@dataclass(frozen=True, slots=True)
class PhaseBlockProjection:
    """Symbolic task-local phase projection for one congruence block.

    The concrete phase set is exactly ``{r + k*g | 0 <= k < count}``, where
    ``r=phase_residue`` and ``g=phase_stride``.  Keeping this representation
    symbolic avoids materializing a task-local phase tuple and, critically,
    never allocates or iterates the global joint period.
    """

    task: str
    period: int
    q_period: int
    block_modulus: int
    block_residue: int
    phase_residue: int
    phase_stride: int
    phase_count: int

    @property
    def first_release_age(self) -> int:
        return (
            int(self.phase_stride) - int(self.phase_residue)
            if int(self.phase_residue)
            else int(self.phase_stride)
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "task": self.task,
            "period": int(self.period),
            "q_period": int(self.q_period),
            "block_modulus": int(self.block_modulus),
            "block_residue": int(self.block_residue),
            "phase_residue": int(self.phase_residue),
            "phase_stride": int(self.phase_stride),
            "phase_count": int(self.phase_count),
        }


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


@lru_cache(maxsize=4096)
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
    # V10.16 normative convention: lcm(empty)=1, hence the highest-priority
    # target has the singleton root phase domain Z_1={0}.
    if not specs:
        return 1
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


def target_release_joint_phase_parameters(
    target_period: int,
    controller_period: int,
    theta: int,
    specs: tuple[CarryTaskSpec, ...],
) -> tuple[int, int, int]:
    """Return ``(n0, q_step, cycle)`` for the exact joint HP phase orbit.

    The parameters are intentionally exposed as pure integer data so later
    certificate layers can stream the orbit without materializing every phase
    vector in memory.
    """
    n0, q_step = _target_release_progression(
        int(target_period), int(controller_period), int(theta)
    )
    cycle = _joint_q_cycle(int(target_period), int(q_step), specs)
    return int(n0), int(q_step), int(cycle)


def target_release_joint_phases_at_q(
    target_period: int,
    specs: tuple[CarryTaskSpec, ...],
    *,
    n0: int,
    q_step: int,
    q: int,
) -> tuple[int, ...]:
    n = int(n0) + int(q) * int(q_step)
    return tuple(
        (-int(n) * int(target_period)) % int(spec.period) for spec in specs
    )


@dataclass(frozen=True, slots=True)
class _ExactJointLoEntryCarryPlan:
    q_periods: tuple[int, ...]
    components: tuple[tuple[int, tuple[int, ...]], ...]
    joint_cycle: int
    ordered_lengths: tuple[int, ...]
    phase_tables: tuple[tuple[int, ...], ...]
    component_carry_rows: tuple[tuple[tuple[int, ...], ...], ...]


@lru_cache(maxsize=128)
def _exact_joint_lo_entry_carry_plan(
    target_period: int,
    q_step: int,
    n0: int,
    specs: tuple[CarryTaskSpec, ...],
) -> _ExactJointLoEntryCarryPlan:
    """Precompute the horizon/switch-independent half of V10.13 exactly."""
    ti = int(target_period)
    step = int(q_step)
    base = int(n0)
    q_periods = tuple(
        int(spec.period) // gcd(int(spec.period), step * ti) for spec in specs
    )

    remaining = set(range(len(specs)))
    components: list[tuple[int, tuple[int, ...]]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.remove(seed)
        members: list[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            linked = [
                other for other in sorted(remaining)
                if gcd(int(q_periods[current]), int(q_periods[other])) > 1
            ]
            for other in linked:
                remaining.remove(other)
                stack.append(other)
        member_tuple = tuple(sorted(members))
        component_period = 1
        for index in member_tuple:
            component_period = lcm(int(component_period), int(q_periods[index]))
        components.append((int(component_period), member_tuple))
    components.sort(key=lambda row: row[1][0])
    component_tuple = tuple(components)

    joint_cycle = 1
    for period in q_periods:
        joint_cycle = lcm(int(joint_cycle), int(period))

    busy = _busy_period_upper(specs, post=False)
    projections = phase_block_task_projections(
        ti, step, base, specs, block_modulus=1, block_residue=0,
    )
    candidate_lengths = {0, 1}
    for projection in projections:
        age = int(projection.first_release_age)
        stride = int(projection.phase_stride)
        while age <= int(busy):
            candidate_lengths.add(int(age))
            age += int(stride)
    ordered_lengths = tuple(sorted(candidate_lengths))

    phase_tables = tuple(
        tuple(
            (-(base + int(r) * step) * ti) % int(spec.period)
            for r in range(int(q_period))
        )
        for spec, q_period in zip(specs, q_periods)
    )

    component_carry_rows: list[tuple[tuple[int, ...], ...]] = []
    for length in ordered_lengths:
        length_rows: list[tuple[int, ...]] = []
        for component_period, member_indices in component_tuple:
            values: list[int] = []
            for residue in range(int(component_period)):
                subtotal = 0
                for index in member_indices:
                    local = int(residue) % int(q_periods[index])
                    phase = int(phase_tables[index][local])
                    subtotal += _count_releases(
                        phase, int(specs[index].period), -int(length), 0
                    ) * int(specs[index].pre_switch_cap)
                values.append(int(subtotal))
            length_rows.append(tuple(values))
        component_carry_rows.append(tuple(length_rows))

    return _ExactJointLoEntryCarryPlan(
        q_periods=tuple(int(value) for value in q_periods),
        components=component_tuple,
        joint_cycle=int(joint_cycle),
        ordered_lengths=ordered_lengths,
        phase_tables=phase_tables,
        component_carry_rows=tuple(component_carry_rows),
    )


def exact_joint_lo_entry_max_with_periodic_future(
    target_period: int,
    q_step: int,
    n0: int,
    specs: tuple[CarryTaskSpec, ...],
    future_by_task_q_residue: tuple[tuple[int, ...], ...],
) -> tuple[int, dict[str, int | tuple[int, ...] | tuple[tuple[str, ...], ...]]]:
    """Evaluate the unchanged V10.13 exact joint maximum without global-Q enumeration.

    Carry-phase structure is independent of the response horizon and switch
    cell, so it is precomputed once per target/controller phase and reused.
    Future interference remains an exact input of every invocation.
    """
    if len(specs) != len(future_by_task_q_residue):
        raise ValueError("JOINT_PERIODIC_FUTURE_DIMENSION_MISMATCH")
    if not specs:
        return 0, {
            "joint_phase_cycle": 1,
            "candidate_lengths": 1,
            "component_periods": (),
            "component_tasks": (),
            "residues_per_candidate": 0,
            "witness_q": 0,
            "witness_length": 0,
        }

    plan = _exact_joint_lo_entry_carry_plan(
        int(target_period), int(q_step), int(n0), specs
    )
    for period, table in zip(plan.q_periods, future_by_task_q_residue):
        if len(table) != int(period):
            raise ValueError("JOINT_PERIODIC_FUTURE_PERIOD_MISMATCH")

    component_future_rows: list[tuple[int, ...]] = []
    for component_period, member_indices in plan.components:
        values: list[int] = []
        for residue in range(int(component_period)):
            subtotal = 0
            for index in member_indices:
                subtotal += int(
                    future_by_task_q_residue[index][
                        int(residue) % int(plan.q_periods[index])
                    ]
                )
            values.append(int(subtotal))
        component_future_rows.append(tuple(values))

    best_value = -1
    best_length = 0
    best_component_residues: tuple[int, ...] = tuple(0 for _ in plan.components)
    for length_index, length in enumerate(plan.ordered_lengths):
        value = -int(length)
        component_residues: list[int] = []
        for component_index, (component_period, _) in enumerate(plan.components):
            carry_row = plan.component_carry_rows[length_index][component_index]
            future_row = component_future_rows[component_index]
            component_best = -1
            component_argmax = 0
            for residue in range(int(component_period)):
                subtotal = int(carry_row[residue]) + int(future_row[residue])
                if subtotal > component_best:
                    component_best = int(subtotal)
                    component_argmax = int(residue)
            value += int(component_best)
            component_residues.append(int(component_argmax))
        if value > best_value:
            best_value = int(value)
            best_length = int(length)
            best_component_residues = tuple(component_residues)

    witness_q = 0
    witness_modulus = 1
    for (component_period, _), residue in zip(plan.components, best_component_residues):
        modulus = int(component_period)
        if modulus == 1:
            continue
        delta = (int(residue) - int(witness_q)) % modulus
        inverse = pow(int(witness_modulus) % modulus, -1, modulus)
        step_count = (delta * inverse) % modulus
        witness_q += int(witness_modulus) * int(step_count)
        witness_modulus *= modulus
        witness_q %= int(witness_modulus)

    return int(best_value), {
        "joint_phase_cycle": int(plan.joint_cycle),
        "candidate_lengths": len(plan.ordered_lengths),
        "component_periods": tuple(int(period) for period, _ in plan.components),
        "component_tasks": tuple(
            tuple(specs[index].name for index in member_indices)
            for _, member_indices in plan.components
        ),
        "residues_per_candidate": sum(int(period) for period, _ in plan.components),
        "witness_q": int(witness_q),
        "witness_length": int(best_length),
    }


def fixed_phase_pre_hi_carry(
    specs: tuple[CarryTaskSpec, ...],
    phases: tuple[int, ...],
    completion_bounds: tuple[int, ...] | None = None,
) -> tuple[int, dict[str, int]]:
    """PRE_HI carry bound for one fixed exact joint phase vector.

    This horizon-independent helper is retained as a fixed-phase audit oracle
    for validating V10.16 block dominance; it is not a default PRE_HI terminal.
    """
    if len(specs) != len(phases):
        raise ValueError("CARRY_PHASE_DIMENSION_MISMATCH")
    if completion_bounds is not None:
        if len(specs) != len(completion_bounds):
            raise ValueError("COMPLETION_BOUND_DIMENSION_MISMATCH")
        if any(int(bound) <= 0 or int(bound) > int(spec.period)
               for spec, bound in zip(specs, completion_bounds)):
            raise CarryInEnvelopeUnresolved("COMPLETION_ENVELOPE_NOT_SINGLE_JOB")

    aggregate_carry, carry_details = fixed_phase_single_switch_backlog(specs, phases)
    completion_carry: int | None = None
    carry = int(aggregate_carry)
    if completion_bounds is not None:
        completion_carry = _independent_completion_carry(
            specs, phases, completion_bounds
        )
        carry = min(int(carry), int(completion_carry))
    return int(carry), {
        "carry_in": int(carry),
        "aggregate_carry_bound": int(aggregate_carry),
        "completion_carry_bound": (
            -1 if completion_carry is None else int(completion_carry)
        ),
        "carry_witness_length": int(carry_details.get("witness_length", 0)),
    }


def fixed_phase_post_switch_future_work(
    specs: tuple[CarryTaskSpec, ...],
    phases: tuple[int, ...],
    horizon: int,
) -> int:
    """Exact-periodic post-switch future work for one fixed phase vector."""
    if len(specs) != len(phases):
        raise ValueError("CARRY_PHASE_DIMENSION_MISMATCH")
    return int(_future_post_switch_work(specs, phases, int(horizon)))


def fixed_phase_pre_hi_interference(
    specs: tuple[CarryTaskSpec, ...],
    phases: tuple[int, ...],
    horizon: int,
    completion_bounds: tuple[int, ...] | None = None,
) -> tuple[int, dict[str, int]]:
    """PRE_HI HP interference for one fixed exact joint phase vector.

    The single-switch aggregate backlog is sound without completion envelopes.
    When a complete single-job completion prefix is available, intersecting it
    with the aggregate carry bound is an independent sound tightening.
    """
    carry, details = fixed_phase_pre_hi_carry(
        specs, phases, completion_bounds
    )
    future = fixed_phase_post_switch_future_work(specs, phases, int(horizon))
    return int(carry) + int(future), {**details, "future": int(future)}


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


def phase_block_task_projection(
    target_period: int,
    q_step: int,
    n0: int,
    spec: CarryTaskSpec,
    *,
    block_modulus: int,
    block_residue: int,
) -> PhaseBlockProjection:
    """Return the complete task-local V10.16 projection without q enumeration."""

    ti = int(target_period)
    step = int(q_step)
    period = int(spec.period)
    modulus = int(block_modulus)
    residue = int(block_residue)
    if modulus <= 0 or residue < 0 or residue >= modulus:
        raise CarryInEnvelopeUnresolved("PHASE_BLOCK_INVALID_CONGRUENCE")

    q_period = period // gcd(period, step * ti)
    phase_stride = gcd(period, modulus * step * ti)
    phase_count = period // phase_stride
    expected_count = q_period // gcd(q_period, modulus)
    if phase_count != expected_count:
        raise CarryInEnvelopeUnresolved("PHASE_BLOCK_PROJECTION_ARITHMETIC_MISMATCH")

    n = int(n0) + residue * step
    base_phase = (-n * ti) % period
    phase_residue = int(base_phase) % int(phase_stride)
    return PhaseBlockProjection(
        task=spec.name,
        period=period,
        q_period=int(q_period),
        block_modulus=modulus,
        block_residue=residue,
        phase_residue=int(phase_residue),
        phase_stride=int(phase_stride),
        phase_count=int(phase_count),
    )


def phase_block_task_projections(
    target_period: int,
    q_step: int,
    n0: int,
    specs: tuple[CarryTaskSpec, ...],
    *,
    block_modulus: int,
    block_residue: int,
) -> tuple[PhaseBlockProjection, ...]:
    return tuple(
        phase_block_task_projection(
            int(target_period), int(q_step), int(n0), spec,
            block_modulus=int(block_modulus), block_residue=int(block_residue),
        )
        for spec in specs
    )


def phase_block_r7_carry_upper(
    specs: tuple[CarryTaskSpec, ...],
    projections: tuple[PhaseBlockProjection, ...],
) -> tuple[int, dict[str, int | str]]:
    """Count-monotone R7 single-switch carry lifting for one phase block.

    For every task projection, all possible pre-zero release ages form one
    arithmetic congruence class.  At each possible boundary we lift the fixed-
    phase release-group coordinates independently.  This is exactly the allowed
    V10.16 relaxation: it may combine coordinates produced by different q values
    inside the block, but it cannot under-approximate any concrete fixed-q R7
    group.  The fixed-phase linear-time switch-prefix recurrence is then reused.
    """

    if len(specs) != len(projections):
        raise ValueError("PHASE_BLOCK_PROJECTION_DIMENSION_MISMATCH")
    if not specs:
        return 0, {
            "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
            "candidate_boundary_count": 0,
            "busy_horizon": 0,
            "witness_length": 0,
        }

    _, _, horizon = _transition_busy_upper(specs)
    groups: dict[int, list[int]] = {}
    for spec, projection in zip(specs, projections):
        if spec.name != projection.task or int(spec.period) != int(projection.period):
            raise ValueError("PHASE_BLOCK_PROJECTION_TASK_MISMATCH")
        delta = int(spec.pre_switch_cap) - int(spec.post_switch_cap)
        # If the task projection is not singleton, absence of a release at this
        # boundary is also possible for some q in the block; max(0,delta) is the
        # sound coordinate-wise upper.  A singleton projection retains a negative
        # HI delta and therefore recovers fixed-phase precision under refinement.
        lifted_delta = int(delta) if int(projection.phase_count) == 1 else max(0, int(delta))
        lifted_lo_endpoint = (
            int(lifted_delta) if spec.criticality == "LO" else 0
        )
        first = int(projection.first_release_age)
        stride = int(projection.phase_stride)
        for age in range(first, int(horizon) + 1, stride):
            row = groups.setdefault(int(age), [0, 0, 0])
            row[0] += int(spec.post_switch_cap)
            row[1] += int(lifted_delta)
            row[2] += int(lifted_lo_endpoint)

    post_demand = 0
    max_switch_delta = 0
    best = 0
    witness = 0
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

    return max(0, int(best)), {
        "candidate_domain_kind": "PROVED_BOUNDARY_UNION",
        "candidate_boundary_count": len(groups),
        "busy_horizon": int(horizon),
        "witness_length": int(witness),
    }


def phase_block_completion_carry_upper(
    specs: tuple[CarryTaskSpec, ...],
    projections: tuple[PhaseBlockProjection, ...],
    completion_bounds: tuple[int, ...],
) -> int:
    """Independent block-level completion carry upper bound.

    Each task takes the worst predecessor age over its *complete* task-local
    projection.  The sum is an independent sound upper bound and is refinement
    monotone because child projections are subsets of parent projections.
    """

    if len(specs) != len(projections) or len(specs) != len(completion_bounds):
        raise ValueError("PHASE_BLOCK_COMPLETION_DIMENSION_MISMATCH")
    total = 0
    for spec, projection, completion in zip(specs, projections, completion_bounds):
        if int(completion) <= 0 or int(completion) > int(spec.period):
            raise CarryInEnvelopeUnresolved("COMPLETION_ENVELOPE_NOT_SINGLE_JOB")
        # Singleton phase zero has no predecessor carry at relative time zero.
        if (
            int(projection.phase_count) == 1
            and int(projection.phase_residue) == 0
        ):
            continue
        age = int(projection.first_release_age)
        residual_time = int(completion) - int(age)
        if residual_time > 0:
            raw = max(int(spec.pre_switch_cap), int(spec.post_switch_cap))
            total += min(int(raw), int(residual_time))
    return int(total)


def phase_block_post_switch_future_upper(
    specs: tuple[CarryTaskSpec, ...],
    projections: tuple[PhaseBlockProjection, ...],
    horizon: int,
) -> int:
    """Exact max-count lifting of PRE_HI post-switch future work."""

    if len(specs) != len(projections):
        raise ValueError("PHASE_BLOCK_PROJECTION_DIMENSION_MISMATCH")
    total = 0
    upper = int(horizon)
    for spec, projection in zip(specs, projections):
        # The symbolic projection is {r+k*g}; r is its smallest phase, so it
        # maximizes the release count in the half-open interval [0,R).
        phase = int(projection.phase_residue)
        if phase >= upper:
            continue
        jobs = ((upper - 1 - phase) // int(spec.period)) + 1
        total += int(jobs) * int(spec.post_switch_cap)
    return int(total)


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
    "PhaseBlockProjection",
    "phase_block_task_projection",
    "phase_block_task_projections",
    "phase_block_r7_carry_upper",
    "phase_block_completion_carry_upper",
    "phase_block_post_switch_future_upper",
    "fixed_phase_single_switch_backlog",
    "fixed_phase_lo_entry_backlog",
    "fixed_phase_pre_hi_carry",
    "fixed_phase_post_switch_future_work",
    "fixed_phase_pre_hi_interference",
    "target_release_joint_phase_parameters",
    "target_release_joint_phases_at_q",
    "exact_joint_lo_entry_max_with_periodic_future",
    "phase_relaxed_lo_entry_carry",
    "phase_relaxed_single_switch_carry",
]
