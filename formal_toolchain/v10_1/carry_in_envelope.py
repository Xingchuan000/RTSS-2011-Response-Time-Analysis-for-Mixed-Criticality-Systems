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


@dataclass(frozen=True, slots=True)
class CRTPhaseFamilyPlan:
    """Exact V10.17 arithmetic plan for one congruence phase family."""

    joint_period: int
    block_modulus: int
    block_residue: int
    family_size: int
    restricted_periods: tuple[int, ...]
    components: tuple[tuple[int, tuple[int, ...]], ...]
    phase_tables: tuple[tuple[int, ...], ...]
    candidate_ages: tuple[int, ...]
    busy_horizon: int
    production_busy_horizon: int
    candidate_count: int
    streaming_residue_work: int

    @property
    def component_periods(self) -> tuple[int, ...]:
        return tuple(int(period) for period, _ in self.components)

    @property
    def residues_per_candidate(self) -> int:
        return sum(int(period) for period, _ in self.components)


class CarryInEnvelopeUnresolved(RuntimeError):
    pass


def prehistory_finite_bound(
    specs: tuple[CarryTaskSpec, ...],
) -> tuple[int, dict[str, object]]:
    """V10.17 finite-prehistory sufficient theorem instantiated with max job caps."""

    if not specs:
        return 0, {
            "utilization_numerator": 0,
            "utilization_denominator": 1,
            "job_cap_sum": 0,
            "job_caps": (),
        }
    caps = tuple(max(int(spec.pre_switch_cap), int(spec.post_switch_cap)) for spec in specs)
    utilization = sum(
        (Fraction(int(cap), int(spec.period)) for spec, cap in zip(specs, caps)),
        Fraction(0, 1),
    )
    if utilization >= 1:
        raise CarryInEnvelopeUnresolved("PREHISTORY_BOUND_UNPROVED")
    cap_sum = sum(int(value) for value in caps)
    ratio = Fraction(int(cap_sum), 1) / (Fraction(1, 1) - utilization)
    bound = (int(ratio.numerator) + int(ratio.denominator) - 1) // int(ratio.denominator)
    return int(bound), {
        "utilization_numerator": int(utilization.numerator),
        "utilization_denominator": int(utilization.denominator),
        "job_cap_sum": int(cap_sum),
        "job_caps": tuple(
            (spec.name, int(cap)) for spec, cap in zip(specs, caps)
        ),
    }


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


def _connected_period_components(
    periods: tuple[int, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Connected components of the V10.17 gcd dependency graph."""

    remaining = {index for index, period in enumerate(periods) if int(period) > 1}
    rows: list[tuple[int, tuple[int, ...]]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        members: list[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            linked = [
                other
                for other in sorted(remaining)
                if gcd(int(periods[current]), int(periods[other])) > 1
            ]
            for other in linked:
                remaining.remove(other)
                stack.append(other)
        member_tuple = tuple(sorted(members))
        component_period = 1
        for index in member_tuple:
            component_period = lcm(int(component_period), int(periods[index]))
        rows.append((int(component_period), member_tuple))
    rows.sort(key=lambda row: row[1][0])
    return tuple(rows)


def _crt_reconstruct(
    component_periods: tuple[int, ...], component_residues: tuple[int, ...]
) -> int:
    value = 0
    modulus_so_far = 1
    for modulus, residue in zip(component_periods, component_residues):
        m = int(modulus)
        if m == 1:
            continue
        delta = (int(residue) - int(value)) % m
        inverse = pow(int(modulus_so_far) % m, -1, m)
        step_count = (delta * inverse) % m
        value += int(modulus_so_far) * int(step_count)
        modulus_so_far *= m
        value %= int(modulus_so_far)
    return int(value)


@lru_cache(maxsize=4096)
def crt_phase_family_plan(
    target_period: int,
    q_step: int,
    n0: int,
    specs: tuple[CarryTaskSpec, ...],
    joint_period: int,
    block_modulus: int,
    block_residue: int,
) -> CRTPhaseFamilyPlan:
    """Build the exact restricted-period/component plan required by V10.17.

    The candidate domain is the complete family release-boundary union.  For a
    concrete q, inserting extra family boundaries only adds switch cells inside
    intervals in which that q has no release, so the fixed-q R7 maximum is
    unchanged while the candidate set is common to every member of the family.
    """

    Q = int(joint_period)
    M = int(block_modulus)
    a = int(block_residue)
    if Q <= 0 or M <= 0 or Q % M != 0 or a < 0 or a >= M:
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_DOMAIN_INVALID")
    K = Q // M
    projections = phase_block_task_projections(
        int(target_period), int(q_step), int(n0), specs,
        block_modulus=M, block_residue=a,
    )
    restricted_periods = tuple(int(row.phase_count) for row in projections)
    restricted_lcm = 1
    for period in restricted_periods:
        restricted_lcm = lcm(int(restricted_lcm), int(period))
    if int(restricted_lcm) != int(K):
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_RESTRICTED_PERIOD_IDENTITY_FAILED")

    components = _connected_period_components(restricted_periods)
    product = 1
    for index, (period, _) in enumerate(components):
        for other_period, _ in components[index + 1:]:
            if gcd(int(period), int(other_period)) != 1:
                raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_COMPONENT_NOT_COPRIME")
        product *= int(period)
    if int(product) != int(K):
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_COMPONENT_PRODUCT_MISMATCH")

    phase_tables: list[tuple[int, ...]] = []
    for spec, period in zip(specs, restricted_periods):
        values: list[int] = []
        for residue in range(int(period)):
            q = int(a) + int(M) * int(residue)
            n = int(n0) + int(q) * int(q_step)
            values.append((-int(n) * int(target_period)) % int(spec.period))
        phase_tables.append(tuple(values))

    _, _, production_busy_horizon = _transition_busy_upper(specs)
    prehistory_bound, _ = prehistory_finite_bound(specs)
    busy_horizon = min(int(production_busy_horizon), int(prehistory_bound))
    candidate_ages: set[int] = set()
    for projection in projections:
        age = int(projection.first_release_age)
        stride = int(projection.phase_stride)
        while age <= int(busy_horizon):
            candidate_ages.add(int(age))
            age += int(stride)
    ordered_ages = tuple(sorted(candidate_ages))
    # one zero-carry candidate plus, for every L, all-pre, exact/open switch
    # cells at every family boundary <=L, and the all-post cell beyond L.
    candidate_count = 1 + sum(2 * (index + 1) + 2 for index in range(len(ordered_ages)))
    residues_per_candidate = sum(int(period) for period, _ in components)
    streaming_residue_work = int(candidate_count) * max(1, int(residues_per_candidate))
    return CRTPhaseFamilyPlan(
        joint_period=Q,
        block_modulus=M,
        block_residue=a,
        family_size=int(K),
        restricted_periods=restricted_periods,
        components=components,
        phase_tables=tuple(phase_tables),
        candidate_ages=ordered_ages,
        busy_horizon=int(busy_horizon),
        production_busy_horizon=int(production_busy_horizon),
        candidate_count=int(candidate_count),
        streaming_residue_work=int(streaming_residue_work),
    )


def _task_r7_candidate_demand(
    spec: CarryTaskSpec,
    phase: int,
    busy_length: int,
    switch_cell_twice_age: int,
) -> int:
    """Exact task demand for one common R7 candidate.

    ``switch_cell_twice_age`` is twice the backward switch age: even values are
    exact integer endpoints, odd values are open-gap representatives.
    """

    L = int(busy_length)
    s2 = int(switch_cell_twice_age)
    total = _count_releases(int(phase), int(spec.period), -L, 0)
    if total <= 0:
        return 0
    if s2 == 0:
        return int(total) * int(spec.pre_switch_cap)
    if s2 > 2 * L:
        return int(total) * int(spec.post_switch_cap)

    if s2 % 2:
        # Open gap at age u-1/2: ages >=u are pre-switch, ages <=u-1 post.
        u = (s2 + 1) // 2
        newer = 0 if u <= 1 else _count_releases(
            int(phase), int(spec.period), -(int(u) - 1), 0
        )
        older_or_boundary = int(total) - int(newer)
        return (
            int(older_or_boundary) * int(spec.pre_switch_cap)
            + int(newer) * int(spec.post_switch_cap)
        )

    # Exact switch endpoint at age u.  LO at the switch retains primary/pre;
    # HI at the switch uses post/high, exactly matching the production R7.
    u = s2 // 2
    newer = 0 if u <= 1 else _count_releases(
        int(phase), int(spec.period), -(int(u) - 1), 0
    )
    endpoint = int(((-int(u) - int(phase)) % int(spec.period)) == 0)
    older = int(total) - int(newer) - int(endpoint)
    endpoint_cap = (
        int(spec.pre_switch_cap)
        if spec.criticality == "LO"
        else int(spec.post_switch_cap)
    )
    return (
        int(older) * int(spec.pre_switch_cap)
        + int(newer) * int(spec.post_switch_cap)
        + int(endpoint) * int(endpoint_cap)
    )


def _iter_crt_r7_candidates(plan: CRTPhaseFamilyPlan):
    # ZERO expands the positive-part branch B_R7=max(0,raw backlog).
    yield ("ZERO", 0, 0)
    ages = plan.candidate_ages
    for index, length in enumerate(ages):
        L = int(length)
        yield ("SWITCH", L, 0)  # switch at zero: all carry releases pre-switch
        for boundary in ages[: index + 1]:
            u = int(boundary)
            yield ("SWITCH", L, 2 * u - 1)  # open gap immediately newer than u
            yield ("SWITCH", L, 2 * u)      # exact endpoint at u
        yield ("SWITCH", L, 2 * L + 1)      # switch before oldest release: all post


def _fixed_phase_r7_on_common_candidate_domain(
    specs: tuple[CarryTaskSpec, ...],
    phases: tuple[int, ...],
    plan: CRTPhaseFamilyPlan,
) -> int:
    best = 0
    for kind, length, switch2 in _iter_crt_r7_candidates(plan):
        if kind == "ZERO":
            value = 0
        else:
            value = -int(length) + sum(
                _task_r7_candidate_demand(spec, phase, int(length), int(switch2))
                for spec, phase in zip(specs, phases)
            )
        if int(value) > int(best):
            best = int(value)
    return max(0, int(best))


def exact_crt_phase_family_pre_hi_max(
    target_period: int,
    q_step: int,
    n0: int,
    specs: tuple[CarryTaskSpec, ...],
    joint_period: int,
    block_modulus: int,
    block_residue: int,
    horizon: int,
) -> tuple[int, dict[str, object]]:
    """Exact V10.17 max_q [fixed-q R7(q) + PRE_HI future(R,q)].

    The V10.17 ACNF candidate grammar is evaluated without materializing a
    candidate-by-residue matrix.  For each busy length, the all-pre component
    vectors are maintained incrementally.  Moving the common switch through
    the family release-boundary union updates only the component residues that
    actually contain a release at that boundary.  Each candidate therefore
    needs only a scan of the exact CRT component residue vectors.

    This is algebraically identical to enumerating every canonical R7
    candidate and every component residue, but uses O(sum L_r) working memory
    instead of O(|C| * sum L_r), and it performs no global-q enumeration.
    """

    plan = crt_phase_family_plan(
        int(target_period), int(q_step), int(n0), specs,
        int(joint_period), int(block_modulus), int(block_residue),
    )
    if not specs:
        return 0, {
            "family_size": 1,
            "restricted_periods": (),
            "component_periods": (),
            "component_tasks": (),
            "candidate_count": 1,
            "residues_per_candidate": 0,
            "winning_candidate": {"kind": "ZERO", "busy_length": 0, "switch_cell_twice_age": 0},
            "component_maxima": (),
            "component_argmax": (),
            "witness_k": 0,
            "witness_q": 0,
            "witness_phases": (),
            "witness_reevaluation": 0,
            "witness_r7": 0,
            "witness_common_domain_r7": 0,
            "witness_future": 0,
            "busy_horizon": 0,
            "production_busy_horizon": 0,
            "streaming_residue_work": 0,
        }

    # Exact task-local future tables.  P_j=1 tasks are constants; the others
    # are accumulated into their gcd-connected CRT component vectors.
    future_tables: list[tuple[int, ...]] = []
    for spec, phase_table in zip(specs, plan.phase_tables):
        values: list[int] = []
        for phase in phase_table:
            p = int(phase)
            if p >= int(horizon):
                values.append(0)
            else:
                jobs = ((int(horizon) - 1 - p) // int(spec.period)) + 1
                values.append(int(jobs) * int(spec.post_switch_cap))
        future_tables.append(tuple(values))

    component_index_by_task = [-1] * len(specs)
    component_future: list[list[int]] = []
    for component_index, (component_period, member_indices) in enumerate(plan.components):
        period = int(component_period)
        row = [0] * period
        for task_index in member_indices:
            component_index_by_task[int(task_index)] = int(component_index)
            local_period = int(plan.restricted_periods[task_index])
            task_future = future_tables[task_index]
            for residue in range(period):
                row[residue] += int(task_future[residue % local_period])
        component_future.append(row)

    constant_indices = tuple(
        index for index, period in enumerate(plan.restricted_periods) if int(period) == 1
    )
    constant_future = sum(int(future_tables[index][0]) for index in constant_indices)

    # For every family boundary, record the unique task-local residue that has
    # a release there.  Phase tables enumerate one exact orbit and are injective
    # over their restricted period, so a task contributes at most one local
    # residue to a given boundary.
    phase_to_local = tuple(
        {int(phase): int(local) for local, phase in enumerate(phase_table)}
        for phase_table in plan.phase_tables
    )
    boundary_updates: dict[int, tuple[tuple[int, int], ...]] = {}
    for age in plan.candidate_ages:
        updates: list[tuple[int, int]] = []
        u = int(age)
        for task_index, spec in enumerate(specs):
            local = phase_to_local[task_index].get((-u) % int(spec.period))
            if local is not None:
                updates.append((int(task_index), int(local)))
        boundary_updates[u] = tuple(updates)

    # Working component carry vectors for the all-pre state of the current busy
    # length.  They are enlarged monotonically as L crosses each family release
    # boundary.  A switch scan copies only O(sum L_r) integers for that L.
    base_component_carry = [
        [0] * int(component_period) for component_period, _ in plan.components
    ]
    base_constant_carry = 0

    def apply_release_delta(
        vectors: list[list[int]],
        constant_value: int,
        task_index: int,
        local_residue: int,
        delta: int,
    ) -> int:
        if int(delta) == 0:
            return int(constant_value)
        local_period = int(plan.restricted_periods[task_index])
        if local_period == 1:
            return int(constant_value) + int(delta)
        component_index = int(component_index_by_task[task_index])
        component_period = int(plan.components[component_index][0])
        vector = vectors[component_index]
        for residue in range(int(local_residue), component_period, local_period):
            vector[residue] += int(delta)
        return int(constant_value)

    def score_candidate(
        vectors: list[list[int]],
        constant_carry: int,
        beta: int,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        total = int(beta) + int(constant_carry) + int(constant_future)
        maxima: list[int] = []
        argmax: list[int] = []
        for carry_vector, future_vector in zip(vectors, component_future):
            local_best: int | None = None
            local_argmax = 0
            for residue in range(len(carry_vector)):
                value = int(carry_vector[residue]) + int(future_vector[residue])
                if local_best is None or int(value) > int(local_best):
                    local_best = int(value)
                    local_argmax = int(residue)
            if local_best is None:
                raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_COMPONENT_EMPTY")
            total += int(local_best)
            maxima.append(int(local_best))
            argmax.append(int(local_argmax))
        return int(total), tuple(maxima), tuple(argmax)

    # Positive-part zero branch: carry is identically zero while future remains
    # exact and task-additive over the same q.
    zero_vectors = [[0] * len(row) for row in component_future]
    best_value, best_component_maxima, best_component_argmax = score_candidate(
        zero_vectors, 0, 0
    )
    best_candidate = ("ZERO", 0, 0)

    ages = plan.candidate_ages
    for length_index, length_value in enumerate(ages):
        length = int(length_value)
        # Crossing L admits exactly the releases at this new family boundary.
        for task_index, local_residue in boundary_updates[length]:
            base_constant_carry = apply_release_delta(
                base_component_carry,
                int(base_constant_carry),
                int(task_index),
                int(local_residue),
                int(specs[task_index].pre_switch_cap),
            )

        state_vectors = [list(row) for row in base_component_carry]
        state_constant = int(base_constant_carry)

        # switch at zero: every carry release in [-L,0) is pre-switch.
        value, maxima, argmax = score_candidate(state_vectors, state_constant, -length)
        if int(value) > int(best_value):
            best_value = int(value)
            best_component_maxima = maxima
            best_component_argmax = argmax
            best_candidate = ("SWITCH", int(length), 0)

        # Move the one common switch from zero towards the oldest release.
        # Before processing boundary u the state is the open cell immediately
        # newer than u.  At exact u, HI endpoints use post/high while LO
        # endpoints remain pre/primary.  Moving past u then changes LO endpoints
        # to post, yielding the next open-cell state.
        for boundary_index in range(length_index + 1):
            boundary = int(ages[boundary_index])

            value, maxima, argmax = score_candidate(
                state_vectors, state_constant, -length
            )
            if int(value) > int(best_value):
                best_value = int(value)
                best_component_maxima = maxima
                best_component_argmax = argmax
                best_candidate = ("SWITCH", int(length), 2 * int(boundary) - 1)

            for task_index, local_residue in boundary_updates[boundary]:
                spec = specs[task_index]
                if spec.criticality == "HI":
                    state_constant = apply_release_delta(
                        state_vectors,
                        int(state_constant),
                        int(task_index),
                        int(local_residue),
                        int(spec.post_switch_cap) - int(spec.pre_switch_cap),
                    )

            value, maxima, argmax = score_candidate(
                state_vectors, state_constant, -length
            )
            if int(value) > int(best_value):
                best_value = int(value)
                best_component_maxima = maxima
                best_component_argmax = argmax
                best_candidate = ("SWITCH", int(length), 2 * int(boundary))

            for task_index, local_residue in boundary_updates[boundary]:
                spec = specs[task_index]
                if spec.criticality == "LO":
                    state_constant = apply_release_delta(
                        state_vectors,
                        int(state_constant),
                        int(task_index),
                        int(local_residue),
                        int(spec.post_switch_cap) - int(spec.pre_switch_cap),
                    )

        # Switch before the oldest release: all releases in the busy suffix are
        # post-switch.  This is the canonical final open cell for this L.
        value, maxima, argmax = score_candidate(state_vectors, state_constant, -length)
        if int(value) > int(best_value):
            best_value = int(value)
            best_component_maxima = maxima
            best_component_argmax = argmax
            best_candidate = ("SWITCH", int(length), 2 * int(length) + 1)

    component_periods = plan.component_periods
    witness_k = _crt_reconstruct(component_periods, best_component_argmax)
    if witness_k < 0 or witness_k >= int(plan.family_size):
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_WITNESS_OUT_OF_RANGE")
    witness_q = (
        int(plan.block_residue) + int(plan.block_modulus) * int(witness_k)
    ) % int(plan.joint_period)
    witness_phases = target_release_joint_phases_at_q(
        int(target_period), specs, n0=int(n0), q_step=int(q_step), q=int(witness_q)
    )
    witness_r7, _ = fixed_phase_single_switch_backlog(specs, witness_phases)
    witness_common_r7 = _fixed_phase_r7_on_common_candidate_domain(
        specs, witness_phases, plan
    )
    if int(witness_r7) != int(witness_common_r7):
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_ACNF_WITNESS_MISMATCH")
    witness_future = fixed_phase_post_switch_future_work(
        specs, witness_phases, int(horizon)
    )
    witness_reevaluation = int(witness_r7) + int(witness_future)
    if int(witness_reevaluation) != int(best_value):
        raise CarryInEnvelopeUnresolved("CRT_PHASE_FAMILY_WITNESS_REEVALUATION_MISMATCH")

    return int(best_value), {
        "family_size": int(plan.family_size),
        "restricted_periods": tuple(int(value) for value in plan.restricted_periods),
        "component_periods": tuple(int(value) for value in component_periods),
        "component_tasks": tuple(
            tuple(specs[index].name for index in members)
            for _, members in plan.components
        ),
        "candidate_count": int(plan.candidate_count),
        "residues_per_candidate": int(plan.residues_per_candidate),
        "winning_candidate": {
            "kind": best_candidate[0],
            "busy_length": int(best_candidate[1]),
            "switch_cell_twice_age": int(best_candidate[2]),
        },
        "component_maxima": tuple(int(value) for value in best_component_maxima),
        "component_argmax": tuple(int(value) for value in best_component_argmax),
        "witness_k": int(witness_k),
        "witness_q": int(witness_q),
        "witness_phases": tuple(int(value) for value in witness_phases),
        "witness_reevaluation": int(witness_reevaluation),
        "witness_r7": int(witness_r7),
        "witness_common_domain_r7": int(witness_common_r7),
        "witness_future": int(witness_future),
        "busy_horizon": int(plan.busy_horizon),
        "production_busy_horizon": int(plan.production_busy_horizon),
        "streaming_residue_work": int(plan.streaming_residue_work),
    }

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
    "CRTPhaseFamilyPlan",
    "PhaseBlockProjection",
    "crt_phase_family_plan",
    "exact_crt_phase_family_pre_hi_max",
    "prehistory_finite_bound",
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
