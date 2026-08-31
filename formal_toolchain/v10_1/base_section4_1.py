"""Exact fixed-priority C-AMC-sem Section 4.1 schedulability test.

This module implements the constant-speed C-AMC-sem analysis in Zhang,
Zheng, and Gu (ACM TECS 2024), Section 4.1, Equations (4), (6), and
(11)--(17).  It deliberately does not reuse AMC-rtb/AMC-max routines.

The deployed V10.1 proof route uses the already-bound fixed-priority order.
No priority reassignment is performed here: this is a schedulability
certificate for the exact priority order used by the frozen runtime.

Time parameters are integral ticks in the deployed model, while execution
costs are represented with :class:`fractions.Fraction` so the implementation
can also reproduce the paper's fractional toy example without floating-point
rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Any, Iterable, Mapping, Sequence


PAPER_REFERENCE = (
    "Zhang, Zheng, Gu. Energy-Aware Adaptive Mixed-Criticality Scheduling "
    "with Semi-Clairvoyance and Graceful Degradation. ACM TECS 23(1), 2024."
)


class Section41ScopeError(ValueError):
    """Raised when the bound task model is outside Section 4.1 assumptions."""


def _q(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(int(value), 1)


def _q_json(value: Fraction | None) -> int | str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return int(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _ceil_ratio(value: Fraction, period: int) -> int:
    if period <= 0:
        raise ValueError("period must be positive")
    return ceil(value / period)


def _floor_ratio(value: Fraction, period: int) -> int:
    if period <= 0:
        raise ValueError("period must be positive")
    return floor(value / period)


@dataclass(frozen=True, slots=True)
class PaperTask:
    """One C-AMC-sem task under the paper's Section 3.1 task model."""

    name: str
    priority: int
    period: int
    deadline: int
    criticality: str
    c_lo: Fraction
    c_hi: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "c_lo", _q(self.c_lo))
        object.__setattr__(self, "c_hi", _q(self.c_hi))
        if not self.name:
            raise Section41ScopeError("task name must be non-empty")
        if self.priority < 0:
            raise Section41ScopeError(f"negative priority index: {self.name}")
        if self.period <= 0 or self.deadline <= 0:
            raise Section41ScopeError(f"non-positive T/D: {self.name}")
        if self.deadline > self.period:
            raise Section41ScopeError(
                f"Section 4.1 assumes constrained deadlines D<=T: {self.name}"
            )
        if self.criticality not in {"LO", "HI"}:
            raise Section41ScopeError(f"invalid criticality: {self.name}")
        if self.c_lo <= 0 or self.c_hi <= 0:
            raise Section41ScopeError(f"non-positive execution cost: {self.name}")
        if self.criticality == "HI" and self.c_hi < self.c_lo:
            raise Section41ScopeError(f"HI task requires C_HI>=C_LO: {self.name}")
        if self.criticality == "LO" and self.c_hi > self.c_lo:
            raise Section41ScopeError(f"LO task requires C_HI<=C_LO: {self.name}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "priority_index": self.priority,
            "period": self.period,
            "deadline": self.deadline,
            "criticality": self.criticality,
            "C_LO": _q_json(self.c_lo),
            "C_HI": _q_json(self.c_hi),
        }


@dataclass(frozen=True, slots=True)
class FixedPointResult:
    value: Fraction
    iterations: int
    converged: bool
    deadline_exceeded: bool
    response_origin: Fraction

    @property
    def response_time(self) -> Fraction:
        return self.value - self.response_origin

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": _q_json(self.value),
            "response_time": _q_json(self.response_time),
            "response_origin": _q_json(self.response_origin),
            "iterations": self.iterations,
            "converged": self.converged,
            "deadline_exceeded": self.deadline_exceeded,
        }


@dataclass(frozen=True, slots=True)
class SwitchCaseResult:
    s: int
    completion_time: Fraction
    response_time: Fraction
    iterations: int
    converged: bool
    deadline_exceeded: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "s": self.s,
            "completion_time": _q_json(self.completion_time),
            "response_time": _q_json(self.response_time),
            "iterations": self.iterations,
            "converged": self.converged,
            "deadline_exceeded": self.deadline_exceeded,
        }


@dataclass(frozen=True, slots=True)
class TaskSection41Certificate:
    target: PaperTask
    r_lo: FixedPointResult
    w_lo: FixedPointResult | None
    case1_candidates: tuple[SwitchCaseResult, ...]
    case2_candidates: tuple[SwitchCaseResult, ...]
    r_hi: Fraction | None
    schedulable: bool
    failure_reason: str | None

    @staticmethod
    def _worst(rows: Sequence[SwitchCaseResult]) -> SwitchCaseResult | None:
        if not rows:
            return None
        return max(rows, key=lambda row: (row.response_time, row.completion_time, row.s))

    def as_dict(self) -> dict[str, Any]:
        worst1 = self._worst(self.case1_candidates)
        worst2 = self._worst(self.case2_candidates)
        return {
            "target": self.target.name,
            "priority_index": self.target.priority,
            "criticality": self.target.criticality,
            "deadline": self.target.deadline,
            "R_LO_eq4": self.r_lo.as_dict(),
            "W_LO_eq12": self.w_lo.as_dict() if self.w_lo is not None else None,
            "case1_eq13_eq16": {
                "candidate_count": len(self.case1_candidates),
                "worst": worst1.as_dict() if worst1 is not None else None,
            },
            "case2_eq14_eq17": {
                "candidate_count": len(self.case2_candidates),
                "worst": worst2.as_dict() if worst2 is not None else None,
            },
            "R_HI_eq15": _q_json(self.r_hi),
            "schedulable": self.schedulable,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class Section41Certificate:
    tasks: tuple[PaperTask, ...]
    task_certificates: tuple[TaskSection41Certificate, ...]

    @property
    def schedulable(self) -> bool:
        return all(row.schedulable for row in self.task_certificates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "c_amc_sem_section4_1_certificate_v1",
            "obligation_id": "BASE_C_AMC_SEM_SECTION4_1_CERTIFICATE",
            "paper_reference": PAPER_REFERENCE,
            "equations": [4, 6, 11, 12, 13, 14, 15, 16, 17],
            "priority_semantics": "fixed priority; smaller priority_index is higher",
            "switch_domains": {
                "case1": "0 <= s < R_i(LO)",
                "case2": "0 <= s < W_i(LO)",
                "candidate_reduction": (
                    "s=0 plus releases of higher-priority LO tasks inside the strict domain; "
                    "this is the Section 5 breakpoint reduction for Equations (16)-(17)"
                ),
            },
            "taskset": [task.as_dict() for task in self.tasks],
            "task_certificates": [row.as_dict() for row in self.task_certificates],
            "all_tasks_schedulable": self.schedulable,
        }


def validate_paper_taskset(tasks: Sequence[PaperTask]) -> tuple[PaperTask, ...]:
    rows = tuple(tasks)
    if not rows:
        raise Section41ScopeError("Section 4.1 requires a non-empty taskset")
    names = [row.name for row in rows]
    priorities = [row.priority for row in rows]
    if len(set(names)) != len(names):
        raise Section41ScopeError("task names must be unique")
    if len(set(priorities)) != len(priorities):
        raise Section41ScopeError("priority indices must be unique")
    ordered = tuple(sorted(rows, key=lambda row: row.priority))
    if ordered != rows:
        raise Section41ScopeError("tasks must be supplied in fixed-priority order")
    return rows


def paper_c_lo_bound(task: Any) -> int:
    """Return the paper-visible ``C_i(LO)`` bound for a frozen V10.1 task.

    The ``mc_stratified_dynamic`` workload deliberately stores the initial
    runtime budget in ``Task.c_lo``.  For HI tasks that value is also the
    semi-clairvoyant NORMAL/ABNORMAL classification threshold and therefore is
    exactly the paper ``C_i(LO)``.  For LO tasks there is no such
    classification threshold: the paper ``C_i(LO)`` must instead bound the
    primary implementation's raw execution demand.  V10.1 freezes that WCET
    envelope as ``actual_demand_upper``.

    Keeping these two meanings separate is essential.  Treating an LO task's
    initial budget as its paper WCET incorrectly rejects precisely the policy
    behaviour that raises future LO-primary service above the initial budget.
    """

    criticality = str(task.criticality)
    if criticality == "HI":
        value = int(task.c_lo)
    elif criticality == "LO":
        value = int(task.actual_demand_upper)
    else:
        raise Section41ScopeError(f"invalid criticality: {task.name}")
    if value <= 0:
        raise Section41ScopeError(f"non-positive paper C_LO binding: {task.name}")
    return value


def bind_paper_taskset(model: Any) -> tuple[PaperTask, ...]:
    """Bind V10.1 ``BoundModel`` tasks to the paper's C-AMC-sem tuple.

    For LO tasks, paper ``C_i(LO)`` is the frozen raw primary-demand WCET
    envelope and paper ``C_i(HI)`` is the degraded/imprecise execution cost.
    For HI tasks, paper ``C_i(LO)`` remains the release-classification
    threshold while ``C_i(HI)`` is the frozen HI-assurance bound.
    """

    rows: list[PaperTask] = []
    for task in model.tasks:
        criticality = str(task.criticality)
        paper_c_lo = Fraction(paper_c_lo_bound(task), 1)
        if criticality == "LO":
            if task.degraded_cost is None:
                raise Section41ScopeError(
                    f"LO task lacks frozen degraded C_HI binding: {task.name}"
                )
            paper_c_hi = Fraction(int(task.degraded_cost), 1)
        elif criticality == "HI":
            paper_c_hi = Fraction(int(task.c_hi), 1)
        else:
            raise Section41ScopeError(f"invalid criticality: {task.name}")
        rows.append(
            PaperTask(
                name=str(task.name),
                priority=int(task.priority),
                period=int(task.period),
                deadline=int(task.deadline),
                criticality=criticality,
                c_lo=paper_c_lo,
                c_hi=paper_c_hi,
            )
        )
    return validate_paper_taskset(tuple(rows))


def _higher_priority(tasks: Sequence[PaperTask], target: PaperTask) -> tuple[PaperTask, ...]:
    return tuple(row for row in tasks if row.priority < target.priority)


def _hp_lo(tasks: Sequence[PaperTask], target: PaperTask) -> tuple[PaperTask, ...]:
    return tuple(row for row in _higher_priority(tasks, target) if row.criticality == "LO")


def _hp_hi(tasks: Sequence[PaperTask], target: PaperTask) -> tuple[PaperTask, ...]:
    return tuple(row for row in _higher_priority(tasks, target) if row.criticality == "HI")


def _solve_fixed_point(
    rhs,
    *,
    initial: Fraction,
    response_origin: Fraction,
    deadline: int,
) -> FixedPointResult:
    """Compute the least fixed point needed by the deadline test.

    There is intentionally no arbitrary iteration cap.  Inside a finite
    response-time horizon the ceiling/floor counters admit only finitely many
    values.  The monotone recurrence therefore either reaches a fixed point or
    crosses the deadline, which is already sufficient to reject the BASE test.
    """

    current = _q(initial)
    origin = _q(response_origin)
    iterations = 0
    if current - origin > deadline:
        return FixedPointResult(current, 0, False, True, origin)
    while True:
        nxt = _q(rhs(current))
        iterations += 1
        if nxt < current:
            raise Section41ScopeError(
                f"Section 4.1 recurrence decreased: current={current}, next={nxt}"
            )
        if nxt - origin > deadline:
            return FixedPointResult(nxt, iterations, False, True, origin)
        if nxt == current:
            return FixedPointResult(nxt, iterations, True, False, origin)
        current = nxt


def eq4_lo_mode_rhs(tasks: Sequence[PaperTask], target: PaperTask, t: Fraction) -> Fraction:
    return target.c_lo + sum(
        (_ceil_ratio(t, row.period) * row.c_lo for row in _higher_priority(tasks, target)),
        Fraction(0, 1),
    )


def eq4_lo_mode_wcrt(tasks: Sequence[PaperTask], target: PaperTask) -> FixedPointResult:
    return _solve_fixed_point(
        lambda t: eq4_lo_mode_rhs(tasks, target, t),
        initial=target.c_lo,
        response_origin=Fraction(0, 1),
        deadline=target.deadline,
    )


def eq6_hp_lo_interference(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    """Equation (6): higher-priority LO-task interference."""

    return sum(
        (
            _ceil_ratio(t, row.period) * row.c_hi
            + (_floor_ratio(Fraction(s, 1), row.period) + 1) * (row.c_lo - row.c_hi)
            for row in _hp_lo(tasks, target)
        ),
        Fraction(0, 1),
    )


def eq11_hp_hi_interference(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    """Equation (11): higher-priority HI-task interference."""

    if t < s:
        raise Section41ScopeError("Equation (11) requires t>=s")
    return sum(
        (
            _ceil_ratio(t, row.period) * row.c_lo
            + _ceil_ratio(t - s, row.period) * (row.c_hi - row.c_lo)
            for row in _hp_hi(tasks, target)
        ),
        Fraction(0, 1),
    )


def eq12_lo_worst_start_rhs(
    tasks: Sequence[PaperTask], target: PaperTask, w: Fraction
) -> Fraction:
    return sum(
        (
            (_floor_ratio(w, row.period) + 1) * row.c_lo
            for row in _higher_priority(tasks, target)
        ),
        Fraction(0, 1),
    )


def eq12_lo_worst_start_time(tasks: Sequence[PaperTask], target: PaperTask) -> FixedPointResult:
    return _solve_fixed_point(
        lambda w: eq12_lo_worst_start_rhs(tasks, target, w),
        initial=Fraction(0, 1),
        response_origin=Fraction(0, 1),
        deadline=target.deadline,
    )


def eq13_case1_rhs(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    return target.c_lo + eq6_hp_lo_interference(tasks, target, s=s, t=t) + eq11_hp_hi_interference(
        tasks, target, s=s, t=t
    )


def eq14_case2_rhs(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    return target.c_hi + eq6_hp_lo_interference(tasks, target, s=s, t=t) + eq11_hp_hi_interference(
        tasks, target, s=s, t=t
    )


def eq16_case1_expanded_rhs(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    lo = sum(
        (
            _ceil_ratio(t, row.period) * row.c_hi
            + (_floor_ratio(Fraction(s, 1), row.period) + 1) * (row.c_lo - row.c_hi)
            for row in _hp_lo(tasks, target)
        ),
        Fraction(0, 1),
    )
    hi = sum(
        (
            _ceil_ratio(t, row.period) * row.c_lo
            + _ceil_ratio(t - s, row.period) * (row.c_hi - row.c_lo)
            for row in _hp_hi(tasks, target)
        ),
        Fraction(0, 1),
    )
    return target.c_lo + lo + hi


def eq17_case2_expanded_rhs(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int, t: Fraction
) -> Fraction:
    return target.c_hi + (
        eq16_case1_expanded_rhs(tasks, target, s=s, t=t) - target.c_lo
    )


def _paper_switch_candidates(
    tasks: Sequence[PaperTask], target: PaperTask, strict_upper: Fraction
) -> tuple[int, ...]:
    """Section 5 breakpoint reduction for the Section 4.1 maximization.

    The s-dependent LO term increases only when a higher-priority LO task is
    released, while the s-dependent HI excess term is non-increasing in s.
    Hence it suffices to evaluate s=0 and higher-priority LO release times in
    the strict interval used by the Section 4.1 prose: [0, upper).
    """

    if strict_upper <= 0:
        return ()
    values = {0}
    for task in _hp_lo(tasks, target):
        release = 0
        while Fraction(release, 1) < strict_upper:
            values.add(release)
            release += task.period
    return tuple(sorted(value for value in values if Fraction(value, 1) < strict_upper))


def _exhaustive_integer_switch_candidates(strict_upper: Fraction) -> tuple[int, ...]:
    """Small-model oracle used by tests for the breakpoint reduction."""

    if strict_upper <= 0:
        return ()
    return tuple(value for value in range(ceil(strict_upper)) if Fraction(value, 1) < strict_upper)


def _solve_case1(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int
) -> SwitchCaseResult:
    initial = max(target.c_lo, Fraction(s, 1))
    result = _solve_fixed_point(
        lambda t: eq16_case1_expanded_rhs(tasks, target, s=s, t=t),
        initial=initial,
        response_origin=Fraction(0, 1),
        deadline=target.deadline,
    )
    return SwitchCaseResult(
        s=s,
        completion_time=result.value,
        response_time=result.response_time,
        iterations=result.iterations,
        converged=result.converged,
        deadline_exceeded=result.deadline_exceeded,
    )


def _solve_case2(
    tasks: Sequence[PaperTask], target: PaperTask, *, s: int
) -> SwitchCaseResult:
    # In Case 2 the target is released no earlier than s, so absolute
    # completion cannot precede s + C_i(HI).  This keeps every recurrence
    # evaluation inside Equation (11)'s t>=s domain.
    initial = Fraction(s, 1) + target.c_hi
    result = _solve_fixed_point(
        lambda t: eq17_case2_expanded_rhs(tasks, target, s=s, t=t),
        initial=initial,
        response_origin=Fraction(s, 1),
        deadline=target.deadline,
    )
    return SwitchCaseResult(
        s=s,
        completion_time=result.value,
        response_time=result.response_time,
        iterations=result.iterations,
        converged=result.converged,
        deadline_exceeded=result.deadline_exceeded,
    )


def _max_response(rows: Iterable[SwitchCaseResult]) -> Fraction | None:
    values = [row.response_time for row in rows]
    return max(values) if values else None


def analyze_task_section4_1(
    tasks: Sequence[PaperTask],
    target: PaperTask,
    *,
    exhaustive_integer_s: bool = False,
) -> TaskSection41Certificate:
    """Compute Equations (4), (12)--(17) for one task."""

    tasks = validate_paper_taskset(tasks)
    if target not in tasks:
        raise Section41ScopeError(f"target not present in taskset: {target.name}")

    r_lo = eq4_lo_mode_wcrt(tasks, target)
    if r_lo.deadline_exceeded or not r_lo.converged:
        return TaskSection41Certificate(
            target=target,
            r_lo=r_lo,
            w_lo=None,
            case1_candidates=(),
            case2_candidates=(),
            r_hi=None,
            schedulable=False,
            failure_reason="R_LO_EXCEEDS_DEADLINE",
        )

    w_lo = eq12_lo_worst_start_time(tasks, target)
    if w_lo.deadline_exceeded or not w_lo.converged:
        return TaskSection41Certificate(
            target=target,
            r_lo=r_lo,
            w_lo=w_lo,
            case1_candidates=(),
            case2_candidates=(),
            r_hi=None,
            schedulable=False,
            failure_reason="W_LO_DOES_NOT_CLOSE_WITHIN_DEADLINE",
        )

    candidate_fn = _exhaustive_integer_switch_candidates if exhaustive_integer_s else None
    if candidate_fn is None:
        case1_s = _paper_switch_candidates(tasks, target, r_lo.value)
        case2_s = _paper_switch_candidates(tasks, target, w_lo.value)
    else:
        case1_s = candidate_fn(r_lo.value)
        case2_s = candidate_fn(w_lo.value)

    case1 = tuple(_solve_case1(tasks, target, s=s) for s in case1_s)
    case2 = tuple(_solve_case2(tasks, target, s=s) for s in case2_s)

    r_hi_parts = [value for value in (_max_response(case1), _max_response(case2)) if value is not None]
    if not r_hi_parts:
        raise Section41ScopeError(f"no valid Case-1/Case-2 switch candidate for {target.name}")
    candidate_failed = any(row.deadline_exceeded or not row.converged for row in (*case1, *case2))
    # A deadline-exceeded recurrence is deliberately stopped before its fixed
    # point because that is already sufficient to reject the BASE test.  Do
    # not mislabel that lower-bound witness as the exact Equation (15) WCRT.
    r_hi = None if candidate_failed else max(r_hi_parts)
    schedulable = (
        (not candidate_failed)
        and r_hi is not None
        and r_lo.value <= target.deadline
        and r_hi <= target.deadline
    )
    return TaskSection41Certificate(
        target=target,
        r_lo=r_lo,
        w_lo=w_lo,
        case1_candidates=case1,
        case2_candidates=case2,
        r_hi=r_hi,
        schedulable=schedulable,
        failure_reason=None if schedulable else "R_HI_EXCEEDS_DEADLINE",
    )


def compute_section4_1_certificate(
    tasks: Sequence[PaperTask], *, exhaustive_integer_s: bool = False
) -> Section41Certificate:
    rows = validate_paper_taskset(tasks)
    certificates: list[TaskSection41Certificate] = []
    for target in rows:
        certificate = analyze_task_section4_1(
            rows, target, exhaustive_integer_s=exhaustive_integer_s
        )
        certificates.append(certificate)
        # The paper test is conjunctive.  Once a task fails its deadline, the
        # BASE route is already insufficient; avoid spending time on lower
        # priority tasks that cannot change that conclusion.
        if not certificate.schedulable:
            break
    return Section41Certificate(rows, tuple(certificates))


def prove_original_c_amc_sem_section4_1(model: Any) -> dict[str, Any]:
    """Run the paper test and return the verifier-facing BASE receipt."""

    tasks = bind_paper_taskset(model)
    certificate = compute_section4_1_certificate(tasks)
    payload = certificate.as_dict()

    # The paper's taskset-level statement is conjunctive over all tasks, but
    # V10.1's primary claim is HI safety only.  Section 4.1 is evaluated in
    # fixed-priority order and ``compute_section4_1_certificate`` stops at the
    # first failed task.  Consequently every successful certificate before
    # that point forms a schedulable fixed-priority prefix.  Any HI target in
    # that prefix is proved independently of lower-priority tasks: lower
    # priorities cannot delay it under the frozen FPPS correspondence.
    cert_by_name = {row.target.name: row for row in certificate.task_certificates}
    hi_safe_targets: list[str] = []
    hi_unresolved_targets: list[str] = []
    for task in tasks:
        if task.criticality != "HI":
            continue
        row = cert_by_name.get(task.name)
        if row is not None and row.schedulable:
            hi_safe_targets.append(task.name)
        else:
            hi_unresolved_targets.append(task.name)

    completion_bound_by_task: dict[str, int] = {}
    completion_bound_basis_by_task: dict[str, dict[str, int]] = {}
    for row in certificate.task_certificates:
        if not row.schedulable or row.r_hi is None:
            continue
        bound = max(row.r_lo.value, row.r_hi)
        if bound.denominator != 1:
            raise Section41ScopeError(
                f"deployed integer-tick Section 4.1 completion bound became fractional: "
                f"{row.target.name}:{bound}"
            )
        completion_bound_by_task[row.target.name] = int(bound)
        completion_bound_basis_by_task[row.target.name] = {
            "R_LO_eq4": int(row.r_lo.value),
            "R_HI_eq15": int(row.r_hi),
            "completion_bound": int(bound),
        }

    payload.update({
        "hi_safety_status": "PASS" if not hi_unresolved_targets else "UNRESOLVED",
        "hi_safe_targets": hi_safe_targets,
        "hi_unresolved_targets": hi_unresolved_targets,
        "completion_bound_by_task": completion_bound_by_task,
        "completion_bound_basis_by_task": completion_bound_basis_by_task,
        "completion_bound_rule": (
            "for every successful Section 4.1 prefix task, dynamic->BASE refinement plus "
            "max(R_i(LO), R_i(HI)) bounds completion from release; lower-priority tasks "
            "cannot invalidate this fixed-priority completion envelope"
        ),
        "hi_prefix_rule": (
            "fixed-priority prefix induction: a BASE-proved HI target appears before "
            "the first failed Section 4.1 task, so every higher-priority task also has "
            "a schedulable Section 4.1 certificate; lower-priority failures cannot "
            "interfere with the target under FPPS"
        ),
    })

    if certificate.schedulable:
        payload.update(
            {
                "status": "PASS",
                "code": "BASE_C_AMC_SEM_SECTION4_1_PASS",
                "reason": None,
            }
        )
    else:
        payload.update(
            {
                # Failure of the all-task sufficient BASE test is not a proof
                # of deployed unsafety.  HI targets that lie in the successful
                # priority prefix remain valid BASE terminal certificates; only
                # unresolved HI targets continue to PCSSC.
                "status": "UNRESOLVED",
                "code": "BASE_C_AMC_SEM_NOT_SUFFICIENT",
                "reason": "SECTION4_1_DEADLINE_TEST_NOT_SATISFIED",
            }
        )
    return payload


__all__ = [
    "PaperTask",
    "Section41ScopeError",
    "TaskSection41Certificate",
    "Section41Certificate",
    "bind_paper_taskset",
    "paper_c_lo_bound",
    "validate_paper_taskset",
    "eq4_lo_mode_rhs",
    "eq4_lo_mode_wcrt",
    "eq6_hp_lo_interference",
    "eq11_hp_hi_interference",
    "eq12_lo_worst_start_rhs",
    "eq12_lo_worst_start_time",
    "eq13_case1_rhs",
    "eq14_case2_rhs",
    "eq16_case1_expanded_rhs",
    "eq17_case2_expanded_rhs",
    "analyze_task_section4_1",
    "compute_section4_1_certificate",
    "prove_original_c_amc_sem_section4_1",
]
