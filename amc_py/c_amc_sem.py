"""C-AMC-sem schedulability analysis for constrained-deadline FPPS tasksets.

This module implements the constant-speed (S=1) analysis in Zhang et al.
(TECS 2024), Section 4.1, Equations (4), (6), (11)--(17), and the switch-
point reduction stated in Section 5's complexity discussion.

The repository's :class:`~amc_py.models.Task` model stores ``c_hi >= c_lo``
for every task, while the production C-AMC-sem runtime represents graceful
LO degradation through ``c_amc_sem_lo_degradation_ratio``.  To keep the
static admission test bit-for-bit aligned with that runtime convention, LO
``C(HI)`` is therefore derived as::

    max(1, min(C(LO), round(C(LO) * XF)))

rather than read from ``Task.c_hi``.  HI tasks continue to use ``Task.c_hi``.

Priority order arguments are always high-to-low, matching the rest of this
codebase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor

from .models import Criticality, SchedulabilityResult, Task
from .rta import compute_r_lo


@dataclass(frozen=True, slots=True)
class CAmcSemTaskAnalysis:
    """Detailed C-AMC-sem response-time result for one task.

    ``r_hi_case1`` and ``r_hi_case2`` are maxima over the paper's candidate
    switch points.  ``r_hi_case2`` is already converted from busy-period
    length ``R_i^s(HI)_2`` to response time ``R_i^s(HI)_2 - s``.
    """

    task_name: str
    r_lo: int | None
    w_lo: int | None
    r_hi_case1: int | None
    r_hi_case2: int | None
    r_hi: int | None
    response_time: int | None
    worst_case: str | None
    worst_switch_time: int | None
    schedulable: bool
    details: str


def validate_c_amc_sem_xf(lo_degradation_ratio: float) -> float:
    """Validate and normalize the C-AMC-sem compensating factor ``XF``."""

    xf = float(lo_degradation_ratio)
    if not (0.0 < xf <= 1.0):
        raise ValueError("lo_degradation_ratio must be in (0, 1]")
    return xf


def c_amc_sem_hi_mode_budget(task: Task, lo_degradation_ratio: float = 0.5) -> int:
    """Return the task's C-AMC-sem ``C(HI)`` in integer runtime ticks.

    HI tasks use their design ``Task.c_hi``.  LO tasks use the same integer
    rounding/clamping rule as ``runtime.py`` and ``event_runtime.py``.
    """

    if task.criticality is Criticality.HI:
        return int(task.c_hi)
    xf = validate_c_amc_sem_xf(lo_degradation_ratio)
    budget = int(round(task.c_lo * xf))
    return max(1, min(int(task.c_lo), budget))


def compute_c_amc_sem_w_lo(
    task: Task,
    higher_priority_tasks: Sequence[Task],
    *,
    max_iter: int = 1000,
) -> int | None:
    """Compute the worst-case LO-mode start time ``W_i(LO)`` (Eq. 12).

    Equation (12) has no own-execution term.  For a highest-priority task its
    smallest solution is therefore zero.  ``None`` means the monotone
    iteration exceeded the task deadline or did not converge.
    """

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if not higher_priority_tasks:
        return 0

    current = sum(int(hp.c_lo) for hp in higher_priority_tasks)
    if current > task.deadline:
        return None

    for _ in range(max_iter):
        next_value = sum(
            (floor(current / hp.period) + 1) * hp.c_lo
            for hp in higher_priority_tasks
        )
        if next_value > task.deadline:
            return None
        if next_value == current:
            return int(next_value)
        current = int(next_value)
    return None


def candidate_c_amc_sem_switch_points(
    higher_priority_tasks: Sequence[Task],
    upper_bound: int,
) -> list[int]:
    """Return the paper-reduced candidate mode-switch points below a bound.

    Zhang et al. state that only ``s`` values corresponding to arrivals of
    higher-priority LO tasks need be checked.  The intervals are half-open
    ``[0, R_i(LO))`` and ``[0, W_i(LO))``.  ``s=0`` is retained even when the
    task has no higher-priority LO tasks (or ``W_i(LO)=0``), which is required
    for the highest-priority and simultaneous-release cases.
    """

    points: set[int] = {0}
    if upper_bound <= 0:
        return [0]

    for hp in higher_priority_tasks:
        if hp.criticality is not Criticality.LO:
            continue
        release = 0
        while release < upper_bound:
            points.add(release)
            release += hp.period
    return sorted(points)


def _solve_case_fixed_point(
    recurrence_fn,
    *,
    start_value: int,
    upper_limit: int,
    max_iter: int = 1000,
) -> int | None:
    """Solve one monotone C-AMC-sem busy-period recurrence.

    Case 2 is bounded by ``s + D_i`` rather than ``D_i`` because the paper's
    busy period starts at zero while the target job may arrive at ``s``.
    """

    if start_value <= 0:
        raise ValueError("start_value must be > 0")
    if upper_limit <= 0:
        raise ValueError("upper_limit must be > 0")

    current = int(start_value)
    for _ in range(max_iter):
        next_value = int(recurrence_fn(current))
        if next_value > upper_limit:
            return None
        if next_value == current:
            return next_value
        current = next_value
    return None


def _c_amc_sem_interference(
    *,
    t: int,
    s: int,
    higher_priority_tasks: Sequence[Task],
    lo_degradation_ratio: float,
) -> int:
    """Compute ``I_L(i,s,t) + I_H(i,s,t)`` from Eqs. (6) and (11)."""

    if t < 0 or s < 0 or t < s:
        raise ValueError("C-AMC-sem interference requires 0 <= s <= t")

    interference = 0
    for hp in higher_priority_tasks:
        if hp.criticality is Criticality.LO:
            c_hi = c_amc_sem_hi_mode_budget(hp, lo_degradation_ratio)
            interference += (
                ceil(t / hp.period) * c_hi
                + (floor(s / hp.period) + 1) * (hp.c_lo - c_hi)
            )
        else:
            interference += (
                ceil(t / hp.period) * hp.c_lo
                + ceil((t - s) / hp.period) * (hp.c_hi - hp.c_lo)
            )
    return int(interference)


def compute_c_amc_sem_case_response_time(
    task: Task,
    higher_priority_tasks: Sequence[Task],
    *,
    s: int,
    case: int,
    lo_degradation_ratio: float = 0.5,
    max_iter: int = 1000,
) -> int | None:
    """Compute one fixed-switch C-AMC-sem response time (Eqs. 16/17).

    Args:
        case: ``1`` for Eq. (16), ``2`` for Eq. (17).

    Returns:
        Case 1 returns ``R_i^s(HI)_1``.  Case 2 returns the actual response
        time ``R_i^s(HI)_2 - s``.  ``None`` means the deadline bound is
        exceeded or the recurrence does not converge.
    """

    xf = validate_c_amc_sem_xf(lo_degradation_ratio)
    if s < 0:
        raise ValueError("s must be >= 0")
    if case not in {1, 2}:
        raise ValueError("case must be 1 or 2")

    own_budget = (
        int(task.c_lo)
        if case == 1
        else c_amc_sem_hi_mode_budget(task, xf)
    )

    def recurrence(t: int) -> int:
        return own_budget + _c_amc_sem_interference(
            t=t,
            s=s,
            higher_priority_tasks=higher_priority_tasks,
            lo_degradation_ratio=xf,
        )

    # The recurrence is meaningful only once t >= s.  For Case 2, the target
    # arrives at s, so its busy-period length is at least s + own_budget.
    start_value = max(own_budget, s) if case == 1 else s + own_budget
    upper_limit = task.deadline if case == 1 else s + task.deadline
    busy_period = _solve_case_fixed_point(
        recurrence,
        start_value=start_value,
        upper_limit=upper_limit,
        max_iter=max_iter,
    )
    if busy_period is None:
        return None
    return busy_period if case == 1 else busy_period - s


def analyze_c_amc_sem_task(
    task: Task,
    higher_priority_tasks: Sequence[Task],
    *,
    lo_degradation_ratio: float = 0.5,
    max_iter: int = 1000,
) -> CAmcSemTaskAnalysis:
    """Analyze one task under C-AMC-sem with a fixed higher-priority set."""

    xf = validate_c_amc_sem_xf(lo_degradation_ratio)
    r_lo = compute_r_lo(task, higher_priority_tasks)
    if r_lo is None:
        return CAmcSemTaskAnalysis(
            task_name=task.name,
            r_lo=None,
            w_lo=None,
            r_hi_case1=None,
            r_hi_case2=None,
            r_hi=None,
            response_time=None,
            worst_case="LO",
            worst_switch_time=None,
            schedulable=False,
            details=f"task {task.name}: R_i(LO) exceeds deadline",
        )

    w_lo = compute_c_amc_sem_w_lo(task, higher_priority_tasks, max_iter=max_iter)
    if w_lo is None:
        return CAmcSemTaskAnalysis(
            task_name=task.name,
            r_lo=r_lo,
            w_lo=None,
            r_hi_case1=None,
            r_hi_case2=None,
            r_hi=None,
            response_time=None,
            worst_case="W_LO",
            worst_switch_time=None,
            schedulable=False,
            details=f"task {task.name}: W_i(LO) does not converge within deadline",
        )

    case1_max = -1
    case1_s: int | None = None
    for s in candidate_c_amc_sem_switch_points(higher_priority_tasks, r_lo):
        response = compute_c_amc_sem_case_response_time(
            task,
            higher_priority_tasks,
            s=s,
            case=1,
            lo_degradation_ratio=xf,
            max_iter=max_iter,
        )
        if response is None:
            return CAmcSemTaskAnalysis(
                task_name=task.name,
                r_lo=r_lo,
                w_lo=w_lo,
                r_hi_case1=None,
                r_hi_case2=None,
                r_hi=None,
                response_time=None,
                worst_case="HI_CASE1",
                worst_switch_time=s,
                schedulable=False,
                details=(
                    f"task {task.name}: C-AMC-sem Case 1 exceeds deadline "
                    f"at switch s={s}"
                ),
            )
        if response > case1_max:
            case1_max = response
            case1_s = s

    case2_max = -1
    case2_s: int | None = None
    for s in candidate_c_amc_sem_switch_points(higher_priority_tasks, w_lo):
        response = compute_c_amc_sem_case_response_time(
            task,
            higher_priority_tasks,
            s=s,
            case=2,
            lo_degradation_ratio=xf,
            max_iter=max_iter,
        )
        if response is None:
            return CAmcSemTaskAnalysis(
                task_name=task.name,
                r_lo=r_lo,
                w_lo=w_lo,
                r_hi_case1=case1_max if case1_max >= 0 else None,
                r_hi_case2=None,
                r_hi=None,
                response_time=None,
                worst_case="HI_CASE2",
                worst_switch_time=s,
                schedulable=False,
                details=(
                    f"task {task.name}: C-AMC-sem Case 2 exceeds deadline "
                    f"at switch s={s}"
                ),
            )
        if response > case2_max:
            case2_max = response
            case2_s = s

    r_hi = max(case1_max, case2_max)
    if case1_max >= case2_max:
        worst_case = "HI_CASE1"
        worst_s = case1_s
    else:
        worst_case = "HI_CASE2"
        worst_s = case2_s

    final_response = max(r_lo, r_hi)
    schedulable = final_response <= task.deadline
    return CAmcSemTaskAnalysis(
        task_name=task.name,
        r_lo=r_lo,
        w_lo=w_lo,
        r_hi_case1=case1_max,
        r_hi_case2=case2_max,
        r_hi=r_hi,
        response_time=final_response if schedulable else None,
        worst_case=worst_case,
        worst_switch_time=worst_s,
        schedulable=schedulable,
        details=(
            f"task {task.name}: R_LO={r_lo}, W_LO={w_lo}, "
            f"R_HI_case1={case1_max}, R_HI_case2={case2_max}, R_HI={r_hi}"
        ),
    )


def c_amc_sem_sched_test(
    ordered_tasks: Sequence[Task],
    *,
    lo_degradation_ratio: float = 0.5,
    max_iter: int = 1000,
) -> SchedulabilityResult:
    """Run the C-AMC-sem schedulability test on a fixed priority order."""

    xf = validate_c_amc_sem_xf(lo_degradation_ratio)
    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        analysis = analyze_c_amc_sem_task(
            task,
            ordered_tasks[:idx],
            lo_degradation_ratio=xf,
            max_iter=max_iter,
        )
        if not analysis.schedulable or analysis.response_time is None:
            return SchedulabilityResult(
                schedulable=False,
                method="c_amc_sem",
                response_times=response_times,
                details=f"{analysis.details}; XF={xf}",
            )
        response_times[task.name] = analysis.response_time

    return SchedulabilityResult(
        schedulable=True,
        method="c_amc_sem",
        response_times=response_times,
        details=f"C-AMC-sem test passed; XF={xf}",
    )


__all__ = [
    "CAmcSemTaskAnalysis",
    "analyze_c_amc_sem_task",
    "c_amc_sem_hi_mode_budget",
    "c_amc_sem_sched_test",
    "candidate_c_amc_sem_switch_points",
    "compute_c_amc_sem_case_response_time",
    "compute_c_amc_sem_w_lo",
    "validate_c_amc_sem_xf",
]
