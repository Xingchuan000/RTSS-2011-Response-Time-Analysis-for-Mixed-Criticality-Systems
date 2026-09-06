"""Regression tests for the TECS'24 C-AMC-sem analysis."""

from __future__ import annotations

from amc_py.c_amc_sem import (
    analyze_c_amc_sem_task,
    c_amc_sem_hi_mode_budget,
    c_amc_sem_sched_test,
)
from amc_py.experiments import evaluate_taskset, resolve_ordering
from amc_py.models import Criticality, Task


def _scaled_paper_toy() -> list[Task]:
    """Paper Table 2 scaled by two so every WCET is an integer tick."""

    # Paper order after OPA: tau1 > tau2 > tau3.
    return [
        Task("tau1", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("tau2", period=40, deadline=40, c_lo=10, c_hi=12, criticality=Criticality.HI),
        Task("tau3", period=200, deadline=200, c_lo=40, c_hi=60, criticality=Criticality.HI),
    ]


def test_lo_hi_budget_matches_runtime_rounding_rule() -> None:
    lo = Task("lo", period=20, deadline=20, c_lo=5, c_hi=5, criticality=Criticality.LO)
    hi = Task("hi", period=20, deadline=20, c_lo=5, c_hi=9, criticality=Criticality.HI)

    # Python/runtime round(2.5) -> 2, then clamp to at least one tick.
    assert c_amc_sem_hi_mode_budget(lo, 0.5) == 2
    assert c_amc_sem_hi_mode_budget(hi, 0.5) == 9


def test_scaled_paper_toy_reproduces_section_4_3_values() -> None:
    tasks = _scaled_paper_toy()

    t1 = analyze_c_amc_sem_task(tasks[0], [], lo_degradation_ratio=0.5)
    t2 = analyze_c_amc_sem_task(tasks[1], tasks[:1], lo_degradation_ratio=0.5)
    t3 = analyze_c_amc_sem_task(tasks[2], tasks[:2], lo_degradation_ratio=0.5)

    assert (t1.r_lo, t1.r_hi) == (2, 2)
    assert (t2.r_lo, t2.w_lo, t2.r_hi_case1, t2.r_hi_case2, t2.r_hi) == (
        14,
        2,
        14,
        15,
        15,
    )
    assert (t3.r_lo, t3.w_lo, t3.r_hi_case1, t3.r_hi_case2, t3.r_hi) == (
        76,
        14,
        78,
        108,
        108,
    )
    assert c_amc_sem_sched_test(tasks, lo_degradation_ratio=0.5).schedulable is True


def test_evaluate_taskset_supports_c_amc_sem_with_opa() -> None:
    tasks = list(reversed(_scaled_paper_toy()))
    result = evaluate_taskset(
        tasks,
        method="c_amc_sem",
        priority_policy="opa",
        c_amc_sem_xf=0.5,
    )

    assert result.schedulable is True
    assert result.method == "c_amc_sem"
    assert "priority_policy=opa" in result.details
    assert "c_amc_sem_xf=0.5" in result.details

    ordered = resolve_ordering(
        tasks,
        priority_policy="opa",
        method="c_amc_sem",
        c_amc_sem_xf=0.5,
    )
    assert c_amc_sem_sched_test(ordered, lo_degradation_ratio=0.5).schedulable is True


def test_c_amc_sem_rejects_invalid_xf() -> None:
    tasks = _scaled_paper_toy()
    for xf in (0.0, -0.1, 1.1):
        try:
            evaluate_taskset(
                tasks,
                method="c_amc_sem",
                priority_policy="dm",
                c_amc_sem_xf=xf,
            )
        except ValueError as exc:
            assert "lo_degradation_ratio" in str(exc)
        else:
            raise AssertionError(f"invalid XF={xf} was accepted")
