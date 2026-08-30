from fractions import Fraction

import pytest

from formal_toolchain.v10_1.base_section4_1 import (
    PaperTask,
    Section41ScopeError,
    analyze_task_section4_1,
    compute_section4_1_certificate,
    eq13_case1_rhs,
    eq14_case2_rhs,
    eq16_case1_expanded_rhs,
    eq17_case2_expanded_rhs,
)


def _paper_toy():
    # Zhang-Zheng-Gu 2024, Table 2 / Section 4.3.
    return (
        PaperTask("tau1", 0, 5, 5, "LO", Fraction(1), Fraction(1, 2)),
        PaperTask("tau2", 1, 20, 20, "HI", Fraction(5), Fraction(6)),
        PaperTask("tau3", 2, 100, 100, "HI", Fraction(20), Fraction(30)),
    )


def test_section4_1_reproduces_paper_toy_example_exactly():
    cert = compute_section4_1_certificate(_paper_toy())
    assert cert.schedulable is True
    by_name = {row.target.name: row for row in cert.task_certificates}

    assert by_name["tau1"].r_lo.value == 1
    assert by_name["tau1"].r_hi == 1

    assert by_name["tau2"].r_lo.value == 7
    assert by_name["tau2"].w_lo.value == 1
    assert by_name["tau2"].r_hi == Fraction(15, 2)

    assert by_name["tau3"].r_lo.value == 38
    assert by_name["tau3"].w_lo.value == 7
    assert by_name["tau3"].r_hi == 54

    worst_case1 = max(by_name["tau3"].case1_candidates, key=lambda row: row.response_time)
    worst_case2 = max(by_name["tau3"].case2_candidates, key=lambda row: row.response_time)
    assert worst_case1.s == 35
    assert worst_case1.completion_time == 39
    assert worst_case2.s == 0
    assert worst_case2.completion_time == 54


def test_eq13_14_decomposition_matches_expanded_eq16_17():
    tasks = _paper_toy()
    target = tasks[-1]
    for s, t in ((0, Fraction(54)), (5, Fraction(109, 2)), (35, Fraction(39))):
        assert eq13_case1_rhs(tasks, target, s=s, t=t) == eq16_case1_expanded_rhs(
            tasks, target, s=s, t=t
        )
        assert eq14_case2_rhs(tasks, target, s=s, t=t) == eq17_case2_expanded_rhs(
            tasks, target, s=s, t=t
        )


@pytest.mark.parametrize("target_index", [0, 1, 2])
def test_section5_breakpoint_reduction_matches_exhaustive_integer_s(target_index):
    tasks = _paper_toy()
    reduced = analyze_task_section4_1(tasks, tasks[target_index], exhaustive_integer_s=False)
    exhaustive = analyze_task_section4_1(tasks, tasks[target_index], exhaustive_integer_s=True)
    assert reduced.r_lo.value == exhaustive.r_lo.value
    assert reduced.w_lo.value == exhaustive.w_lo.value
    assert reduced.r_hi == exhaustive.r_hi
    assert reduced.schedulable == exhaustive.schedulable


def test_switch_domain_is_strict_not_closed_at_rlo_or_wlo():
    tasks = _paper_toy()
    row = analyze_task_section4_1(tasks, tasks[-1])
    assert all(candidate.s < row.r_lo.value for candidate in row.case1_candidates)
    assert all(candidate.s < row.w_lo.value for candidate in row.case2_candidates)
    assert 38 not in {candidate.s for candidate in row.case1_candidates}
    assert 7 not in {candidate.s for candidate in row.case2_candidates}


def test_constrained_deadline_scope_is_enforced():
    with pytest.raises(Section41ScopeError, match="constrained deadlines"):
        PaperTask("bad", 0, 10, 11, "HI", Fraction(1), Fraction(2))


def test_lo_and_hi_execution_ordering_scope_is_enforced():
    with pytest.raises(Section41ScopeError, match="LO task requires"):
        PaperTask("bad_lo", 0, 10, 10, "LO", Fraction(1), Fraction(2))
    with pytest.raises(Section41ScopeError, match="HI task requires"):
        PaperTask("bad_hi", 0, 10, 10, "HI", Fraction(2), Fraction(1))


def test_failed_base_test_is_a_non_safety_conclusion():
    tasks = (
        PaperTask("hi0", 0, 5, 5, "HI", Fraction(4), Fraction(5)),
        PaperTask("hi1", 1, 5, 5, "HI", Fraction(4), Fraction(5)),
    )
    cert = compute_section4_1_certificate(tasks)
    assert cert.schedulable is False
    assert cert.task_certificates[-1].failure_reason in {
        "R_LO_EXCEEDS_DEADLINE",
        "R_HI_EXCEEDS_DEADLINE",
    }


def test_v10_binding_uses_frozen_lo_degraded_cost_not_taskbound_c_hi():
    from types import SimpleNamespace
    from formal_toolchain.v10_1.base_section4_1 import bind_paper_taskset

    model = SimpleNamespace(
        tasks=(
            SimpleNamespace(
                name="lo",
                priority=0,
                period=10,
                deadline=10,
                criticality="LO",
                c_lo=6,
                c_hi=9,
                degraded_cost=3,
                actual_demand_upper=8,
            ),
            SimpleNamespace(
                name="hi",
                priority=1,
                period=20,
                deadline=20,
                criticality="HI",
                c_lo=4,
                c_hi=7,
                degraded_cost=None,
            ),
        )
    )
    rows = bind_paper_taskset(model)
    assert rows[0].c_lo == 8
    assert rows[0].c_hi == 3
    assert rows[1].c_lo == 4
    assert rows[1].c_hi == 7



def test_dynamic_lo_initial_budget_is_not_misbound_as_paper_c_lo():
    from types import SimpleNamespace
    from formal_toolchain.v10_1.base_section4_1 import bind_paper_taskset, paper_c_lo_bound

    lo = SimpleNamespace(
        name="mc_sd_lo_3", priority=0, period=10_000, deadline=10_000,
        criticality="LO", c_lo=1097, c_hi=1097, degraded_cost=548,
        actual_demand_upper=1646,
    )
    hi = SimpleNamespace(
        name="mc_sd_hi_0", priority=1, period=11_000, deadline=11_000,
        criticality="HI", c_lo=661, c_hi=2210, degraded_cost=None,
        actual_demand_upper=2210,
    )
    rows = bind_paper_taskset(SimpleNamespace(tasks=(lo, hi)))

    # LO Task.c_lo is the initial runtime budget in mc_stratified_dynamic.
    # The raw primary-demand envelope is the sound paper WCET.
    assert paper_c_lo_bound(lo) == 1646
    assert rows[0].c_lo == 1646
    assert rows[0].c_hi == 548

    # HI C_LO must remain the semi-clairvoyant classification threshold.
    assert paper_c_lo_bound(hi) == 661
    assert rows[1].c_lo == 661
    assert rows[1].c_hi == 2210

def test_verifier_facing_failure_is_unresolved_not_unsafe():
    from types import SimpleNamespace
    from formal_toolchain.v10_1.base_section4_1 import prove_original_c_amc_sem_section4_1

    model = SimpleNamespace(
        tasks=(
            SimpleNamespace(
                name="hi0", priority=0, period=5, deadline=5,
                criticality="HI", c_lo=4, c_hi=5, degraded_cost=None,
            ),
            SimpleNamespace(
                name="hi1", priority=1, period=5, deadline=5,
                criticality="HI", c_lo=4, c_hi=5, degraded_cost=None,
            ),
        )
    )
    result = prove_original_c_amc_sem_section4_1(model)
    assert result["status"] == "UNRESOLVED"
    assert result["code"] == "BASE_C_AMC_SEM_NOT_SUFFICIENT"
    assert result["all_tasks_schedulable"] is False
