from amc_py.models import Criticality, Task
from amc_py.priorities import sort_by_deadline_monotonic
from amc_py.rta import (
    analyze_hi_mode,
    analyze_lo_mode,
    compute_r_hi,
    compute_r_lo,
    solve_fixed_point,
)


def test_solve_fixed_point_converges() -> None:
    # 方程 R = 1 + ceil(R/10)，固定点应为 2。
    result = solve_fixed_point(lambda r: 1 + ((r + 9) // 10), start_value=1, deadline=10)
    assert result == 2


def test_solve_fixed_point_deadline_exceeded() -> None:
    # 方程单调增长且会超过截止期，应返回 None。
    result = solve_fixed_point(lambda r: r + 1, start_value=1, deadline=5, max_iter=10)
    assert result is None


def test_lo_and_hi_mode_analysis() -> None:
    tasks = [
        Task("tau1", period=5, deadline=5, c_lo=1, c_hi=2, criticality=Criticality.HI),
        Task("tau2", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
    ]
    ordered = sort_by_deadline_monotonic(tasks)

    lo_result = analyze_lo_mode(tasks, ordered)
    hi_result = analyze_hi_mode(tasks, ordered)

    assert lo_result.schedulable is True
    assert hi_result.schedulable is True
    assert lo_result.response_times["tau2"] == 3
    assert hi_result.response_times["tau2"] == 5


def test_compute_r_lo_and_r_hi() -> None:
    hp = Task("hp", period=5, deadline=5, c_lo=1, c_hi=2, criticality=Criticality.HI)
    lp = Task("lp", period=12, deadline=12, c_lo=2, c_hi=4, criticality=Criticality.HI)

    assert compute_r_lo(lp, [hp]) == 3
    assert compute_r_hi(lp, [hp]) == 8
