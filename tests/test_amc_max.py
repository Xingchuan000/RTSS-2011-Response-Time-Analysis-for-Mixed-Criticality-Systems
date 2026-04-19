from amc_py.amc import amc_max_sched_test, amc_rtb_sched_test
from amc_py.models import Criticality, Task
from amc_py.priorities import sort_by_deadline_monotonic


def test_amc_max_not_weaker_than_amc_rtb() -> None:
    # 在多组代表性任务集上检查：若 AMC-rtb 可调度，则 AMC-max 也应可调度。
    tasksets = [
        [
            Task("a1", 5, 5, 1, 1, Criticality.LO),
            Task("a2", 7, 7, 1, 3, Criticality.HI),
            Task("a3", 20, 18, 2, 4, Criticality.HI),
        ],
        [
            Task("b1", 4, 4, 1, 1, Criticality.LO),
            Task("b2", 6, 6, 1, 2, Criticality.HI),
            Task("b3", 15, 15, 2, 5, Criticality.HI),
        ],
        [
            Task("c1", 3, 3, 1, 1, Criticality.LO),
            Task("c2", 8, 8, 2, 4, Criticality.HI),
            Task("c3", 18, 16, 2, 5, Criticality.HI),
        ],
    ]

    for tasks in tasksets:
        ordered = sort_by_deadline_monotonic(tasks)
        rtb_sched = amc_rtb_sched_test(ordered).schedulable
        max_sched = amc_max_sched_test(ordered).schedulable

        if rtb_sched:
            assert max_sched is True


def test_amc_max_and_amc_rtb_comparison_example() -> None:
    # 对该任务集，两者都可调度，且 AMC-max 作为更精细分析不应给出更差判定。
    tasks = [
        Task("tau_lo", period=5, deadline=5, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("tau_hi1", period=7, deadline=7, c_lo=1, c_hi=4, criticality=Criticality.HI),
        Task("tau_hi2", period=30, deadline=25, c_lo=2, c_hi=4, criticality=Criticality.HI),
    ]
    ordered = sort_by_deadline_monotonic(tasks)

    rtb_result = amc_rtb_sched_test(ordered)
    max_result = amc_max_sched_test(ordered)

    assert rtb_result.schedulable is True
    assert max_result.schedulable is True
