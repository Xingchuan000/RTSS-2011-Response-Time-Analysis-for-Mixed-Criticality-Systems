from amc_py.amc import amc_rtb_sched_test
from amc_py.models import Criticality, Task
from amc_py.priorities import sort_by_deadline_monotonic
from amc_py.smc import smc_sched_test


def test_amc_rtb_not_weaker_than_smc_example() -> None:
    # 该示例体现 AMC-rtb 相比 SMC 的优势：
    # - LO 高优先级干扰在 AMC-rtb 中由 R_LO 截断。
    # - 同一任务集下，SMC 不可调度而 AMC-rtb 可调度。
    tasks = [
        Task("tau_lo", period=5, deadline=5, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("tau_hi1", period=7, deadline=7, c_lo=1, c_hi=4, criticality=Criticality.HI),
        Task("tau_hi2", period=30, deadline=25, c_lo=2, c_hi=4, criticality=Criticality.HI),
    ]
    ordered = sort_by_deadline_monotonic(tasks)

    smc_result = smc_sched_test(ordered)
    amc_rtb_result = amc_rtb_sched_test(ordered)

    assert smc_result.schedulable is False
    assert amc_rtb_result.schedulable is True


def test_amc_rtb_dominates_smc_on_reference_sets() -> None:
    # 在一组代表性任务集上验证“若 SMC 可调度，则 AMC-rtb 也应可调度”。
    tasksets = [
        [
            Task("a1", 5, 5, 1, 1, Criticality.LO),
            Task("a2", 10, 10, 2, 3, Criticality.HI),
        ],
        [
            Task("b1", 4, 4, 1, 2, Criticality.LO),
            Task("b2", 8, 8, 2, 3, Criticality.HI),
            Task("b3", 12, 12, 2, 4, Criticality.HI),
        ],
        [
            Task("c1", 3, 3, 1, 1, Criticality.LO),
            Task("c2", 6, 6, 1, 2, Criticality.HI),
            Task("c3", 15, 14, 2, 4, Criticality.HI),
        ],
    ]

    for tasks in tasksets:
        ordered = sort_by_deadline_monotonic(tasks)
        smc_sched = smc_sched_test(ordered).schedulable
        amc_sched = amc_rtb_sched_test(ordered).schedulable

        if smc_sched:
            assert amc_sched is True
