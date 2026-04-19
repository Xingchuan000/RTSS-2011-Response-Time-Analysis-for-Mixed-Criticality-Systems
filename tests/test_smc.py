from amc_py.models import Criticality, Task
from amc_py.generator import generate_taskset
from amc_py.priorities import sort_by_deadline_monotonic
from amc_py.smc import (
    compute_smc_no_response_time,
    compute_smc_response_time,
    smc_no_sched_test,
    smc_sched_test,
)


def test_smc_and_smc_no_difference() -> None:
    # 关键示例：
    # - 高优先级 LO 任务具有较大的 c_hi（用于体现 SMC-no 的保守性）
    # - 低优先级 HI 任务在 SMC 下可调度，在 SMC-no 下不可调度
    tasks = [
        Task("tau_lo", period=5, deadline=5, c_lo=1, c_hi=4, criticality=Criticality.LO),
        Task("tau_hi", period=10, deadline=9, c_lo=1, c_hi=2, criticality=Criticality.HI),
    ]
    ordered = sort_by_deadline_monotonic(tasks)

    tau_hi = ordered[1]
    hp = [ordered[0]]

    r_smc = compute_smc_response_time(tau_hi, hp)
    r_smc_no = compute_smc_no_response_time(tau_hi, hp)

    assert r_smc == 3
    assert r_smc_no is None

    smc_result = smc_sched_test(ordered)
    smc_no_result = smc_no_sched_test(ordered)

    assert smc_result.schedulable is True
    assert smc_no_result.schedulable is False


def test_smc_no_is_more_conservative_than_smc_when_lo_has_separate_chi() -> None:
    tasks = [
        Task("tau_lo", period=10, deadline=10, c_lo=1, c_hi=2, criticality=Criticality.LO),
        Task("tau_hi", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
    ]
    ordered = sort_by_deadline_monotonic(tasks)
    smc_r = compute_smc_response_time(ordered[1], ordered[:1])
    smc_no_r = compute_smc_no_response_time(ordered[1], ordered[:1])
    assert smc_r is not None and smc_no_r is not None
    assert smc_no_r >= smc_r


def test_smc_and_smc_no_do_not_collapse_under_scaled_lo_hi_budget() -> None:
    # 避免依赖实验模块（其依赖 matplotlib/pandas），这里直接比较 SMC 与 SMC-NO 的响应时间计算结果。
    # 由于随机任务集不保证每次都出现差异，采用多种子搜索一个“可区分”样本即可。
    found_difference = False
    for seed in range(20):
        taskset = generate_taskset(
            num_tasks=12,
            total_util=0.6,
            min_period=10,
            max_period=1000,
            time_scale=100,
            cf=2.0,
            cp=0.5,
            seed=seed,
            deadline_mode="implicit",
            criticality_assignment="bernoulli",
            lo_hi_budget_policy="scaled_by_cf",
        )
        ordered = sort_by_deadline_monotonic(taskset)
        for idx, task in enumerate(ordered):
            if task.criticality is not Criticality.HI:
                continue
            r_smc = compute_smc_response_time(task, ordered[:idx])
            r_smc_no = compute_smc_no_response_time(task, ordered[:idx])
            if r_smc != r_smc_no:
                found_difference = True
                break
        if found_difference:
            break

    assert found_difference is True
