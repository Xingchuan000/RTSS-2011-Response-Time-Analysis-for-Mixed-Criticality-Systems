"""SMC 与 SMC-no 响应时间分析（阶段 B）。"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from .models import Criticality, SchedulabilityResult, Task
from .rta import solve_fixed_point


def _wcet_at_criticality(task: Task, level: Criticality) -> int:
    """返回任务在给定关键级下采用的 WCET。"""

    return task.c_hi if level is Criticality.HI else task.c_lo


def compute_smc_response_time(task: Task, higher_priority_tasks: Sequence[Task]) -> int | None:
    """计算 SMC 响应时间。

    SMC 递推式（双关键级情形）：
    R = C_i(L_i) + Σ_{j∈hp(i)} ceil(R/T_j) * C_j(min(L_i, L_j))

    直观解释：
    - 计算目标任务 i 时，干扰任务 j 的执行预算不会超过两者关键级中的“较低级”。
    - 因此当 i 是 HI、j 是 LO 时，j 仅按 C_j(LO) 计入干扰。
    """

    task_level = task.criticality
    task_cost = _wcet_at_criticality(task, task_level)

    def recurrence(r_value: int) -> int:
        interference = 0
        for hp_task in higher_priority_tasks:
            hp_jobs = ceil(r_value / hp_task.period)
            effective_level = (
                Criticality.LO
                if hp_task.criticality is Criticality.LO or task_level is Criticality.LO
                else Criticality.HI
            )
            interference += hp_jobs * _wcet_at_criticality(hp_task, effective_level)
        return task_cost + interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task_cost,
        deadline=task.deadline,
    )


def smc_sched_test(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对已排序任务执行 SMC 可调度性测试。"""

    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        r_value = compute_smc_response_time(task, ordered_tasks[:idx])
        if r_value is None:
            return SchedulabilityResult(
                schedulable=False,
                method="smc",
                response_times=response_times,
                details=f"任务 {task.name} 在 SMC 下超过截止期",
            )
        response_times[task.name] = r_value

    return SchedulabilityResult(
        schedulable=True,
        method="smc",
        response_times=response_times,
        details="SMC 测试通过",
    )


def compute_smc_no_response_time(task: Task, higher_priority_tasks: Sequence[Task]) -> int | None:
    """计算 SMC-no 响应时间。

    SMC-no 递推式：
    R = C_i(L_i) + Σ_{j∈hp(i)} ceil(R/T_j) * C_j(L_i)

    与 SMC 的差异：
    - 不再使用 min(L_i, L_j)，而是统一按目标任务 i 的关键级取干扰 WCET。
    - 因此当 i 是 HI 时，LO 高优先级任务也可能以 C_j(HI) 参与干扰，通常更保守。
    """

    task_level = task.criticality
    task_cost = _wcet_at_criticality(task, task_level)

    def recurrence(r_value: int) -> int:
        interference = 0
        for hp_task in higher_priority_tasks:
            hp_jobs = ceil(r_value / hp_task.period)
            interference += hp_jobs * _wcet_at_criticality(hp_task, task_level)
        return task_cost + interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task_cost,
        deadline=task.deadline,
    )


def smc_no_sched_test(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对已排序任务执行 SMC-no 可调度性测试。"""

    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        r_value = compute_smc_no_response_time(task, ordered_tasks[:idx])
        if r_value is None:
            return SchedulabilityResult(
                schedulable=False,
                method="smc_no",
                response_times=response_times,
                details=f"任务 {task.name} 在 SMC-no 下超过截止期",
            )
        response_times[task.name] = r_value

    return SchedulabilityResult(
        schedulable=True,
        method="smc_no",
        response_times=response_times,
        details="SMC-no 测试通过",
    )
