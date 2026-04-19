"""响应时间分析（Response-Time Analysis, RTA）基础实现（阶段 B）。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil

from .models import Criticality, SchedulabilityResult, Task


def solve_fixed_point(
    recurrence_fn: Callable[[int], int],
    start_value: int,
    deadline: int,
    max_iter: int = 1000,
) -> int | None:
    """通用固定点求解器。

    参数说明：
    - recurrence_fn: 递推函数 f(R)，例如响应时间方程右侧。
    - start_value: 初值，通常取任务自身执行时间 C_i。
    - deadline: 截止期上界，超过则视为不可调度。
    - max_iter: 最大迭代次数，防止异常情况下无限循环。

    返回：
    - 收敛且不超过 deadline：返回固定点值。
    - 未收敛或中途超过 deadline：返回 None。
    """

    if start_value <= 0:
        raise ValueError("start_value 必须为正整数")
    if deadline <= 0:
        raise ValueError("deadline 必须为正整数")
    if max_iter <= 0:
        raise ValueError("max_iter 必须为正整数")

    current = start_value
    for _ in range(max_iter):
        next_value = recurrence_fn(current)

        # 响应时间分析中，若中间结果已经超过截止期，可提前判失败。
        if next_value > deadline:
            return None

        # 到达固定点：R_{k+1} = R_k。
        if next_value == current:
            return next_value

        current = next_value

    # 达到最大迭代次数仍未收敛。
    return None


def compute_r_lo(task: Task, higher_priority_tasks: Sequence[Task]) -> int | None:
    """计算任务在 LO 模式下的响应时间 R_LO。

    公式思想（经典固定优先级 RTA）：
    R = C_i(LO) + Σ_{j∈hp(i)} ceil(R / T_j) * C_j(LO)
    """

    def recurrence(r_value: int) -> int:
        interference = 0
        for hp_task in higher_priority_tasks:
            jobs = ceil(r_value / hp_task.period)
            interference += jobs * hp_task.c_lo
        return task.c_lo + interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task.c_lo,
        deadline=task.deadline,
    )


def analyze_lo_mode(tasks: Sequence[Task], ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对整个任务集执行 LO 模式分析。

    约定：
    - ordered_tasks 必须是按优先级从高到低排序后的列表。
    - LO 模式下，所有任务都按 C(LO) 计算。
    """

    _ = tasks  # 预留参数，便于后续保持统一接口。

    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        r_lo = compute_r_lo(task, ordered_tasks[:idx])
        if r_lo is None:
            return SchedulabilityResult(
                schedulable=False,
                method="lo_mode",
                response_times=response_times,
                details=f"任务 {task.name} 在 LO 模式下超过截止期",
            )
        response_times[task.name] = r_lo

    return SchedulabilityResult(
        schedulable=True,
        method="lo_mode",
        response_times=response_times,
        details="LO 模式分析通过",
    )


def compute_r_hi(task: Task, higher_priority_hi_tasks: Sequence[Task]) -> int | None:
    """计算 HI 任务在 HI 模式下的响应时间 R_HI。

    公式思想：
    R = C_i(HI) + Σ_{j∈hp_HI(i)} ceil(R / T_j) * C_j(HI)

    其中 hp_HI(i) 仅包含高优先级 HI 任务。
    """

    if task.criticality is not Criticality.HI:
        raise ValueError("compute_r_hi 仅适用于 HI 任务")

    def recurrence(r_value: int) -> int:
        interference = 0
        for hp_task in higher_priority_hi_tasks:
            jobs = ceil(r_value / hp_task.period)
            interference += jobs * hp_task.c_hi
        return task.c_hi + interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task.c_hi,
        deadline=task.deadline,
    )


def analyze_hi_mode(tasks: Sequence[Task], ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对任务集执行 HI 模式分析。

    规则：
    - 仅 HI 任务需要被分析并满足截止期。
    - 干扰只来自更高优先级的 HI 任务。
    """

    _ = tasks

    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        if task.criticality is Criticality.LO:
            continue

        hp_hi_tasks = [
            hp_task
            for hp_task in ordered_tasks[:idx]
            if hp_task.criticality is Criticality.HI
        ]

        r_hi = compute_r_hi(task, hp_hi_tasks)
        if r_hi is None:
            return SchedulabilityResult(
                schedulable=False,
                method="hi_mode",
                response_times=response_times,
                details=f"任务 {task.name} 在 HI 模式下超过截止期",
            )
        response_times[task.name] = r_hi

    return SchedulabilityResult(
        schedulable=True,
        method="hi_mode",
        response_times=response_times,
        details="HI 模式分析通过",
    )


def _single_budget_wcet(task: Task) -> int:
    """返回论文 CrMPO baseline 的“单一执行时间参数”。

    定义：
    - HI 任务：使用 C(HI)
    - LO 任务：使用 C(LO)
    """

    return task.c_hi if task.criticality is Criticality.HI else task.c_lo


def compute_r_classic_single_budget(task: Task, higher_priority_tasks: Sequence[Task]) -> int | None:
    """计算经典基线（Classic）响应时间。

    依据 Baruah 等 RTSS'11 论文中 CrMPO 对比方法描述：
    - 每个任务仅使用一个执行时间参数（HI 用 C(HI)，LO 用 C(LO)）。
    - 然后套用标准固定优先级响应时间方程。

    公式：
    R = C_i + Σ_{j∈hp(i)} ceil(R / T_j) * C_j
    """

    task_cost = _single_budget_wcet(task)

    def recurrence(r_value: int) -> int:
        interference = 0
        for hp_task in higher_priority_tasks:
            jobs = ceil(r_value / hp_task.period)
            interference += jobs * _single_budget_wcet(hp_task)
        return task_cost + interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task_cost,
        deadline=task.deadline,
    )


def analyze_classic_hi_budget(
    tasks: Sequence[Task],
    ordered_tasks: Sequence[Task],
) -> SchedulabilityResult:
    """对整个任务集执行 Classic 单一预算分析。"""

    _ = tasks
    response_times: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        r_value = compute_r_classic_single_budget(task, ordered_tasks[:idx])
        if r_value is None:
            return SchedulabilityResult(
                schedulable=False,
                method="classic_hi",
                response_times=response_times,
                details=f"任务 {task.name} 在 classic_single_budget 分析下超过截止期",
            )
        response_times[task.name] = r_value

    return SchedulabilityResult(
        schedulable=True,
        method="classic_hi",
        response_times=response_times,
        details="classic_single_budget 分析通过",
    )
