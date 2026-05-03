"""AMC-rtb 与 AMC-max 响应时间分析（阶段 C）。"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor

from .models import Criticality, SchedulabilityResult, Task
from .rta import compute_r_hi, compute_r_lo, solve_fixed_point


def build_design_r_lo_map(ordered_tasks: Sequence[Task]) -> dict[str, int]:
    """按优先级顺序计算并返回设计时 `R_LO` 映射。

    说明：
    - 该函数用于运行时 safety checker 的前置输入；
    - 只返回 HI 任务的 `R_LO`，与运行时检查公式需求一致；
    - 若任意任务 `R_LO` 不可解，则立即抛出异常，避免静默退化。
    """

    r_lo_map: dict[str, int] = {}
    for idx, task in enumerate(ordered_tasks):
        r_lo = compute_r_lo(task, ordered_tasks[:idx])
        if r_lo is None:
            raise ValueError(f"任务 {task.name} 的设计时 R_LO 不可解，无法构造运行时安全检查器")
        if task.criticality is Criticality.HI:
            r_lo_map[task.name] = r_lo
    return r_lo_map


def compute_amc_rtb_response_time(
    task: Task,
    higher_priority_tasks: Sequence[Task],
    r_lo_map: dict[str, int],
) -> int | None:
    """计算 AMC-rtb（Method 1）下单任务响应时间。

    公式实现思路：
    1. 若任务为 LO，则退化为 R_LO。
    2. 若任务为 HI，模式切换方程为：
       R = C_i(HI)
         + Σ_{j∈hp_LO(i)} ceil(R_LO_i / T_j) * C_j(LO)
         + Σ_{j∈hp_HI(i)} ceil(R / T_j)   * C_j(HI)
    3. 最终返回 max(R_LO_i, R_HI_i, R_MC_i)。
    """

    r_lo_i = r_lo_map.get(task.name)
    if r_lo_i is None:
        raise ValueError(f"r_lo_map 中缺少任务 {task.name} 的 R_LO")

    # LO 任务在 AMC-rtb 中直接使用 LO 响应时间。
    if task.criticality is Criticality.LO:
        return r_lo_i if r_lo_i <= task.deadline else None

    hp_lo = [hp_task for hp_task in higher_priority_tasks if hp_task.criticality is Criticality.LO]
    hp_hi = [hp_task for hp_task in higher_priority_tasks if hp_task.criticality is Criticality.HI]

    def recurrence(r_value: int) -> int:
        interference_lo = sum(ceil(r_lo_i / hp_task.period) * hp_task.c_lo for hp_task in hp_lo)
        interference_hi = sum(ceil(r_value / hp_task.period) * hp_task.c_hi for hp_task in hp_hi)
        return task.c_hi + interference_lo + interference_hi

    # 以 C_i(HI) 为起点求解模式切换固定点。
    r_mc = solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task.c_hi,
        deadline=task.deadline,
    )
    if r_mc is None:
        return None

    # HI-only 方程下的响应时间上界。
    r_hi = compute_r_hi(task, hp_hi)
    if r_hi is None:
        return None

    final_r = max(r_lo_i, r_hi, r_mc)
    return final_r if final_r <= task.deadline else None


def amc_rtb_sched_test(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对已排序任务执行 AMC-rtb 可调度性测试。"""

    r_lo_map: dict[str, int] = {}
    response_times: dict[str, int] = {}

    # 第一阶段：先得到所有任务 R_LO。
    for idx, task in enumerate(ordered_tasks):
        r_lo = compute_r_lo(task, ordered_tasks[:idx])
        if r_lo is None:
            return SchedulabilityResult(
                schedulable=False,
                method="amc_rtb",
                response_times=response_times,
                details=f"任务 {task.name} 的 R_LO 超过截止期",
            )
        r_lo_map[task.name] = r_lo

    # 第二阶段：按优先级顺序计算 AMC-rtb 响应时间。
    for idx, task in enumerate(ordered_tasks):
        r_value = compute_amc_rtb_response_time(task, ordered_tasks[:idx], r_lo_map)
        if r_value is None:
            return SchedulabilityResult(
                schedulable=False,
                method="amc_rtb",
                response_times=response_times,
                details=f"任务 {task.name} 在 AMC-rtb 下超过截止期",
            )
        response_times[task.name] = r_value

    return SchedulabilityResult(
        schedulable=True,
        method="amc_rtb",
        response_times=response_times,
        details="AMC-rtb 测试通过",
    )


def candidate_switch_points(task_i: Task, ordered_tasks: Sequence[Task], r_lo_i: int) -> list[int]:
    """构造 AMC-max 候选模式切换点集合 S。

    根据论文与 mceval 思路：
    - 在 [0, R_LO_i) 区间内，枚举高优先级 LO 任务的释放点 k*T_j。
    - 去重并排序。
    """

    task_idx = ordered_tasks.index(task_i)
    switch_points: set[int] = set()

    for hp_task in ordered_tasks[:task_idx]:
        if hp_task.criticality is Criticality.HI:
            continue

        max_k = ceil(r_lo_i / hp_task.period) - 1
        for k in range(max_k + 1):
            switch_points.add(k * hp_task.period)

    # 若不存在 LO 干扰候选点，仍保留 s=0 以保证函数可计算。
    if not switch_points:
        switch_points.add(0)

    return sorted(switch_points)


def compute_M(task_k: Task, s: int, t: int) -> int:
    """计算 AMC-max 公式中的 M(k, s, t)。

    公式（双关键级，隐式截止期/一般截止期兼容）：
    M = min(
        ceil((t - s - (T_k - D_k)) / T_k + 1),
        ceil(t / T_k)
    )
    """

    m1 = ceil(((t - s - (task_k.period - task_k.deadline)) / task_k.period) + 1)
    m2 = ceil(t / task_k.period)

    # M 表示 HI 执行预算可出现的作业个数，物理上不应为负。
    return max(0, min(m1, m2))


def compute_amc_max_response_time_for_switch(task_i: Task, ordered_tasks: Sequence[Task], s: int) -> int | None:
    """在固定切换点 s 下计算 AMC-max 响应时间上界。"""

    if task_i.criticality is Criticality.LO:
        task_idx = ordered_tasks.index(task_i)
        return compute_r_lo(task_i, ordered_tasks[:task_idx])

    task_idx = ordered_tasks.index(task_i)
    hp_tasks = ordered_tasks[:task_idx]
    hp_lo = [task for task in hp_tasks if task.criticality is Criticality.LO]
    hp_hi = [task for task in hp_tasks if task.criticality is Criticality.HI]

    lo_interference = sum((floor(s / hp_task.period) + 1) * hp_task.c_lo for hp_task in hp_lo)

    def recurrence(r_value: int) -> int:
        hi_interference = 0
        for hp_task in hp_hi:
            m_value = compute_M(hp_task, s, r_value)
            jobs_total = ceil(r_value / hp_task.period)
            hi_interference += m_value * hp_task.c_hi + (jobs_total - m_value) * hp_task.c_lo
        return task_i.c_hi + lo_interference + hi_interference

    return solve_fixed_point(
        recurrence_fn=recurrence,
        start_value=task_i.c_hi + lo_interference,
        deadline=task_i.deadline,
    )


def compute_amc_max_response_time(task_i: Task, ordered_tasks: Sequence[Task]) -> int | None:
    """计算任务在 AMC-max（Method 2）下的响应时间。"""

    task_idx = ordered_tasks.index(task_i)
    hp_tasks = ordered_tasks[:task_idx]

    r_lo_i = compute_r_lo(task_i, hp_tasks)
    if r_lo_i is None:
        return None

    if task_i.criticality is Criticality.LO:
        return r_lo_i

    switch_points = candidate_switch_points(task_i, ordered_tasks, r_lo_i)

    r_max = task_i.c_hi
    for s in switch_points:
        r_s = compute_amc_max_response_time_for_switch(task_i, ordered_tasks, s)
        if r_s is None:
            return None
        r_max = max(r_max, r_s)

    hp_hi = [task for task in hp_tasks if task.criticality is Criticality.HI]
    r_hi_i = compute_r_hi(task_i, hp_hi)
    if r_hi_i is None:
        return None

    final_r = max(r_max, r_lo_i, r_hi_i)
    return final_r if final_r <= task_i.deadline else None


def amc_max_sched_test(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """对已排序任务执行 AMC-max 可调度性测试。"""

    response_times: dict[str, int] = {}
    for task in ordered_tasks:
        r_value = compute_amc_max_response_time(task, ordered_tasks)
        if r_value is None:
            return SchedulabilityResult(
                schedulable=False,
                method="amc_max",
                response_times=response_times,
                details=f"任务 {task.name} 在 AMC-max 下超过截止期",
            )
        response_times[task.name] = r_value

    return SchedulabilityResult(
        schedulable=True,
        method="amc_max",
        response_times=response_times,
        details="AMC-max 测试通过",
    )
