"""UB-H&L 可行性测试（阶段 B）。"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Criticality, Task
from .priorities import sort_by_deadline_monotonic
from .rta import analyze_hi_mode, analyze_lo_mode


def ub_l_test(tasks: Sequence[Task]) -> bool:
    """执行 UB-L（LO 模式可行性）测试。

    实现策略：
    - 使用 DM 排序。
    - 对排序后的任务集运行 LO 模式响应时间分析。
    """

    ordered_tasks = sort_by_deadline_monotonic(tasks)
    result = analyze_lo_mode(tasks, ordered_tasks)
    return result.schedulable


def ub_h_test(tasks: Sequence[Task]) -> bool:
    """执行 UB-H（HI 模式可行性）测试。

    实现策略：
    - 使用 DM 排序。
    - 仅要求 HI 任务在 HI 模式分析通过。
    """

    ordered_tasks = sort_by_deadline_monotonic(tasks)
    result = analyze_hi_mode(tasks, ordered_tasks)
    return result.schedulable


def ub_hl_test(tasks: Sequence[Task]) -> bool:
    """执行 UB-H&L 综合测试。

    判定规则：
    - LO 模式通过 且 HI 模式通过，才认为任务集通过 UB-H&L。
    """

    has_hi_task = any(task.criticality is Criticality.HI for task in tasks)

    # 对仅 LO 的任务集，UB-H 等价于 vacuously true；只需检查 UB-L。
    if not has_hi_task:
        return ub_l_test(tasks)

    return ub_l_test(tasks) and ub_h_test(tasks)
