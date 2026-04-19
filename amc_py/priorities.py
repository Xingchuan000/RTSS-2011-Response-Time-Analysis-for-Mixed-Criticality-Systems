"""优先级分配与排序工具（阶段 B）。

本模块提供静态优先级排序函数：
1. Deadline Monotonic（DM）
2. Criticality Monotonic（CM）
3. CrMPO（关键级优先，再按截止期）
"""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from typing import Callable

from .models import Criticality, PriorityAssignmentResult, SchedulabilityResult, Task


def _criticality_rank(task: Task) -> int:
    """将关键级映射为可排序的整数。

    约定：
    - HI 任务优先级排序时应排在 LO 前面，因此 HI 的 rank 更小。
    - LO 的 rank 更大。
    """

    return 0 if task.criticality is Criticality.HI else 1


def sort_by_deadline_monotonic(tasks: Iterable[Task]) -> list[Task]:
    """按截止期升序排序（截止期越小，静态优先级越高）。"""

    return sorted(tasks, key=lambda task: task.deadline)


def sort_by_criticality_monotonic(tasks: Iterable[Task]) -> list[Task]:
    """按关键级排序（HI 优先于 LO）。

    这里保持 Python 稳定排序特性：
    - 关键级相同的任务，保持输入顺序不变。
    """

    return sorted(tasks, key=_criticality_rank)


def sort_by_crmpo(tasks: Iterable[Task]) -> list[Task]:
    """CrMPO 排序：先按关键级（HI 在前），再按截止期（小者在前）。

    该行为与 mceval 参考实现一致：
    - primary key: criticality (descending by level semantics)
    - secondary key: deadline (ascending)
    """

    return sorted(tasks, key=lambda task: (_criticality_rank(task), task.deadline))


def reindex_priorities(tasks: Iterable[Task]) -> dict[str, int]:
    """为已排序任务重建优先级编号。

    返回字典含义：
    - key: 任务名
    - value: 静态优先级索引（0 表示最高优先级）
    """

    priority_map: dict[str, int] = {}
    for idx, task in enumerate(tasks):
        priority_map[task.name] = idx
    return priority_map


def _to_schedulable_flag(result: SchedulabilityResult | bool) -> bool:
    """统一提取可调度性布尔值。"""

    if isinstance(result, bool):
        return result
    return result.schedulable


def lowest_priority_candidates(tasks: Iterable[Task]) -> list[Task]:
    """返回可作为当前最低优先级的候选任务列表。

    该函数目前直接返回输入顺序的浅拷贝，便于后续扩展剪枝策略。
    """

    return list(tasks)


def _move_task_to_index(tasks: list[Task], source_idx: int, target_idx: int) -> list[Task]:
    """把列表中 source_idx 的任务移动到 target_idx 位置并返回新列表。"""

    updated = tasks.copy()
    task = updated.pop(source_idx)
    updated.insert(target_idx, task)
    return updated


def is_task_lowest_priority_feasible(
    ordered_tasks: list[Task],
    candidate_idx: int,
    lowest_priority_idx: int,
    lowest_priority_test_fn: Callable[..., SchedulabilityResult | bool],
) -> tuple[bool, list[Task]]:
    """判断候选任务是否可以被放到当前最低优先级位置。

    这是标准 Audsley 语义中的关键操作：
    - 在“尚未分配优先级”的候选集合里，选择一个任务暂时放到当前最低优先级。
    - 只验证“该候选任务在当前假设下是否可调度”，若通过则该任务可被固定在该层级。

    返回：
    - 第一个值：是否可行。
    - 第二个值：尝试后的任务顺序（便于调用方在可行时直接复用）。
    """

    trial_order = _move_task_to_index(ordered_tasks, candidate_idx, lowest_priority_idx)
    # 兼容旧回调签名：
    # - 新签名：fn(trial_order, lowest_priority_idx)
    # - 旧签名：fn(trial_order)
    param_count = len(inspect.signature(lowest_priority_test_fn).parameters)
    if param_count >= 2:
        result = lowest_priority_test_fn(trial_order, lowest_priority_idx)
    else:
        result = lowest_priority_test_fn(trial_order)
    return _to_schedulable_flag(result), trial_order


def audsley_opa(
    tasks: Iterable[Task],
    lowest_priority_test_fn: Callable[..., SchedulabilityResult | bool],
) -> PriorityAssignmentResult:
    """执行 Audsley OPA 最优优先级分配。

    输入：
    - tasks: 待分配任务集合（无需预排序）。
    - lowest_priority_test_fn:
      - 推荐签名：`fn(trial_order, lowest_priority_idx) -> bool | SchedulabilityResult`；
      - 兼容签名：`fn(trial_order) -> bool | SchedulabilityResult`（会退化为旧行为）。

    输出：
    - PriorityAssignmentResult：
      - success=True 时，`priorities` 给出每个任务的最终优先级索引（0 为最高）。
      - success=False 时，`priorities` 为空。
    """

    ordered = list(tasks)
    n = len(ordered)

    # 从最低优先级位置（n-1）向最高优先级位置（0）逐层确定任务。
    for target_idx in range(n - 1, -1, -1):
        assigned = False

        for candidate_idx in range(target_idx + 1):
            # 每轮仅验证“候选任务作为当前最低优先级”是否可行；
            # 这正是 Audsley OPA 与“直接测试完整固定顺序”的核心区别。
            schedulable, trial_order = is_task_lowest_priority_feasible(
                ordered_tasks=ordered,
                candidate_idx=candidate_idx,
                lowest_priority_idx=target_idx,
                lowest_priority_test_fn=lowest_priority_test_fn,
            )

            if schedulable:
                ordered = trial_order
                assigned = True
                break

        if not assigned:
            return PriorityAssignmentResult(
                success=False,
                method="opa",
                priorities={},
                details=f"无法为优先级位置 {target_idx} 找到可行候选任务",
            )

    return PriorityAssignmentResult(
        success=True,
        method="opa",
        priorities=reindex_priorities(ordered),
        details="OPA 分配成功",
    )
