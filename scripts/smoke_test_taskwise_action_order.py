"""校验 taskwise single 动作空间的槽位顺序。

本脚本只做一件事：验证 `build_budget_action_space(..., action_space="single",
include_explicit_noop=True)` 的输出顺序与 taskwise Q 网络的拼接顺序完全一致。
如果这里漂移，taskwise 网络输出就会和环境动作语义错位，因此必须独立做守卫。
"""

from __future__ import annotations

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space


def main() -> None:
    """构造 3 个任务并检查 increase / decrease / noop 的固定槽位。"""

    ordered_tasks = [
        Task(name="task_0", period=10, deadline=10, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task(name="task_1", period=20, deadline=20, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task(name="task_2", period=40, deadline=40, c_lo=4, c_hi=4, criticality=Criticality.HI),
    ]
    actions = build_budget_action_space(
        ordered_tasks,
        action_space="single",
        include_explicit_noop=True,
        budget_increase_ratio=0.025,
        budget_decrease_ratio=0.015,
    )

    task_count = len(ordered_tasks)
    for task_index in range(task_count):
        action = actions[task_index]
        assert action.increase_idx == task_index, f"increase 槽位 {task_index} 未对齐到对应任务"
        assert action.decrease_indices == (), f"increase 槽位 {task_index} 不应包含 decrease 任务"

    for task_index in range(task_count):
        action = actions[task_count + task_index]
        assert action.increase_idx is None, f"decrease 槽位 {task_index} 不应包含 increase 任务"
        assert action.decrease_indices == (task_index,), f"decrease 槽位 {task_index} 未对齐到对应任务"

    assert actions[2 * task_count].is_noop, "最后一个动作槽位必须是 explicit noop"
    print("Taskwise action order: PASS")


if __name__ == "__main__":
    main()
