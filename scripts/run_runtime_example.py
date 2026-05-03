"""运行时仿真示例脚本（阶段运行时模拟 · 第 6 轮）。

脚本目标：
1. 演示 nominal scenario（所有 job 按 C(LO) 执行）；
2. 演示 single HI overrun scenario（指定 HI job 超限触发 LO->HI 切换）；
3. 用统一接口 `compare_static_and_runtime` 同时输出静态分析与运行时仿真摘要。

使用方式（推荐在仓库根目录执行）：
    PYTHONPATH=. python scripts/run_runtime_example.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# 为了支持“直接运行脚本”的场景，这里把项目根目录显式加入 sys.path。
# 这样即使用户尚未执行 `pip install -e .`，也能导入本地 amc_py 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py import (
    RuntimeConfig,
    compare_static_and_runtime,
    make_nominal_scenario,
    make_single_hi_overrun_scenario,
)
from amc_py.models import Criticality, Task


def _build_demo_tasks() -> list[Task]:
    """构造示例任务集。

    任务集设计要点：
    - 包含 1 个 HI 任务 `tau_hi`，可用于触发模式切换；
    - 包含 2 个 LO 任务，便于观察切换后 drop/suppress 行为；
    - 参数保持温和，nominal 场景下在给定时间窗内通常不会 miss。
    """

    return [
        Task(
            name="tau_hi",
            period=10,
            deadline=10,
            c_lo=1,
            c_hi=3,
            criticality=Criticality.HI,
        ),
        Task(
            name="tau_lo_a",
            period=12,
            deadline=12,
            c_lo=2,
            c_hi=2,
            criticality=Criticality.LO,
        ),
        Task(
            name="tau_lo_b",
            period=20,
            deadline=20,
            c_lo=2,
            c_hi=2,
            criticality=Criticality.LO,
        ),
    ]


def _print_taskset(tasks: list[Task]) -> None:
    """打印任务集参数，便于人工核查输入。"""

    print("任务集定义：")
    for task in tasks:
        print(
            f"  - {task.name}: T={task.period}, D={task.deadline}, "
            f"C_LO={task.c_lo}, C_HI={task.c_hi}, L={task.criticality.value}"
        )


def _print_job_table(comparison) -> None:
    """打印已释放 jobs 的关键字段。

    表头字段说明：
    - `release/deadline`: 绝对释放时刻与绝对截止时刻；
    - `executed/actual`: 仿真结束时累计执行量与目标实际执行量；
    - `completion`: 完成时刻（None 表示未完成）；
    - `dropped/drop_time`: 是否被 HI 切换丢弃及丢弃时刻。
    """

    print("job 执行摘要：")
    for job in comparison.runtime_result.jobs:
        print(
            f"  - {job.task.name}[{job.release_index}] "
            f"release={job.release_time}, deadline={job.absolute_deadline}, "
            f"executed={job.executed_time}/{job.actual_cost}, "
            f"completion={job.completion_time}, dropped={job.dropped}, drop_time={job.drop_time}"
        )


def _run_one_case(
    case_name: str,
    tasks: list[Task],
    method: str,
    priority_policy: str,
    scenario,
    config: RuntimeConfig,
) -> None:
    """执行单个场景并打印“静态分析 vs 运行时仿真”的并列结果。"""

    print(f"\n=== 场景：{case_name} ===")
    print(f"method={method}, priority_policy={priority_policy}")
    print(f"scenario={scenario.name}")

    comparison = compare_static_and_runtime(
        tasks=tasks,
        method=method,
        priority_policy=priority_policy,
        scenario=scenario,
        config=config,
    )

    print(f"ordered_task_names={comparison.ordered_task_names}")
    print(
        "static_result: "
        f"schedulable={comparison.static_result.schedulable}, "
        f"details={comparison.static_result.details}"
    )
    print(
        "runtime_summary: "
        f"final_mode={comparison.runtime_result.final_mode.value}, "
        f"mode_switched={comparison.mode_switched()}, "
        f"deadline_missed={comparison.deadline_missed()}, "
        f"num_misses={len(comparison.runtime_result.deadline_misses)}, "
        f"num_dropped={len(comparison.runtime_result.dropped_jobs())}"
    )

    if comparison.runtime_result.mode_switch is None:
        print("mode_switch_event=None")
    else:
        event = comparison.runtime_result.mode_switch
        print(
            "mode_switch_event="
            f"(time={event.switch_time}, task={event.triggering_task}, "
            f"release_index={event.triggering_release_index}, "
            f"executed_at_switch={event.executed_at_switch})"
        )

    if comparison.runtime_result.deadline_misses:
        print("deadline_misses:")
        for miss in comparison.runtime_result.deadline_misses:
            print(
                f"  - task={miss.task}, release_index={miss.release_index}, "
                f"deadline={miss.absolute_deadline}, mode={miss.mode_at_miss.value}, "
                f"executed_at_miss={miss.executed_at_miss}"
            )
    else:
        print("deadline_misses=None")

    _print_job_table(comparison)


def main() -> None:
    """示例主入口：连续演示 nominal 与 single HI overrun 两个场景。"""

    tasks = _build_demo_tasks()
    method = "amc_rtb"
    priority_policy = "dm"
    # 固定一个中等时间窗，既能看到多个 release，又不会让输出过长。
    config = RuntimeConfig(end_time=30, capture_trace=False)

    print("=== Runtime Simulation Example ===")
    _print_taskset(tasks)

    # 场景 1：nominal（不会触发模式切换）。
    _run_one_case(
        case_name="nominal",
        tasks=tasks,
        method=method,
        priority_policy=priority_policy,
        scenario=make_nominal_scenario(),
        config=config,
    )

    # 场景 2：single HI overrun（指定 tau_hi 的第 0 次 release 跑到 C(HI)）。
    _run_one_case(
        case_name="single_hi_overrun",
        tasks=tasks,
        method=method,
        priority_policy=priority_policy,
        scenario=make_single_hi_overrun_scenario(
            task_name="tau_hi",
            release_index=0,
            overrun_to="c_hi",
        ),
        config=config,
    )


if __name__ == "__main__":
    main()
