"""单任务集示例脚本（阶段 D）。

脚本用途：
- 快速生成一个随机任务集
- 分别运行多种分析器
- 打印每种分析器的可调度结果与响应时间
"""

from __future__ import annotations

from pathlib import Path
import sys

# 把项目根目录加入模块搜索路径，确保直接执行脚本时可导入 amc_py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.experiments import evaluate_taskset
from amc_py.generator import generate_taskset


def main() -> None:
    """执行单示例并打印结果。"""

    # 这里固定随机种子，确保每次运行结果可复现。
    taskset = generate_taskset(
        num_tasks=6,
        total_util=0.75,
        min_period=10,
        max_period=200,
        cf=2.0,
        cp=0.5,
        seed=42,
    )

    print("=== 单任务集示例 ===")
    for task in taskset:
        print(
            f"{task.name}: T={task.period}, D={task.deadline}, "
            f"C_LO={task.c_lo}, C_HI={task.c_hi}, L={task.criticality.value}"
        )

    methods = ["ub_hl", "smc", "smc_no", "amc_rtb", "amc_max"]
    print("\n=== 分析结果（priority_policy=dm）===")
    for method in methods:
        result = evaluate_taskset(taskset, method=method, priority_policy="dm")
        print(f"[{method}] schedulable={result.schedulable}, details={result.details}")
        print(f"response_times={result.response_times}")


if __name__ == "__main__":
    main()
