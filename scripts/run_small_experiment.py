"""小规模实验脚本（阶段 D）。

脚本用途：
1. 运行一个小规模 utilization sweep
2. 导出原始结果 CSV
3. 计算并导出 weighted schedulability CSV
4. 生成两张示例图
"""

from __future__ import annotations

from pathlib import Path
import sys

# 把项目根目录加入模块搜索路径，确保直接执行脚本时可导入 amc_py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.experiments import (
    compute_weighted_schedulability,
    plot_schedulable_percentage,
    plot_weighted_schedulability,
    run_utilization_sweep,
)


def main() -> None:
    """执行小规模 sweep 实验并落盘结果文件。"""

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用较小样本量，确保本地几秒内可完成。
    util_values = [0.3, 0.5, 0.7, 0.9]
    results = run_utilization_sweep(
        util_values=util_values,
        num_tasksets=20,
        num_tasks=8,
        cf=2.0,
        cp=0.5,
        min_period=10,
        max_period=1000,
        method="amc_rtb",
        priority_policy="dm",
        seed=123,
    )

    raw_csv = output_dir / "small_util_sweep.csv"
    results.to_csv(raw_csv, index=False)

    weighted = compute_weighted_schedulability(results, group_col="sweep_value")
    weighted_csv = output_dir / "small_util_sweep_weighted.csv"
    weighted.to_csv(weighted_csv, index=False)

    fig1 = plot_schedulable_percentage(
        results,
        x_col="sweep_value",
        output_path=output_dir / "small_util_schedulable_percentage.png",
        title="Small Utilization Sweep - Schedulable Percentage",
    )
    fig2 = plot_weighted_schedulability(
        weighted,
        x_col="sweep_value",
        output_path=output_dir / "small_util_weighted_schedulability.png",
        title="Small Utilization Sweep - Weighted Schedulability",
    )

    print("=== 小规模实验完成 ===")
    print(f"原始结果 CSV: {raw_csv.resolve()}")
    print(f"加权统计 CSV: {weighted_csv.resolve()}")
    print(f"可调度率图: {fig1.resolve() if fig1 else '未生成'}")
    print(f"加权可调度率图: {fig2.resolve() if fig2 else '未生成'}")


if __name__ == "__main__":
    main()
