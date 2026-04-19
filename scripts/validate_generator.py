"""阶段3：任务集生成器统计校验脚本。

脚本目标：
1. 批量生成任务集并统计关键分布；
2. 输出 CSV、图表与 Markdown 报告；
3. 验证生成器是否符合配置预期。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# 允许直接运行脚本时导入本地包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 在无写权限环境中显式指定 matplotlib 缓存目录，避免权限告警。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from amc_py.generator import (
    GenerationConfig,
    load_generation_config,
    make_generation_config,
    generate_taskset,
    taskset_total_util,
)
from amc_py.models import Criticality


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Validate generator distribution statistics")
    parser.add_argument("--mode", choices=["fast", "paper"], default="fast", help="生成器模式")
    parser.add_argument("--config", type=str, default=None, help="自定义 YAML 配置路径")
    parser.add_argument("--num-tasksets", type=int, default=200, help="生成任务集数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/generator_validation",
        help="统计结果输出目录",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> GenerationConfig:
    """根据参数加载最终使用的生成配置。

    优先读取用户显式传入的 YAML；否则回退到 fast/paper 预设。
    """

    if args.config:
        return load_generation_config(args.config)
    return make_generation_config(args.mode)


def summarize_series(series: pd.Series) -> dict[str, float]:
    """生成常用分布摘要统计。"""

    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.5)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
    }


def plot_distributions(taskset_df: pd.DataFrame, task_df: pd.DataFrame, output_path: Path) -> None:
    """绘制生成器关键分布图并落盘。"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(taskset_df["hi_ratio"], bins=20, color="#4C72B0", alpha=0.85)
    axes[0, 0].set_title("HI Ratio Distribution")
    axes[0, 0].set_xlabel("HI ratio per taskset")

    axes[0, 1].hist(task_df["period"], bins=20, color="#55A868", alpha=0.85)
    axes[0, 1].set_title("Period Distribution")
    axes[0, 1].set_xlabel("period")

    axes[1, 0].hist(task_df["deadline_ratio"], bins=20, color="#C44E52", alpha=0.85)
    axes[1, 0].set_title("Deadline/Period Ratio Distribution")
    axes[1, 0].set_xlabel("D/T")

    axes[1, 1].hist(taskset_df["actual_util_lo"], bins=20, color="#8172B2", alpha=0.85)
    axes[1, 1].set_title("Actual LO Utilization Distribution")
    axes[1, 1].set_xlabel("U_LO")

    for ax in axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    config: GenerationConfig,
    num_tasksets: int,
    summary: dict[str, dict[str, float]],
    outputs: dict[str, Path],
) -> None:
    """输出 Markdown 统计报告。"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Generator Validation Report",
                "",
                "## 1. 配置概览",
                "",
                f"- num_tasksets: {num_tasksets}",
                f"- num_tasks: {config.num_tasks}",
                f"- total_util: {config.total_util}",
                f"- period_range: [{config.min_period}, {config.max_period}]",
                f"- time_scale: {config.time_scale}",
                f"- cf: {config.cf}",
                f"- cp: {config.cp}",
                f"- deadline_mode: {config.deadline_mode}",
                f"- deadline_ratio_min: {config.deadline_ratio_min}",
                f"- criticality_assignment: {config.criticality_assignment}",
                f"- lo_hi_budget_policy: {config.lo_hi_budget_policy}",
                "",
                "## 2. 分布摘要",
                "",
                f"- HI ratio: {summary['hi_ratio']}",
                f"- period: {summary['period']}",
                f"- deadline_ratio: {summary['deadline_ratio']}",
                f"- actual_util_lo: {summary['actual_util_lo']}",
                f"- actual_util_hi: {summary['actual_util_hi']}",
                "",
                "## 3. 输出文件",
                "",
                f"- taskset_stats.csv: {outputs['taskset_csv'].as_posix()}",
                f"- task_stats.csv: {outputs['task_csv'].as_posix()}",
                f"- plot: {outputs['plot'].as_posix()}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """执行生成器统计校验主流程。"""

    args = parse_args()
    config = load_config(args)

    if args.num_tasksets <= 0:
        raise ValueError("--num-tasksets 必须为正整数")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taskset_records: list[dict[str, float]] = []
    task_records: list[dict[str, float]] = []

    # 固定 base_seed，并在每个任务集上递增，确保可复现且互相独立。
    # 该校验脚本的目标是观察总体分布，不要求跨任务集“完全无相关”，
    # 因此使用简单稳定的线性种子序列即可。
    for taskset_id in range(args.num_tasksets):
        seed = args.seed + taskset_id
        taskset = generate_taskset(
            num_tasks=config.num_tasks,
            total_util=config.total_util,
            min_period=config.min_period,
            max_period=config.max_period,
            time_scale=config.time_scale,
            cf=config.cf,
            cp=config.cp,
            seed=seed,
            deadline_mode=config.deadline_mode,
            deadline_ratio_min=config.deadline_ratio_min,
            criticality_assignment=config.criticality_assignment,
            lo_hi_budget_policy=config.lo_hi_budget_policy,
        )

        # 统计任务集层指标（HI 占比、离散化后的真实利用率）。
        hi_count = sum(task.criticality is Criticality.HI for task in taskset)
        hi_ratio = hi_count / len(taskset)
        util_lo = taskset_total_util(taskset, mode=Criticality.LO)
        util_hi = taskset_total_util(taskset, mode=Criticality.HI)

        taskset_records.append(
            {
                "taskset_id": taskset_id,
                "seed": seed,
                "hi_count": float(hi_count),
                "hi_ratio": float(hi_ratio),
                "actual_util_lo": float(util_lo),
                "actual_util_hi": float(util_hi),
            }
        )

        # 统计任务层指标（周期、截止期比例、关键级分布）。
        for task in taskset:
            task_records.append(
                {
                    "taskset_id": float(taskset_id),
                    "period": float(task.period),
                    "deadline": float(task.deadline),
                    "deadline_ratio": float(task.deadline / task.period),
                    "criticality_hi": float(task.criticality is Criticality.HI),
                }
            )

    taskset_df = pd.DataFrame(taskset_records)
    task_df = pd.DataFrame(task_records)

    taskset_csv = output_dir / "taskset_stats.csv"
    task_csv = output_dir / "task_stats.csv"
    plot_path = output_dir / "generator_validation_plots.png"
    report_path = output_dir / "generator_validation_report.md"

    taskset_df.to_csv(taskset_csv, index=False)
    task_df.to_csv(task_csv, index=False)
    plot_distributions(taskset_df, task_df, plot_path)

    summary = {
        "hi_ratio": summarize_series(taskset_df["hi_ratio"]),
        "period": summarize_series(task_df["period"]),
        "deadline_ratio": summarize_series(task_df["deadline_ratio"]),
        "actual_util_lo": summarize_series(taskset_df["actual_util_lo"]),
        "actual_util_hi": summarize_series(taskset_df["actual_util_hi"]),
    }
    write_report(
        report_path=report_path,
        config=config,
        num_tasksets=args.num_tasksets,
        summary=summary,
        outputs={"taskset_csv": taskset_csv, "task_csv": task_csv, "plot": plot_path},
    )

    print("=== 生成器统计校验完成 ===")
    print(f"taskset_stats.csv: {taskset_csv.resolve()}")
    print(f"task_stats.csv: {task_csv.resolve()}")
    print(f"plots: {plot_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
