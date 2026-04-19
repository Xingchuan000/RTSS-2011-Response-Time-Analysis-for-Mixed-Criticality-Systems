"""RTSS'11 论文 Fig.1~Fig.5 复现实验统一入口（阶段4）。

脚本功能：
1. 统一命令行入口：`--figure fig1|fig2|fig3|fig4|fig5`；
2. 每张图都输出原始数据、聚合结果和图像文件；
3. Fig.2~Fig.4 按论文口径执行“util 层内聚合 + 跨 util 加权聚合”；
4. Fig.5 强制使用 arbitrary deadline（允许 D<T）。

说明：
- 为了可追溯性，脚本会同时把运行参数写入对应的 notes 文档。
- 默认输出根目录为 `outputs/`，各图产物放在 `outputs/figX/` 下。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

# 允许直接运行脚本时导入本地项目模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 在受限环境中显式设置 matplotlib 缓存目录，避免权限告警。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from amc_py.aggregation import aggregate_by_util, aggregate_weighted_schedulability
from amc_py.experiments import evaluate_taskset
from amc_py.generator import (
    DeadlineMode,
    GenerationConfig,
    generate_taskset,
    load_generation_config,
    make_generation_config,
    taskset_total_util,
)
from amc_py.models import Criticality

# 无图形界面环境下强制使用 Agg，避免 CI 或服务器环境报错。
matplotlib.use("Agg")


@dataclass(frozen=True)
class MethodPolicy:
    """记录“分析方法 + 优先级策略 + 图例标签”的绑定关系。"""

    method: str
    priority_policy: str
    label: str


@dataclass(frozen=True)
class FigurePreset:
    """每张图 sweep 参数预设。"""

    figure: str
    x_axis: str
    y_axis_metric: str
    deadline_mode: DeadlineMode
    util_values: list[float]
    cf_values: list[float]
    cp_values: list[float]
    n_values: list[int]
    num_tasks: int | None = None


def default_methods() -> list[MethodPolicy]:
    """返回论文对比中常用的方法集合。"""

    return [
        MethodPolicy("ub_hl", "dm", "UB-H&L"),
        MethodPolicy("amc_max", "opa", "AMC-max"),
        MethodPolicy("amc_rtb", "opa", "AMC-rtb"),
        MethodPolicy("smc", "opa", "SMC"),
        MethodPolicy("smc_no", "opa", "SMC-NO"),
        MethodPolicy("crmpo_baseline", "crmpo", "CrMPO"),
    ]


def _series(start: float, stop: float, step: float) -> list[float]:
    """生成闭区间等差数列，并规避浮点累计误差。"""

    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 3))
        current += step
    return values


def get_figure_preset(figure: str, mode: str) -> FigurePreset:
    """按 figure + mode 返回 sweep 预设。"""

    paper_util_values = _series(0.025, 0.975, 0.025)
    fast_util_values = [0.2, 0.4, 0.6, 0.8]

    presets: dict[tuple[str, str], FigurePreset] = {
        # 论文口径（RTSS'11）
        ("paper", "fig1"): FigurePreset(
            figure="fig1",
            x_axis="util",
            y_axis_metric="schedulable_ratio",
            deadline_mode="implicit",
            util_values=paper_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[],
            num_tasks=20,
        ),
        ("paper", "fig2"): FigurePreset(
            figure="fig2",
            x_axis="cf",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=paper_util_values,
            cf_values=_series(1.0, 5.0, 0.5),
            cp_values=[],
            n_values=[],
            num_tasks=20,
        ),
        ("paper", "fig3"): FigurePreset(
            figure="fig3",
            x_axis="cp",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=paper_util_values,
            cf_values=[],
            cp_values=_series(0.05, 0.95, 0.1),
            n_values=[],
            num_tasks=20,
        ),
        ("paper", "fig4"): FigurePreset(
            figure="fig4",
            x_axis="num_tasks",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=paper_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[8, 24, 40, 56, 72, 88],
            num_tasks=None,
        ),
        ("paper", "fig5"): FigurePreset(
            figure="fig5",
            x_axis="util",
            y_axis_metric="schedulable_ratio",
            deadline_mode="arbitrary_paper",
            util_values=paper_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[],
            num_tasks=20,
        ),
        # fast 模式用于本地快速验证
        ("fast", "fig1"): FigurePreset(
            figure="fig1",
            x_axis="util",
            y_axis_metric="schedulable_ratio",
            deadline_mode="implicit",
            util_values=fast_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[],
            num_tasks=12,
        ),
        ("fast", "fig2"): FigurePreset(
            figure="fig2",
            x_axis="cf",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=fast_util_values,
            cf_values=[1.0, 2.0, 3.0, 4.0],
            cp_values=[],
            n_values=[],
            num_tasks=12,
        ),
        ("fast", "fig3"): FigurePreset(
            figure="fig3",
            x_axis="cp",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=fast_util_values,
            cf_values=[],
            cp_values=[0.2, 0.4, 0.6, 0.8],
            n_values=[],
            num_tasks=12,
        ),
        ("fast", "fig4"): FigurePreset(
            figure="fig4",
            x_axis="num_tasks",
            y_axis_metric="weighted_schedulability",
            deadline_mode="implicit",
            util_values=fast_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[4, 8, 12, 16],
            num_tasks=None,
        ),
        ("fast", "fig5"): FigurePreset(
            figure="fig5",
            x_axis="util",
            y_axis_metric="schedulable_ratio",
            deadline_mode="arbitrary_paper",
            util_values=fast_util_values,
            cf_values=[],
            cp_values=[],
            n_values=[],
            num_tasks=12,
        ),
    }

    key = (mode, figure)
    if key not in presets:
        raise ValueError(f"未找到 preset: mode={mode}, figure={figure}")
    return presets[key]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Reproduce RTSS'11 Fig.1~Fig.5")
    parser.add_argument("--figure", choices=["fig1", "fig2", "fig3", "fig4", "fig5"], required=True)
    parser.add_argument("--mode", choices=["fast", "paper"], default="fast")
    parser.add_argument("--config", type=str, default=None, help="可选：生成器 YAML 配置路径")
    parser.add_argument("--num-tasksets", type=int, default=80, help="每个 sweep 点任务集个数")
    parser.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser.add_argument("--output-root", type=str, default="outputs")
    return parser.parse_args()


def resolve_config(mode: str, config_path: str | None) -> GenerationConfig:
    """解析实验生成配置。

    优先级：
    1. 若传入 `--config`，读取 YAML；
    2. 否则使用 `fast/paper` 预设。
    """

    if config_path:
        return load_generation_config(config_path)
    return make_generation_config(mode)


def make_runtime_generation_config(
    base: GenerationConfig,
    preset: FigurePreset,
    *,
    total_util: float | None = None,
    num_tasks: int | None = None,
    cf: float | None = None,
    cp: float | None = None,
) -> GenerationConfig:
    """构造运行时配置，确保 figure preset 语义真正落到生成器。

    设计目标：
    - base 提供全局默认参数；
    - preset 仅负责覆盖图级别语义（特别是 deadline_mode）；
    - 外层 sweep（util/cf/cp/n）通过具名参数局部覆写。
    这样可以避免“metadata 写了 deadline_mode，但生成器实际没用上”的偏差。
    """

    return GenerationConfig(
        num_tasks=num_tasks if num_tasks is not None else base.num_tasks,
        total_util=total_util if total_util is not None else base.total_util,
        min_period=base.min_period,
        max_period=base.max_period,
        time_scale=base.time_scale,
        cf=cf if cf is not None else base.cf,
        cp=cp if cp is not None else base.cp,
        deadline_mode=preset.deadline_mode,
        deadline_ratio_min=base.deadline_ratio_min,
        criticality_assignment=base.criticality_assignment,
        lo_hi_budget_policy=base.lo_hi_budget_policy,
    )


def deterministic_seed(base_seed: int, outer_idx: int, util_idx: int, taskset_id: int) -> int:
    """为嵌套 sweep 生成稳定且可追溯的局部随机种子。

    通过多层偏移把外层 sweep 维度编码进种子，保证：
    - 同一配置可复现；
    - 不同 sweep 点彼此相对独立。
    """

    return base_seed + outer_idx * 100_000 + util_idx * 1_000 + taskset_id


def evaluate_methods_on_taskset(
    taskset: list,
    methods: list[MethodPolicy],
) -> list[dict[str, Any]]:
    """在同一任务集上评估多个方法，保证横向比较公平。"""

    records: list[dict[str, Any]] = []
    for item in methods:
        result = evaluate_taskset(taskset, method=item.method, priority_policy=item.priority_policy)
        records.append(
            {
                "method": item.method,
                "method_label": item.label,
                "priority_policy": item.priority_policy,
                "schedulable": bool(result.schedulable),
                "details": result.details,
            }
        )
    return records


def plot_multi_method_curve(
    aggregated: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
) -> None:
    """绘制多方法曲线图。"""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method_label, group in aggregated.groupby("method_label"):
        group_sorted = group.sort_values(x_col)
        ax.plot(group_sorted[x_col], group_sorted[y_col], marker="o", linewidth=2, label=method_label)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figure_notes(
    report_path: Path,
    figure: str,
    args: argparse.Namespace,
    config: GenerationConfig,
    preset: FigurePreset,
    output_dir: Path,
) -> None:
    """生成每张图的复现说明文档。"""

    runtime_num_tasks = preset.num_tasks if preset.num_tasks is not None else config.num_tasks

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                f"# {figure.upper()} Reproduction Notes",
                "",
                f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
                f"- figure: {figure}",
                f"- mode: {args.mode}",
                f"- num_tasksets: {args.num_tasksets}",
                f"- seed: {args.seed}",
                f"- num_tasks(default): {config.num_tasks}",
                f"- num_tasks(runtime): {runtime_num_tasks}",
                f"- total_util(default): {config.total_util}",
                f"- period_range: [{config.min_period}, {config.max_period}]",
                f"- cf(default): {config.cf}",
                f"- cp(default): {config.cp}",
                f"- time_scale: {config.time_scale}",
                f"- x_axis: {preset.x_axis}",
                f"- y_axis_metric: {preset.y_axis_metric}",
                f"- deadline_mode(default): {config.deadline_mode}",
                f"- deadline_mode(runtime): {preset.deadline_mode}",
                f"- deadline_ratio_min(default): {config.deadline_ratio_min}",
                f"- criticality_assignment(default): {config.criticality_assignment}",
                f"- lo_hi_budget_policy(default): {config.lo_hi_budget_policy}",
                "",
                "## 输出文件",
                f"- {output_dir / 'raw_results.csv'}",
                f"- {output_dir / 'aggregated_results.csv'}",
                f"- {output_dir / f'{figure}.png'}",
            ]
        ),
        encoding="utf-8",
    )


def write_figure_metadata(
    output_dir: Path,
    figure: str,
    args: argparse.Namespace,
    config: GenerationConfig,
    preset: FigurePreset,
) -> None:
    """输出结构化元数据，避免 figure 指标口径混淆。"""

    generator_profile = f"file:{args.config}" if args.config else f"mode:{args.mode}"
    runtime_num_tasks = preset.num_tasks if preset.num_tasks is not None else config.num_tasks
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "figure_name": figure,
        "mode": args.mode,
        "x_axis": preset.x_axis,
        "y_axis_metric": preset.y_axis_metric,
        "deadline_mode": preset.deadline_mode,
        "generator_profile": generator_profile,
        "criticality_assignment": config.criticality_assignment,
        "num_tasksets": args.num_tasksets,
        "seed": args.seed,
        "num_tasks_runtime": runtime_num_tasks,
        "default_config": {
            "num_tasks": config.num_tasks,
            "total_util": config.total_util,
            "min_period": config.min_period,
            "max_period": config.max_period,
            "time_scale": config.time_scale,
            "cf": config.cf,
            "cp": config.cp,
            "deadline_mode": config.deadline_mode,
            "deadline_ratio_min": config.deadline_ratio_min,
            "criticality_assignment": config.criticality_assignment,
            "lo_hi_budget_policy": config.lo_hi_budget_policy,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_fig1(
    args: argparse.Namespace,
    config: GenerationConfig,
    preset: FigurePreset,
    methods: list[MethodPolicy],
    output_dir: Path,
) -> None:
    """Fig.1：util sweep（同一批任务集上比较多方法）。

    注意：每个 util 点上，多种方法共享同一批任务集，避免横向比较的样本偏差。
    """

    runtime_num_tasks = preset.num_tasks if preset.num_tasks is not None else config.num_tasks
    records: list[dict[str, Any]] = []

    for util_idx, util_value in enumerate(preset.util_values):
        for taskset_id in range(args.num_tasksets):
            seed = deterministic_seed(args.seed, util_idx, 0, taskset_id)
            runtime_cfg = make_runtime_generation_config(
                config,
                preset,
                total_util=float(util_value),
                num_tasks=runtime_num_tasks,
            )
            # 统一使用 runtime_cfg 下发到生成器，避免局部字段遗漏。
            taskset = generate_taskset(
                num_tasks=runtime_cfg.num_tasks,
                total_util=runtime_cfg.total_util,
                min_period=runtime_cfg.min_period,
                max_period=runtime_cfg.max_period,
                time_scale=runtime_cfg.time_scale,
                cf=runtime_cfg.cf,
                cp=runtime_cfg.cp,
                seed=seed,
                deadline_mode=runtime_cfg.deadline_mode,
                deadline_ratio_min=runtime_cfg.deadline_ratio_min,
                criticality_assignment=runtime_cfg.criticality_assignment,
                lo_hi_budget_policy=runtime_cfg.lo_hi_budget_policy,
            )

            base = {
                "figure": "fig1",
                "sweep_type": "util",
                "sweep_value": float(util_value),
                "util_level": float(util_value),
                "taskset_id": taskset_id,
                "seed": seed,
                "num_tasks": runtime_num_tasks,
                "cf": runtime_cfg.cf,
                "cp": runtime_cfg.cp,
                "actual_total_util_lo": taskset_total_util(taskset),
                "actual_total_util_hi": taskset_total_util(taskset, mode=Criticality.HI),
                "deadline_mode": runtime_cfg.deadline_mode,
            }

            for method_record in evaluate_methods_on_taskset(taskset, methods):
                records.append({**base, **method_record})

    raw = pd.DataFrame(records)
    raw.to_csv(output_dir / "raw_results.csv", index=False)

    aggregated = aggregate_by_util(
        raw,
        util_col="sweep_value",
        outer_group_cols=["method", "method_label"],
        weight_col="actual_total_util_lo",
        indicator_col="schedulable",
    )
    aggregated.to_csv(output_dir / "aggregated_results.csv", index=False)

    plot_multi_method_curve(
        aggregated,
        x_col="sweep_value",
        y_col="schedulable_ratio",
        title="Fig.1 Utilization Sweep",
        output_path=output_dir / "fig1.png",
    )


def run_nested_weighted_sweep(
    figure: str,
    sweep_type: str,
    sweep_values: list[float | int],
    args: argparse.Namespace,
    config: GenerationConfig,
    preset: FigurePreset,
    methods: list[MethodPolicy],
    output_dir: Path,
) -> None:
    """Fig.2/3/4：带 util 嵌套层的 weighted sweep。

    核心流程：
    1. 外层 sweep（CF / CP / N）；
    2. 内层 util sweep（论文口径的加权基础）；
    3. 在每个任务集上评估多方法；
    4. 两层聚合：
       - util 层内聚合：`aggregate_by_util`
       - 跨 util 聚合：`aggregate_weighted_schedulability`
    """

    records: list[dict[str, Any]] = []
    default_num_tasks = preset.num_tasks if preset.num_tasks is not None else config.num_tasks

    for outer_idx, sweep_value in enumerate(sweep_values):
        for util_idx, util_level in enumerate(preset.util_values):
            for taskset_id in range(args.num_tasksets):
                seed = deterministic_seed(args.seed, outer_idx, util_idx, taskset_id)

                local_num_tasks = default_num_tasks
                local_cf = config.cf
                local_cp = config.cp

                if sweep_type == "cf":
                    local_cf = float(sweep_value)
                elif sweep_type == "cp":
                    local_cp = float(sweep_value)
                elif sweep_type == "n":
                    local_num_tasks = int(sweep_value)
                else:
                    raise ValueError(f"不支持的 sweep_type={sweep_type}")

                # Fig.2~Fig.4 也统一走 preset.deadline_mode，避免仅 Fig.5 生效。
                taskset = generate_taskset(
                    num_tasks=local_num_tasks,
                    total_util=float(util_level),
                    min_period=config.min_period,
                    max_period=config.max_period,
                    time_scale=config.time_scale,
                    cf=local_cf,
                    cp=local_cp,
                    seed=seed,
                    deadline_mode=preset.deadline_mode,
                    deadline_ratio_min=config.deadline_ratio_min,
                    criticality_assignment=config.criticality_assignment,
                    lo_hi_budget_policy=config.lo_hi_budget_policy,
                )

                base = {
                    "figure": figure,
                    "sweep_type": sweep_type,
                    "sweep_value": float(sweep_value),
                    "util_level": float(util_level),
                    "taskset_id": taskset_id,
                    "seed": seed,
                    "num_tasks": local_num_tasks,
                    "cf": local_cf,
                    "cp": local_cp,
                    "actual_total_util_lo": taskset_total_util(taskset),
                    "actual_total_util_hi": taskset_total_util(taskset, mode=Criticality.HI),
                    "deadline_mode": preset.deadline_mode,
                }

                for method_record in evaluate_methods_on_taskset(taskset, methods):
                    records.append({**base, **method_record})

    raw = pd.DataFrame(records)
    raw.to_csv(output_dir / "raw_results.csv", index=False)

    util_layer = aggregate_by_util(
        raw,
        util_col="util_level",
        outer_group_cols=["method", "method_label", "sweep_value"],
        weight_col="actual_total_util_lo",
        indicator_col="schedulable",
    )
    util_layer.to_csv(output_dir / "util_layer_aggregated.csv", index=False)

    aggregated = aggregate_weighted_schedulability(
        util_layer,
        util_col="util_level",
        outer_group_cols=["method", "method_label", "sweep_value"],
    )
    aggregated.to_csv(output_dir / "aggregated_results.csv", index=False)

    plot_multi_method_curve(
        aggregated,
        x_col="sweep_value",
        y_col="weighted_schedulability",
        title=f"{figure.upper()} {sweep_type.upper()} Sweep",
        output_path=output_dir / f"{figure}.png",
    )


def run_fig5(
    args: argparse.Namespace,
    config: GenerationConfig,
    preset: FigurePreset,
    methods: list[MethodPolicy],
    output_dir: Path,
) -> None:
    """Fig.5：arbitrary deadline util sweep（允许 D<T）。

    preset.deadline_mode 在 paper 模式下为 arbitrary_paper，
    因此生成器会按 HI:[c_hi,T]、LO:[c_lo,T] 的论文口径采样截止期。
    """

    runtime_num_tasks = preset.num_tasks if preset.num_tasks is not None else config.num_tasks
    records: list[dict[str, Any]] = []

    for util_idx, util_value in enumerate(preset.util_values):
        for taskset_id in range(args.num_tasksets):
            seed = deterministic_seed(args.seed, util_idx, 0, taskset_id)
            runtime_cfg = make_runtime_generation_config(
                config,
                preset,
                total_util=float(util_value),
                num_tasks=runtime_num_tasks,
            )
            taskset = generate_taskset(
                num_tasks=runtime_cfg.num_tasks,
                total_util=runtime_cfg.total_util,
                min_period=runtime_cfg.min_period,
                max_period=runtime_cfg.max_period,
                time_scale=runtime_cfg.time_scale,
                cf=runtime_cfg.cf,
                cp=runtime_cfg.cp,
                seed=seed,
                deadline_mode=runtime_cfg.deadline_mode,
                deadline_ratio_min=runtime_cfg.deadline_ratio_min,
                criticality_assignment=runtime_cfg.criticality_assignment,
                lo_hi_budget_policy=runtime_cfg.lo_hi_budget_policy,
            )

            base = {
                "figure": "fig5",
                "sweep_type": "util",
                "sweep_value": float(util_value),
                "util_level": float(util_value),
                "taskset_id": taskset_id,
                "seed": seed,
                "num_tasks": runtime_num_tasks,
                "cf": runtime_cfg.cf,
                "cp": runtime_cfg.cp,
                "actual_total_util_lo": taskset_total_util(taskset),
                "actual_total_util_hi": taskset_total_util(taskset, mode=Criticality.HI),
                "deadline_mode": runtime_cfg.deadline_mode,
                "has_deadline_less_than_period": any(task.deadline < task.period for task in taskset),
            }

            for method_record in evaluate_methods_on_taskset(taskset, methods):
                records.append({**base, **method_record})

    raw = pd.DataFrame(records)
    raw.to_csv(output_dir / "raw_results.csv", index=False)

    aggregated = aggregate_by_util(
        raw,
        util_col="sweep_value",
        outer_group_cols=["method", "method_label"],
        weight_col="actual_total_util_lo",
        indicator_col="schedulable",
    )
    aggregated.to_csv(output_dir / "aggregated_results.csv", index=False)

    plot_multi_method_curve(
        aggregated,
        x_col="sweep_value",
        y_col="schedulable_ratio",
        title="Fig.5 Arbitrary Deadline Utilization Sweep",
        output_path=output_dir / "fig5.png",
    )


def main() -> None:
    """主入口：根据 `--figure` 分发对应复现实验。"""

    args = parse_args()
    if args.num_tasksets <= 0:
        raise ValueError("--num-tasksets 必须为正整数")

    config = resolve_config(args.mode, args.config)
    preset = get_figure_preset(args.figure, args.mode)
    methods = default_methods()

    output_root = Path(args.output_root)
    output_dir = output_root / args.figure
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.figure == "fig1":
        run_fig1(args, config, preset, methods, output_dir)
    elif args.figure == "fig2":
        run_nested_weighted_sweep("fig2", "cf", preset.cf_values, args, config, preset, methods, output_dir)
    elif args.figure == "fig3":
        run_nested_weighted_sweep("fig3", "cp", preset.cp_values, args, config, preset, methods, output_dir)
    elif args.figure == "fig4":
        run_nested_weighted_sweep("fig4", "n", preset.n_values, args, config, preset, methods, output_dir)
    elif args.figure == "fig5":
        run_fig5(args, config, preset, methods, output_dir)
    else:
        raise ValueError(f"不支持的 figure={args.figure}")

    # 落盘复现说明文档。
    note_path = PROJECT_ROOT / "reports" / f"{args.figure}_reproduction_notes.md"
    write_figure_notes(note_path, args.figure, args, config, preset, output_dir)
    write_figure_metadata(output_dir, args.figure, args, config, preset)

    print(f"=== {args.figure} 复现实验完成 ===")
    print(f"raw: {(output_dir / 'raw_results.csv').resolve()}")
    print(f"aggregated: {(output_dir / 'aggregated_results.csv').resolve()}")
    print(f"figure: {(output_dir / f'{args.figure}.png').resolve()}")
    print(f"notes: {note_path.resolve()}")
    print(f"metadata: {(output_dir / 'metadata.json').resolve()}")


if __name__ == "__main__":
    main()
