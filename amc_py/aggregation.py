"""阶段5：统计聚合模块。

本模块把 weighted schedulability 统计过程拆成两层：
1. util 层内聚合（同一 util 值下的统计）；
2. 跨 util 的加权聚合（得到某个 sweep 点的最终 weighted schedulability）。

这样做的好处：
- 统计口径清晰，可单独测试每一层；
- 实验脚本只负责采样与落盘，不耦合统计公式；
- 便于后续扩展更多聚合维度（例如 method、参数 sweep、随机种子分组等）。
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _normalize_group_cols(group_cols: Sequence[str] | None) -> list[str]:
    """把可选分组列参数标准化为列表。"""

    if group_cols is None:
        return []
    return list(group_cols)


def aggregate_by_util(
    results: pd.DataFrame,
    util_col: str = "util_level",
    outer_group_cols: Sequence[str] | None = None,
    weight_col: str = "actual_total_util_lo",
    indicator_col: str = "schedulable",
) -> pd.DataFrame:
    """第一层：在 util 层内聚合统计。

    输入要求：
    - `results` 至少包含 util 列、权重列、可调度指示列。
    - `indicator_col` 可为 bool/int，内部会转换为 int(0/1)。

    输出字段：
    - util_sum: 该 util 层所有样本权重和
    - weighted_success_sum: Σ(U_i * I_i)
    - taskset_count: 样本数量
    - success_count: 可调度样本数量
    - schedulable_ratio: success_count / taskset_count
    - weighted_schedulability_at_util: weighted_success_sum / util_sum
    """

    group_cols = _normalize_group_cols(outer_group_cols)

    required = set(group_cols + [util_col, weight_col, indicator_col])
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results 缺少必要列：{sorted(missing)}")

    if results.empty:
        # 返回带预期列名的空表，方便调用方串联处理。
        return pd.DataFrame(
            columns=group_cols
            + [
                util_col,
                "util_sum",
                "weighted_success_sum",
                "taskset_count",
                "success_count",
                "schedulable_ratio",
                "weighted_schedulability_at_util",
            ]
        )

    work = results.copy()
    work["indicator"] = work[indicator_col].astype(int)
    work["weighted_success"] = work[weight_col] * work["indicator"]

    grouped = work.groupby(group_cols + [util_col], as_index=False).agg(
        util_sum=(weight_col, "sum"),
        weighted_success_sum=("weighted_success", "sum"),
        taskset_count=("indicator", "count"),
        success_count=("indicator", "sum"),
    )

    grouped["schedulable_ratio"] = grouped["success_count"] / grouped["taskset_count"]
    grouped["weighted_schedulability_at_util"] = (
        grouped["weighted_success_sum"] / grouped["util_sum"]
    )

    return grouped


def aggregate_weighted_schedulability(
    util_aggregated: pd.DataFrame,
    util_col: str = "util_level",
    outer_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """第二层：跨 util 层做 weighted schedulability 聚合。

    输入应为 `aggregate_by_util()` 的输出。

    输出字段：
    - util_sum: 跨 util 的总权重
    - weighted_success_sum: 跨 util 的 Σ(U_i * I_i)
    - taskset_count: 跨 util 的总样本数
    - success_count: 跨 util 的总成功数
    - util_level_count: util 层数量
    - schedulable_ratio: success_count / taskset_count
    - weighted_schedulability: weighted_success_sum / util_sum
    """

    group_cols = _normalize_group_cols(outer_group_cols)

    required = set(group_cols + [util_col, "util_sum", "weighted_success_sum", "taskset_count", "success_count"])
    missing = required - set(util_aggregated.columns)
    if missing:
        raise ValueError(f"util_aggregated 缺少必要列：{sorted(missing)}")

    if util_aggregated.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "util_sum",
                "weighted_success_sum",
                "taskset_count",
                "success_count",
                "util_level_count",
                "schedulable_ratio",
                "weighted_schedulability",
            ]
        )

    if group_cols:
        grouped = util_aggregated.groupby(group_cols, as_index=False).agg(
            util_sum=("util_sum", "sum"),
            weighted_success_sum=("weighted_success_sum", "sum"),
            taskset_count=("taskset_count", "sum"),
            success_count=("success_count", "sum"),
            util_level_count=(util_col, "nunique"),
        )
    else:
        # 无外层分组时，直接对全表做一次总聚合。
        grouped = pd.DataFrame(
            [
                {
                    "util_sum": float(util_aggregated["util_sum"].sum()),
                    "weighted_success_sum": float(util_aggregated["weighted_success_sum"].sum()),
                    "taskset_count": int(util_aggregated["taskset_count"].sum()),
                    "success_count": int(util_aggregated["success_count"].sum()),
                    "util_level_count": int(util_aggregated[util_col].nunique()),
                }
            ]
        )

    grouped["schedulable_ratio"] = grouped["success_count"] / grouped["taskset_count"]
    grouped["weighted_schedulability"] = grouped["weighted_success_sum"] / grouped["util_sum"]

    return grouped
