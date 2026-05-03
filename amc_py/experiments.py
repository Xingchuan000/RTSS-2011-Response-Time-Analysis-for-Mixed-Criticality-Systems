"""实验批处理、统计与绘图模块（阶段 D）。

本模块在阶段 C 的统一评估入口基础上，补齐阶段 D 需要的实验能力：
1. 统一评估接口 `evaluate_taskset`
2. 多种 sweep 实验接口（利用率/CF/CP/任务数）
3. weighted schedulability 统计
4. 可视化绘图（可调度率曲线 + 加权可调度率曲线）
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
from typing import Literal

# 在导入 matplotlib 前设置可写缓存目录，避免权限相关告警。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .aggregation import aggregate_by_util
from .amc import (
    amc_max_sched_test,
    amc_rtb_sched_test,
    compute_amc_max_response_time,
    compute_amc_rtb_response_time,
)
from .generator import generate_taskset, taskset_total_util
from .models import Criticality, PriorityAssignmentResult, SchedulabilityResult, Task
from .priorities import audsley_opa, sort_by_crmpo, sort_by_deadline_monotonic
from .rta import analyze_classic_hi_budget, analyze_hi_mode, analyze_lo_mode, compute_r_lo
from .smc import (
    compute_smc_no_response_time,
    compute_smc_response_time,
    smc_no_sched_test,
    smc_sched_test,
)

# 在无图形界面的环境里使用 Agg 后端，避免脚本运行时报错。
matplotlib.use("Agg")

MethodFn = Callable[[Sequence[Task]], SchedulabilityResult]
OpaLowestPriorityTestFn = Callable[[list[Task], int], bool]
MethodName = Literal["ub_hl", "smc", "smc_no", "amc_rtb", "amc_max", "crmpo_baseline"]
PriorityPolicyName = Literal["dm", "crmpo", "opa"]

# 方法与优先级策略兼容矩阵（阶段 2.2）。
# 约定：
# - 键是分析方法。
# - 值是该方法允许搭配的优先级策略集合。
METHOD_POLICY_MATRIX: dict[str, set[str]] = {
    "ub_hl": {"dm", "crmpo"},
    "smc": {"dm", "crmpo", "opa"},
    "smc_no": {"dm", "crmpo", "opa"},
    "amc_rtb": {"dm", "crmpo", "opa"},
    "amc_max": {"dm", "crmpo", "opa"},
    "crmpo_baseline": {"crmpo"},
}



def _ub_hl_sched_test_on_ordered(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """在指定优先级顺序下执行 UB-H&L 测试。"""

    # 第一步：先做 LO 模式分析；若失败可直接返回。
    lo_result = analyze_lo_mode(ordered_tasks, ordered_tasks)
    if not lo_result.schedulable:
        return SchedulabilityResult(
            schedulable=False,
            method="ub_hl",
            response_times=lo_result.response_times,
            details=f"UB-L 失败：{lo_result.details}",
        )

    # 第二步：再做 HI 模式分析；仅当 LO 与 HI 都通过时整体通过。
    hi_result = analyze_hi_mode(ordered_tasks, ordered_tasks)
    if not hi_result.schedulable:
        merged = dict(lo_result.response_times)
        merged.update(hi_result.response_times)
        return SchedulabilityResult(
            schedulable=False,
            method="ub_hl",
            response_times=merged,
            details=f"UB-H 失败：{hi_result.details}",
        )

    merged = dict(lo_result.response_times)
    merged.update(hi_result.response_times)
    return SchedulabilityResult(
        schedulable=True,
        method="ub_hl",
        response_times=merged,
        details="UB-H&L 测试通过",
    )


def _crmpo_baseline_sched_test_on_ordered(ordered_tasks: Sequence[Task]) -> SchedulabilityResult:
    """在给定顺序下执行 CrMPO baseline 响应时间分析。

    实现口径对齐 RTSS'11 论文中的 CrMPO 对比方法：
    - 优先级由 CrMPO 决定（在入口层强制校验）。
    - 响应时间分析按 Classic 公式，每任务仅使用一个预算参数：
      HI 任务取 `C(HI)`，LO 任务取 `C(LO)`。
    """

    result = analyze_classic_hi_budget(ordered_tasks, ordered_tasks)
    return SchedulabilityResult(
        schedulable=result.schedulable,
        method="crmpo_baseline",
        response_times=result.response_times,
        details=f"CrMPO baseline: {result.details}",
    )


def _validate_evaluation_inputs(tasks: Sequence[Task], method: str, priority_policy: str) -> None:
    """统一校验评估入口输入，保证报错信息一致且可读。"""

    if not tasks:
        raise ValueError("tasks 不能为空，至少需要一个任务")

    task_names = [task.name for task in tasks]
    if len(set(task_names)) != len(task_names):
        raise ValueError("tasks 中存在重复任务名，无法建立稳定优先级映射")

    if method not in METHOD_POLICY_MATRIX:
        supported = ", ".join(sorted(METHOD_POLICY_MATRIX))
        raise ValueError(f"不支持的 method={method}，可选值：{supported}")

    supported_policies = METHOD_POLICY_MATRIX[method]
    if priority_policy not in supported_policies:
        supported = ", ".join(sorted(supported_policies))
        raise ValueError(
            f"method={method} 不支持 priority_policy={priority_policy}，可选值：{supported}"
        )



def _resolve_method(method: str) -> MethodFn:
    """把 method 字符串映射为具体分析函数。"""

    method_map: dict[str, MethodFn] = {
        "ub_hl": _ub_hl_sched_test_on_ordered,
        "smc": smc_sched_test,
        "smc_no": smc_no_sched_test,
        "amc_rtb": amc_rtb_sched_test,
        "amc_max": amc_max_sched_test,
        "crmpo_baseline": _crmpo_baseline_sched_test_on_ordered,
    }
    try:
        return method_map[method]
    except KeyError as exc:
        supported = ", ".join(sorted(method_map.keys()))
        raise ValueError(f"不支持的 method={method}，可选值：{supported}") from exc



def _smc_lowest_priority_task_schedulable(trial_order: list[Task], lowest_priority_idx: int) -> bool:
    """SMC 下仅判定“当前最低优先级候选任务”是否可调度。"""

    task = trial_order[lowest_priority_idx]
    hp_tasks = trial_order[:lowest_priority_idx]
    return compute_smc_response_time(task, hp_tasks) is not None


def _smc_no_lowest_priority_task_schedulable(
    trial_order: list[Task], lowest_priority_idx: int
) -> bool:
    """SMC-no 下仅判定“当前最低优先级候选任务”是否可调度。"""

    task = trial_order[lowest_priority_idx]
    hp_tasks = trial_order[:lowest_priority_idx]
    return compute_smc_no_response_time(task, hp_tasks) is not None


def _amc_rtb_lowest_priority_task_schedulable(
    trial_order: list[Task], lowest_priority_idx: int
) -> bool:
    """AMC-rtb 下仅判定“当前最低优先级候选任务”是否可调度。"""

    task = trial_order[lowest_priority_idx]
    hp_tasks = trial_order[:lowest_priority_idx]
    r_lo = compute_r_lo(task, hp_tasks)
    if r_lo is None:
        return False
    return compute_amc_rtb_response_time(task, hp_tasks, {task.name: r_lo}) is not None


def _amc_max_lowest_priority_task_schedulable(
    trial_order: list[Task], lowest_priority_idx: int
) -> bool:
    """AMC-max 下仅判定“当前最低优先级候选任务”是否可调度。"""

    task = trial_order[lowest_priority_idx]
    prefix = trial_order[: lowest_priority_idx + 1]
    return compute_amc_max_response_time(task, prefix) is not None


def _resolve_opa_lowest_priority_test(method: str) -> OpaLowestPriorityTestFn:
    """解析 OPA 使用的候选最低优先级可行性测试函数。"""

    method_map: dict[str, OpaLowestPriorityTestFn] = {
        "smc": _smc_lowest_priority_task_schedulable,
        "smc_no": _smc_no_lowest_priority_task_schedulable,
        "amc_rtb": _amc_rtb_lowest_priority_task_schedulable,
        "amc_max": _amc_max_lowest_priority_task_schedulable,
    }
    try:
        return method_map[method]
    except KeyError as exc:
        supported = ", ".join(sorted(method_map.keys()))
        raise ValueError(
            f"method={method} 不支持 OPA 候选任务测试，可选值：{supported}"
        ) from exc


def _resolve_ordering(
    tasks: Sequence[Task],
    priority_policy: str,
    method: str,
) -> tuple[list[Task], PriorityAssignmentResult | None]:
    """内部实现：根据优先级策略生成有序任务列表，并附带 OPA 分配结果。

    说明：
    - 该函数是 `resolve_ordering()` 公共接口的底层实现，继续保留私有版本的原因是
      `evaluate_taskset()` 在 OPA 失败时需要拿到完整的 `PriorityAssignmentResult`，
      以便把失败原因回填进 `SchedulabilityResult.details`。
    - 公共调用路径应通过 `resolve_ordering()` 访问；runtime 等外部模块也应统一使用
      公共接口，避免与此处 tuple 返回值耦合。

    返回：
    - 一个二元组 `(ordered_tasks, opa_result)`：
      * `ordered_tasks`: 按最终优先级从高到低排列的任务列表。OPA 失败时为空列表。
      * `opa_result`:     仅在 `priority_policy == "opa"` 时有意义，其余策略为 None。
    """

    # DM：按截止期升序排序。
    if priority_policy == "dm":
        return sort_by_deadline_monotonic(tasks), None

    # CrMPO：先关键级再截止期。
    if priority_policy == "crmpo":
        return sort_by_crmpo(tasks), None

    # OPA：使用 Audsley 逐层试探最低优先级任务。
    if priority_policy == "opa":
        lowest_priority_test_fn = _resolve_opa_lowest_priority_test(method)
        opa_result = audsley_opa(tasks, lowest_priority_test_fn)
        if not opa_result.success:
            # OPA 失败时仍把 opa_result 返回，便于上层拼装详细错误信息。
            return [], opa_result

        priority_map = opa_result.priorities
        ordered = sorted(tasks, key=lambda task: priority_map[task.name])
        return ordered, opa_result

    raise ValueError("不支持的 priority_policy，可选值：dm, crmpo, opa")


def resolve_ordering(
    tasks: Sequence[Task],
    priority_policy: str,
    method: str,
) -> list[Task]:
    """公共接口：根据优先级策略返回有序任务列表。

    设计目的：
    - 为 runtime 仿真器、外部实验脚本等下游模块提供统一的优先级解析入口，
      让它们无需自己重复实现 DM / CrMPO / OPA 的分派逻辑。
    - 与 `evaluate_taskset()` 共用同一套底层实现（`_resolve_ordering`），
      保证静态分析与运行时仿真看到的优先级顺序一致。

    参数：
    - tasks: 原始任务列表（未排序）；函数内部不会修改此列表。
    - priority_policy: `dm | crmpo | opa` 中之一。
    - method: 分析方法名；仅当 `priority_policy == "opa"` 时被使用，
      用于选择 Audsley 搜索所需的“候选最低优先级任务可行性测试”。

    返回：
    - `list[Task]`：按最终优先级从高到低排列的任务列表。

    异常：
    - `ValueError`:   `priority_policy` 不在支持集合内，或 OPA 组合下 `method` 不支持 OPA。
    - `RuntimeError`: `priority_policy == "opa"` 且 Audsley 未能成功分配优先级；
      异常 message 形如 `"OPA 分配失败：{opa_result.details}"`，
      调用方可直接用 `str(exc)` 拿到可读错误描述。
    """

    ordered, opa_result = _resolve_ordering(tasks, priority_policy, method)

    # OPA 失败时，公共接口选择以异常显式通知调用方，避免出现“返回空列表但不报错”的歧义。
    # `evaluate_taskset()` 会捕获这个异常并转成不可调度的 SchedulabilityResult。
    if priority_policy == "opa" and opa_result is not None and not opa_result.success:
        raise RuntimeError(f"OPA 分配失败：{opa_result.details}")

    return ordered



def evaluate_taskset(tasks: Sequence[Task], method: str, priority_policy: str) -> SchedulabilityResult:
    """统一任务集评估入口。

    参数：
    - tasks: 原始任务列表。
    - method: `ub_hl | smc | smc_no | amc_rtb | amc_max | crmpo_baseline`。
    - priority_policy: `dm | crmpo | opa`。

    返回：
    - SchedulabilityResult，其中 `details` 会带上 method/policy 元信息，
      便于后续汇总和调试。
    """

    # 先做参数合法性与方法/策略组合校验，拦截明显非法的调用。
    _validate_evaluation_inputs(tasks, method, priority_policy)

    # 选择具体分析函数（UB-H&L / SMC / AMC-rtb 等）。
    method_fn = _resolve_method(method)

    # 通过公共 API `resolve_ordering()` 得到已排序任务列表；
    # OPA 失败会以 RuntimeError 形式抛出，这里捕获后转成 SchedulabilityResult 的失败路径，
    # 保证对外可见的 `evaluate_taskset()` 行为与重构前完全一致。
    try:
        ordered_tasks = resolve_ordering(tasks, priority_policy, method)
    except RuntimeError as exc:
        return SchedulabilityResult(
            schedulable=False,
            method=method,
            response_times={},
            details=str(exc),
        )

    # 真正执行可调度性测试，并把 method/policy 元信息前置到 details 里。
    result = method_fn(ordered_tasks)
    details = f"method={method}, priority_policy={priority_policy}, {result.details}"
    return SchedulabilityResult(
        schedulable=result.schedulable,
        method=method,
        response_times=result.response_times,
        details=details,
    )



def _records_to_dataframe(records: list[dict[str, object]]) -> pd.DataFrame:
    """把实验记录列表转为 DataFrame，并统一列顺序。"""

    frame = pd.DataFrame(records)
    preferred = [
        "sweep_type",
        "sweep_value",
        "taskset_id",
        "seed",
        "method",
        "priority_policy",
        "num_tasks",
        "target_total_util",
        "actual_total_util_lo",
        "actual_total_util_hi",
        "cf",
        "cp",
        "min_period",
        "max_period",
        "schedulable",
        "details",
    ]

    # 仅保留当前数据里实际出现的列，避免 KeyError。
    ordered_cols = [col for col in preferred if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered_cols]
    return frame[ordered_cols + remaining]



def _run_generic_sweep(
    sweep_type: str,
    sweep_values: Sequence[float | int],
    num_tasksets: int,
    num_tasks: int,
    total_util: float,
    cf: float,
    cp: float,
    min_period: int,
    max_period: int,
    method: str,
    priority_policy: str,
    seed: int = 0,
) -> pd.DataFrame:
    """通用 sweep 执行器。

    设计说明：
    - 通过 `sweep_type` 决定哪一个参数随 sweep 值变化。
    - 其余参数保持常量。
    - 每个 sweep 值生成 `num_tasksets` 个随机任务集并逐一评估。
    """

    if num_tasksets <= 0:
        raise ValueError("num_tasksets 必须为正整数")

    records: list[dict[str, object]] = []
    base_seed = seed

    for sweep_value in sweep_values:
        for taskset_id in range(num_tasksets):
            # 使用可重复的种子策略：
            # base_seed + sweep位置偏移 + taskset索引。
            local_seed = base_seed + int(taskset_id) + int(len(records))

            # 根据 sweep 类型更新本轮参数。
            local_num_tasks = num_tasks
            local_total_util = total_util
            local_cf = cf
            local_cp = cp

            if sweep_type == "util":
                local_total_util = float(sweep_value)
            elif sweep_type == "cf":
                local_cf = float(sweep_value)
            elif sweep_type == "cp":
                local_cp = float(sweep_value)
            elif sweep_type == "n":
                local_num_tasks = int(sweep_value)
            else:
                raise ValueError("不支持的 sweep_type，可选值：util, cf, cp, n")

            # 生成任务集并执行可调度性评估。
            taskset = generate_taskset(
                num_tasks=local_num_tasks,
                total_util=local_total_util,
                min_period=min_period,
                max_period=max_period,
                cf=local_cf,
                cp=local_cp,
                seed=local_seed,
                # sweep API 默认沿用论文主线中的隐式截止期语义（D=T）。
                deadline_mode="implicit",
            )

            eval_result = evaluate_taskset(taskset, method=method, priority_policy=priority_policy)

            # 记录目标利用率与实际离散化后利用率，便于结果解释。
            record: dict[str, object] = {
                "sweep_type": sweep_type,
                "sweep_value": float(sweep_value),
                "taskset_id": taskset_id,
                "seed": local_seed,
                "method": method,
                "priority_policy": priority_policy,
                "num_tasks": local_num_tasks,
                "target_total_util": local_total_util,
                "actual_total_util_lo": taskset_total_util(taskset),
                "actual_total_util_hi": taskset_total_util(taskset, mode=Criticality.HI),
                "cf": local_cf,
                "cp": local_cp,
                "min_period": min_period,
                "max_period": max_period,
                "schedulable": eval_result.schedulable,
                "details": eval_result.details,
            }
            records.append(record)

    return _records_to_dataframe(records)



def run_utilization_sweep(
    util_values: Sequence[float],
    num_tasksets: int = 100,
    num_tasks: int = 8,
    cf: float = 2.0,
    cp: float = 0.5,
    min_period: int = 10,
    max_period: int = 1000,
    method: str = "amc_rtb",
    priority_policy: str = "dm",
    seed: int = 0,
) -> pd.DataFrame:
    """执行总利用率 sweep。"""

    return _run_generic_sweep(
        sweep_type="util",
        sweep_values=util_values,
        num_tasksets=num_tasksets,
        num_tasks=num_tasks,
        total_util=util_values[0] if util_values else 0.5,
        cf=cf,
        cp=cp,
        min_period=min_period,
        max_period=max_period,
        method=method,
        priority_policy=priority_policy,
        seed=seed,
    )



def run_cf_sweep(
    cf_values: Sequence[float],
    num_tasksets: int = 100,
    num_tasks: int = 8,
    total_util: float = 0.7,
    cp: float = 0.5,
    min_period: int = 10,
    max_period: int = 1000,
    method: str = "amc_rtb",
    priority_policy: str = "dm",
    seed: int = 0,
) -> pd.DataFrame:
    """执行 CF（C(HI)/C(LO) 比例）sweep。"""

    return _run_generic_sweep(
        sweep_type="cf",
        sweep_values=cf_values,
        num_tasksets=num_tasksets,
        num_tasks=num_tasks,
        total_util=total_util,
        cf=cf_values[0] if cf_values else 2.0,
        cp=cp,
        min_period=min_period,
        max_period=max_period,
        method=method,
        priority_policy=priority_policy,
        seed=seed,
    )



def run_cp_sweep(
    cp_values: Sequence[float],
    num_tasksets: int = 100,
    num_tasks: int = 8,
    total_util: float = 0.7,
    cf: float = 2.0,
    min_period: int = 10,
    max_period: int = 1000,
    method: str = "amc_rtb",
    priority_policy: str = "dm",
    seed: int = 0,
) -> pd.DataFrame:
    """执行 CP（HI 任务比例）sweep。"""

    return _run_generic_sweep(
        sweep_type="cp",
        sweep_values=cp_values,
        num_tasksets=num_tasksets,
        num_tasks=num_tasks,
        total_util=total_util,
        cf=cf,
        cp=cp_values[0] if cp_values else 0.5,
        min_period=min_period,
        max_period=max_period,
        method=method,
        priority_policy=priority_policy,
        seed=seed,
    )



def run_taskset_size_sweep(
    taskset_sizes: Sequence[int],
    num_tasksets: int = 100,
    total_util: float = 0.7,
    cf: float = 2.0,
    cp: float = 0.5,
    min_period: int = 10,
    max_period: int = 1000,
    method: str = "amc_rtb",
    priority_policy: str = "dm",
    seed: int = 0,
) -> pd.DataFrame:
    """执行任务数量 sweep。"""

    return _run_generic_sweep(
        sweep_type="n",
        sweep_values=taskset_sizes,
        num_tasksets=num_tasksets,
        num_tasks=taskset_sizes[0] if taskset_sizes else 8,
        total_util=total_util,
        cf=cf,
        cp=cp,
        min_period=min_period,
        max_period=max_period,
        method=method,
        priority_policy=priority_policy,
        seed=seed,
    )



def compute_weighted_schedulability(
    results: pd.DataFrame,
    group_col: str = "sweep_value",
) -> pd.DataFrame:
    """兼容接口：按单层分组计算 weighted schedulability。

    说明：
    - 阶段5已把聚合逻辑拆到 `amc_py.aggregation`。
    - 该函数保留原签名，内部复用新模块，避免影响旧脚本与测试。
    """

    grouped = aggregate_by_util(
        results=results,
        util_col=group_col,
        outer_group_cols=None,
        weight_col="actual_total_util_lo",
        indicator_col="schedulable",
    ).rename(columns={"weighted_success_sum": "weighted_sum"})

    return grouped[
        [
            group_col,
            "util_sum",
            "weighted_sum",
            "taskset_count",
            "schedulable_ratio",
            "weighted_schedulability_at_util",
        ]
    ].rename(columns={"weighted_schedulability_at_util": "weighted_schedulability"})



def plot_schedulable_percentage(
    results: pd.DataFrame,
    x_col: str = "sweep_value",
    output_path: str | Path | None = None,
    title: str = "Schedulable Percentage",
) -> Path | None:
    """绘制可调度率曲线。"""

    required_cols = {x_col, "schedulable"}
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f"results 缺少必要列：{sorted(missing)}")

    stat = (
        results.groupby(x_col, as_index=False)["schedulable"]
        .mean()
        .rename(columns={"schedulable": "schedulable_percentage"})
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(stat[x_col], stat["schedulable_percentage"], marker="o", linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Schedulable Percentage")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if output_path is None:
        plt.close(fig)
        return None

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path



def plot_weighted_schedulability(
    weighted_results: pd.DataFrame,
    x_col: str = "sweep_value",
    output_path: str | Path | None = None,
    title: str = "Weighted Schedulability",
) -> Path | None:
    """绘制 weighted schedulability 曲线。"""

    required_cols = {x_col, "weighted_schedulability"}
    missing = required_cols - set(weighted_results.columns)
    if missing:
        raise ValueError(f"weighted_results 缺少必要列：{sorted(missing)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        weighted_results[x_col],
        weighted_results["weighted_schedulability"],
        marker="s",
        linewidth=2,
        color="#d95f02",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel("Weighted Schedulability")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    if output_path is None:
        plt.close(fig)
        return None

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
