"""观测与动作元数据导出工具。

本模块只做“语义名称导出”，不参与任何训练、推理或奖励计算逻辑。
这样 VIPER 落盘时可以复用现有状态/动作语义，而不是在脚本里再手写一套映射。
"""

from __future__ import annotations

from collections.abc import Sequence

from amc_py.models import Task
from amc_py.rl.actions import BudgetAction
from amc_py.rl.feature_config import (
    OBSERVATION_MODE_V10_BASIC,
    OBSERVATION_MODE_V11_FULL_10D,
    OBSERVATION_MODE_V11_LITE_6D,
    OBSERVATION_MODE_V11_NO_MAX_9D,
    OBSERVATION_MODE_V11_NO_PRIORITY_9D,
    OBSERVATION_MODE_V11_NO_RISK_9D,
    OBSERVATION_MODE_V11_NO_RISK_NO_UTIL_8D,
    OBSERVATION_MODE_V11_NO_UTIL_9D,
    OBSERVATION_MODE_V12_FULL_14D,
    OBSERVATION_MODE_V13_RH_17D,
    FeatureConfig,
    V11_GLOBAL_FEATURE_NAMES,
    V11_LITE_GLOBAL_FEATURE_NAMES,
    V11_LITE_PER_TASK_FEATURE_NAMES,
    V11_NO_MAX_GLOBAL_FEATURE_NAMES,
    V11_NO_MAX_PER_TASK_FEATURE_NAMES,
    V11_NO_PRIORITY_GLOBAL_FEATURE_NAMES,
    V11_NO_PRIORITY_PER_TASK_FEATURE_NAMES,
    V11_NO_RISK_GLOBAL_FEATURE_NAMES,
    V11_NO_RISK_NO_UTIL_GLOBAL_FEATURE_NAMES,
    V11_NO_RISK_NO_UTIL_PER_TASK_FEATURE_NAMES,
    V11_NO_RISK_PER_TASK_FEATURE_NAMES,
    V11_NO_UTIL_GLOBAL_FEATURE_NAMES,
    V11_NO_UTIL_PER_TASK_FEATURE_NAMES,
    V11_PER_TASK_FEATURE_NAMES,
    V12_GLOBAL_FEATURE_NAMES,
    V12_PER_TASK_FEATURE_NAMES,
    V13_RH_17D_GLOBAL_FEATURE_NAMES,
    V13_RH_17D_PER_TASK_FEATURE_NAMES,
)


def _feature_name_groups(feature_config: FeatureConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """按 observation_mode 返回“每任务特征名 + 全局特征名”。

    这里显式列出所有计划要求支持的模式，目的是保证新增模式时如果忘记补映射，
    会直接在这里报错，而不是悄悄输出错误名称。
    """

    mode = feature_config.observation_mode
    if mode == OBSERVATION_MODE_V10_BASIC:
        return ("budget_norm", "recent_cost_norm"), ()
    if mode == OBSERVATION_MODE_V11_FULL_10D:
        return V11_PER_TASK_FEATURE_NAMES, V11_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_NO_RISK_9D:
        return V11_NO_RISK_PER_TASK_FEATURE_NAMES, V11_NO_RISK_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_NO_UTIL_9D:
        return V11_NO_UTIL_PER_TASK_FEATURE_NAMES, V11_NO_UTIL_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_NO_MAX_9D:
        return V11_NO_MAX_PER_TASK_FEATURE_NAMES, V11_NO_MAX_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_NO_PRIORITY_9D:
        return V11_NO_PRIORITY_PER_TASK_FEATURE_NAMES, V11_NO_PRIORITY_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_NO_RISK_NO_UTIL_8D:
        return V11_NO_RISK_NO_UTIL_PER_TASK_FEATURE_NAMES, V11_NO_RISK_NO_UTIL_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V11_LITE_6D:
        return V11_LITE_PER_TASK_FEATURE_NAMES, V11_LITE_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V12_FULL_14D:
        return V12_PER_TASK_FEATURE_NAMES, V12_GLOBAL_FEATURE_NAMES
    if mode == OBSERVATION_MODE_V13_RH_17D:
        return V13_RH_17D_PER_TASK_FEATURE_NAMES, V13_RH_17D_GLOBAL_FEATURE_NAMES
    raise ValueError(f"不支持的 observation_mode: {mode}")


def build_observation_feature_names(
    ordered_tasks: Sequence[Task],
    feature_config: FeatureConfig,
) -> tuple[str, ...]:
    """构造与状态向量严格同序的特征名称列表。

    规则：
    - task 级特征按 `T00.<task_name>.<feature_name>` 这种稳定格式展开；
    - global 特征按 `global.<feature_name>` 追加到末尾；
    - 顺序必须与 `build_observation(...)` 实际输出一致，否则树规则会错位。
    """

    per_task_names, global_names = _feature_name_groups(feature_config)
    names: list[str] = []
    for task_index, task in enumerate(ordered_tasks):
        prefix = f"T{task_index:02d}.{task.name}"
        for feature_name in per_task_names:
            names.append(f"{prefix}.{feature_name}")
    for feature_name in global_names:
        names.append(f"global.{feature_name}")
    return tuple(names)


def build_action_definitions(actions: Sequence[BudgetAction]) -> list[dict[str, object]]:
    """把动作空间定义导出为稳定的 JSON 结构。"""

    definitions: list[dict[str, object]] = []
    for action in actions:
        definitions.append(
            {
                "action_id": int(action.action_id),
                "action_space_type": str(action.action_space_type),
                "is_noop": bool(action.is_noop),
                "increase_task": action.increase_task,
                "decrease_tasks": list(action.decrease_tasks),
                "increase_idx": action.increase_idx,
                "decrease_indices": list(action.decrease_indices),
                "increase_ratio": float(action.increase_ratio),
                "decrease_ratio": float(action.decrease_ratio),
                "is_constraint_guided_pair": bool(action.is_constraint_guided_pair),
                "constraint_guided_increase_rank": action.constraint_guided_increase_rank,
                "is_residual_ranked": bool(action.is_residual_ranked),
                "residual_action_type": action.residual_action_type,
                "residual_rank": action.residual_rank,
            }
        )
    return definitions
