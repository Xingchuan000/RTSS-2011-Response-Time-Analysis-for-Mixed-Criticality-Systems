"""观测与动作元数据导出工具。

本模块只做“语义名称导出”，不参与任何训练、推理或奖励计算逻辑。
这样 VIPER 落盘时可以复用现有状态/动作语义，而不是在脚本里再手写一套映射。
"""

from __future__ import annotations

from collections.abc import Sequence

from amc_py.models import Task
from amc_py.rl.actions import BudgetAction
from amc_py.rl.feature_config import FeatureConfig, observation_feature_groups


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

    return build_observation_feature_names_from_task_names(
        [task.name for task in ordered_tasks],
        feature_config,
    )


def build_observation_feature_names_from_task_names(
    task_names: Sequence[str],
    feature_config: FeatureConfig,
) -> tuple[str, ...]:
    per_task_names, global_names = observation_feature_groups(
        feature_config.observation_mode
    )
    names: list[str] = []
    for task_index, task_name in enumerate(task_names):
        prefix = f"T{task_index:02d}.{task_name}"
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
