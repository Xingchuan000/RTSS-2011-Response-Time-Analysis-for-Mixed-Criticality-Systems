"""workload 层最小抽象。

该模块只定义 workload provider 与其返回结果的数据形状。
这里故意不引入 DQN、agent、env 或训练逻辑，确保依赖方向始终是
“训练层依赖 workload 层”，而不是 workload 层反向依赖训练层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from amc_py.models import Task
from amc_py.rl.observation import NormalizationBounds
from amc_py.runtime_scenarios import ExecutionScenario


@dataclass(frozen=True, slots=True)
class WorkloadBundle:
    """描述某个 seed 下完整实例化后的 workload 结果。

    说明：
    - `tasks` 保持“未排序任务集”语义，后续是否做 DM/OPA 排序由 experiment 层决定；
    - `scenario` 与 `normalization_bounds` 必须与 `tasks` 同源构造，避免训练层再做拼装；
    - `metadata` 只承载调试和追踪信息，不承载训练配置。
    """

    tasks: tuple[Task, ...]
    scenario: ExecutionScenario
    normalization_bounds: NormalizationBounds
    taskset_seed: int | None = None
    scenario_seed: int | None = None
    attempts: int = 1
    metadata: dict[str, Any] | None = None


class WorkloadProvider(Protocol):
    """定义 workload provider 的统一工厂接口。"""

    name: str

    def build(self, seed: int) -> WorkloadBundle:
        """根据外部 seed 构造任务集、场景、归一化边界与元数据。"""
