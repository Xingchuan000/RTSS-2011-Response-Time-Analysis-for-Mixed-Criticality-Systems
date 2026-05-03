"""RL 交互过程中的基础数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Agent 在一次决策点看到的观测。"""

    time: int
    state_vector: tuple[float, ...]
    raw_budgets: Mapping[str, int]
    raw_recent_costs: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    """一次 step 的返回数据，保留 reward/done/info 结构。"""

    observation: AgentObservation
    reward: float
    done: bool
    info: dict
