"""预算动作 agent 抽象与基线实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from amc_py.rl.actions import BudgetAction
from amc_py.rl.types import AgentObservation


class BudgetAgent:
    """预算动作 agent 抽象基类。"""

    def select_action(self, observation: AgentObservation) -> BudgetAction | None:
        """根据观测选择一个动作，或返回 None 表示本轮不动作。"""

        raise NotImplementedError


@dataclass(slots=True)
class NoOpBudgetAgent(BudgetAgent):
    """不执行任何预算调整的基线 agent。"""

    def select_action(self, observation: AgentObservation) -> BudgetAction | None:  # noqa: ARG002
        """始终返回 None，表示不更新预算。"""

        return None


@dataclass(slots=True)
class RandomBudgetAgent(BudgetAgent):
    """在动作空间中均匀随机采样动作的基线 agent。"""

    actions: tuple[BudgetAction, ...]
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化独立随机数生成器，保证给定 seed 可复现。"""

        self._rng = random.Random(self.seed)

    def select_action(self, observation: AgentObservation) -> BudgetAction | None:  # noqa: ARG002
        """从动作空间随机选取一个动作；若空间为空则返回 None。"""

        if not self.actions:
            return None
        return self._rng.choice(self.actions)


@dataclass(slots=True)
class HeuristicBudgetAgent(BudgetAgent):
    """基于任务压力分数的启发式预算 agent。"""

    actions: tuple[BudgetAction, ...]
    eps: float = 1e-6

    def select_action(self, observation: AgentObservation) -> BudgetAction | None:
        """选择“增压最高任务 + 降压最低两任务”的动作。"""

        if not self.actions:
            return None
        task_names = list(observation.raw_budgets.keys())
        if len(task_names) < 3:
            return None

        pressure: dict[str, float] = {}
        for task_name in task_names:
            budget = float(observation.raw_budgets[task_name])
            recent_cost = float(observation.raw_recent_costs.get(task_name, 0))
            pressure[task_name] = recent_cost / max(budget, self.eps)

        increase_task = max(task_names, key=lambda name: pressure[name])
        decrease_candidates = [name for name in task_names if name != increase_task]
        decrease_tasks = tuple(sorted(decrease_candidates, key=lambda name: pressure[name])[:2])
        decrease_task_set = set(decrease_tasks)

        for action in self.actions:
            if action.increase_task != increase_task:
                continue
            if set(action.decrease_tasks) == decrease_task_set:
                return action
        return None
