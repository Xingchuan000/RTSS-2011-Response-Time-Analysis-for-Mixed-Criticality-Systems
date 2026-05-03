"""阶段 7：RL 风格 reset/step 环境接口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from amc_py.amc import build_design_r_lo_map
from amc_py.budget_runtime import BudgetState
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Task
from amc_py.rl.actions import BudgetAction, apply_budget_action_candidate, build_budget_action_space
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.observation import NormalizationBounds, build_observation
from amc_py.rl.safety import (
    RuntimeBudgetSafetyChecker,
    has_effective_budget_updates,
    merge_budget_candidate,
)
from amc_py.rl.types import AgentObservation, AgentStepResult
from amc_py.runtime_models import RuntimeConfig
from amc_py.runtime_scenarios import ExecutionScenario


@dataclass(slots=True)
class AmcBudgetEnv:
    """为 DQN 预集成准备的 AMC 预算环境。"""

    ordered_tasks: Sequence[Task]
    scenario: ExecutionScenario
    runtime_config: RuntimeConfig
    agent_period: int = 10
    check_safety: bool = True
    safety_checker: RuntimeBudgetSafetyChecker | None = None
    normalization_bounds: NormalizationBounds | None = None
    _actions: tuple[BudgetAction, ...] = field(init=False, repr=False)
    _monitor: RuntimeMonitor = field(init=False, repr=False)
    _engine: EventRuntimeEngine | None = field(init=False, default=None, repr=False)
    _done: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """初始化固定动作空间。"""

        self._actions = build_budget_action_space(self.ordered_tasks)

    @property
    def action_space_size(self) -> int:
        """返回离散动作空间大小。"""

        return len(self._actions)

    def valid_action_mask(self) -> tuple[bool, ...]:
        """返回当前时刻每个离散动作是否可通过安全检查。"""

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")
        if not self.check_safety:
            return tuple(True for _ in self._actions)

        checker = self._ensure_checker()
        mask: list[bool] = []
        for action in self._actions:
            # 仅复用环境现有动作到候选预算的映射，不引入额外决策语义。
            updates = apply_budget_action_candidate(
                action=action,
                budget_state=self._engine.runtime_budgets,
                ordered_tasks=self.ordered_tasks,
            )
            if not has_effective_budget_updates(self._engine.runtime_budgets, updates):
                mask.append(False)
                continue
            merged = merge_budget_candidate(self._engine.runtime_budgets, updates)
            mask.append(checker.validate_candidate(merged).accepted)
        return tuple(mask)

    def _ensure_checker(self) -> RuntimeBudgetSafetyChecker:
        """获取环境使用的安全检查器。"""

        if self.safety_checker is not None:
            return self.safety_checker
        design_r_lo = build_design_r_lo_map(self.ordered_tasks)
        return RuntimeBudgetSafetyChecker(ordered_tasks=self.ordered_tasks, design_r_lo=design_r_lo)

    def reset(self, seed: int | None = None) -> AgentObservation:  # noqa: ARG002
        """重置环境并返回初始观测。"""

        self._monitor = RuntimeMonitor()
        self._engine = EventRuntimeEngine.build(
            ordered_tasks=self.ordered_tasks,
            scenario=self.scenario,
            config=self.runtime_config,
            budget_state=BudgetState.from_tasks(self.ordered_tasks),
            monitor=self._monitor,
        )
        # 先处理 time=0 的边界事件，再返回首次观测，避免 agent 早于首批 release 决策。
        self._engine.run_until(0, include_boundary=True)
        self._done = False
        return build_observation(
            time=self._engine.current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
        )

    def step(self, action_id: int | None) -> AgentStepResult:
        """执行一步动作并推进到下一次 agent 激活点。"""

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")
        if self._done:
            raise RuntimeError("环境已结束，请先 reset")

        accepted = False
        reject_reason: str | None = None

        if action_id is not None:
            if action_id < 0 or action_id >= len(self._actions):
                raise ValueError(f"非法 action_id={action_id}")
            action = self._actions[action_id]
            updates = apply_budget_action_candidate(
                action=action,
                budget_state=self._engine.runtime_budgets,
                ordered_tasks=self.ordered_tasks,
            )
            merged = merge_budget_candidate(self._engine.runtime_budgets, updates)
            if self.check_safety:
                report = self._ensure_checker().validate_candidate(merged)
                accepted = report.accepted
                if not accepted:
                    reject_reason = report.reason
            else:
                accepted = True
            if accepted:
                self._engine.apply_budget_updates(updates)

        target_time = min(self._engine.current_time + self.agent_period, self.runtime_config.end_time or 0)
        # step 返回的 next observation 必须包含目标决策时刻的普通事件处理结果。
        self._engine.run_until(target_time, include_boundary=True)
        reward = self._monitor.consume_reward()

        current_time = self._engine.current_time
        if self.runtime_config.end_time is None:
            done = False
        else:
            done = current_time >= self.runtime_config.end_time
        self._done = done

        runtime_result = self._engine.finish()
        observation = build_observation(
            time=current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
        )
        info = {
            "time": current_time,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "mode_changes": runtime_result.mode_change_count(),
            "lo_cancellations": runtime_result.lo_job_cancellation_count(),
            "deadline_misses": len(runtime_result.deadline_misses),
        }
        return AgentStepResult(observation=observation, reward=reward, done=done, info=info)
