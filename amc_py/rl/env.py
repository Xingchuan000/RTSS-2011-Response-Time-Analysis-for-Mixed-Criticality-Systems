"""阶段 7：RL 风格 reset/step 环境接口。"""

from __future__ import annotations

from collections import Counter
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
    _action_log: list[dict[str, object]] = field(init=False, repr=False)
    _mask_log: list[dict[str, object]] = field(init=False, repr=False)
    _mask_reject_reasons: Counter[str] = field(init=False, repr=False)
    _last_mask_details: list[dict[str, object]] = field(init=False, repr=False)
    _safety_checked_actions: int = field(init=False, default=0, repr=False)
    _safety_accepted_actions: int = field(init=False, default=0, repr=False)
    _safety_rejected_actions: int = field(init=False, default=0, repr=False)
    _selected_invalid_mask_actions: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        """初始化固定动作空间。"""

        self._actions = build_budget_action_space(self.ordered_tasks)
        self._action_log = []
        self._mask_log = []
        self._mask_reject_reasons = Counter()
        self._last_mask_details = []

    @property
    def action_log(self) -> list[dict[str, object]]:
        """返回 DQN 路径的动作日志。"""

        return self._action_log

    @property
    def mask_log(self) -> list[dict[str, object]]:
        """返回每次 valid_action_mask 计算时的统计日志。"""

        return self._mask_log

    def debug_statistics(self) -> dict[str, int | float | bool]:
        """汇总评估脚本需要的 mask/safety 统计。

        这里不在脚本层做“猜测式”推导，而是直接返回环境在运行过程中原位记录的
        统计值，确保 DQN CSV 与调试 trace 使用同一份事实来源。
        """

        valid_counts = [int(row["valid_action_count"]) for row in self._mask_log]
        invalid_counts = [int(row["masked_action_count"]) for row in self._mask_log]
        mask_checks = len(self._mask_log)
        return {
            "check_safety": self.check_safety,
            "safety_checked_actions": self._safety_checked_actions,
            "safety_accepted_actions": self._safety_accepted_actions,
            "safety_rejected_actions": self._safety_rejected_actions,
            "valid_action_count_mean": (sum(valid_counts) / mask_checks) if mask_checks > 0 else 0.0,
            "masked_action_count_mean": (sum(invalid_counts) / mask_checks) if mask_checks > 0 else 0.0,
            "masked_action_count_max": max(invalid_counts) if invalid_counts else 0,
            "mask_rejection_rate_mean": (
                (sum(invalid_counts) / (mask_checks * len(self._actions))) if mask_checks > 0 else 0.0
            ),
            "selected_invalid_mask_actions": self._selected_invalid_mask_actions,
        }

    @property
    def action_space_size(self) -> int:
        """返回离散动作空间大小。"""

        return len(self._actions)

    def valid_action_mask(self) -> tuple[bool, ...]:
        """返回当前时刻每个离散动作是否可通过安全检查。

        这里除了返回布尔 mask，还会同步记录：
        - 每个 action 被判 invalid 的具体原因；
        - 本轮 mask 计算里 valid / invalid 动作数量；
        - 被 mask 掉的原因计数分布。

        这样 DQN 评估在 `rejected_actions=0` 时也能区分两种情况：
        - 动作确实在 mask 过滤后仍然安全；
        - 动作本应被 mask 掉，但选择器与 mask 语义出现偏差。
        """

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")
        checker = self._ensure_checker() if self.check_safety else None
        mask: list[bool] = []
        mask_details: list[dict[str, object]] = []
        reject_reason_counts: Counter[str] = Counter()
        for action in self._actions:
            # 仅复用环境现有动作到候选预算的映射，不引入额外决策语义。
            updates = apply_budget_action_candidate(
                action=action,
                budget_state=self._engine.runtime_budgets,
                ordered_tasks=self.ordered_tasks,
            )
            budget_before = dict(self._engine.runtime_budgets.budgets)
            candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
            if not has_effective_budget_updates(self._engine.runtime_budgets, updates):
                mask.append(False)
                reject_reason_counts["no_effective_budget_change"] += 1
                mask_details.append(
                    {
                        "action_id": action.action_id,
                        "valid": False,
                        "reject_reason": "no_effective_budget_change",
                        "updates": dict(updates),
                        "budget_before": budget_before,
                        "candidate_budgets": candidate_budgets,
                    }
                )
                continue
            if not self.check_safety:
                mask.append(True)
                mask_details.append(
                    {
                        "action_id": action.action_id,
                        "valid": True,
                        "reject_reason": None,
                        "updates": dict(updates),
                        "budget_before": budget_before,
                        "candidate_budgets": candidate_budgets,
                    }
                )
                continue
            report = checker.validate_candidate(candidate_budgets)
            mask.append(report.accepted)
            if not report.accepted:
                reject_reason_counts[report.reason] += 1
            mask_details.append(
                {
                    "action_id": action.action_id,
                    "valid": report.accepted,
                    "reject_reason": None if report.accepted else report.reason,
                    "reject_diagnostics": list(report.diagnostics),
                    "updates": dict(updates),
                    "budget_before": budget_before,
                    "candidate_budgets": candidate_budgets,
                }
            )
        valid_action_count = sum(mask)
        masked_action_count = len(mask) - valid_action_count
        self._last_mask_details = mask_details
        self._mask_reject_reasons.update(reject_reason_counts)
        self._mask_log.append(
            {
                "time": self._engine.current_time,
                "total_actions": len(self._actions),
                "valid_action_count": valid_action_count,
                "masked_action_count": masked_action_count,
                "reject_reason_counts": dict(reject_reason_counts),
            }
        )
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
        self._action_log = []
        self._mask_log = []
        self._mask_reject_reasons = Counter()
        self._last_mask_details = []
        self._safety_checked_actions = 0
        self._safety_accepted_actions = 0
        self._safety_rejected_actions = 0
        self._selected_invalid_mask_actions = 0
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

        action_time = self._engine.current_time
        accepted = False
        reject_reason: str | None = None
        reject_diagnostics: tuple[dict[str, str | int | float], ...] = ()
        updates: dict[str, int] = {}
        budget_before = dict(self._engine.runtime_budgets.budgets)
        candidate_budgets = dict(budget_before)
        valid_action_count = 0
        masked_action_count = 0
        selected_action_was_mask_valid = action_id is None
        action_was_checked = False

        if action_id is not None:
            if action_id < 0 or action_id >= len(self._actions):
                raise ValueError(f"非法 action_id={action_id}")
            if self._last_mask_details:
                valid_action_count = int(self._mask_log[-1]["valid_action_count"])
                masked_action_count = int(self._mask_log[-1]["masked_action_count"])
                selected_action_was_mask_valid = bool(self._last_mask_details[action_id]["valid"])
                if not selected_action_was_mask_valid:
                    self._selected_invalid_mask_actions += 1
            action = self._actions[action_id]
            updates = apply_budget_action_candidate(
                action=action,
                budget_state=self._engine.runtime_budgets,
                ordered_tasks=self.ordered_tasks,
            )
            candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
            if self.check_safety:
                action_was_checked = True
                self._safety_checked_actions += 1
                report = self._ensure_checker().validate_candidate(candidate_budgets)
                accepted = report.accepted
                if not accepted:
                    reject_reason = report.reason
                    reject_diagnostics = report.diagnostics
                    self._safety_rejected_actions += 1
                else:
                    self._safety_accepted_actions += 1
            else:
                accepted = True
            if accepted:
                self._engine.apply_budget_updates(updates)
        else:
            if self._mask_log:
                valid_action_count = int(self._mask_log[-1]["valid_action_count"])
                masked_action_count = int(self._mask_log[-1]["masked_action_count"])

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
        budget_after = dict(self._engine.runtime_budgets.budgets)
        observation = build_observation(
            time=current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
        )
        info = {
            "time": current_time,
            "action_time": action_time,
            "action_id": action_id,
            "accepted": accepted,
            "updates": dict(updates),
            "budget_before": budget_before,
            "candidate_budgets": candidate_budgets,
            "budget_after": budget_after,
            "check_safety": self.check_safety,
            "safety_checked": action_was_checked,
            "valid_action_count": valid_action_count,
            "masked_action_count": masked_action_count,
            "selected_action_was_mask_valid": selected_action_was_mask_valid,
            "reject_reason": reject_reason,
            "reject_diagnostics": list(reject_diagnostics),
            "mode_changes": runtime_result.mode_change_count(),
            "lo_cancellations": runtime_result.lo_job_cancellation_count(),
            "deadline_misses": len(runtime_result.deadline_misses),
        }
        action_log_entry = dict(info)
        action_log_entry["time"] = action_time
        self._action_log.append(action_log_entry)
        return AgentStepResult(observation=observation, reward=reward, done=done, info=info)
