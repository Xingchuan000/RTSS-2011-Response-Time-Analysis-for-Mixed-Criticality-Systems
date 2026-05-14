"""阶段 7：RL 风格 reset/step 环境接口。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
import math
import numpy as np

from amc_py.amc import build_design_r_lo_map
from amc_py.budget_runtime import BudgetState
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.rl.actions import (
    BudgetAction,
    action_violates_hi_decrease_guard,
    apply_budget_action_candidate,
    build_budget_action_space,
)
from amc_py.rl.constraint_guided_pair import (
    ConstraintGuidedResolvedAction,
    ConstraintGuidedTransferCandidate,
    enumerate_constraint_guided_transfer_candidates,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.feature_state import RuntimeFeatureState
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.observation import NormalizationBounds, build_observation
from amc_py.rl.reward_config import RewardModeConfig, evaluate_reward_expression, load_reward_mode_config
from amc_py.rl.safety import (
    RuntimeBudgetSafetyChecker,
    merge_budget_candidate,
)
from amc_py.rl.types import AgentObservation, AgentStepResult
from amc_py.runtime_models import RuntimeConfig
from amc_py.runtime_scenarios import ExecutionScenario


@dataclass(frozen=True)
class CandidateBudgetUpdateDiagnosis:
    """候选预算更新诊断结果。

    该结构用于为 constraint-guided pair 提供“拒绝原因 + 违反约束行”的可解释信息，
    仅用于诊断，不改变环境原有动作执行语义。
    """

    accepted: bool
    reason: str
    normalized_reason: str
    violated_row_index: int | None
    violation_amount: float
    lhs_value: float | None
    bound_value: float | None
    slack_value: float | None
    row_coefficients: tuple[float, ...]
    task_names: tuple[str, ...]


def _normalize_candidate_reject_reason(reason: str) -> str:
    """把带前缀细节的 reject reason 归一化到稳定类别。"""

    text = str(reason or "unknown")
    if text.startswith("hi_lo_mode_violation"):
        return "hi_lo_mode_violation"
    if text.startswith("hi_mode_switch_violation"):
        return "hi_mode_switch_violation"
    if text.startswith("lo_mode_violation"):
        return "lo_mode_violation"
    if text.startswith("budget_floor_violation"):
        return "budget_floor_violation"
    if text.startswith("budget_upper_bound_violation"):
        return "budget_upper_bound_violation"
    if text.startswith("no_effective_budget_change"):
        return "no_effective_budget_change"
    if text.startswith("decrease_hi_forbidden"):
        return "decrease_hi_forbidden"
    if text.startswith("incremental_constraint_violation"):
        return "incremental_constraint_violation"
    if text.startswith("valid"):
        return "valid"
    return text or "unknown"


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
    reward_mode: str = "mendes"
    action_space: Literal["triple", "pair", "single", "constraint_guided_pair", "constraint_guided_transfer"] = "triple"
    # constraint-guided transfer 动作空间参数（支持旧名 constraint_guided_pair 作为 alias）。
    constraint_guided_pair_top_k_risk: int = 3
    constraint_guided_pair_top_k_decrease: int = 5
    constraint_guided_pair_prefer_lo: bool = False
    constraint_guided_pair_include_hi_risk_boost: bool = False
    constraint_guided_pair_allow_increase_only_when_safe: bool = False
    mask_detail_mode: Literal["minimal", "full"] = "minimal"
    budget_increase_ratio: float = 0.10
    budget_decrease_ratio: float = 0.05
    include_explicit_noop: bool = False
    budget_floor_ratio: float = 0.0
    forbid_decreasing_hi_budgets: bool = False
    # v11 特征配置。默认保持 v10_basic，确保旧实验行为不变。
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
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
    _selected_explicit_noop_actions: int = field(init=False, default=0, repr=False)
    _no_safe_action_steps: int = field(init=False, default=0, repr=False)
    _prev_job_start_count: int = field(init=False, default=0, repr=False)
    _prev_lo_overrun_count: int = field(init=False, default=0, repr=False)
    _prev_hi_overrun_count: int = field(init=False, default=0, repr=False)
    _prev_mode_changes: int = field(init=False, default=0, repr=False)
    _prev_lo_cancellations: int = field(init=False, default=0, repr=False)
    _prev_deadline_misses: int = field(init=False, default=0, repr=False)
    _task_index: dict[str, int] = field(init=False, repr=False)
    _task_names: tuple[str, ...] = field(init=False, repr=False)
    _task_upper_bounds: tuple[int, ...] = field(init=False, repr=False)
    _initial_budgets: dict[str, int] = field(init=False, repr=False)
    _safety_matrix_a: np.ndarray | None = field(init=False, default=None, repr=False)
    _safety_bounds: np.ndarray | None = field(init=False, default=None, repr=False)
    _reward_mode_config: RewardModeConfig = field(init=False, repr=False)
    # v11 运行时特征缓存（EMA/history/event window）。
    _feature_state: RuntimeFeatureState | None = field(init=False, default=None, repr=False)
    _last_observation: AgentObservation | None = field(init=False, default=None, repr=False)
    _last_constraint_guided_transfer_candidates: tuple[ConstraintGuidedTransferCandidate, ...] = field(
        init=False, default=(), repr=False
    )

    def __post_init__(self) -> None:
        """初始化固定动作空间。"""
        if self.budget_floor_ratio < 0.0 or self.budget_floor_ratio > 1.0:
            raise ValueError("budget_floor_ratio must be in [0, 1]")
        # 加载奖励模式配置（权重 + 公式）。
        # 这样 reward 计算方式可由配置文件驱动，而非写死在代码中。
        self._reward_mode_config = load_reward_mode_config(self.reward_mode)

        if self.action_space == "constraint_guided_pair":
            self.action_space = "constraint_guided_transfer"
        self._actions = build_budget_action_space(
            self.ordered_tasks,
            action_space=self.action_space,
            budget_increase_ratio=self.budget_increase_ratio,
            budget_decrease_ratio=self.budget_decrease_ratio,
            include_explicit_noop=self.include_explicit_noop,
            constraint_guided_pair_top_k_risk=self.constraint_guided_pair_top_k_risk,
            constraint_guided_pair_top_k_decrease=self.constraint_guided_pair_top_k_decrease,
            constraint_guided_pair_include_hi_risk_boost=self.constraint_guided_pair_include_hi_risk_boost,
        )
        self._action_log = []
        self._mask_log = []
        self._mask_reject_reasons = Counter()
        self._last_mask_details = []
        self._task_names = tuple(task.name for task in self.ordered_tasks)
        self._task_index = {name: idx for idx, name in enumerate(self._task_names)}
        self._task_upper_bounds = tuple(
            (
                (task.c_hi if task.criticality is Criticality.HI and task.c_hi > 0 else task.deadline)
                if task.criticality is Criticality.HI
                else task.deadline
            )
            for task in self.ordered_tasks
        )
        # 记录环境初始预算，用于阶段 2 的预算变化归一化惩罚。
        self._initial_budgets = {task.name: task.c_lo for task in self.ordered_tasks}
        self._last_observation = None

    def _resolve_constraint_guided_pair_action(
        self,
        action: BudgetAction,
        *,
        check_safety: bool,
    ) -> ConstraintGuidedResolvedAction:
        """将 constraint-guided transfer 槽位动作解析为当前状态下可执行的 bundled 更新。"""

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")
        if self._last_observation is None:
            return ConstraintGuidedResolvedAction(
                valid=False,
                reject_reason="constraint_guided_no_observation",
                slot_id=action.action_id,
                increase_rank=action.constraint_guided_increase_rank,
                increase_idx=None,
                decrease_indices=(),
                updates={},
                candidate_budgets=dict(self._engine.runtime_budgets.budgets),
                single_increase_was_safe=False,
                safety_checked=False,
                diagnosis_reason=None,
                violated_row_index=None,
            )
        if self.feature_config.observation_mode != "v11_full_10d":
            return ConstraintGuidedResolvedAction(
                valid=False,
                reject_reason="constraint_guided_unsupported_observation_mode",
                slot_id=action.action_id,
                increase_rank=action.constraint_guided_increase_rank,
                increase_idx=None,
                decrease_indices=(),
                updates={},
                candidate_budgets=dict(self._engine.runtime_budgets.budgets),
                single_increase_was_safe=False,
                safety_checked=False,
                diagnosis_reason=None,
                violated_row_index=None,
            )
        candidates = enumerate_constraint_guided_transfer_candidates(
            tasks=self.ordered_tasks,
            current_budgets=self._engine.runtime_budgets.budgets,
            safety_checker=self._ensure_checker() if check_safety else None,
            runtime_features=self,
            budget_increase_ratio=self.budget_increase_ratio,
            budget_decrease_ratio=self.budget_decrease_ratio,
            budget_floor_ratio=self.budget_floor_ratio,
            top_k_risk=self.constraint_guided_pair_top_k_risk,
            top_k_decrease=self.constraint_guided_pair_top_k_decrease,
            prefer_lo=self.constraint_guided_pair_prefer_lo,
            include_hi_risk_boost=self.constraint_guided_pair_include_hi_risk_boost,
            validate_safety=check_safety,
        )
        self._last_constraint_guided_transfer_candidates = candidates
        slot_index = int(action.constraint_guided_increase_rank or 0)
        found = None
        for candidate in candidates:
            if candidate.slot_index == slot_index:
                found = candidate
                break
        if found is None:
            return ConstraintGuidedResolvedAction(
                valid=False,
                reject_reason="constraint_guided_no_increase_candidate",
                slot_id=action.action_id,
                increase_rank=slot_index,
                increase_idx=None,
                decrease_indices=(),
                updates={},
                candidate_budgets=dict(self._engine.runtime_budgets.budgets),
                single_increase_was_safe=False,
                safety_checked=False,
                diagnosis_reason=None,
                violated_row_index=None,
            )
        pair_budgets = dict(found.candidate_budgets)
        valid = bool(found.accepted)
        reason = found.reject_reason or "constraint_guided_transfer_rejected"
        if valid:
            updates = {
                name: int(value)
                for name, value in pair_budgets.items()
                if int(value) != int(self._engine.runtime_budgets.budgets[name])
            }
        else:
            updates = {}
        return ConstraintGuidedResolvedAction(
            valid=bool(valid),
            reject_reason=None if valid else reason,
            slot_id=action.action_id,
            increase_rank=found.increase_rank,
            increase_idx=found.increase_task_idx,
            decrease_indices=found.decrease_task_indices,
            updates=updates,
            candidate_budgets=pair_budgets,
            single_increase_was_safe=found.single_increase_accepted,
            safety_checked=check_safety,
            diagnosis_reason=None if valid else reason,
            violated_row_index=found.violated_row_index,
        )

    def _compute_normalized_budget_change(
        self,
        *,
        budget_before: dict[str, int],
        candidate_budgets: dict[str, int],
        initial_budgets: dict[str, int],
        changed_task_ids: Sequence[str],
    ) -> float:
        """按文档口径计算预算改变量归一化和。

        公式严格为：
        sum(|candidate - before| / initial)

        其中 initial 为 reset 时初始预算，用于把不同任务尺度统一到可比较量纲。
        """

        normalized_total = 0.0
        for task_id in changed_task_ids:
            denominator = max(float(initial_budgets[task_id]), 1e-9)
            normalized_total += abs(float(candidate_budgets[task_id]) - float(budget_before[task_id])) / denominator
        return normalized_total

    def _compute_budget_drift_mean(self, *, budgets: dict[str, int], initial_budgets: dict[str, int]) -> float:
        """计算预算漂移惩罚项中的平均欠配比值。

        公式：
        mean_i max(0, 1 - B_i / B_i_initial)
        """

        if not self._task_names:
            return 0.0
        drift_total = 0.0
        for task_name in self._task_names:
            current_budget = float(budgets[task_name])
            initial_budget = max(float(initial_budgets[task_name]), 1e-9)
            drift_total += max(0.0, 1.0 - current_budget / initial_budget)
        return drift_total / float(len(self._task_names))

    def _budget_floor_violation(
        self,
        *,
        updates: dict[str, int],
    ) -> str | None:
        """检查局部预算更新是否违反 episode 初始预算下限。

        约束语义严格遵循文档：
        - `budget_floor_ratio <= 0.0` 时，表示不启用 floor，直接返回 `None`；
        - 只检查本步 `updates` 中被修改的任务，因为未修改任务不可能在当前 step 突然跌破 floor；
        - 下界值按 `ceil(initial_budget * budget_floor_ratio)` 计算，避免整数预算被向下取整后放松过多；
        - 一旦发现第一个违规任务，立即返回带任务名的 reject_reason，便于日志定位。
        """

        if self.budget_floor_ratio <= 0.0:
            return None

        for task_name, candidate_budget in updates.items():
            initial_budget = self._initial_budgets[task_name]
            floor_value = max(1, math.ceil(initial_budget * self.budget_floor_ratio))
            if candidate_budget < floor_value:
                return f"budget_floor_violation:{task_name}"
        return None

    def check_candidate_budget_update(
        self,
        *,
        new_budgets: dict[str, int],
    ) -> tuple[bool, str]:
        """检查候选预算向量在当前环境下是否可行。

        返回值语义严格固定为 `(is_valid, reject_reason)`：
        - `is_valid=True` 时 `reject_reason=="valid"`；
        - `is_valid=False` 时 `reject_reason` 仅取以下之一：
          `incremental_constraint_violation` / `budget_floor_violation`
          / `budget_upper_bound_violation` / `no_effective_budget_change`
          / `decrease_hi_forbidden` / `unknown`。

        设计约束：
        1. 只做计划内校验，不引入额外兜底规则；
        2. 复用 `valid_action_mask` 与 `step` 的同一套硬约束语义；
        3. 该方法用于 diagnostic-only 场景，不修改环境状态。
        """

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")

        budget_before = dict(self._engine.runtime_budgets.budgets)
        updates: dict[str, int] = {}
        for task_name in self._task_names:
            if task_name not in new_budgets:
                return False, "unknown"
            old_value = int(budget_before[task_name])
            new_value = int(new_budgets[task_name])
            if old_value != new_value:
                updates[task_name] = new_value

        if not updates:
            return False, "no_effective_budget_change"

        has_hi_decrease = False
        for task_name, new_value in updates.items():
            idx = self._task_index[task_name]
            old_value = int(budget_before[task_name])
            if new_value < old_value and self.ordered_tasks[idx].criticality is Criticality.HI:
                has_hi_decrease = True
                break
        if self.forbid_decreasing_hi_budgets and has_hi_decrease:
            return False, "decrease_hi_forbidden"

        for task_name, new_value in updates.items():
            idx = self._task_index[task_name]
            upper_bound = int(self._task_upper_bounds[idx])
            if new_value < 1 or new_value > upper_bound:
                return False, "budget_upper_bound_violation"

        floor_reject = self._budget_floor_violation(updates=updates)
        if floor_reject is not None:
            return False, "budget_floor_violation"

        if not self.check_safety:
            return True, "valid"

        checker = self._ensure_checker()
        report = checker.validate_candidate(dict(new_budgets))
        if report.accepted:
            return True, "valid"
        # 这里严格按 safety checker 的 `report.reason` 前缀做映射，不再从 diagnostics 猜测。
        # 目的：
        # 1) 与计划文档保持一致；
        # 2) 保留后续脚本对具体约束类型（HI/LO）做细分统计的能力；
        # 3) 同时输出统一聚合口径 `incremental_constraint_violation`。
        reason = str(report.reason)
        if (
            reason.startswith("hi_lo_mode_violation")
            or reason.startswith("hi_mode_switch_violation")
            or reason.startswith("lo_mode_violation")
        ):
            return False, "incremental_constraint_violation"
        if reason:
            return False, reason
        return False, "unknown"

    def diagnose_candidate_budget_update(
        self,
        *,
        new_budgets: dict[str, int],
    ) -> CandidateBudgetUpdateDiagnosis:
        """诊断候选预算更新并返回最严重违反约束行的信息。"""

        accepted, reason = self.check_candidate_budget_update(new_budgets=new_budgets)
        normalized_reason = _normalize_candidate_reject_reason(reason)
        task_names = tuple(task.name for task in self.ordered_tasks)
        empty_coefficients = tuple(0.0 for _ in task_names)

        if accepted:
            return CandidateBudgetUpdateDiagnosis(
                accepted=True,
                reason="valid",
                normalized_reason="valid",
                violated_row_index=None,
                violation_amount=0.0,
                lhs_value=None,
                bound_value=None,
                slack_value=None,
                row_coefficients=empty_coefficients,
                task_names=task_names,
            )

        if self._engine is None:
            raise RuntimeError("环境尚未 reset")
        if self._safety_matrix_a is None or self._safety_bounds is None:
            if not self.check_safety:
                return CandidateBudgetUpdateDiagnosis(
                    accepted=False,
                    reason=reason,
                    normalized_reason=normalized_reason,
                    violated_row_index=None,
                    violation_amount=0.0,
                    lhs_value=None,
                    bound_value=None,
                    slack_value=None,
                    row_coefficients=empty_coefficients,
                    task_names=task_names,
                )
            checker = self._ensure_checker()
            self._safety_matrix_a, self._safety_bounds = checker.build_linear_constraints()

        assert self._safety_matrix_a is not None
        assert self._safety_bounds is not None

        budget_vector = np.array([float(new_budgets[name]) for name in self._task_names], dtype=np.float64)
        lhs = self._safety_matrix_a @ budget_vector
        violation = lhs - self._safety_bounds
        worst_row = int(np.argmax(violation))
        violation_amount = float(violation[worst_row])

        current_vector = np.array(
            [float(self._engine.runtime_budgets.budgets[name]) for name in self._task_names],
            dtype=np.float64,
        )
        current_lhs = self._safety_matrix_a @ current_vector
        slack = self._safety_bounds - current_lhs

        violated_row_index: int | None = worst_row if violation_amount > 0.0 else None
        return CandidateBudgetUpdateDiagnosis(
            accepted=False,
            reason=reason,
            normalized_reason=normalized_reason,
            violated_row_index=violated_row_index,
            violation_amount=max(0.0, violation_amount),
            lhs_value=float(lhs[worst_row]),
            bound_value=float(self._safety_bounds[worst_row]),
            slack_value=float(slack[worst_row]),
            row_coefficients=tuple(float(x) for x in self._safety_matrix_a[worst_row, :]),
            task_names=task_names,
        )

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
            "action_space_type": self.action_space,
            "action_count": len(self._actions),
            "constraint_guided_pair_top_k_risk": self.constraint_guided_pair_top_k_risk,
            "constraint_guided_pair_top_k_decrease": self.constraint_guided_pair_top_k_decrease,
            "constraint_guided_pair_prefer_lo": self.constraint_guided_pair_prefer_lo,
            "constraint_guided_pair_include_hi_risk_boost": self.constraint_guided_pair_include_hi_risk_boost,
            "constraint_guided_pair_allow_increase_only_when_safe": (
                self.constraint_guided_pair_allow_increase_only_when_safe
            ),
            "budget_increase_ratio": self.budget_increase_ratio,
            "budget_decrease_ratio": self.budget_decrease_ratio,
            "budget_floor_ratio": self.budget_floor_ratio,
            "valid_action_count_mean": (sum(valid_counts) / mask_checks) if mask_checks > 0 else 0.0,
            "masked_action_count_mean": (sum(invalid_counts) / mask_checks) if mask_checks > 0 else 0.0,
            "masked_action_count_max": max(invalid_counts) if invalid_counts else 0,
            "mask_rejection_rate_mean": (
                (sum(invalid_counts) / (mask_checks * len(self._actions))) if mask_checks > 0 else 0.0
            ),
            "selected_invalid_mask_actions": self._selected_invalid_mask_actions,
            # 显式 noop（动作空间中的真实动作）与 action_id=None（无合法动作时被动空转）语义不同，
            # 这里单独统计显式 noop，供阶段 1 验证“agent 是否主动选择不调整预算”。
            "selected_explicit_noop_actions": self._selected_explicit_noop_actions,
            "selected_explicit_noop_rate": (
                (self._selected_explicit_noop_actions / mask_checks) if mask_checks > 0 else 0.0
            ),
            "no_safe_action_steps": self._no_safe_action_steps,
            # HI 预算保护约束统计：
            # - count: 运行期间累计有多少动作因“decrease 命中 HI”被 mask；
            # - rate : 上述 count 在“总候选动作数(mask_checks * action_count)”中的占比。
            # 这样在不看原始日志的情况下，也能从 CSV 直接判断该约束对动作空间的压缩强度。
            "masked_decrease_hi_forbidden_count": int(self._mask_reject_reasons.get("decrease_hi_forbidden", 0)),
            "masked_decrease_hi_forbidden_rate": (
                (int(self._mask_reject_reasons.get("decrease_hi_forbidden", 0)) / (mask_checks * len(self._actions)))
                if mask_checks > 0
                else 0.0
            ),
            # budget floor 统计：
            # - count: 有多少候选动作因为“会把某任务预算压到初始预算 floor 以下”而被 mask；
            # - rate : 上述 count 在总候选动作数中的占比。
            # 这两个字段用于训练/评估后直接判断 floor 约束是否真实参与了动作过滤。
            "masked_budget_floor_violation_count": int(self._mask_reject_reasons.get("budget_floor_violation", 0)),
            "masked_budget_floor_violation_rate": (
                (int(self._mask_reject_reasons.get("budget_floor_violation", 0)) / (mask_checks * len(self._actions)))
                if mask_checks > 0
                else 0.0
            ),
            "masked_constraint_guided_no_increase_candidate_count": int(
                self._mask_reject_reasons.get("constraint_guided_no_increase_candidate", 0)
            ),
            "masked_constraint_guided_no_decrease_candidate_count": int(
                self._mask_reject_reasons.get("constraint_guided_no_decrease_candidate", 0)
            ),
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
        budget_before = dict(self._engine.runtime_budgets.budgets)
        budget_array = np.array([float(budget_before[name]) for name in self._task_names], dtype=np.float64)
        if self.check_safety:
            assert checker is not None
            if self._safety_matrix_a is None or self._safety_bounds is None:
                # 缓存 A/b：约束不随 step 变化，只需要构建一次。
                self._safety_matrix_a, self._safety_bounds = checker.build_linear_constraints()
            current_lhs = self._safety_matrix_a @ budget_array
            slack = self._safety_bounds - current_lhs
        else:
            slack = None
        for action in self._actions:
            # constraint-guided pair 动作槽位需要在当前观测下动态解析。
            if action.is_constraint_guided_pair:
                resolved = self._resolve_constraint_guided_pair_action(action, check_safety=self.check_safety)
                mask.append(bool(resolved.valid))
                if not resolved.valid and resolved.reject_reason is not None:
                    reject_reason_counts[_normalize_candidate_reject_reason(resolved.reject_reason)] += 1
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": bool(resolved.valid),
                            "reject_reason": resolved.reject_reason,
                            "is_noop": False,
                            "is_constraint_guided_pair": True,
                            "increase_idx": resolved.increase_idx,
                            "decrease_indices": tuple(resolved.decrease_indices),
                            "single_increase_was_safe": resolved.single_increase_was_safe,
                            "violated_row_index": resolved.violated_row_index,
                        }
                    )
                else:
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": bool(resolved.valid),
                            "reject_reason": resolved.reject_reason,
                            "updates": dict(resolved.updates),
                            "budget_before": budget_before,
                            "candidate_budgets": dict(resolved.candidate_budgets),
                            "is_noop": False,
                            "is_constraint_guided_pair": True,
                            "increase_idx": resolved.increase_idx,
                            "decrease_indices": tuple(resolved.decrease_indices),
                            "increase_rank": resolved.increase_rank,
                            "single_increase_was_safe": resolved.single_increase_was_safe,
                            "diagnosis_reason": resolved.diagnosis_reason,
                            "violated_row_index": resolved.violated_row_index,
                        }
                    )
                continue
            # 阶段 1 语义：显式 noop 必须始终合法。
            # 1) 不能再走“是否有效更新”的过滤；
            # 2) 不能再走预算安全检查；
            # 3) mask 详情里仍记录预算前后（相同）与 is_noop 标记，便于调试。
            if action.is_noop:
                mask.append(True)
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": True,
                            "reject_reason": None,
                            "is_noop": True,
                        }
                    )
                else:
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": True,
                            "reject_reason": None,
                            "updates": {},
                            "budget_before": budget_before,
                            "candidate_budgets": dict(budget_before),
                            "is_noop": True,
                        }
                    )
                continue
            # HI 预算保护（阶段 3 核心规则）：
            # - 触发条件：action.decrease_indices 中存在任意 HI 任务；
            # - 处理方式：直接 mask=False，并记录 reject_reason=decrease_hi_forbidden；
            # - 语义边界：只约束 decrease，不限制 increase，因此“给 HI 任务加预算”仍然允许。
            # 注意：显式 noop 已在前面提前放行，不受此规则影响。
            if action_violates_hi_decrease_guard(
                action=action,
                ordered_tasks=self.ordered_tasks,
                forbid_decreasing_hi_budgets=self.forbid_decreasing_hi_budgets,
            ):
                mask.append(False)
                reject_reason_counts["decrease_hi_forbidden"] += 1
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": "decrease_hi_forbidden",
                            "is_noop": False,
                        }
                    )
                else:
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": "decrease_hi_forbidden",
                            "updates": {},
                            "budget_before": budget_before,
                            "candidate_budgets": dict(budget_before),
                            "is_noop": False,
                        }
                    )
                continue
            updates: dict[str, int] = {}
            delta_pairs: list[tuple[int, float]] = []
            reject_reason: str | None = None
            if action.increase_idx is not None:
                inc_idx = action.increase_idx
                inc_name = self._task_names[inc_idx]
                old_inc = int(budget_before[inc_name])
                inc_value = math.ceil(old_inc * (1.0 + action.increase_ratio))
                inc_value = min(inc_value, self._task_upper_bounds[inc_idx])
                inc_value = max(1, inc_value)
                updates[inc_name] = inc_value
                if inc_value == old_inc:
                    reject_reason = "no_effective_budget_change"
                delta_pairs.append((inc_idx, float(inc_value - old_inc)))
            for dec_idx in action.decrease_indices:
                dec_name = self._task_names[dec_idx]
                old_dec = int(budget_before[dec_name])
                dec_value = math.floor(old_dec * (1.0 - action.decrease_ratio))
                dec_value = max(1, dec_value)
                updates[dec_name] = dec_value
                if dec_value == old_dec:
                    reject_reason = "no_effective_budget_change"
                delta_pairs.append((dec_idx, float(dec_value - old_dec)))
            if reject_reason is not None:
                mask.append(False)
                reject_reason_counts["no_effective_budget_change"] += 1
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": "no_effective_budget_change",
                            "is_noop": False,
                        }
                    )
                else:
                    candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": "no_effective_budget_change",
                            "updates": dict(updates),
                            "budget_before": budget_before,
                            "candidate_budgets": candidate_budgets,
                            "is_noop": False,
                        }
                    )
                continue
            # budget floor 必须在 safety check 之前执行。
            # 原因是它属于“动作可行性硬约束”，不是 reward 正则，也不是 safety checker 的线性约束。
            # 因此一旦违反 floor，应直接把动作标记为 invalid，并保留带任务名的 reject_reason。
            floor_reject_reason = self._budget_floor_violation(updates=updates)
            if floor_reject_reason is not None:
                mask.append(False)
                reject_reason_counts["budget_floor_violation"] += 1
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": floor_reject_reason,
                            "is_noop": False,
                        }
                    )
                else:
                    candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": False,
                            "reject_reason": floor_reject_reason,
                            "updates": dict(updates),
                            "budget_before": budget_before,
                            "candidate_budgets": candidate_budgets,
                            "is_noop": False,
                        }
                    )
                continue
            if not self.check_safety:
                mask.append(True)
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": True,
                            "reject_reason": None,
                            "is_noop": False,
                        }
                    )
                else:
                    candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": True,
                            "reject_reason": None,
                            "updates": dict(updates),
                            "budget_before": budget_before,
                            "candidate_budgets": candidate_budgets,
                            "is_noop": False,
                        }
                    )
                continue
            assert slack is not None
            delta_lhs = np.zeros_like(slack)
            for idx, delta in delta_pairs:
                delta_lhs += self._safety_matrix_a[:, idx] * delta  # type: ignore[index]
            accepted = bool(np.all(delta_lhs <= slack))
            if not accepted:
                reject_reason = "incremental_constraint_violation"
                reject_reason_counts[reject_reason] += 1
            mask.append(accepted)
            if self.mask_detail_mode == "minimal":
                mask_details.append(
                    {
                        "action_id": action.action_id,
                        "valid": accepted,
                        "reject_reason": reject_reason,
                        "is_noop": False,
                    }
                )
            else:
                candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
                report = checker.validate_candidate(candidate_budgets)
                mask_details.append(
                    {
                        "action_id": action.action_id,
                        "valid": accepted,
                        "reject_reason": None if accepted else reject_reason,
                        "reject_diagnostics": list(report.diagnostics),
                        "updates": dict(updates),
                        "budget_before": budget_before,
                        "candidate_budgets": candidate_budgets,
                        "is_noop": False,
                    }
                )
        valid_action_count = sum(mask)
        masked_action_count = len(mask) - valid_action_count
        if valid_action_count == 0:
            self._no_safe_action_steps += 1
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
        # 阶段 7：将默认 checker 缓存在 env 实例上，避免每次 mask/step 都重复构造。
        # 约束：只在调用方未显式注入 `safety_checker` 时才懒加载一次默认 checker。
        design_r_lo = build_design_r_lo_map(self.ordered_tasks)
        self.safety_checker = RuntimeBudgetSafetyChecker(
            ordered_tasks=self.ordered_tasks,
            design_r_lo=design_r_lo,
        )
        return self.safety_checker

    def _compute_current_safety_margin_min(self) -> float:
        """计算当前预算向量在线性安全约束下的最小归一化裕量。

        语义与实现计划保持一致：
        - check_safety=False 时直接返回 1.0；
        - 复用 valid_action_mask 中已缓存的 A/b，避免重复构建约束；
        - 不吞异常，若 checker 构建失败应直接暴露问题。
        """

        if not self.check_safety:
            return 1.0
        if self._engine is None:
            return 1.0

        checker = self._ensure_checker()
        if self._safety_matrix_a is None or self._safety_bounds is None:
            self._safety_matrix_a, self._safety_bounds = checker.build_linear_constraints()

        budget_array = np.array(
            [float(self._engine.runtime_budgets.budgets[name]) for name in self._task_names],
            dtype=np.float64,
        )
        lhs = self._safety_matrix_a @ budget_array
        slack = self._safety_bounds - lhs
        denom = np.maximum(1.0, np.abs(self._safety_bounds))
        margin = float(np.min(slack / denom))
        return float(np.clip(margin, 0.0, 1.0))

    def _update_feature_state(
        self,
        *,
        delta_job_start: int,
        delta_mode_changes: int,
        delta_lo_cancellations: int,
        delta_hi_overrun: int,
        delta_lo_overrun: int,
    ) -> None:
        """在每个 step 结束后更新 v11 运行时特征缓存。

        更新顺序与文档一致：
        1. 先按“每任务新样本计数”判断当前 step 是否出现新样本；
        2. 仅对有新样本的任务更新 ema_cost / overrun_ema；
        3. 仅对有新样本的任务追加 recent_cost 到 cost_history；
        3. 追加全局事件窗口计数。
        """

        if self._feature_state is None:
            return
        if self._engine is None:
            return

        cfg = self.feature_config
        feature_state = self._feature_state

        for task in self.ordered_tasks:
            task_name = task.name
            # current_completion_count 表示该任务截至当前时刻总共产生了多少个“新执行样本”。
            # last_seen_completion_count 表示 feature_state 已消费到哪个样本版本号。
            # 只有 current > last_seen 才说明“本 step 区间内真的出现了新样本”。
            current_completion_count = self._monitor.completed_job_count_by_task.get(task_name, 0)
            last_seen_completion_count = feature_state.last_seen_completion_count.get(task_name, 0)

            feature_state.init_task(task_name, init_cost=float(task.c_lo))
            if current_completion_count <= last_seen_completion_count:
                # 没有新样本：跳过该任务的 EMA/history 更新，避免重复采样污染。
                continue

            budget = float(self._engine.runtime_budgets.budgets[task_name])
            recent_cost = float(self._monitor.recent_execution.get(task_name, 0.0))

            old_ema = feature_state.ema_cost[task_name]
            feature_state.ema_cost[task_name] = cfg.ema_alpha * recent_cost + (1.0 - cfg.ema_alpha) * old_ema

            overrun_flag = 1.0 if recent_cost > budget else 0.0
            old_overrun_ema = feature_state.overrun_ema[task_name]
            feature_state.overrun_ema[task_name] = (
                cfg.overrun_ema_alpha * overrun_flag + (1.0 - cfg.overrun_ema_alpha) * old_overrun_ema
            )

            feature_state.cost_history[task_name].append(recent_cost)
            # 标记该任务的样本版本号已被消费到 current_completion_count。
            feature_state.last_seen_completion_count[task_name] = current_completion_count

        feature_state.append_event_window(
            mode_changes=delta_mode_changes,
            lo_cancellations=delta_lo_cancellations,
            hi_overruns=delta_hi_overrun,
            lo_overruns=delta_lo_overrun,
            job_starts=delta_job_start,
        )

    def reset(self, seed: int | None = None) -> AgentObservation:  # noqa: ARG002
        """重置环境并返回初始观测。"""

        # 奖励参数不再写死在代码中，而是按 reward_mode 从配置文件加载。
        # 这样后续调参只需改配置文件，不需要改动 Python 代码。
        self._monitor = RuntimeMonitor(reward_mode=self.reward_mode)
        self._engine = EventRuntimeEngine.build(
            ordered_tasks=self.ordered_tasks,
            scenario=self.scenario,
            config=self.runtime_config,
            budget_state=BudgetState.from_tasks(self.ordered_tasks),
            monitor=self._monitor,
        )
        # 先处理 time=0 的边界事件，再返回首次观测，避免 agent 早于首批 release 决策。
        self._engine.run_until(0, include_boundary=True)
        # reset 后把“本轮 episode 的初始预算快照”固定下来。
        # 后续每一步 budget_change_norm 都以该快照作归一化分母来源。
        self._initial_budgets = dict(self._engine.runtime_budgets.budgets)
        self._done = False
        self._action_log = []
        self._mask_log = []
        self._mask_reject_reasons = Counter()
        self._last_mask_details = []
        self._safety_checked_actions = 0
        self._safety_accepted_actions = 0
        self._safety_rejected_actions = 0
        self._selected_invalid_mask_actions = 0
        self._selected_explicit_noop_actions = 0
        self._no_safe_action_steps = 0
        self._prev_job_start_count = 0
        self._prev_lo_overrun_count = 0
        self._prev_hi_overrun_count = 0
        self._prev_mode_changes = 0
        self._prev_lo_cancellations = 0
        self._prev_deadline_misses = 0
        # v11 特征缓存在每个 episode reset 后重置，避免跨 episode 污染。
        self._feature_state = RuntimeFeatureState(
            history_k=self.feature_config.history_k,
            event_window=self.feature_config.event_window,
        )
        for task in self.ordered_tasks:
            init_cost = float(task.c_lo)
            self._feature_state.init_task(task.name, init_cost=init_cost)
            self._feature_state.cost_history[task.name].append(init_cost)
            # reset 时把“已消费样本版本号”对齐到 monitor 当前计数，
            # 避免后续 step 把 reset 前的旧样本误判为“本 interval 新样本”。
            self._feature_state.last_seen_completion_count[task.name] = (
                self._monitor.completed_job_count_by_task.get(task.name, 0)
            )
        observation = build_observation(
            time=self._engine.current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
            feature_state=self._feature_state,
            feature_config=self.feature_config,
            safety_margin_min=self._compute_current_safety_margin_min(),
        )
        self._last_observation = observation
        return observation

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
        is_explicit_noop_action = False

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
            if action.is_noop:
                # 显式 noop 的执行语义是“主动接受但不改预算”：
                # - 不做 safety check；
                # - 不调用 apply_budget_updates；
                # - 仅在统计上标记为一次被主动选择的 noop。
                # 这样训练/评估可以区分：
                # - action_id=None 的隐式 noop（通常因无合法动作）；
                # - action_id!=None 的显式 noop（agent 明确选择“不调整预算”）。
                accepted = True
                is_explicit_noop_action = True
                self._selected_explicit_noop_actions += 1
                updates = {}
                candidate_budgets = dict(budget_before)
            elif action.is_constraint_guided_pair:
                # constraint-guided pair 由 env 在当前状态下动态解析，
                # 解析过程不改运行时预算，只有解析成功且 valid=True 才真正 apply。
                resolved = self._resolve_constraint_guided_pair_action(action, check_safety=self.check_safety)
                accepted = bool(resolved.valid)
                reject_reason = resolved.reject_reason
                updates = dict(resolved.updates) if resolved.valid else {}
                candidate_budgets = dict(resolved.candidate_budgets)
                action_was_checked = bool(resolved.safety_checked)
                if action_was_checked:
                    self._safety_checked_actions += 1
                if accepted:
                    if action_was_checked:
                        self._safety_accepted_actions += 1
                    self._engine.apply_budget_updates(updates)
                else:
                    normalized_reject = _normalize_candidate_reject_reason(reject_reason or "unknown")
                    if action_was_checked and normalized_reject in {
                        "incremental_constraint_violation",
                        "hi_lo_mode_violation",
                        "hi_mode_switch_violation",
                        "lo_mode_violation",
                    }:
                        self._safety_rejected_actions += 1
            else:
                # 运行时执行路径与 mask 路径做同样判定：
                # 即便调用方绕过了 valid_action_mask（例如手工传 action_id），
                # 这里也会拒绝 HI decrease 动作，确保系统级语义一致。
                if action_violates_hi_decrease_guard(
                    action=action,
                    ordered_tasks=self.ordered_tasks,
                    forbid_decreasing_hi_budgets=self.forbid_decreasing_hi_budgets,
                ):
                    accepted = False
                    reject_reason = "decrease_hi_forbidden"
                    updates = {}
                    candidate_budgets = dict(budget_before)
                else:
                    updates = apply_budget_action_candidate(
                        action=action,
                        budget_state=self._engine.runtime_budgets,
                        ordered_tasks=self.ordered_tasks,
                    )
                    candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)
                    # step() 的 floor 兜底与 mask 路径必须保持同一语义：
                    # 即便调用方绕过了 valid_action_mask，这里也要拒绝任何会跌破 floor 的动作。
                    floor_reject_reason = self._budget_floor_violation(updates=updates)
                    if floor_reject_reason is not None:
                        accepted = False
                        reject_reason = floor_reject_reason
                    elif self.check_safety:
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
        current_time = self._engine.current_time
        if self.runtime_config.end_time is None:
            done = False
        else:
            done = current_time >= self.runtime_config.end_time
        self._done = done

        runtime_result = self._engine.finish()
        mode_changes = runtime_result.mode_change_count()
        lo_cancellations = runtime_result.lo_job_cancellation_count()
        deadline_misses = len(runtime_result.deadline_misses)
        delta_job_start = self._monitor.job_start_count - self._prev_job_start_count
        delta_lo_overrun = self._monitor.lo_overrun_count - self._prev_lo_overrun_count
        delta_hi_overrun = self._monitor.hi_overrun_count - self._prev_hi_overrun_count
        delta_mode_changes = mode_changes - self._prev_mode_changes
        delta_lo_cancellations = lo_cancellations - self._prev_lo_cancellations
        delta_deadline_misses = deadline_misses - self._prev_deadline_misses
        # interval_time 表示“本次 step 覆盖了多长模拟时间”。
        # 这里严格按文档采用 max(1.0, current_time - action_time)：
        # 1) 分母不会为 0；
        # 2) 后续 mode_change_rate 使用该值归一化，避免短 interval 下数值异常放大。
        interval_time = max(1.0, float(current_time - action_time))
        # delta_total_jobs 表示“本次 interval 内启动了多少个 job”。
        # 同样使用 max(1.0, delta_job_start) 作为归一化分母，确保 overrun/miss 比率可计算。
        delta_total_jobs = max(1.0, float(delta_job_start))
        # LO overrun rate：interval 内 LO overrun 次数 / interval 内 job_start 次数。
        lo_overrun_rate = float(delta_lo_overrun) / delta_total_jobs
        # HI overrun rate：interval 内 HI overrun 次数 / interval 内 job_start 次数。
        hi_overrun_rate = float(delta_hi_overrun) / delta_total_jobs
        # mode change rate：interval 内模式切换次数 / interval 覆盖时长。
        # 该变量保留是为了兼容可能已存在的旧 reward 公式。
        mode_change_rate = float(delta_mode_changes) / interval_time
        # mode_change_per_job：interval 内模式切换次数 / interval 内 job_start 次数。
        # 这是当前更稳的主推荐指标：每次 mode change 的惩罚会随着该 interval 的 job 数归一化，
        # 避免“单次 mode change 在 job 很少时对 reward 造成过大离散冲击”。
        mode_change_per_job = float(delta_mode_changes) / delta_total_jobs
        # LO cancellation rate：interval 内 LO 任务取消次数 / interval 内 job_start 次数。
        lo_cancellation_rate = float(delta_lo_cancellations) / delta_total_jobs
        # deadline miss rate：interval 内 deadline miss 次数 / interval 内 job_start 次数。
        deadline_miss_rate = float(delta_deadline_misses) / delta_total_jobs
        # invalid_action：动作经过安全检查且未被接受时记为 1.0，否则记为 0.0。
        # 这样可以把“动作被拒绝”作为一个可在奖励公式里直接惩罚的显式信号。
        invalid_action = 1.0 if action_was_checked and not accepted else 0.0

        step_reward_job_start = 0.0
        step_reward_lo_overrun = 0.0
        step_reward_hi_overrun = 0.0
        step_reward_mode_change = 0.0
        step_reward_lo_cancellation = 0.0
        step_reward_deadline_miss = 0.0
        # 为了兼容已有 monitor 行为，这里仍调用 consume_reward 清空累计器；
        # 真正用于训练的 reward 数值由“配置文件中的公式”重新计算。
        _ = self._monitor.consume_reward()
        event_job_start_reward = self._monitor.reward_weights.job_start * delta_job_start
        event_lo_overrun_reward = self._monitor.reward_weights.lo_overrun * delta_lo_overrun
        event_hi_overrun_reward = self._monitor.reward_weights.hi_overrun * delta_hi_overrun
        paper_reward = evaluate_reward_expression(
            self._reward_mode_config.paper_reward_formula,
            {
                "delta_job_start": float(delta_job_start),
                "delta_lo_overrun": float(delta_lo_overrun),
                "delta_hi_overrun": float(delta_hi_overrun),
                "event_job_start_reward": float(event_job_start_reward),
                "event_lo_overrun_reward": float(event_lo_overrun_reward),
                "event_hi_overrun_reward": float(event_hi_overrun_reward),
            },
        )
        reward = paper_reward
        budget_change_norm = self._compute_normalized_budget_change(
            budget_before=budget_before,
            candidate_budgets=candidate_budgets,
            initial_budgets=self._initial_budgets,
            changed_task_ids=tuple(updates.keys()),
        )
        # 下面分量用于日志可解释性：
        # 这里不再记录旧 event-based 分量，而是记录“与当前 interval 公式同口径”的主要分量，
        # 这样 episode 级别的 reward_*_sum 才能反映当前训练目标，而不是被 event_weights=0 误导为全 0。
        # 注意：这些分量是诊断口径，最终总 reward 仍以 step_reward_formula 的表达式求值为准。
        budget_after = dict(self._engine.runtime_budgets.budgets)
        reward_parameters = self._reward_mode_config.reward_parameters
        # job_start_weight 对应公式中的正向 job 启动奖励系数。
        job_start_weight = float(reward_parameters.get("job_start_weight", 0.0))
        # 以下 penalty 系数与 interval 公式中的各惩罚项一一对应。
        lo_overrun_penalty = float(reward_parameters.get("lo_overrun_penalty", 0.0))
        hi_overrun_penalty = float(reward_parameters.get("hi_overrun_penalty", 0.0))
        mode_change_penalty = float(reward_parameters.get("mode_change_penalty", 0.0))
        lo_cancellation_penalty = float(reward_parameters.get("lo_cancellation_penalty", 0.0))
        deadline_miss_penalty = float(reward_parameters.get("deadline_miss_penalty", 0.0))
        invalid_action_penalty = float(reward_parameters.get("invalid_action_penalty", 0.0))
        noop_bonus = float(reward_parameters.get("noop_bonus", 0.0))
        budget_change_penalty = float(reward_parameters.get("budget_change_penalty", 0.0))
        budget_drift_penalty = float(reward_parameters.get("budget_drift_penalty", 0.0))
        noop_bonus_if_noop = noop_bonus if is_explicit_noop_action else 0.0
        budget_change_penalty_value = budget_change_penalty * budget_change_norm
        budget_drift_mean = self._compute_budget_drift_mean(
            budgets=budget_after,
            initial_budgets=self._initial_budgets,
        )
        budget_drift_penalty_value = budget_drift_penalty * budget_drift_mean
        # interval 公式分量（用于日志拆分）：
        # - lo/hi/job_start 保留 event 分量并叠加 interval 分量，兼容旧 reward mode 的日志语义；
        # - mode change 惩罚明确使用按 job 归一化后的 mode_change_per_job；
        # - 各项符号与公式保持一致（惩罚项为负）。
        step_reward_job_start = event_job_start_reward + job_start_weight * float(delta_job_start)
        step_reward_lo_overrun = event_lo_overrun_reward - lo_overrun_penalty * lo_overrun_rate
        step_reward_hi_overrun = event_hi_overrun_reward - hi_overrun_penalty * hi_overrun_rate
        step_reward_mode_change = -mode_change_penalty * mode_change_per_job
        step_reward_lo_cancellation = -lo_cancellation_penalty * lo_cancellation_rate
        step_reward_deadline_miss = -deadline_miss_penalty * deadline_miss_rate
        step_reward_invalid_action = -invalid_action_penalty * invalid_action
        # reward_variables 是“奖励表达式可见变量表”：
        # - 继续保留原有 event/paper 变量，确保旧 reward 配置不受影响；
        # - 新增 interval 差分与 rate 变量，供 interval-based reward 使用；
        # - 后续再把 reward_parameters 动态合并进来，使 JSON 可直接引用参数名。
        reward_variables: dict[str, float | bool] = {
            "paper_reward": float(paper_reward),
            "noop_bonus_if_noop": float(noop_bonus_if_noop),
            "budget_change_penalty": float(budget_change_penalty),
            "budget_change_norm": float(budget_change_norm),
            "budget_drift_penalty": float(budget_drift_penalty),
            "budget_drift_mean": float(budget_drift_mean),
            "is_explicit_noop_action": bool(is_explicit_noop_action),
            "event_job_start_reward": float(event_job_start_reward),
            "event_lo_overrun_reward": float(event_lo_overrun_reward),
            "event_hi_overrun_reward": float(event_hi_overrun_reward),
            "delta_job_start": float(delta_job_start),
            "delta_lo_overrun": float(delta_lo_overrun),
            "delta_hi_overrun": float(delta_hi_overrun),
            "delta_mode_changes": float(delta_mode_changes),
            "delta_lo_cancellations": float(delta_lo_cancellations),
            "delta_deadline_misses": float(delta_deadline_misses),
            "interval_time": float(interval_time),
            "delta_total_jobs": float(delta_total_jobs),
            "lo_overrun_rate": float(lo_overrun_rate),
            "hi_overrun_rate": float(hi_overrun_rate),
            "mode_change_rate": float(mode_change_rate),
            "mode_change_per_job": float(mode_change_per_job),
            "lo_cancellation_rate": float(lo_cancellation_rate),
            "deadline_miss_rate": float(deadline_miss_rate),
            "invalid_action": float(invalid_action),
        }
        # 必须把 reward_parameters 合并进变量表：
        # 这样 step_reward_formula 可以直接写 `lo_overrun_penalty * lo_overrun_rate`，
        # 而无需在公式里硬编码数值，便于后续仅通过配置调参。
        for key, value in reward_parameters.items():
            if isinstance(value, (int, float, bool)):
                reward_variables[key] = float(value)
        reward = evaluate_reward_expression(
            self._reward_mode_config.step_reward_formula,
            reward_variables,
        )
        self._prev_job_start_count = self._monitor.job_start_count
        self._prev_lo_overrun_count = self._monitor.lo_overrun_count
        self._prev_hi_overrun_count = self._monitor.hi_overrun_count
        self._prev_mode_changes = mode_changes
        self._prev_lo_cancellations = lo_cancellations
        self._prev_deadline_misses = deadline_misses
        self._update_feature_state(
            delta_job_start=delta_job_start,
            delta_mode_changes=delta_mode_changes,
            delta_lo_cancellations=delta_lo_cancellations,
            delta_hi_overrun=delta_hi_overrun,
            delta_lo_overrun=delta_lo_overrun,
        )
        safety_margin_min = self._compute_current_safety_margin_min()

        observation = build_observation(
            time=current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
            feature_state=self._feature_state,
            feature_config=self.feature_config,
            safety_margin_min=safety_margin_min,
        )
        self._last_observation = observation
        info = {
            "time": current_time,
            "action_time": action_time,
            "action_id": action_id,
            "accepted": accepted,
            # 统一输出 is_noop，明确区分：
            # - 显式 noop：action_id 为具体编号且 is_noop=True；
            # - 隐式 noop：action_id=None（通常是无合法动作）；
            # - 普通预算动作：is_noop=False。
            "is_noop": bool(action_id is None or is_explicit_noop_action),
            # 显式 noop 的单独标记：用于训练/评估统计，避免仅靠 action_id 推断语义。
            "is_explicit_noop_action": bool(is_explicit_noop_action),
            # 约定：只有预算真的发生更新时才会出现非空 updates；
            # 显式/隐式 noop 和被拒绝动作都应保持 updates={}。
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
            "mode_changes": mode_changes,
            "lo_cancellations": lo_cancellations,
            "deadline_misses": deadline_misses,
            "interval_time": float(interval_time),
            "delta_total_jobs": float(delta_total_jobs),
            "delta_job_start": float(delta_job_start),
            "delta_lo_overrun": float(delta_lo_overrun),
            "delta_hi_overrun": float(delta_hi_overrun),
            "delta_mode_changes": float(delta_mode_changes),
            "delta_lo_cancellations": float(delta_lo_cancellations),
            "delta_deadline_misses": float(delta_deadline_misses),
            "lo_overrun_rate": float(lo_overrun_rate),
            "hi_overrun_rate": float(hi_overrun_rate),
            "mode_change_rate": float(mode_change_rate),
            "mode_change_per_job": float(mode_change_per_job),
            "lo_cancellation_rate": float(lo_cancellation_rate),
            "deadline_miss_rate": float(deadline_miss_rate),
            "invalid_action": float(invalid_action),
            "reward": float(reward),
            "step_reward_total": reward,
            "paper_reward": paper_reward,
            "noop_reward_bonus": noop_bonus_if_noop,
            "budget_change_norm": budget_change_norm,
            "budget_change_penalty_value": budget_change_penalty_value,
            "budget_drift_mean": budget_drift_mean,
            "budget_drift_penalty_value": budget_drift_penalty_value,
            "reward_after_regularization": reward,
            "step_reward_job_start": step_reward_job_start,
            "step_reward_lo_overrun": step_reward_lo_overrun,
            "step_reward_hi_overrun": step_reward_hi_overrun,
            "step_reward_mode_change": step_reward_mode_change,
            "step_reward_lo_cancellation": step_reward_lo_cancellation,
            "step_reward_deadline_miss": step_reward_deadline_miss,
            # invalid_action 惩罚单独记录，便于分析“动作被拒绝”对 reward 的影响权重。
            "step_reward_invalid_action": step_reward_invalid_action,
            "observation_mode": self.feature_config.observation_mode,
            "state_dim": len(observation.state_vector),
            "feature_safety_margin_min": float(safety_margin_min),
        }
        action_log_entry = dict(info)
        action_log_entry["time"] = action_time
        self._action_log.append(action_log_entry)
        return AgentStepResult(observation=observation, reward=reward, done=done, info=info)
