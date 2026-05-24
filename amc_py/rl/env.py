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
from amc_py.rl.feature_config import (
    OBSERVATION_MODE_V12_FULL_14D,
    FeatureConfig,
    supports_task_structured_features,
)
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
    action_space: Literal[
        "triple",
        "pair",
        "single",
        "constraint_guided_pair",
        "constraint_guided_transfer",
        "residual_ranked",
        "residual_safe_ranked",
        "residual_anchor_mc_lo_2",
    ] = "triple"
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
    enable_residual_safety_fallback: bool = False
    residual_guard_hi_pressure_delta_limit: float = 0.03
    residual_guard_hi_pressure_abs_limit: float = 0.30
    residual_guard_reject_decrease_pressure_threshold: float = 0.05
    residual_guard_use_hi_pressure_max: bool = False
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
    _residual_guard_rejected_actions: int = field(init=False, default=0, repr=False)
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
    # 动作空间名称缓存：
    # - `action_space` 可能在初始化时经过 alias 归一化（例如 constraint_guided_pair -> transfer）；
    # - 将最终名称固化到 `action_space_name`，供 step/info 日志稳定读取。
    action_space_name: str = field(init=False, repr=False, default="unknown")
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
        # 保存归一化后的动作空间名称，供日志与下游 CSV 使用。
        self.action_space_name = str(self.action_space)
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
        # constraint-guided 依赖任务级结构化 observation。
        # 新增的 v11 消融模式虽然维度更小，但仍然属于同一语义家族，
        # 因此这里不应只硬编码 v11_full_10d，而应统一走 helper 判断。
        if not supports_task_structured_features(self.feature_config.observation_mode):
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

    def _resolve_residual_ranked_action(
        self,
        action: BudgetAction,
        *,
        budget_state: BudgetState,
    ) -> tuple[dict[str, int], dict[str, int], str | None, BudgetAction | None]:
        """将 residual_ranked 槽位动作解析为具体预算更新。

        返回：
        - updates
        - candidate_budgets
        - reject_reason
        - concrete_action
        """

        budget_before = dict(budget_state.budgets)
        if action.is_noop or action.residual_action_type == "noop":
            return {}, dict(budget_before), None, action

        concrete: BudgetAction | None = None
        if action.residual_action_type == "increase_lo_risk":
            increase_idx, reject_reason = self._select_residual_increase_target(
                budgets=budget_before,
                criticality=Criticality.LO,
                rank=int(action.residual_rank or 0),
            )
            if reject_reason is not None or increase_idx is None:
                return {}, dict(budget_before), reject_reason or "residual_rank_no_lo_target", None
            concrete = self._make_residual_concrete_action(
                slot_action=action,
                increase_idx=increase_idx,
                decrease_indices=(),
            )
        elif action.residual_action_type == "decrease_lowest_risk":
            decrease_indices, reject_reason = self._select_residual_decrease_targets(
                budgets=budget_before,
                pool="global_low_risk",
                start_rank=int(action.residual_rank or 0),
                count=1,
                allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
            )
            if reject_reason is not None:
                return {}, dict(budget_before), reject_reason, None
            concrete = self._make_residual_concrete_action(
                slot_action=action,
                increase_idx=None,
                decrease_indices=decrease_indices,
            )
        elif action.residual_action_type == "decrease_lo_lowest_risk":
            decrease_indices, reject_reason = self._select_residual_decrease_targets(
                budgets=budget_before,
                pool="lo_low_risk",
                start_rank=int(action.residual_rank or 0),
                count=1,
                allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
            )
            if reject_reason is not None:
                return {}, dict(budget_before), reject_reason, None
            concrete = self._make_residual_concrete_action(
                slot_action=action,
                increase_idx=None,
                decrease_indices=decrease_indices,
            )
        elif action.residual_action_type in {
            "transfer_to_lo_risk_from_global_low",
            "transfer_to_lo_risk_from_lo_low",
            "transfer_to_lo_risk_from_global_low2",
        }:
            increase_idx, reject_reason = self._select_residual_increase_target(
                budgets=budget_before,
                criticality=Criticality.LO,
                rank=int(action.residual_rank or 0),
            )
            if reject_reason is not None or increase_idx is None:
                return {}, dict(budget_before), reject_reason or "residual_rank_no_lo_target", None
            if action.residual_action_type == "transfer_to_lo_risk_from_lo_low":
                decrease_pool = "lo_low_risk"
            else:
                decrease_pool = "global_low_risk"
            decrease_indices, reject_reason = self._select_residual_decrease_targets(
                budgets=budget_before,
                pool=decrease_pool,
                start_rank=int(action.residual_decrease_rank or 0),
                count=int(action.residual_decrease_count or 1),
                exclude_indices={increase_idx},
                allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
            )
            if reject_reason is not None:
                return {}, dict(budget_before), reject_reason, None
            concrete = self._make_residual_concrete_action(
                slot_action=action,
                increase_idx=increase_idx,
                decrease_indices=decrease_indices,
            )
        else:
            return {}, dict(budget_before), f"unknown_residual_action_type:{action.residual_action_type}", None

        if concrete is None:
            return {}, dict(budget_before), "residual_concrete_action_missing", None
        updates = apply_budget_action_candidate(
            action=concrete,
            budget_state=budget_state,
            ordered_tasks=self.ordered_tasks,
        )
        candidate_budgets = merge_budget_candidate(budget_state, updates)
        return (updates, candidate_budgets, None, concrete)

    def _select_residual_increase_target(
        self,
        *,
        budgets: dict[str, int],
        criticality: Criticality,
        rank: int,
    ) -> tuple[int | None, str | None]:
        """按风险排名选择 residual increase 目标。"""

        ranked = self._rank_task_indices_by_risk(
            budgets=budgets,
            criticality=criticality,
            descending=True,
        )
        if rank >= len(ranked):
            return None, f"residual_rank_no_{criticality.name.lower()}_target"
        return ranked[rank], None

    def _select_residual_decrease_targets(
        self,
        *,
        budgets: dict[str, int],
        pool: str,
        start_rank: int,
        count: int,
        exclude_indices: set[int] | None = None,
        allow_hi_decrease: bool = True,
    ) -> tuple[tuple[int, ...], str | None]:
        """按低风险池选择 residual decrease 目标集合。"""

        chosen_exclusions = exclude_indices or set()
        if pool == "global_low_risk":
            ranked = self._rank_task_indices_by_risk(budgets=budgets, criticality=None, descending=False)
        elif pool == "lo_low_risk":
            ranked = self._rank_task_indices_by_risk(
                budgets=budgets,
                criticality=Criticality.LO,
                descending=False,
            )
        elif pool == "hi_low_risk":
            ranked = self._rank_task_indices_by_risk(
                budgets=budgets,
                criticality=Criticality.HI,
                descending=False,
            )
        else:
            return (), f"unknown_residual_decrease_pool:{pool}"

        filtered = [idx for idx in ranked if idx not in chosen_exclusions]
        # 当策略配置禁止降低 HI 预算时，global/lo 池在 residual 解析阶段就提前排除 HI，
        # 避免动作在解析后才因 decrease_hi_forbidden 大量失效，提升 transfer 槽位可用率。
        if not allow_hi_decrease:
            filtered = [
                idx
                for idx in filtered
                if self.ordered_tasks[idx].criticality is not Criticality.HI
            ]
        selected = tuple(filtered[start_rank : start_rank + count])
        if len(selected) < count:
            return (), f"residual_rank_no_decrease_target:{pool}"
        return selected, None

    def _budget_candidate_reject_reason(
        self,
        *,
        action: BudgetAction | None,
        updates: dict[str, int],
        budget_before: dict[str, int],
        candidate_budgets: dict[str, int],
        hi_pressure_threshold: float,
        lo_pressure_threshold: float,
    ) -> str | None:
        """统一预算候选的拒绝原因判定。

        这个函数的目的只有一个：把“一个候选预算更新是否可接受”的判定收敛到单点，
        让 safe residual 的 resolver、mask、step 三条路径共用同一套语义，减少偏差。
        """

        if action is not None and action_violates_hi_decrease_guard(
            action=action,
            ordered_tasks=self.ordered_tasks,
            forbid_decreasing_hi_budgets=self.forbid_decreasing_hi_budgets,
        ):
            return "decrease_hi_forbidden"

        residual_guard_reason = self._residual_safety_guard_reject_reason(
            action=action,
            budget_before=budget_before,
            candidate_budgets=candidate_budgets,
            hi_pressure_threshold=hi_pressure_threshold,
            lo_pressure_threshold=lo_pressure_threshold,
        )
        if residual_guard_reason is not None:
            return residual_guard_reason

        floor_reject_reason = self._budget_floor_violation(updates=updates)
        if floor_reject_reason is not None:
            return floor_reject_reason

        if self.check_safety:
            report = self._ensure_checker().validate_candidate(candidate_budgets)
            if not report.accepted:
                return report.reason
        return None

    def _resolve_residual_safe_ranked_action(
        self,
        action: BudgetAction,
        budget_before: dict[str, int],
        *,
        hi_pressure_threshold: float,
        lo_pressure_threshold: float,
    ) -> tuple[dict[str, int], dict[str, int], str | None, BudgetAction | None]:
        """把 safe residual 槽位动作解析为“第 k 个安全候选”。

        关键语义：
        1. 先按 LO 风险排序生成 increase 候选；
        2. 再组合 decrease 候选形成 concrete action；
        3. 逐个执行 guard/checker 过滤，只保留真正可执行候选；
        4. rank-k 选第 k 个安全候选，而不是第 k 个原始风险候选。
        """

        if action.is_noop or action.residual_action_type == "noop":
            return {}, dict(budget_before), None, action
        is_anchor_action = action.residual_action_type == "direct_safe_increase_anchor"
        if not ((action.residual_action_type or "").startswith("safe_") or is_anchor_action):
            return {}, dict(budget_before), f"unsupported_safe_action_type:{action.residual_action_type}", None

        if is_anchor_action:
            if action.increase_idx is None:
                return {}, dict(budget_before), "residual_anchor_missing_increase_idx", None
            inc_ranked = [int(action.increase_idx)]
        else:
            inc_ranked = self._rank_task_indices_by_risk(
                budgets=budget_before,
                criticality=Criticality.LO,
                descending=True,
            )
            if not inc_ranked:
                return {}, dict(budget_before), "residual_safe_no_candidate", None

        safe_candidates: list[tuple[float, float, int, dict[str, int], dict[str, int], BudgetAction]] = []

        for increase_idx in inc_ranked:
            decrease_combos: list[tuple[int, ...]] = [()]
            if action.residual_action_type == "safe_transfer_global_low_to_lo_risk":
                dec, reason = self._select_residual_decrease_targets(
                    budgets=budget_before,
                    pool="global_low_risk",
                    start_rank=int(action.residual_decrease_rank or 0),
                    count=int(action.residual_decrease_count or 1),
                    exclude_indices={increase_idx},
                    allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
                )
                if reason is None:
                    decrease_combos = [dec]
                else:
                    decrease_combos = []
            elif action.residual_action_type == "safe_transfer_lo_low_to_lo_risk":
                dec, reason = self._select_residual_decrease_targets(
                    budgets=budget_before,
                    pool="lo_low_risk",
                    start_rank=int(action.residual_decrease_rank or 0),
                    count=int(action.residual_decrease_count or 1),
                    exclude_indices={increase_idx},
                    allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
                )
                if reason is None:
                    decrease_combos = [dec]
                else:
                    decrease_combos = []
            elif action.residual_action_type == "safe_transfer_global_low2_to_lo_risk":
                dec, reason = self._select_residual_decrease_targets(
                    budgets=budget_before,
                    pool="global_low_risk",
                    start_rank=int(action.residual_decrease_rank or 0),
                    count=int(action.residual_decrease_count or 2),
                    exclude_indices={increase_idx},
                    allow_hi_decrease=not self.forbid_decreasing_hi_budgets,
                )
                if reason is None:
                    decrease_combos = [dec]
                else:
                    decrease_combos = []
            elif action.residual_action_type in {"safe_increase_lo_risk", "direct_safe_increase_anchor"}:
                decrease_combos = [()]
            else:
                return {}, dict(budget_before), f"unknown_safe_action_type:{action.residual_action_type}", None

            for dec_indices in decrease_combos:
                concrete = self._make_residual_concrete_action(
                    slot_action=action,
                    increase_idx=increase_idx,
                    decrease_indices=tuple(dec_indices),
                )
                updates = apply_budget_action_candidate(
                    action=concrete,
                    budget_state=self._engine.runtime_budgets,  # type: ignore[arg-type]
                    ordered_tasks=self.ordered_tasks,
                )
                candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)  # type: ignore[arg-type]
                reject_reason = self._budget_candidate_reject_reason(
                    action=concrete,
                    updates=updates,
                    budget_before=budget_before,
                    candidate_budgets=candidate_budgets,
                    hi_pressure_threshold=hi_pressure_threshold,
                    lo_pressure_threshold=lo_pressure_threshold,
                )
                if reject_reason is not None:
                    continue

                inc_risk = self._task_exec_budget_ratio(task_index=increase_idx, budgets=budget_before)
                if dec_indices:
                    dec_risk = min(
                        self._task_exec_budget_ratio(task_index=idx, budgets=budget_before) for idx in dec_indices
                    )
                else:
                    dec_risk = 0.0
                safe_candidates.append((inc_risk, dec_risk, increase_idx, updates, candidate_budgets, concrete))

        if not safe_candidates:
            return {}, dict(budget_before), "residual_safe_no_candidate", None

        safe_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        rank = int(action.residual_rank or 0)
        if rank >= len(safe_candidates):
            return {}, dict(budget_before), "residual_safe_rank_out_of_range", None

        selected = safe_candidates[rank]
        return selected[3], selected[4], None, selected[5]

    def _rank_lo_utility_increase_candidates(
        self,
        *,
        budgets: dict[str, int],
    ) -> list[int]:
        """按“提升收益潜力”对 LO increase 候选排序（高分在前）。"""

        scored: list[tuple[float, int]] = []
        for idx, task in enumerate(self.ordered_tasks):
            if task.criticality is not Criticality.LO:
                continue
            current = float(budgets[task.name])
            initial = float(self._initial_budgets.get(task.name, current))
            # 正向漂移惩罚：预算已经被加过很多的任务，后续增配优先级下调。
            positive_drift = max(0.0, current / max(initial, 1.0) - 1.0)
            pressure = self._task_exec_budget_ratio(task_index=idx, budgets=budgets)
            score = pressure - 0.25 * positive_drift
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored]

    def _rank_lo_redundant_decrease_candidates(
        self,
        *,
        budgets: dict[str, int],
    ) -> list[int]:
        """按“冗余可回收性”对 LO decrease 候选排序（高分在前）。"""

        scored: list[tuple[float, int]] = []
        for idx, task in enumerate(self.ordered_tasks):
            if task.criticality is not Criticality.LO:
                continue
            current = float(budgets[task.name])
            initial = float(self._initial_budgets.get(task.name, current))
            # 仅回收超出初始预算的部分，避免训练初期就压缩 baseline。
            if current <= initial:
                continue
            pressure = self._task_exec_budget_ratio(task_index=idx, budgets=budgets)
            positive_drift = max(0.0, current / max(initial, 1.0) - 1.0)
            redundancy_score = positive_drift + max(0.0, 1.0 - pressure)
            scored.append((redundancy_score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored]

    def _resolve_residual_safe_adjust_action(
        self,
        action: BudgetAction,
        budget_before: dict[str, int],
        *,
        hi_pressure_threshold: float,
        lo_pressure_threshold: float,
    ) -> tuple[dict[str, int], dict[str, int], str | None, BudgetAction | None]:
        """把 residual_safe_adjust_15a 槽位动作解析为具体 increase/decrease 动作。"""

        if action.is_noop or action.residual_action_type == "noop":
            return {}, dict(budget_before), None, action

        action_type = action.residual_action_type or ""
        if action_type == "safe_increase_lo_utility":
            ranked_candidates = self._rank_lo_utility_increase_candidates(budgets=budget_before)
            safe_candidates: list[tuple[int, dict[str, int], dict[str, int], BudgetAction]] = []
            for increase_idx in ranked_candidates:
                concrete = self._make_residual_concrete_action(
                    slot_action=action,
                    increase_idx=increase_idx,
                    decrease_indices=(),
                )
                updates = apply_budget_action_candidate(
                    action=concrete,
                    budget_state=self._engine.runtime_budgets,  # type: ignore[arg-type]
                    ordered_tasks=self.ordered_tasks,
                )
                candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)  # type: ignore[arg-type]
                reject_reason = self._budget_candidate_reject_reason(
                    action=concrete,
                    updates=updates,
                    budget_before=budget_before,
                    candidate_budgets=candidate_budgets,
                    hi_pressure_threshold=hi_pressure_threshold,
                    lo_pressure_threshold=lo_pressure_threshold,
                )
                if reject_reason is None:
                    safe_candidates.append((increase_idx, updates, candidate_budgets, concrete))
            if not safe_candidates:
                return {}, dict(budget_before), "residual_safe_adjust_no_candidate", None
            rank = int(action.residual_rank or 0)
            if rank >= len(safe_candidates):
                return {}, dict(budget_before), "residual_safe_adjust_rank_out_of_range", None
            selected = safe_candidates[rank]
            return selected[1], selected[2], None, selected[3]

        if action_type == "safe_decrease_lo_redundant":
            ranked_candidates = self._rank_lo_redundant_decrease_candidates(budgets=budget_before)
            safe_candidates: list[tuple[int, dict[str, int], dict[str, int], BudgetAction]] = []
            for decrease_idx in ranked_candidates:
                concrete = self._make_residual_concrete_action(
                    slot_action=action,
                    increase_idx=None,
                    decrease_indices=(decrease_idx,),
                )
                updates = apply_budget_action_candidate(
                    action=concrete,
                    budget_state=self._engine.runtime_budgets,  # type: ignore[arg-type]
                    ordered_tasks=self.ordered_tasks,
                )
                candidate_budgets = merge_budget_candidate(self._engine.runtime_budgets, updates)  # type: ignore[arg-type]
                reject_reason = self._budget_candidate_reject_reason(
                    action=concrete,
                    updates=updates,
                    budget_before=budget_before,
                    candidate_budgets=candidate_budgets,
                    hi_pressure_threshold=hi_pressure_threshold,
                    lo_pressure_threshold=lo_pressure_threshold,
                )
                if reject_reason is None:
                    safe_candidates.append((decrease_idx, updates, candidate_budgets, concrete))
            if not safe_candidates:
                return {}, dict(budget_before), "residual_safe_adjust_no_candidate", None
            rank = int(action.residual_rank or 0)
            if rank >= len(safe_candidates):
                return {}, dict(budget_before), "residual_safe_adjust_rank_out_of_range", None
            selected = safe_candidates[rank]
            return selected[1], selected[2], None, selected[3]

        return {}, dict(budget_before), f"unsupported_safe_adjust_action_type:{action.residual_action_type}", None

    def _make_residual_concrete_action(
        self,
        *,
        slot_action: BudgetAction,
        increase_idx: int | None,
        decrease_indices: tuple[int, ...],
    ) -> BudgetAction:
        """根据 residual 槽位动作和解析索引构造 concrete action。"""

        increase_task = self.ordered_tasks[increase_idx].name if increase_idx is not None else None
        decrease_tasks = tuple(self.ordered_tasks[idx].name for idx in decrease_indices)
        return BudgetAction(
            action_id=slot_action.action_id,
            increase_task=increase_task,
            decrease_tasks=decrease_tasks,
            increase_idx=increase_idx,
            decrease_indices=decrease_indices,
            increase_ratio=slot_action.increase_ratio,
            decrease_ratio=slot_action.decrease_ratio,
            action_space_type=slot_action.action_space_type,
            is_noop=slot_action.is_noop,
            is_residual_ranked=True,
            residual_action_type=slot_action.residual_action_type,
            residual_rank=slot_action.residual_rank,
            residual_decrease_rank=slot_action.residual_decrease_rank,
            residual_decrease_count=slot_action.residual_decrease_count,
            residual_decrease_pool=slot_action.residual_decrease_pool,
        )

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

    def _compute_lo_pressure_mean(
        self,
        *,
        budgets: dict[str, int],
        threshold: float,
    ) -> float:
        """计算 LO task 的 cancellation pressure 均值。

        定义：
        pressure_i = max(0, estimated_exec / budget_i - threshold)

        其中 estimated_exec 使用 max(recent_execution, ema_cost)：
        - recent_execution：反映最近一次真实运行样本的瞬时风险；
        - ema_cost：反映更平滑的长期执行成本趋势；
        - 取 max 可减少“风险被低估”的概率。
        """

        if self._feature_state is None:
            return 0.0

        pressures: list[float] = []
        for task in self.ordered_tasks:
            if task.criticality is not Criticality.LO:
                continue
            task_name = task.name
            # 预算分母做下界保护，避免除零并与项目其它比值计算口径一致。
            budget = max(1.0, float(budgets[task_name]))
            recent_exec = float(self._monitor.recent_execution.get(task_name, 0.0))
            ema_exec = float(self._feature_state.ema_cost.get(task_name, float(task.c_lo)))
            estimated_exec = max(recent_exec, ema_exec)
            pressure = max(0.0, estimated_exec / budget - float(threshold))
            pressures.append(pressure)

        if not pressures:
            return 0.0
        return float(sum(pressures) / len(pressures))

    def _compute_lo_pressure_stats(
        self,
        *,
        budgets: dict[str, int],
        pressure_threshold: float,
        near_cancel_threshold: float,
    ) -> dict[str, float]:
        """计算 LO task 的 dense risk 指标集合。

        返回三个互补指标：
        - lo_pressure_mean：所有 LO pressure 的平均值，反映总体风险水平；
        - lo_pressure_max：所有 LO pressure 的最大值，专门捕捉“最危险任务”；
        - lo_near_cancel_rate：estimated_exec / budget 超过 near_cancel_threshold 的 LO 任务比例。

        指标定义：
        pressure_i = max(0, estimated_exec / budget_i - pressure_threshold)
        estimated_exec = max(recent_execution, ema_cost)
        """

        if self._feature_state is None:
            return {
                "lo_pressure_mean": 0.0,
                "lo_pressure_max": 0.0,
                "lo_near_cancel_rate": 0.0,
            }

        pressures: list[float] = []
        near_cancel_count = 0
        lo_count = 0

        for task in self.ordered_tasks:
            if task.criticality is not Criticality.LO:
                continue

            lo_count += 1
            task_name = task.name
            # 与现有 pressure 计算一致：budget 分母做下界保护，避免除零和数值放大。
            budget = max(1.0, float(budgets[task_name]))
            recent_exec = float(self._monitor.recent_execution.get(task_name, 0.0))
            ema_exec = float(self._feature_state.ema_cost.get(task_name, float(task.c_lo)))
            # 取 max 以避免瞬时低估执行风险，保持 reward shaping 的保守性。
            estimated_exec = max(recent_exec, ema_exec)

            ratio = estimated_exec / budget
            pressure = max(0.0, ratio - float(pressure_threshold))
            pressures.append(pressure)

            if ratio > float(near_cancel_threshold):
                near_cancel_count += 1

        if lo_count == 0:
            return {
                "lo_pressure_mean": 0.0,
                "lo_pressure_max": 0.0,
                "lo_near_cancel_rate": 0.0,
            }

        return {
            "lo_pressure_mean": float(sum(pressures) / len(pressures)) if pressures else 0.0,
            "lo_pressure_max": float(max(pressures)) if pressures else 0.0,
            "lo_near_cancel_rate": float(near_cancel_count / lo_count),
        }

    def _compute_hi_mode_pressure_stats(
        self,
        *,
        budgets: dict[str, int],
        threshold: float,
    ) -> dict[str, float]:
        """计算 HI 任务 mode 风险统计量（均值 + 最大值）。

        返回字段：
        - hi_mode_pressure_mean
        - hi_mode_pressure_max

        定义：
        pressure_i = max(0, estimated_exec / budget_i - threshold)
        """

        if self._feature_state is None:
            return {
                "hi_mode_pressure_mean": 0.0,
                "hi_mode_pressure_max": 0.0,
            }

        pressures: list[float] = []
        for task in self.ordered_tasks:
            if task.criticality is not Criticality.HI:
                continue
            task_name = task.name
            # 预算分母做下界保护，避免除零并与项目其它比值计算口径一致。
            budget = max(1.0, float(budgets[task_name]))
            recent_exec = float(self._monitor.recent_execution.get(task_name, 0.0))
            ema_exec = float(self._feature_state.ema_cost.get(task_name, float(task.c_lo)))
            estimated_exec = max(recent_exec, ema_exec)
            pressure = max(0.0, estimated_exec / budget - float(threshold))
            pressures.append(pressure)

        if not pressures:
            return {
                "hi_mode_pressure_mean": 0.0,
                "hi_mode_pressure_max": 0.0,
            }
        return {
            "hi_mode_pressure_mean": float(sum(pressures) / len(pressures)),
            "hi_mode_pressure_max": float(max(pressures)),
        }

    def _compute_hi_mode_pressure_mean(
        self,
        *,
        budgets: dict[str, int],
        threshold: float,
    ) -> float:
        """计算 HI task 的 mode-change pressure 均值。"""

        stats = self._compute_hi_mode_pressure_stats(budgets=budgets, threshold=threshold)
        return float(stats["hi_mode_pressure_mean"])

    def _compute_task_pressure(
        self,
        *,
        task_index: int,
        budgets: dict[str, int],
        threshold: float,
    ) -> float:
        """计算单个任务 pressure，用于 residual safety guard。"""

        if self._feature_state is None:
            return 0.0
        task = self.ordered_tasks[task_index]
        budget = max(1.0, float(budgets[task.name]))
        recent_exec = float(self._monitor.recent_execution.get(task.name, 0.0))
        ema_exec = float(self._feature_state.ema_cost.get(task.name, float(task.c_lo)))
        estimated_exec = max(recent_exec, ema_exec)
        return float(max(0.0, estimated_exec / budget - float(threshold)))

    def _task_exec_budget_ratio(
        self,
        *,
        task_index: int,
        budgets: dict[str, int],
    ) -> float:
        """计算任务执行开销与预算比值，用于 residual 风险排序。"""

        if self._feature_state is None:
            return 0.0
        task = self.ordered_tasks[task_index]
        budget = max(1.0, float(budgets[task.name]))
        recent_exec = float(self._monitor.recent_execution.get(task.name, 0.0))
        ema_exec = float(self._feature_state.ema_cost.get(task.name, float(task.c_lo)))
        estimated_exec = max(recent_exec, ema_exec)
        return float(estimated_exec / budget)

    def _rank_task_indices_by_risk(
        self,
        *,
        budgets: dict[str, int],
        criticality: Criticality | None = None,
        descending: bool = True,
    ) -> list[int]:
        """按风险从高到低/低到高排序任务索引。"""

        indices: list[int] = []
        for idx, task in enumerate(self.ordered_tasks):
            if criticality is not None and task.criticality is not criticality:
                continue
            indices.append(idx)
        return sorted(
            indices,
            key=lambda idx: self._task_exec_budget_ratio(task_index=idx, budgets=budgets),
            reverse=descending,
        )

    def _residual_safety_guard_reject_reason(
        self,
        *,
        action: BudgetAction | None,
        budget_before: dict[str, int],
        candidate_budgets: dict[str, int],
        hi_pressure_threshold: float,
        lo_pressure_threshold: float,
    ) -> str | None:
        """判断 residual action 是否需要 fallback。"""

        if not self.enable_residual_safety_fallback:
            return None
        if action is None or action.is_noop:
            return None
        hi_before_stats = self._compute_hi_mode_pressure_stats(
            budgets=budget_before,
            threshold=hi_pressure_threshold,
        )
        hi_after_stats = self._compute_hi_mode_pressure_stats(
            budgets=candidate_budgets,
            threshold=hi_pressure_threshold,
        )
        hi_key = "hi_mode_pressure_max" if self.residual_guard_use_hi_pressure_max else "hi_mode_pressure_mean"
        hi_before = float(hi_before_stats[hi_key])
        hi_after = float(hi_after_stats[hi_key])
        if hi_after > hi_before + float(self.residual_guard_hi_pressure_delta_limit):
            return "residual_guard_hi_pressure_delta"
        if hi_after > float(self.residual_guard_hi_pressure_abs_limit):
            return "residual_guard_hi_pressure_abs"
        for dec_idx in action.decrease_indices:
            task = self.ordered_tasks[dec_idx]
            threshold = hi_pressure_threshold if task.criticality is Criticality.HI else lo_pressure_threshold
            dec_pressure = self._compute_task_pressure(
                task_index=dec_idx,
                budgets=budget_before,
                threshold=threshold,
            )
            if dec_pressure > float(self.residual_guard_reject_decrease_pressure_threshold):
                return "residual_guard_decrease_risky_task"
        return None

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
            # residual_ranked 在 mask 阶段复用 resolver，确保“槽位语义 -> concrete action”与 step 完全一致。
            if action.is_residual_ranked:
                is_safe_adjust_action = action.action_space_type == "residual_safe_adjust_15a"
                is_safe_residual_action = (
                    (action.residual_action_type or "").startswith("safe_")
                    or action.residual_action_type == "direct_safe_increase_anchor"
                )
                if is_safe_adjust_action:
                    updates, candidate_budgets, resolve_reject_reason, concrete_action = (
                        self._resolve_residual_safe_adjust_action(
                            action,
                            budget_before=budget_before,
                            hi_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                            ),
                            lo_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                            ),
                        )
                    )
                elif is_safe_residual_action:
                    updates, candidate_budgets, resolve_reject_reason, concrete_action = (
                        self._resolve_residual_safe_ranked_action(
                            action,
                            budget_before=budget_before,
                            hi_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                            ),
                            lo_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                            ),
                        )
                    )
                else:
                    updates, candidate_budgets, resolve_reject_reason, concrete_action = self._resolve_residual_ranked_action(
                        action,
                        budget_state=self._engine.runtime_budgets,
                    )
                valid = resolve_reject_reason is None
                reject_reason = resolve_reject_reason
                if valid and concrete_action is not None and not is_safe_residual_action:
                    if action_violates_hi_decrease_guard(
                        action=concrete_action,
                        ordered_tasks=self.ordered_tasks,
                        forbid_decreasing_hi_budgets=self.forbid_decreasing_hi_budgets,
                    ):
                        valid = False
                        reject_reason = "decrease_hi_forbidden"
                # residual safety guard 前移到 mask：
                # 这样 DQN 在采样阶段就看不到“step 必拒绝”的 residual 动作，
                # 避免 replay 中混入 mask 通过但 step reject 的不一致样本。
                if valid and concrete_action is not None and not is_safe_residual_action:
                    residual_guard_reason = self._residual_safety_guard_reject_reason(
                        action=concrete_action,
                        budget_before=budget_before,
                        candidate_budgets=candidate_budgets,
                        hi_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                        ),
                        lo_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                        ),
                    )
                    if residual_guard_reason is not None:
                        valid = False
                        reject_reason = residual_guard_reason
                mask.append(valid)
                if not valid and reject_reason is not None:
                    reject_reason_counts[_normalize_candidate_reject_reason(reject_reason)] += 1
                if self.mask_detail_mode == "minimal":
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": valid,
                            "reject_reason": reject_reason,
                            "is_noop": bool(action.residual_action_type == "noop"),
                            "is_residual_ranked": True,
                            "residual_action_type": action.residual_action_type,
                            "residual_rank": action.residual_rank,
                            "resolved_increase_task": concrete_action.increase_task if concrete_action else None,
                            "resolved_decrease_tasks": (
                                tuple(concrete_action.decrease_tasks) if concrete_action else ()
                            ),
                            "increase_idx": concrete_action.increase_idx if concrete_action else None,
                            "decrease_indices": (
                                tuple(concrete_action.decrease_indices) if concrete_action else ()
                            ),
                            "safe_candidate": bool(
                                resolve_reject_reason is None and is_safe_residual_action
                            ),
                            "safe_reject_reason": resolve_reject_reason
                            if is_safe_residual_action
                            else None,
                        }
                    )
                else:
                    mask_details.append(
                        {
                            "action_id": action.action_id,
                            "valid": valid,
                            "reject_reason": reject_reason,
                            "updates": dict(updates),
                            "budget_before": budget_before,
                            "candidate_budgets": dict(candidate_budgets),
                            "is_noop": bool(action.residual_action_type == "noop"),
                            "is_residual_ranked": True,
                            "residual_action_type": action.residual_action_type,
                            "residual_rank": action.residual_rank,
                            "resolved_increase_task": concrete_action.increase_task if concrete_action else None,
                            "resolved_decrease_tasks": (
                                tuple(concrete_action.decrease_tasks) if concrete_action else ()
                            ),
                            "increase_idx": concrete_action.increase_idx if concrete_action else None,
                            "decrease_indices": (
                                tuple(concrete_action.decrease_indices) if concrete_action else ()
                            ),
                            "safe_candidate": bool(
                                resolve_reject_reason is None and is_safe_residual_action
                            ),
                            "safe_reject_reason": resolve_reject_reason
                            if is_safe_residual_action
                            else None,
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
            # task_cancel_ema 使用 overrun 事件作为 cancellation pressure proxy：
            # 该统计与 completion 新样本无关，因此必须每个 step 都更新，避免漏掉纯 overrun 事件。
            current_overrun_count = self._monitor.overrun_count_by_task.get(task_name, 0)
            last_seen_overrun_count = feature_state.last_seen_overrun_count.get(task_name, 0)
            overrun_event_flag = 1.0 if current_overrun_count > last_seen_overrun_count else 0.0
            old_cancel_ema = feature_state.task_cancel_ema.get(task_name, 0.0)
            feature_state.task_cancel_ema[task_name] = (
                cfg.overrun_ema_alpha * overrun_event_flag + (1.0 - cfg.overrun_ema_alpha) * old_cancel_ema
            )
            feature_state.last_seen_overrun_count[task_name] = current_overrun_count
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

    def _compute_safe_inc_possible_by_task(self) -> dict[str, bool]:
        """计算每个任务在当前预算下执行 single increase 后是否可通过既有安全检查。"""

        if self._engine is None:
            return {task.name: False for task in self.ordered_tasks}
        if not self.check_safety:
            return {task.name: True for task in self.ordered_tasks}

        result: dict[str, bool] = {}
        before = dict(self._engine.runtime_budgets.budgets)
        for task in self.ordered_tasks:
            task_name = task.name
            current = int(before[task_name])
            candidate = int(math.ceil(current * (1.0 + self.budget_increase_ratio)))
            upper = int(self._task_upper_bounds[self._task_index[task_name]])
            candidate = min(candidate, upper)
            if candidate <= current:
                result[task_name] = False
                continue
            new_budgets = dict(before)
            new_budgets[task_name] = candidate
            diag = self.diagnose_candidate_budget_update(new_budgets=new_budgets)
            result[task_name] = bool(diag.accepted)
        return result

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
        self._residual_guard_rejected_actions = 0
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
            self._feature_state.last_seen_overrun_count[task.name] = self._monitor.overrun_count_by_task.get(
                task.name,
                0,
            )
        safe_inc_possible_by_task = (
            self._compute_safe_inc_possible_by_task()
            if self.feature_config.observation_mode == OBSERVATION_MODE_V12_FULL_14D
            else None
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
            initial_budgets=self._initial_budgets,
            safe_inc_possible_by_task=safe_inc_possible_by_task,
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
        resolved_increase_task: str | None = None
        resolved_decrease_tasks: tuple[str, ...] = ()
        resolved_increase_idx: int | None = None
        resolved_decrease_indices: tuple[int, ...] = ()

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
            # 对静态动作空间（single/pair/triple）提前写入 resolved 目标，
            # 让 validation policy logging 在非 residual 空间下也能输出具体任务信息。
            resolved_increase_task = action.increase_task
            resolved_decrease_tasks = tuple(action.decrease_tasks)
            resolved_increase_idx = action.increase_idx
            resolved_decrease_indices = tuple(action.decrease_indices)
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
            elif action.is_residual_ranked:
                is_safe_adjust_action = action.action_space_type == "residual_safe_adjust_15a"
                is_safe_residual_action = (
                    (action.residual_action_type or "").startswith("safe_")
                    or action.residual_action_type == "direct_safe_increase_anchor"
                )
                if is_safe_adjust_action:
                    (
                        updates,
                        candidate_budgets,
                        resolve_reject_reason,
                        concrete_action,
                    ) = self._resolve_residual_safe_adjust_action(
                        action,
                        budget_before=budget_before,
                        hi_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                        ),
                        lo_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                        ),
                    )
                elif is_safe_residual_action:
                    (
                        updates,
                        candidate_budgets,
                        resolve_reject_reason,
                        concrete_action,
                    ) = self._resolve_residual_safe_ranked_action(
                        action,
                        budget_before=budget_before,
                        hi_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                        ),
                        lo_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                        ),
                    )
                else:
                    (
                        updates,
                        candidate_budgets,
                        resolve_reject_reason,
                        concrete_action,
                    ) = self._resolve_residual_ranked_action(
                        action,
                        budget_state=self._engine.runtime_budgets,
                    )
                resolved_increase_task = concrete_action.increase_task if concrete_action is not None else None
                resolved_decrease_tasks = tuple(concrete_action.decrease_tasks) if concrete_action is not None else ()
                resolved_increase_idx = concrete_action.increase_idx if concrete_action is not None else None
                resolved_decrease_indices = (
                    tuple(concrete_action.decrease_indices) if concrete_action is not None else ()
                )
                if resolve_reject_reason is not None:
                    accepted = False
                    reject_reason = resolve_reject_reason
                    updates = {}
                    candidate_budgets = dict(budget_before)
                else:
                    if concrete_action is None:
                        accepted = False
                        reject_reason = "residual_concrete_action_missing"
                        updates = {}
                        candidate_budgets = dict(budget_before)
                    elif is_safe_residual_action:
                        accepted = True
                    else:
                        residual_guard_reason = self._residual_safety_guard_reject_reason(
                            action=concrete_action,
                            budget_before=budget_before,
                            candidate_budgets=candidate_budgets,
                            hi_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                            ),
                            lo_pressure_threshold=float(
                                self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                            ),
                        )
                        if residual_guard_reason is not None:
                            accepted = False
                            reject_reason = residual_guard_reason
                            updates = {}
                            candidate_budgets = dict(budget_before)
                            action_was_checked = True
                            self._residual_guard_rejected_actions += 1
                        else:
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
                # residual_ranked 分支在动作被接受后，必须显式把 updates 应用到 runtime budget。
                # 否则会出现日志显示 accepted=True，但预算状态实际未变化的问题。
                if accepted:
                    self._engine.apply_budget_updates(updates)
                elif is_safe_residual_action and reject_reason is not None:
                    reject_reason = f"safe_mask_step_mismatch:{reject_reason}"
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
                    residual_guard_reason = self._residual_safety_guard_reject_reason(
                        action=action,
                        budget_before=budget_before,
                        candidate_budgets=candidate_budgets,
                        hi_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                        ),
                        lo_pressure_threshold=float(
                            self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                        ),
                    )
                    if residual_guard_reason is not None:
                        accepted = False
                        reject_reason = residual_guard_reason
                        updates = {}
                        candidate_budgets = dict(budget_before)
                        action_was_checked = True
                        self._residual_guard_rejected_actions += 1
                    else:
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
        # 二次项惩罚系数：用于抑制 mode change 突发尖峰（spike）。
        # 该项与线性 mode_change_penalty 叠加，形成“低频轻罚、高频重罚”的非线性约束。
        mode_change_spike_penalty = float(reward_parameters.get("mode_change_spike_penalty", 0.0))
        lo_cancellation_penalty = float(reward_parameters.get("lo_cancellation_penalty", 0.0))
        deadline_miss_penalty = float(reward_parameters.get("deadline_miss_penalty", 0.0))
        invalid_action_penalty = float(reward_parameters.get("invalid_action_penalty", 0.0))
        noop_bonus = float(reward_parameters.get("noop_bonus", 0.0))
        budget_change_penalty = float(reward_parameters.get("budget_change_penalty", 0.0))
        budget_drift_penalty = float(reward_parameters.get("budget_drift_penalty", 0.0))
        lo_pressure_penalty = float(reward_parameters.get("lo_pressure_penalty", 0.0))
        lo_pressure_threshold = float(reward_parameters.get("lo_pressure_threshold", 0.8))
        lo_pressure_max_penalty = float(reward_parameters.get("lo_pressure_max_penalty", 0.0))
        lo_near_cancel_penalty = float(reward_parameters.get("lo_near_cancel_penalty", 0.0))
        lo_near_cancel_threshold = float(reward_parameters.get("lo_near_cancel_threshold", 0.9))
        hi_mode_pressure_penalty = float(reward_parameters.get("hi_mode_pressure_penalty", 0.0))
        hi_mode_pressure_threshold = float(reward_parameters.get("hi_mode_pressure_threshold", 0.8))
        noop_bonus_if_noop = noop_bonus if is_explicit_noop_action else 0.0
        budget_change_penalty_value = budget_change_penalty * budget_change_norm
        budget_drift_mean = self._compute_budget_drift_mean(
            budgets=budget_after,
            initial_budgets=self._initial_budgets,
        )
        budget_drift_penalty_value = budget_drift_penalty * budget_drift_mean
        # 这里明确使用 action 生效后的 budget_after，让 shaping 反映“当前动作导致的即时风险状态”。
        lo_pressure_stats = self._compute_lo_pressure_stats(
            budgets=budget_after,
            pressure_threshold=lo_pressure_threshold,
            near_cancel_threshold=lo_near_cancel_threshold,
        )
        lo_pressure_mean = float(lo_pressure_stats["lo_pressure_mean"])
        lo_pressure_max = float(lo_pressure_stats["lo_pressure_max"])
        lo_near_cancel_rate = float(lo_pressure_stats["lo_near_cancel_rate"])
        hi_mode_pressure_mean = self._compute_hi_mode_pressure_mean(
            budgets=budget_after,
            threshold=hi_mode_pressure_threshold,
        )
        lo_pressure_penalty_value = lo_pressure_penalty * lo_pressure_mean
        lo_pressure_max_penalty_value = lo_pressure_max_penalty * lo_pressure_max
        lo_near_cancel_penalty_value = lo_near_cancel_penalty * lo_near_cancel_rate
        hi_mode_pressure_penalty_value = hi_mode_pressure_penalty * hi_mode_pressure_mean
        # mode-change 二次惩罚项的实际值（已包含 mode_change_per_job^2），
        # 单独拆出来用于日志记录，便于后续分析“稳定性代价”在总 reward 中的占比。
        mode_change_spike_penalty_value = (
            mode_change_spike_penalty * mode_change_per_job * mode_change_per_job
        )
        # interval 公式分量（用于日志拆分）：
        # - lo/hi/job_start 保留 event 分量并叠加 interval 分量，兼容旧 reward mode 的日志语义；
        # - mode change 惩罚明确使用按 job 归一化后的 mode_change_per_job；
        # - 各项符号与公式保持一致（惩罚项为负）。
        step_reward_job_start = event_job_start_reward + job_start_weight * float(delta_job_start)
        step_reward_lo_overrun = event_lo_overrun_reward - lo_overrun_penalty * lo_overrun_rate
        step_reward_hi_overrun = event_hi_overrun_reward - hi_overrun_penalty * hi_overrun_rate
        step_reward_mode_change = (
            -mode_change_penalty * mode_change_per_job
            -mode_change_spike_penalty_value
        )
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
            "lo_pressure_mean": float(lo_pressure_mean),
            "lo_pressure_max": float(lo_pressure_max),
            "lo_near_cancel_rate": float(lo_near_cancel_rate),
            "hi_mode_pressure_mean": float(hi_mode_pressure_mean),
            "lo_pressure_penalty": float(lo_pressure_penalty),
            "lo_pressure_max_penalty": float(lo_pressure_max_penalty),
            "lo_near_cancel_penalty": float(lo_near_cancel_penalty),
            "hi_mode_pressure_penalty": float(hi_mode_pressure_penalty),
            "lo_pressure_threshold": float(lo_pressure_threshold),
            "lo_near_cancel_threshold": float(lo_near_cancel_threshold),
            "hi_mode_pressure_threshold": float(hi_mode_pressure_threshold),
            "lo_pressure_penalty_value": float(lo_pressure_penalty_value),
            "lo_pressure_max_penalty_value": float(lo_pressure_max_penalty_value),
            "lo_near_cancel_penalty_value": float(lo_near_cancel_penalty_value),
            "hi_mode_pressure_penalty_value": float(hi_mode_pressure_penalty_value),
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
        safe_inc_possible_by_task = (
            self._compute_safe_inc_possible_by_task()
            if self.feature_config.observation_mode == OBSERVATION_MODE_V12_FULL_14D
            else None
        )

        observation = build_observation(
            time=current_time,
            ordered_tasks=self.ordered_tasks,
            budget_state=self._engine.runtime_budgets,
            monitor=self._monitor,
            bounds=self.normalization_bounds,
            feature_state=self._feature_state,
            feature_config=self.feature_config,
            safety_margin_min=safety_margin_min,
            initial_budgets=self._initial_budgets,
            safe_inc_possible_by_task=safe_inc_possible_by_task,
        )
        self._last_observation = observation
        info = {
            "time": current_time,
            "action_time": action_time,
            "action_id": action_id,
            # 统一写入动作空间名称，便于 validation 侧跨 action_space 做日志聚合。
            "action_space": getattr(self, "action_space_name", "unknown"),
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
            "residual_guard_enabled": bool(self.enable_residual_safety_fallback),
            "residual_guard_rejected": bool(reject_reason and str(reject_reason).startswith("residual_guard_")),
            "residual_guard_rejected_actions": int(self._residual_guard_rejected_actions),
            "residual_guard_hi_pressure_delta_limit": float(self.residual_guard_hi_pressure_delta_limit),
            "residual_guard_hi_pressure_abs_limit": float(self.residual_guard_hi_pressure_abs_limit),
            "residual_action_type": (
                getattr(action, "residual_action_type", None) if action_id is not None else None
            ),
            "residual_rank": getattr(action, "residual_rank", None) if action_id is not None else None,
            "residual_decrease_rank": (
                getattr(action, "residual_decrease_rank", None) if action_id is not None else None
            ),
            "residual_decrease_count": (
                getattr(action, "residual_decrease_count", None) if action_id is not None else None
            ),
            "residual_decrease_pool": (
                getattr(action, "residual_decrease_pool", None) if action_id is not None else None
            ),
            # 兼容通用字段命名：validation 统计统一读这组 key，无需区分 residual/非 residual 分支。
            "resolved_increase_task": resolved_increase_task,
            "resolved_decrease_tasks": tuple(resolved_decrease_tasks),
            "increase_idx": resolved_increase_idx,
            "decrease_indices": tuple(resolved_decrease_indices),
            "residual_resolved_increase_task": resolved_increase_task,
            "residual_resolved_decrease_tasks": tuple(resolved_decrease_tasks),
            "residual_resolved_increase_idx": resolved_increase_idx,
            "residual_resolved_decrease_indices": tuple(resolved_decrease_indices),
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
            "lo_pressure_mean": lo_pressure_mean,
            "lo_pressure_max": lo_pressure_max,
            "lo_near_cancel_rate": lo_near_cancel_rate,
            "hi_mode_pressure_mean": hi_mode_pressure_mean,
            "lo_pressure_penalty_value": lo_pressure_penalty_value,
            "lo_pressure_max_penalty_value": lo_pressure_max_penalty_value,
            "lo_near_cancel_penalty_value": lo_near_cancel_penalty_value,
            "hi_mode_pressure_penalty_value": hi_mode_pressure_penalty_value,
            # mode-change spike 惩罚系数与分量值都写入 info，方便训练日志直接聚合。
            "mode_change_spike_penalty": float(mode_change_spike_penalty),
            "mode_change_spike_penalty_value": float(mode_change_spike_penalty_value),
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
