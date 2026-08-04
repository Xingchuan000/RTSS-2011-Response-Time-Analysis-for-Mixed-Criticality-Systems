"""阶段 6：agent-driven runtime wrapper。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math

from amc_py.budget_runtime import BudgetState
from amc_py.amc import build_design_r_lo_map
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.rl.actions import apply_budget_action_candidate
from amc_py.rl.action_execution import BudgetActionExecutionConfig, evaluate_budget_action
from amc_py.rl.agents import BudgetAgent
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.lo_service_reward import LoServiceRewardTracker
from amc_py.rl.observation import NormalizationBounds, build_observation
from amc_py.rl.reward_config import evaluate_reward_expression, load_reward_mode_config
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, merge_budget_candidate
from amc_py.runtime_models import RuntimeConfig, SimulationResult
from amc_py.qamc.models import QAmcProfileBundle
from amc_py.qamc.rl_contract import validate_qamc_rl_semantics
from amc_py.runtime_scenarios import ExecutionScenario


RECOVERY_REWARD_DEFAULT_VARIABLES: dict[str, float | bool] = {
    "budget_under_drift_mean": 0.0,
    "budget_over_drift_mean": 0.0,
    "budget_abs_drift_deadzone_mean": 0.0,
    "budget_over_drift_deadzone_mean": 0.0,
    "budget_abs_drift_mean": 0.0,
    "budget_abs_drift_deadzone": 0.05,
    "budget_abs_drift_penalty": 0.0,
    "budget_abs_drift_penalty_value": 0.0,
    "budget_drift_mean": 0.0,
    "budget_drift_penalty_value": 0.0,
    "budget_change_norm": 0.0,
    "budget_change_penalty": 0.0,
    "over_budget_dwell_penalty": 0.0,
    "over_budget_dwell_deadzone": 0.05,
    "over_increase_penalty": 0.0,
    "over_increase_deadzone": 0.05,
    "over_increase_excess": 0.0,
    "budget_soft_cap_ratio": 0.0,
    "budget_soft_cap_penalty": 0.0,
    "budget_soft_cap_increase_excess": 0.0,
    "budget_soft_cap_penalty_value": 0.0,
    "is_soft_cap_increase_action": 0.0,
    "budget_soft_cap_dwell_penalty": 0.0,
    "budget_soft_cap_dwell_max_penalty": 0.0,
    "budget_soft_cap_dwell_excess_mean": 0.0,
    "budget_soft_cap_dwell_excess_max": 0.0,
    "budget_soft_cap_dwell_task_count": 0.0,
    "budget_soft_cap_dwell_task_rate": 0.0,
    "budget_soft_cap_dwell_penalty_value": 0.0,
    "budget_soft_cap_dwell_max_penalty_value": 0.0,
    "budget_soft_cap_dwell_total_penalty_value": 0.0,
    "is_soft_cap_dwell_state": 0.0,
    "safe_recovery_decrease": 0.0,
    "recovery_decrease_target_count": 0.0,
    "recovery_decrease_excess_before_mean": 0.0,
    "unsafe_decrease_full": 0.0,
    "unsafe_decrease_penalty": 0.0,
    "unsafe_decrease_lo_near_cancel_threshold": 0.95,
    "unsafe_decrease_hi_pressure_threshold": 0.9,
    "pingpong_action": 0.0,
    "pingpong_penalty": 0.0,
    "concentration_penalty": 0.0,
    "concentration_window": 3.0,
    "increase_concentration_excess": 0.0,
    "consecutive_increase_count_for_target": 0.0,
    "lo_pressure_mean": 0.0,
    "lo_pressure_max": 0.0,
    "lo_near_cancel_rate": 0.0,
    "hi_mode_pressure_mean": 0.0,
    "lo_pressure_penalty": 0.0,
    "lo_pressure_threshold": 0.8,
    "lo_pressure_max_penalty": 0.0,
    "lo_near_cancel_penalty": 0.0,
    "lo_near_cancel_threshold": 0.9,
    "hi_mode_pressure_penalty": 0.0,
    "hi_mode_pressure_threshold": 0.8,
    "lo_budget_cancellation_rate": 0.0,
    "lo_active_drop_rate": 0.0,
    "lo_release_drop_rate": 0.0,
    "delta_lo_budget_cancellations": 0.0,
    "delta_lo_active_dropped_on_mode_switch": 0.0,
    "delta_lo_release_dropped_in_degraded_mode": 0.0,
    "active_lo_job_count": 0.0,
    "active_lo_job_rate": 0.0,
    "active_lo_work_ratio": 0.0,
    "active_lo_under_hi_pressure": 0.0,
    "active_lo_under_hi_pressure_penalty": 0.0,
    "active_lo_under_hi_pressure_penalty_value": 0.0,
    "is_explicit_noop_action": 0.0,
    "delta_lo_released_jobs": 0.0,
    "delta_lo_finalized_jobs": 0.0,
    "delta_lo_service_quality_sum": 0.0,
    "delta_lo_equiv_jne": 0.0,
    "delta_lo_zero_service_jobs": 0.0,
    "delta_lo_partial_service_jobs": 0.0,
    "lo_service_quality_per_finalized_job": 0.0,
    "lo_equiv_jne_per_finalized_job": 0.0,
    "cumulative_lo_service_quality_sum": 0.0,
    "cumulative_lo_equiv_jne": 0.0,
    "cumulative_lo_finalized_jobs": 0.0,
    "delta_lo_deadline_misses": 0.0,
    "delta_hi_deadline_misses": 0.0,
    "lo_deadline_miss_rate": 0.0,
    "hi_deadline_miss_rate": 0.0,
}


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    """agent 驱动仿真的配置。"""

    agent_period: int = 10
    end_time: int = 1000
    check_safety: bool = True
    reward_mode: str = "mendes"
    # 与训练 env 同步的阶段 3 开关：
    # True 时，凡是 decrease 集合触及 HI 任务的动作都会被拒绝。
    forbid_decreasing_hi_budgets: bool = False
    # 与 DQN env 一致：若 >0，则拒绝任何使任务 budget 低于 episode 初始 budget * ratio 的动作。
    budget_floor_ratio: float = 0.0
    budget_rounding_mode: str = "ceil_floor"
    min_budget_delta: int = 1
    enable_deploy_cap_mask: bool = False
    deploy_cap_mask_ratio: float = 4.0
    deploy_cap_mask_criticality: str = "lo"
    budget_update_source: str = "UNSPECIFIED"
    action_space: str = "single"
    step_guard_semantics: str = "checked"


@dataclass(slots=True)
class AgentRuntimeResult:
    """agent 驱动仿真结果。"""

    runtime_result: SimulationResult
    accepted_actions: int
    rejected_actions: int
    noop_actions: int
    total_reward: float
    action_log: list[dict] = field(default_factory=list)
    safety_checked_actions: int = 0
    safety_accepted_actions: int = 0
    safety_rejected_actions: int = 0


@dataclass(slots=True)
class _ActionRewardState:
    """保存要与下一个 runtime interval 配对的动作正则上下文。"""

    ordered_tasks: Sequence[Task]
    initial_budgets: dict[str, int]
    last_direction: str | None = None
    last_task: str | None = None
    consecutive_increase_by_task: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.consecutive_increase_by_task = {task.name: 0 for task in self.ordered_tasks}

    def neutral(self) -> dict[str, float | bool]:
        return {
            "noop_bonus_if_noop": 0.0,
            "is_explicit_noop_action": False,
            "budget_change_norm": 0.0,
            "budget_drift_mean": 0.0,
            "budget_under_drift_mean": 0.0,
            "budget_over_drift_mean": 0.0,
            "budget_over_drift_deadzone_mean": 0.0,
            "budget_abs_drift_mean": 0.0,
            "budget_abs_drift_deadzone_mean": 0.0,
            "budget_abs_drift_penalty_value": 0.0,
            "over_increase_excess": 0.0,
            "budget_soft_cap_increase_excess": 0.0,
            "budget_soft_cap_penalty_value": 0.0,
            "is_soft_cap_increase_action": 0.0,
            "budget_soft_cap_dwell_excess_mean": 0.0,
            "budget_soft_cap_dwell_excess_max": 0.0,
            "budget_soft_cap_dwell_task_count": 0.0,
            "budget_soft_cap_dwell_task_rate": 0.0,
            "budget_soft_cap_dwell_penalty_value": 0.0,
            "budget_soft_cap_dwell_max_penalty_value": 0.0,
            "budget_soft_cap_dwell_total_penalty_value": 0.0,
            "is_soft_cap_dwell_state": 0.0,
            "pingpong_action": 0.0,
            "increase_concentration_excess": 0.0,
            "consecutive_increase_count_for_target": 0.0,
            "invalid_action": 0.0,
            "is_budget_action": 0.0,
            "is_increase_action": 0.0,
            "is_decrease_action": 0.0,
            "is_transfer_action": 0.0,
            "decrease_hits_hi": 0.0,
            "decrease_hits_lo": 0.0,
            "decrease_task_count": 0.0,
            "unsafe_decrease": 0.0,
        }

    def _budget_ratio(self, budgets: dict[str, int], task_name: str) -> float:
        initial = max(float(self.initial_budgets.get(task_name, 0)), 1e-9)
        return float(budgets.get(task_name, 0)) / initial

    def _budget_drift_stats(
        self,
        budgets: dict[str, int],
        *,
        deadzone: float,
    ) -> dict[str, float]:
        if not self.initial_budgets:
            return {
                "budget_drift_mean": 0.0,
                "budget_under_drift_mean": 0.0,
                "budget_over_drift_mean": 0.0,
                "budget_over_drift_deadzone_mean": 0.0,
                "budget_abs_drift_mean": 0.0,
                "budget_abs_drift_deadzone_mean": 0.0,
            }
        dz = max(0.0, float(deadzone))
        under_total = 0.0
        over_total = 0.0
        over_deadzone_total = 0.0
        abs_total = 0.0
        abs_deadzone_total = 0.0
        for task_name in self.initial_budgets:
            ratio = self._budget_ratio(budgets, task_name)
            under = max(0.0, 1.0 - ratio)
            over = max(0.0, ratio - 1.0)
            abs_drift = abs(ratio - 1.0)
            under_total += under
            over_total += over
            over_deadzone_total += max(0.0, over - dz)
            abs_total += abs_drift
            abs_deadzone_total += max(0.0, abs_drift - dz)
        denom = float(len(self.initial_budgets))
        return {
            "budget_drift_mean": under_total / denom,
            "budget_under_drift_mean": under_total / denom,
            "budget_over_drift_mean": over_total / denom,
            "budget_over_drift_deadzone_mean": over_deadzone_total / denom,
            "budget_abs_drift_mean": abs_total / denom,
            "budget_abs_drift_deadzone_mean": abs_deadzone_total / denom,
        }

    def build(
        self,
        *,
        action: object | None,
        accepted: bool,
        safety_checked: bool,
        budget_before: dict[str, int],
        candidate_budgets: dict[str, int],
        budget_after: dict[str, int],
        updates: dict[str, int],
        reward_parameters: dict[str, float],
    ) -> dict[str, float | bool]:
        """按 env.step 的动作口径生成下一 interval 使用的变量。"""

        variables = self.neutral()
        is_explicit_noop = action is not None and bool(getattr(action, "is_noop", False))
        is_budget_action = action is not None and not is_explicit_noop
        increase_task = getattr(action, "increase_task", None) if action is not None else None
        decrease_tasks = tuple(getattr(action, "decrease_tasks", ())) if action is not None else ()
        is_increase_action = is_budget_action and increase_task is not None and not decrease_tasks
        is_decrease_action = is_budget_action and increase_task is None and bool(decrease_tasks)
        is_transfer_action = is_budget_action and increase_task is not None and bool(decrease_tasks)

        changed_total = 0.0
        for task_name in updates:
            denominator = max(float(self.initial_budgets[task_name]), 1e-9)
            changed_total += abs(
                float(candidate_budgets[task_name]) - float(budget_before[task_name])
            ) / denominator

        abs_deadzone = float(reward_parameters.get("budget_abs_drift_deadzone", 0.0))
        drift = self._budget_drift_stats(budget_after, deadzone=abs_deadzone)
        over_dwell_deadzone = float(
            reward_parameters.get("over_budget_dwell_deadzone", abs_deadzone)
        )
        over_dwell = self._budget_drift_stats(
            budget_after, deadzone=over_dwell_deadzone
        )["budget_over_drift_deadzone_mean"]

        over_increase_deadzone = float(reward_parameters.get("over_increase_deadzone", 0.05))
        over_increase_excess = 0.0
        if is_increase_action and increase_task is not None:
            over_increase_excess = max(
                0.0,
                self._budget_ratio(budget_before, increase_task)
                - 1.0
                - over_increase_deadzone,
            )

        current_direction: str | None = None
        current_task: str | None = None
        if is_increase_action and increase_task is not None:
            current_direction = "increase"
            current_task = increase_task
        elif is_decrease_action and len(decrease_tasks) == 1:
            current_direction = "decrease"
            current_task = decrease_tasks[0]
        pingpong_action = float(
            current_direction is not None
            and current_task is not None
            and self.last_direction is not None
            and self.last_task == current_task
            and self.last_direction != current_direction
        )

        concentration_window = int(reward_parameters.get("concentration_window", 3))
        consecutive_count = 0
        concentration_excess = 0.0
        if accepted and is_increase_action and increase_task is not None:
            consecutive_count = self.consecutive_increase_by_task[increase_task] + 1
            concentration_excess = max(0.0, float(consecutive_count - concentration_window))
            self.consecutive_increase_by_task[increase_task] = consecutive_count
        else:
            for task_name in self.consecutive_increase_by_task:
                self.consecutive_increase_by_task[task_name] = 0

        if accepted and current_direction is not None:
            self.last_direction = current_direction
            self.last_task = current_task

        soft_cap_ratio = float(reward_parameters.get("budget_soft_cap_ratio", 0.0))
        soft_cap_increase_excess = 0.0
        if soft_cap_ratio > 0.0 and is_increase_action and increase_task is not None:
            soft_cap_increase_excess = max(
                0.0, self._budget_ratio(budget_before, increase_task) - soft_cap_ratio
            )
        lo_excesses = [
            max(0.0, self._budget_ratio(budget_after, task.name) - soft_cap_ratio)
            for task in self.ordered_tasks
            if task.criticality is Criticality.LO and soft_cap_ratio > 0.0
        ]
        soft_cap_count = sum(value > 0.0 for value in lo_excesses)
        soft_cap_mean = float(sum(lo_excesses) / len(lo_excesses)) if lo_excesses else 0.0
        soft_cap_max = float(max(lo_excesses)) if lo_excesses else 0.0

        task_criticality = {task.name: task.criticality for task in self.ordered_tasks}
        decrease_hits_hi = any(
            task_criticality.get(name) is Criticality.HI for name in decrease_tasks
        )
        decrease_hits_lo = any(
            task_criticality.get(name) is Criticality.LO for name in decrease_tasks
        )
        invalid_action = float(safety_checked and not accepted)
        noop_bonus = float(reward_parameters.get("noop_bonus", 0.0))
        abs_penalty = float(reward_parameters.get("budget_abs_drift_penalty", 0.0))
        soft_cap_penalty = float(reward_parameters.get("budget_soft_cap_penalty", 0.0))
        soft_cap_dwell_penalty = float(
            reward_parameters.get("budget_soft_cap_dwell_penalty", 0.0)
        )
        soft_cap_dwell_max_penalty = float(
            reward_parameters.get("budget_soft_cap_dwell_max_penalty", 0.0)
        )
        soft_cap_dwell_value = soft_cap_dwell_penalty * soft_cap_mean
        soft_cap_dwell_max_value = soft_cap_dwell_max_penalty * soft_cap_max

        variables.update(
            {
                "noop_bonus_if_noop": noop_bonus if is_explicit_noop else 0.0,
                "is_explicit_noop_action": is_explicit_noop,
                "budget_change_norm": changed_total,
                **drift,
                "budget_over_drift_deadzone_mean": over_dwell,
                "budget_abs_drift_penalty_value": (
                    abs_penalty * drift["budget_abs_drift_deadzone_mean"]
                ),
                "over_increase_excess": over_increase_excess,
                "budget_soft_cap_increase_excess": soft_cap_increase_excess,
                "budget_soft_cap_penalty_value": (
                    soft_cap_penalty * soft_cap_increase_excess
                ),
                "is_soft_cap_increase_action": float(soft_cap_increase_excess > 0.0),
                "budget_soft_cap_dwell_excess_mean": soft_cap_mean,
                "budget_soft_cap_dwell_excess_max": soft_cap_max,
                "budget_soft_cap_dwell_task_count": float(soft_cap_count),
                "budget_soft_cap_dwell_task_rate": (
                    float(soft_cap_count / len(lo_excesses)) if lo_excesses else 0.0
                ),
                "budget_soft_cap_dwell_penalty_value": soft_cap_dwell_value,
                "budget_soft_cap_dwell_max_penalty_value": soft_cap_dwell_max_value,
                "budget_soft_cap_dwell_total_penalty_value": (
                    soft_cap_dwell_value + soft_cap_dwell_max_value
                ),
                "is_soft_cap_dwell_state": float(soft_cap_max > 0.0),
                "pingpong_action": pingpong_action,
                "increase_concentration_excess": concentration_excess,
                "consecutive_increase_count_for_target": float(consecutive_count),
                "invalid_action": invalid_action,
                "is_budget_action": float(is_budget_action),
                "is_increase_action": float(is_increase_action),
                "is_decrease_action": float(is_decrease_action),
                "is_transfer_action": float(is_transfer_action),
                "decrease_hits_hi": float(decrease_hits_hi),
                "decrease_hits_lo": float(decrease_hits_lo),
                "decrease_task_count": float(len(decrease_tasks)),
                "unsafe_decrease": float(is_decrease_action and decrease_hits_hi),
            }
        )
        return variables


def _default_safety_checker(ordered_tasks: Sequence[Task]) -> RuntimeBudgetSafetyChecker:
    """构造默认安全检查器：必须使用 AMC-rtb 设计时 `R_LO`。"""

    design_r_lo = build_design_r_lo_map(ordered_tasks)
    return RuntimeBudgetSafetyChecker(ordered_tasks=ordered_tasks, design_r_lo=design_r_lo)


def _budget_floor_violation(
    *,
    updates: dict[str, int],
    initial_budgets: dict[str, int],
    budget_floor_ratio: float,
) -> str | None:
    """检查局部预算更新是否会让任务预算跌破 episode 初始 floor。

    该 helper 与 `AmcBudgetEnv._budget_floor_violation()` 保持同一口径：
    - ratio<=0 时不启用约束；
    - 只检查本步被更新的任务；
    - floor 使用 `ceil(initial * ratio)`；
    - 返回值包含任务名，便于 wrapper 日志和 DQN env 日志对齐。
    """

    if budget_floor_ratio <= 0.0:
        return None
    for task_name, candidate_budget in updates.items():
        floor_value = max(1, math.ceil(initial_budgets[task_name] * budget_floor_ratio))
        if candidate_budget < floor_value:
            return f"budget_floor_violation:{task_name}"
    return None


def _ensure_reward_variables(reward_variables: dict[str, float | bool]) -> dict[str, float | bool]:
    """补齐 reward expression 需要的默认变量。

    runtime wrapper 的 baseline / noop 路径没有 env.py 那么完整的 shaping 上下文，
    因此这里用默认值兜底，避免新 reward mode 因变量缺失直接崩溃。
    """

    for key, value in RECOVERY_REWARD_DEFAULT_VARIABLES.items():
        reward_variables.setdefault(key, value)
    return reward_variables


def simulate_ordered_taskset_with_agent(
    *,
    ordered_tasks: Sequence[Task],
    scenario: ExecutionScenario,
    agent: BudgetAgent,
    runtime_config: RuntimeConfig,
    agent_config: AgentRuntimeConfig,
    safety_checker: RuntimeBudgetSafetyChecker | None = None,
    bounds: NormalizationBounds | None = None,
    qamc_profile_bundle: QAmcProfileBundle | None = None,
) -> AgentRuntimeResult:
    """以固定 agent 周期驱动 runtime，并记录动作接受/拒绝统计。"""

    # 奖励参数按 reward_mode 从配置文件读取，避免 runtime wrapper 内部硬编码。
    monitor = RuntimeMonitor(reward_mode=agent_config.reward_mode)
    reward_mode_config = load_reward_mode_config(agent_config.reward_mode)
    runtime_cfg = replace(runtime_config, end_time=agent_config.end_time)
    validate_qamc_rl_semantics(
        semantics=runtime_cfg.semantics,
        action_space=agent_config.action_space,
        check_safety=agent_config.check_safety,
        step_guard_semantics=agent_config.step_guard_semantics,
        budget_rounding_mode=agent_config.budget_rounding_mode,
        min_budget_delta=agent_config.min_budget_delta,
    )
    build_kwargs = {
        "ordered_tasks": ordered_tasks,
        "scenario": scenario,
        "config": runtime_cfg,
        "budget_state": BudgetState.from_tasks(ordered_tasks),
        "monitor": monitor,
    }
    if qamc_profile_bundle is not None:
        build_kwargs["qamc_profile_bundle"] = qamc_profile_bundle
    engine = EventRuntimeEngine.build(**build_kwargs)
    engine.run_until(0, include_boundary=True)
    lo_service_tracker = LoServiceRewardTracker()
    lo_service_tracker.prime(engine.finish())
    # 固定记录本轮 episode 的初始预算快照。
    # 后续任何 floor 判断都必须相对这份初始值，而不是相对当前已漂移的 runtime budget。
    initial_budgets = dict(engine.runtime_budgets.budgets)
    action_reward_state = _ActionRewardState(
        ordered_tasks=ordered_tasks,
        initial_budgets=initial_budgets,
    )
    pending_action_reward_variables = action_reward_state.neutral()

    checker = safety_checker or _default_safety_checker(ordered_tasks)
    accepted_actions = 0
    rejected_actions = 0
    noop_actions = 0
    total_reward = 0.0
    action_log: list[dict] = []
    safety_checked_actions = 0
    safety_accepted_actions = 0
    safety_rejected_actions = 0

    current_tick = 0
    prev_job_start_count = 0
    prev_lo_overrun_count = 0
    prev_hi_overrun_count = 0
    prev_mode_changes = 0
    prev_lo_cancellations = 0
    prev_deadline_misses = 0
    prev_lo_deadline_misses = 0
    prev_hi_deadline_misses = 0
    previous_reward_time = 0
    while current_tick < agent_config.end_time:
        # 决策点前先处理同一时刻边界事件，保证观测语义稳定。
        engine.run_until(current_tick, include_boundary=True)
        runtime_snapshot = engine.finish()
        lo_service_delta = lo_service_tracker.consume(
            runtime_snapshot,
            terminal=False,
        )
        mode_changes = runtime_snapshot.mode_change_count()
        lo_cancellations = runtime_snapshot.lo_job_cancellation_count()
        deadline_misses = len(runtime_snapshot.deadline_misses)
        task_criticality = {task.name: task.criticality for task in ordered_tasks}
        lo_deadline_misses = sum(
            1
            for miss in runtime_snapshot.deadline_misses
            if task_criticality.get(miss.task) is Criticality.LO
        )
        hi_deadline_misses = sum(
            1
            for miss in runtime_snapshot.deadline_misses
            if task_criticality.get(miss.task) is not Criticality.LO
        )
        delta_job_start = monitor.job_start_count - prev_job_start_count
        delta_lo_overrun = monitor.lo_overrun_count - prev_lo_overrun_count
        delta_hi_overrun = monitor.hi_overrun_count - prev_hi_overrun_count
        delta_mode_changes = mode_changes - prev_mode_changes
        delta_lo_cancellations = lo_cancellations - prev_lo_cancellations
        delta_deadline_misses = deadline_misses - prev_deadline_misses
        delta_lo_deadline_misses = lo_deadline_misses - prev_lo_deadline_misses
        delta_hi_deadline_misses = hi_deadline_misses - prev_hi_deadline_misses
        # 本次 reward 对应 previous_reward_time 到当前决策点之间的 interval。
        # 动作正则上下文由上一个决策点产生，并与该 interval 配对。
        interval_time = max(1.0, float(engine.current_time - previous_reward_time))
        delta_total_jobs = max(1.0, float(delta_job_start))
        lo_overrun_rate = float(delta_lo_overrun) / delta_total_jobs
        hi_overrun_rate = float(delta_hi_overrun) / delta_total_jobs
        mode_change_rate = float(delta_mode_changes) / interval_time
        mode_change_per_job = float(delta_mode_changes) / delta_total_jobs
        lo_cancellation_rate = float(delta_lo_cancellations) / delta_total_jobs
        deadline_miss_rate = float(delta_deadline_misses) / delta_total_jobs
        lo_deadline_miss_rate = float(delta_lo_deadline_misses) / delta_total_jobs
        hi_deadline_miss_rate = float(delta_hi_deadline_misses) / delta_total_jobs
        # 与训练环境保持一致：runtime wrapper 的奖励由“配置文件公式”计算。
        _ = (
            delta_job_start,
            delta_lo_overrun,
            delta_hi_overrun,
            delta_mode_changes,
            delta_lo_cancellations,
            delta_deadline_misses,
        )
        _ = monitor.consume_reward()
        event_job_start_reward = monitor.reward_weights.job_start * delta_job_start
        event_lo_overrun_reward = monitor.reward_weights.lo_overrun * delta_lo_overrun
        event_hi_overrun_reward = monitor.reward_weights.hi_overrun * delta_hi_overrun
        paper_reward = evaluate_reward_expression(
            reward_mode_config.paper_reward_formula,
            {
                "delta_job_start": float(delta_job_start),
                "delta_lo_overrun": float(delta_lo_overrun),
                "delta_hi_overrun": float(delta_hi_overrun),
                "event_job_start_reward": float(event_job_start_reward),
                "event_lo_overrun_reward": float(event_lo_overrun_reward),
                "event_hi_overrun_reward": float(event_hi_overrun_reward),
            },
        )
        # 与 env.py 保持同口径的 reward 变量表，避免新 reward mode 在 wrapper 路径报 Unknown variable。
        reward_variables: dict[str, float | bool] = {
            "paper_reward": float(paper_reward),
            "budget_change_penalty": float(reward_mode_config.reward_parameters.get("budget_change_penalty", 0.0)),
            "budget_drift_penalty": float(reward_mode_config.reward_parameters.get("budget_drift_penalty", 0.0)),
            "event_job_start_reward": float(event_job_start_reward),
            "event_lo_overrun_reward": float(event_lo_overrun_reward),
            "event_hi_overrun_reward": float(event_hi_overrun_reward),
            "delta_job_start": float(delta_job_start),
            "delta_lo_overrun": float(delta_lo_overrun),
            "delta_hi_overrun": float(delta_hi_overrun),
            "delta_mode_changes": float(delta_mode_changes),
            "delta_lo_cancellations": float(delta_lo_cancellations),
            "delta_deadline_misses": float(delta_deadline_misses),
            "delta_lo_deadline_misses": float(delta_lo_deadline_misses),
            "delta_hi_deadline_misses": float(delta_hi_deadline_misses),
            "interval_time": float(interval_time),
            "delta_total_jobs": float(delta_total_jobs),
            "lo_overrun_rate": float(lo_overrun_rate),
            "hi_overrun_rate": float(hi_overrun_rate),
            "mode_change_rate": float(mode_change_rate),
            "mode_change_per_job": float(mode_change_per_job),
            "lo_cancellation_rate": float(lo_cancellation_rate),
            "deadline_miss_rate": float(deadline_miss_rate),
            "lo_deadline_miss_rate": float(lo_deadline_miss_rate),
            "hi_deadline_miss_rate": float(hi_deadline_miss_rate),
            "delta_lo_released_jobs": float(lo_service_delta.released_jobs),
            "delta_lo_finalized_jobs": float(lo_service_delta.finalized_jobs),
            "delta_lo_service_quality_sum": float(lo_service_delta.service_quality_sum),
            "delta_lo_equiv_jne": float(lo_service_delta.equiv_jne),
            "delta_lo_zero_service_jobs": float(lo_service_delta.zero_service_jobs),
            "delta_lo_partial_service_jobs": float(lo_service_delta.partial_service_jobs),
            "lo_service_quality_per_finalized_job": float(
                lo_service_delta.service_quality_per_finalized_job
            ),
            "lo_equiv_jne_per_finalized_job": float(
                lo_service_delta.equiv_jne_per_finalized_job
            ),
            "cumulative_lo_service_quality_sum": float(
                lo_service_tracker.cumulative_service_quality_sum
            ),
            "cumulative_lo_equiv_jne": float(lo_service_tracker.cumulative_equiv_jne),
            "cumulative_lo_finalized_jobs": float(
                lo_service_tracker.cumulative_finalized_jobs
            ),
        }
        reward_variables.update(pending_action_reward_variables)
        # 奖励参数也并入变量表，使 JSON 公式可直接引用参数名（如 mode_change_spike_penalty）。
        reward_variables.update(reward_mode_config.reward_parameters)
        _ensure_reward_variables(reward_variables)
        step_reward = evaluate_reward_expression(
            reward_mode_config.step_reward_formula,
            reward_variables,
        )
        total_reward += step_reward
        prev_job_start_count = monitor.job_start_count
        prev_lo_overrun_count = monitor.lo_overrun_count
        prev_hi_overrun_count = monitor.hi_overrun_count
        prev_mode_changes = mode_changes
        prev_lo_cancellations = lo_cancellations
        prev_deadline_misses = deadline_misses
        prev_lo_deadline_misses = lo_deadline_misses
        prev_hi_deadline_misses = hi_deadline_misses
        previous_reward_time = engine.current_time

        observation = build_observation(
            time=engine.current_time,
            ordered_tasks=ordered_tasks,
            budget_state=engine.runtime_budgets,
            monitor=monitor,
            bounds=bounds,
            rh_risk_context={
                "hi_mode_pressure_mean": 0.0,
                "hi_mode_pressure_max": 0.0,
                "active_lo_job_rate": 0.0,
                "active_lo_work_ratio": 0.0,
                "active_lo_under_hi_pressure": 0.0,
                "recent_active_drop_rate": 0.0,
                "recent_budget_cancellation_rate": 0.0,
                "recent_release_drop_rate": 0.0,
            },
        )
        action = agent.select_action(observation)
        action_accepted = False
        action_safety_checked = False
        action_budget_before = dict(engine.runtime_budgets.budgets)
        action_candidate_budgets = dict(action_budget_before)
        action_budget_after = dict(action_budget_before)
        action_updates: dict[str, int] = {}

        # 统一 noop 语义（阶段 1）：
        # 1) `action is None`：隐式 noop，通常表示“当前 agent 不给预算动作”；
        # 2) `action.is_noop=True`：显式 noop，表示“agent 主动选择动作空间中的 noop 动作”。
        #
        # 这两类 noop 在运行时都必须满足同一执行约束：
        # - 不进行预算安全检查（因为没有候选预算变更）；
        # - 不调用 `apply_budget_updates`（避免制造伪更新事件）；
        # - `updates` 固定为空字典，`budget_before == budget_after`。
        #
        # 但统计口径上要区分“显式/隐式”：
        # - `noop_actions`：两类 noop 都计入；
        # - `is_explicit_noop`：仅显式 noop 为 True，供后续 rate 分析使用。
        if action is None or bool(getattr(action, "is_noop", False)):
            budget_snapshot = dict(engine.runtime_budgets.budgets)
            action_accepted = action is not None
            action_budget_before = dict(budget_snapshot)
            action_candidate_budgets = dict(budget_snapshot)
            action_budget_after = dict(budget_snapshot)
            noop_actions += 1
            action_log.append(
                {
                    "time": current_tick,
                    # 隐式 noop 不算“显式决策被接受”；显式 noop 记为 accepted=True。
                    # 这样可以保留“agent 确实给了一个合法动作（noop）”这一事实。
                    "accepted": action is not None,
                    "noop": True,
                    "is_explicit_noop": action is not None,
                    "action_id": None if action is None else action.action_id,
                    "updates": {},
                    # 对 noop 来说，候选预算与执行后预算都应与决策前一致。
                    "budget_before": budget_snapshot,
                    "candidate_budgets": budget_snapshot,
                    "budget_after": budget_snapshot,
                    "check_safety": agent_config.check_safety,
                    "safety_checked": False,
                    "reject_reason": None,
                    "lo_budget_cancellation_rate": 0.0,
                    "lo_active_drop_rate": 0.0,
                    "lo_release_drop_rate": 0.0,
                    "delta_lo_budget_cancellations": 0.0,
                    "delta_lo_active_dropped_on_mode_switch": 0.0,
                    "delta_lo_release_dropped_in_degraded_mode": 0.0,
                    "active_lo_job_count": 0.0,
                    "active_lo_job_rate": 0.0,
                    "active_lo_work_ratio": 0.0,
                    "active_lo_under_hi_pressure": 0.0,
                    "active_lo_under_hi_pressure_penalty_value": 0.0,
                }
            )
        else:
            budget_before = dict(engine.runtime_budgets.budgets)
            execution = evaluate_budget_action(
                action=action,
                ordered_tasks=ordered_tasks,
                budget_state=engine.runtime_budgets,
                initial_budgets=initial_budgets,
                config=BudgetActionExecutionConfig(
                    rounding_mode=agent_config.budget_rounding_mode,
                    min_budget_delta=agent_config.min_budget_delta,
                    budget_floor_ratio=agent_config.budget_floor_ratio,
                    fixed_floor_by_task=(
                        {
                            name: profile.full_quality_isolated_wcet
                            for name, profile in qamc_profile_bundle.profiles.items()
                        }
                        if qamc_profile_bundle is not None
                        else {}
                    ),
                    forbid_decreasing_hi_budgets=agent_config.forbid_decreasing_hi_budgets,
                    enable_deploy_cap_mask=agent_config.enable_deploy_cap_mask,
                    deploy_cap_mask_ratio=agent_config.deploy_cap_mask_ratio,
                    deploy_cap_mask_criticality=agent_config.deploy_cap_mask_criticality,
                    check_safety=agent_config.check_safety,
                ),
                safety_checker=checker,
            )
            accepted = execution.accepted
            reject_reason = execution.reject_reason
            updates = dict(execution.updates)
            merged = dict(execution.candidate_budgets)
            safety_checked = execution.safety_checked
            action_accepted = accepted
            action_safety_checked = safety_checked
            action_budget_before = dict(budget_before)
            action_candidate_budgets = dict(merged)
            action_updates = dict(updates)
            reject_diagnostics = execution.diagnostics
            if safety_checked:
                safety_checked_actions += 1
                if accepted:
                    safety_accepted_actions += 1
                else:
                    safety_rejected_actions += 1

            if accepted:
                engine.apply_budget_updates(updates, source=agent_config.budget_update_source)
                accepted_actions += 1
            else:
                rejected_actions += 1
            budget_after = dict(engine.runtime_budgets.budgets)
            action_budget_after = dict(budget_after)
            action_log.append(
                {
                    "time": current_tick,
                    "accepted": accepted,
                    "noop": False,
                    "action_id": action.action_id,
                    "reject_reason": reject_reason,
                    "reject_diagnostics": list(reject_diagnostics),
                    "updates": dict(updates),
                    "budget_before": budget_before,
                    "candidate_budgets": dict(merged),
                    "budget_after": budget_after,
                    "check_safety": agent_config.check_safety,
                    "safety_checked": safety_checked,
                    "lo_budget_cancellation_rate": 0.0,
                    "lo_active_drop_rate": 0.0,
                    "lo_release_drop_rate": 0.0,
                    "delta_lo_budget_cancellations": 0.0,
                    "delta_lo_active_dropped_on_mode_switch": 0.0,
                    "delta_lo_release_dropped_in_degraded_mode": 0.0,
                    "active_lo_job_count": 0.0,
                    "active_lo_job_rate": 0.0,
                    "active_lo_work_ratio": 0.0,
                    "active_lo_under_hi_pressure": 0.0,
                    "active_lo_under_hi_pressure_penalty_value": 0.0,
                }
            )

        pending_action_reward_variables = action_reward_state.build(
            action=action,
            accepted=action_accepted,
            safety_checked=action_safety_checked,
            budget_before=action_budget_before,
            candidate_budgets=action_candidate_budgets,
            budget_after=action_budget_after,
            updates=action_updates,
            reward_parameters=reward_mode_config.reward_parameters,
        )
        current_tick += agent_config.agent_period

    engine.run_until(agent_config.end_time)
    final_snapshot = engine.finish()
    final_lo_service_delta = lo_service_tracker.consume(
        final_snapshot,
        terminal=True,
    )
    final_mode_changes = final_snapshot.mode_change_count()
    final_lo_cancellations = final_snapshot.lo_job_cancellation_count()
    final_deadline_misses = len(final_snapshot.deadline_misses)
    task_criticality = {task.name: task.criticality for task in ordered_tasks}
    final_lo_deadline_misses = sum(
        1
        for miss in final_snapshot.deadline_misses
        if task_criticality.get(miss.task) is Criticality.LO
    )
    final_hi_deadline_misses = sum(
        1
        for miss in final_snapshot.deadline_misses
        if task_criticality.get(miss.task) is not Criticality.LO
    )
    final_delta_job_start = monitor.job_start_count - prev_job_start_count
    final_delta_lo_overrun = monitor.lo_overrun_count - prev_lo_overrun_count
    final_delta_hi_overrun = monitor.hi_overrun_count - prev_hi_overrun_count
    final_delta_mode_changes = final_mode_changes - prev_mode_changes
    final_delta_lo_cancellations = final_lo_cancellations - prev_lo_cancellations
    final_delta_deadline_misses = final_deadline_misses - prev_deadline_misses
    final_delta_lo_deadline_misses = final_lo_deadline_misses - prev_lo_deadline_misses
    final_delta_hi_deadline_misses = final_hi_deadline_misses - prev_hi_deadline_misses
    _ = (
        final_delta_job_start,
        final_delta_lo_overrun,
        final_delta_hi_overrun,
        final_delta_mode_changes,
        final_delta_lo_cancellations,
        final_delta_deadline_misses,
    )
    _ = monitor.consume_reward()
    final_event_job_start_reward = monitor.reward_weights.job_start * final_delta_job_start
    final_event_lo_overrun_reward = monitor.reward_weights.lo_overrun * final_delta_lo_overrun
    final_event_hi_overrun_reward = monitor.reward_weights.hi_overrun * final_delta_hi_overrun
    final_delta_total_jobs = max(1.0, float(final_delta_job_start))
    final_interval_time = max(
        1.0, float(engine.current_time - previous_reward_time)
    )
    final_lo_overrun_rate = float(final_delta_lo_overrun) / final_delta_total_jobs
    final_hi_overrun_rate = float(final_delta_hi_overrun) / final_delta_total_jobs
    final_mode_change_rate = float(final_delta_mode_changes) / final_interval_time
    final_mode_change_per_job = float(final_delta_mode_changes) / final_delta_total_jobs
    final_lo_cancellation_rate = float(final_delta_lo_cancellations) / final_delta_total_jobs
    final_deadline_miss_rate = float(final_delta_deadline_misses) / final_delta_total_jobs
    final_lo_deadline_miss_rate = float(final_delta_lo_deadline_misses) / final_delta_total_jobs
    final_hi_deadline_miss_rate = float(final_delta_hi_deadline_misses) / final_delta_total_jobs
    final_paper_reward = evaluate_reward_expression(
        reward_mode_config.paper_reward_formula,
        {
            "delta_job_start": float(final_delta_job_start),
            "delta_lo_overrun": float(final_delta_lo_overrun),
            "delta_hi_overrun": float(final_delta_hi_overrun),
            "event_job_start_reward": float(final_event_job_start_reward),
            "event_lo_overrun_reward": float(final_event_lo_overrun_reward),
            "event_hi_overrun_reward": float(final_event_hi_overrun_reward),
        },
    )
    final_reward_variables: dict[str, float | bool] = {
                "paper_reward": float(final_paper_reward),
                "budget_change_penalty": float(reward_mode_config.reward_parameters.get("budget_change_penalty", 0.0)),
                "budget_drift_penalty": float(reward_mode_config.reward_parameters.get("budget_drift_penalty", 0.0)),
                "event_job_start_reward": float(final_event_job_start_reward),
                "event_lo_overrun_reward": float(final_event_lo_overrun_reward),
                "event_hi_overrun_reward": float(final_event_hi_overrun_reward),
                "delta_job_start": float(final_delta_job_start),
                "delta_lo_overrun": float(final_delta_lo_overrun),
                "delta_hi_overrun": float(final_delta_hi_overrun),
                "delta_mode_changes": float(final_delta_mode_changes),
                "delta_lo_cancellations": float(final_delta_lo_cancellations),
                "delta_deadline_misses": float(final_delta_deadline_misses),
                "delta_lo_deadline_misses": float(final_delta_lo_deadline_misses),
                "delta_hi_deadline_misses": float(final_delta_hi_deadline_misses),
                "delta_total_jobs": float(final_delta_total_jobs),
                "interval_time": float(final_interval_time),
                "lo_overrun_rate": float(final_lo_overrun_rate),
                "hi_overrun_rate": float(final_hi_overrun_rate),
                "mode_change_rate": float(final_mode_change_rate),
                "mode_change_per_job": float(final_mode_change_per_job),
                "lo_cancellation_rate": float(final_lo_cancellation_rate),
                "deadline_miss_rate": float(final_deadline_miss_rate),
                "lo_deadline_miss_rate": float(final_lo_deadline_miss_rate),
                "hi_deadline_miss_rate": float(final_hi_deadline_miss_rate),
                "delta_lo_released_jobs": float(final_lo_service_delta.released_jobs),
                "delta_lo_finalized_jobs": float(final_lo_service_delta.finalized_jobs),
                "delta_lo_service_quality_sum": float(
                    final_lo_service_delta.service_quality_sum
                ),
                "delta_lo_equiv_jne": float(final_lo_service_delta.equiv_jne),
                "delta_lo_zero_service_jobs": float(
                    final_lo_service_delta.zero_service_jobs
                ),
                "delta_lo_partial_service_jobs": float(
                    final_lo_service_delta.partial_service_jobs
                ),
                "lo_service_quality_per_finalized_job": float(
                    final_lo_service_delta.service_quality_per_finalized_job
                ),
                "lo_equiv_jne_per_finalized_job": float(
                    final_lo_service_delta.equiv_jne_per_finalized_job
                ),
                "cumulative_lo_service_quality_sum": float(
                    lo_service_tracker.cumulative_service_quality_sum
                ),
                "cumulative_lo_equiv_jne": float(lo_service_tracker.cumulative_equiv_jne),
                "cumulative_lo_finalized_jobs": float(
                    lo_service_tracker.cumulative_finalized_jobs
                ),
                **reward_mode_config.reward_parameters,
            }
    final_reward_variables.update(pending_action_reward_variables)
    final_reward_variables.update(reward_mode_config.reward_parameters)
    _ensure_reward_variables(final_reward_variables)
    total_reward += evaluate_reward_expression(
        reward_mode_config.step_reward_formula,
        final_reward_variables,
    )

    return AgentRuntimeResult(
        runtime_result=engine.finish(),
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        noop_actions=noop_actions,
        total_reward=total_reward,
        action_log=action_log,
        safety_checked_actions=safety_checked_actions,
        safety_accepted_actions=safety_accepted_actions,
        safety_rejected_actions=safety_rejected_actions,
    )
