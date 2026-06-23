"""阶段 6：agent-driven runtime wrapper。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from amc_py.budget_runtime import BudgetState
from amc_py.amc import build_design_r_lo_map
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Task
from amc_py.rl.actions import action_violates_hi_decrease_guard, apply_budget_action_candidate
from amc_py.rl.agents import BudgetAgent
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.observation import NormalizationBounds, build_observation
from amc_py.rl.reward_config import evaluate_reward_expression, load_reward_mode_config
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, merge_budget_candidate
from amc_py.runtime_models import RuntimeConfig, SimulationResult
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
    "is_explicit_noop_action": 0.0,
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
) -> AgentRuntimeResult:
    """以固定 agent 周期驱动 runtime，并记录动作接受/拒绝统计。"""

    # 奖励参数按 reward_mode 从配置文件读取，避免 runtime wrapper 内部硬编码。
    monitor = RuntimeMonitor(reward_mode=agent_config.reward_mode)
    reward_mode_config = load_reward_mode_config(agent_config.reward_mode)
    runtime_cfg = RuntimeConfig(
        end_time=agent_config.end_time,
        jobs_per_task=runtime_config.jobs_per_task,
        hyperperiod_limit=runtime_config.hyperperiod_limit,
        capture_trace=runtime_config.capture_trace,
        capture_debug_events=runtime_config.capture_debug_events,
        stop_at_first_miss=runtime_config.stop_at_first_miss,
        drop_lo_jobs_on_hi_switch=runtime_config.drop_lo_jobs_on_hi_switch,
        semantics=runtime_config.semantics,
        record_dropped_lo_releases=runtime_config.record_dropped_lo_releases,
    )
    engine = EventRuntimeEngine.build(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=runtime_cfg,
        budget_state=BudgetState.from_tasks(ordered_tasks),
        monitor=monitor,
    )
    # 固定记录本轮 episode 的初始预算快照。
    # 后续任何 floor 判断都必须相对这份初始值，而不是相对当前已漂移的 runtime budget。
    initial_budgets = dict(engine.runtime_budgets.budgets)

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
    while current_tick < agent_config.end_time:
        # 决策点前先处理同一时刻边界事件，保证观测语义稳定。
        engine.run_until(current_tick, include_boundary=True)
        runtime_snapshot = engine.finish()
        mode_changes = runtime_snapshot.mode_change_count()
        lo_cancellations = runtime_snapshot.lo_job_cancellation_count()
        deadline_misses = len(runtime_snapshot.deadline_misses)
        delta_job_start = monitor.job_start_count - prev_job_start_count
        delta_lo_overrun = monitor.lo_overrun_count - prev_lo_overrun_count
        delta_hi_overrun = monitor.hi_overrun_count - prev_hi_overrun_count
        delta_mode_changes = mode_changes - prev_mode_changes
        delta_lo_cancellations = lo_cancellations - prev_lo_cancellations
        delta_deadline_misses = deadline_misses - prev_deadline_misses
        # 与训练环境对齐：构造 interval 级时间与归一化分母。
        # 这里 action_time 即当前决策点时间，current_time 在 run_until 后通常等于该值，
        # 因此 interval_time 最小设为 1.0，保证任何 reward rate 都不会出现除零。
        action_time = current_tick
        interval_time = max(1.0, float(engine.current_time - action_time))
        delta_total_jobs = max(1.0, float(delta_job_start))
        lo_overrun_rate = float(delta_lo_overrun) / delta_total_jobs
        hi_overrun_rate = float(delta_hi_overrun) / delta_total_jobs
        mode_change_rate = float(delta_mode_changes) / interval_time
        mode_change_per_job = float(delta_mode_changes) / delta_total_jobs
        lo_cancellation_rate = float(delta_lo_cancellations) / delta_total_jobs
        deadline_miss_rate = float(delta_deadline_misses) / delta_total_jobs
        # wrapper 路径当前没有“动作被拒绝”输入到 reward 计算前，因此保持 0.0。
        invalid_action = 0.0
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
            "noop_bonus_if_noop": 0.0,
            "budget_change_penalty": float(reward_mode_config.reward_parameters.get("budget_change_penalty", 0.0)),
            "budget_change_norm": 0.0,
            "budget_drift_penalty": float(reward_mode_config.reward_parameters.get("budget_drift_penalty", 0.0)),
            "budget_drift_mean": 0.0,
            "is_explicit_noop_action": False,
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

        observation = build_observation(
            time=engine.current_time,
            ordered_tasks=ordered_tasks,
            budget_state=engine.runtime_budgets,
            monitor=monitor,
            bounds=bounds,
        )
        action = agent.select_action(observation)

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
                }
            )
        else:
            budget_before = dict(engine.runtime_budgets.budgets)
            accepted = True
            reject_reason: str | None = None
            reject_diagnostics: tuple[dict[str, str | int | float], ...] = ()
            safety_checked = False
            if action_violates_hi_decrease_guard(
                action=action,
                ordered_tasks=ordered_tasks,
                forbid_decreasing_hi_budgets=agent_config.forbid_decreasing_hi_budgets,
            ):
                # 与 env 保持完全一致：
                # - 命中 HI decrease 立即拒绝；
                # - reject_reason 固定为 decrease_hi_forbidden；
                # - 不再进入 safety checker，避免被其它 reject_reason 覆盖。
                accepted = False
                reject_reason = "decrease_hi_forbidden"
                updates = {}
                merged = dict(budget_before)
            else:
                updates = apply_budget_action_candidate(
                    action=action,
                    budget_state=engine.runtime_budgets,
                    ordered_tasks=ordered_tasks,
                )
                merged = merge_budget_candidate(engine.runtime_budgets, updates)
                floor_reject_reason = _budget_floor_violation(
                    updates=updates,
                    initial_budgets=initial_budgets,
                    budget_floor_ratio=agent_config.budget_floor_ratio,
                )
                if floor_reject_reason is not None:
                    # floor rejection 不属于 safety checker rejection。
                    # 因此这里只记录 accepted=False 与 reject_reason，不增加 safety 统计计数。
                    accepted = False
                    reject_reason = floor_reject_reason
                    reject_diagnostics = ()
                elif agent_config.check_safety:
                    safety_checked = True
                    safety_checked_actions += 1
                    report = checker.validate_candidate(merged)
                    accepted = report.accepted
                    reject_reason = None if accepted else report.reason
                    reject_diagnostics = report.diagnostics
                    if accepted:
                        safety_accepted_actions += 1
                    else:
                        safety_rejected_actions += 1
                else:
                    accepted = True

            if accepted:
                engine.apply_budget_updates(updates)
                accepted_actions += 1
            else:
                rejected_actions += 1
            budget_after = dict(engine.runtime_budgets.budgets)
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
                }
            )

        current_tick += agent_config.agent_period

    engine.run_until(agent_config.end_time)
    final_snapshot = engine.finish()
    final_mode_changes = final_snapshot.mode_change_count()
    final_lo_cancellations = final_snapshot.lo_job_cancellation_count()
    final_deadline_misses = len(final_snapshot.deadline_misses)
    final_delta_job_start = monitor.job_start_count - prev_job_start_count
    final_delta_lo_overrun = monitor.lo_overrun_count - prev_lo_overrun_count
    final_delta_hi_overrun = monitor.hi_overrun_count - prev_hi_overrun_count
    final_delta_mode_changes = final_mode_changes - prev_mode_changes
    final_delta_lo_cancellations = final_lo_cancellations - prev_lo_cancellations
    final_delta_deadline_misses = final_deadline_misses - prev_deadline_misses
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
    final_interval_time = 1.0
    final_lo_overrun_rate = float(final_delta_lo_overrun) / final_delta_total_jobs
    final_hi_overrun_rate = float(final_delta_hi_overrun) / final_delta_total_jobs
    final_mode_change_rate = float(final_delta_mode_changes) / final_interval_time
    final_mode_change_per_job = float(final_delta_mode_changes) / final_delta_total_jobs
    final_lo_cancellation_rate = float(final_delta_lo_cancellations) / final_delta_total_jobs
    final_deadline_miss_rate = float(final_delta_deadline_misses) / final_delta_total_jobs
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
    total_reward += evaluate_reward_expression(
        reward_mode_config.step_reward_formula,
        _ensure_reward_variables(
            {
                "paper_reward": float(final_paper_reward),
                "noop_bonus_if_noop": 0.0,
                "budget_change_penalty": float(reward_mode_config.reward_parameters.get("budget_change_penalty", 0.0)),
                "budget_change_norm": 0.0,
                "budget_drift_penalty": float(reward_mode_config.reward_parameters.get("budget_drift_penalty", 0.0)),
                "budget_drift_mean": 0.0,
                "is_explicit_noop_action": False,
                "event_job_start_reward": float(final_event_job_start_reward),
                "event_lo_overrun_reward": float(final_event_lo_overrun_reward),
                "event_hi_overrun_reward": float(final_event_hi_overrun_reward),
                "delta_job_start": float(final_delta_job_start),
                "delta_lo_overrun": float(final_delta_lo_overrun),
                "delta_hi_overrun": float(final_delta_hi_overrun),
                "delta_mode_changes": float(final_delta_mode_changes),
                "delta_lo_cancellations": float(final_delta_lo_cancellations),
                "delta_deadline_misses": float(final_delta_deadline_misses),
                "delta_total_jobs": float(final_delta_total_jobs),
                "interval_time": float(final_interval_time),
                "lo_overrun_rate": float(final_lo_overrun_rate),
                "hi_overrun_rate": float(final_hi_overrun_rate),
                "mode_change_rate": float(final_mode_change_rate),
                "mode_change_per_job": float(final_mode_change_per_job),
                "lo_cancellation_rate": float(final_lo_cancellation_rate),
                "deadline_miss_rate": float(final_deadline_miss_rate),
                "invalid_action": 0.0,
                **reward_mode_config.reward_parameters,
            }
        ),
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
