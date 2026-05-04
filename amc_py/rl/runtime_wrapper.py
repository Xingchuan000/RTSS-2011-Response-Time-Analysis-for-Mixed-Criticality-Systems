"""阶段 6：agent-driven runtime wrapper。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from amc_py.budget_runtime import BudgetState
from amc_py.amc import build_design_r_lo_map
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Task
from amc_py.rl.actions import apply_budget_action_candidate
from amc_py.rl.agents import BudgetAgent
from amc_py.rl.monitor import RuntimeMonitor
from amc_py.rl.observation import NormalizationBounds, build_observation
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, merge_budget_candidate
from amc_py.runtime_models import RuntimeConfig, SimulationResult
from amc_py.runtime_scenarios import ExecutionScenario


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    """agent 驱动仿真的配置。"""

    agent_period: int = 10
    end_time: int = 1000
    check_safety: bool = True
    reward_mode: str = "mendes"


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

    monitor = RuntimeMonitor()
    runtime_cfg = RuntimeConfig(
        end_time=agent_config.end_time,
        jobs_per_task=runtime_config.jobs_per_task,
        hyperperiod_limit=runtime_config.hyperperiod_limit,
        capture_trace=runtime_config.capture_trace,
        stop_at_first_miss=runtime_config.stop_at_first_miss,
        drop_lo_jobs_on_hi_switch=runtime_config.drop_lo_jobs_on_hi_switch,
        semantics=runtime_config.semantics,
    )
    engine = EventRuntimeEngine.build(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=runtime_cfg,
        budget_state=BudgetState.from_tasks(ordered_tasks),
        monitor=monitor,
    )

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
        if agent_config.reward_mode == "mendes":
            step_reward = (
                0.1 * delta_job_start
                - 1.0 * delta_lo_overrun
                - 2.0 * delta_hi_overrun
            )
        elif agent_config.reward_mode == "event_delta":
            step_reward = (
                -5.0 * delta_mode_changes
                - 2.0 * delta_lo_cancellations
                - 20.0 * delta_deadline_misses
                + 0.05 * delta_job_start
            )
        elif agent_config.reward_mode == "event_delta_no_job_start":
            step_reward = (
                -5.0 * delta_mode_changes
                - 2.0 * delta_lo_cancellations
                - 20.0 * delta_deadline_misses
            )
        else:
            raise ValueError(f"不支持的 reward_mode: {agent_config.reward_mode}")
        total_reward += step_reward
        _ = monitor.consume_reward()
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

        if action is None:
            noop_actions += 1
            action_log.append(
                {
                    "time": current_tick,
                    "accepted": False,
                    "noop": True,
                    "action_id": None,
                    "updates": {},
                    "budget_before": dict(engine.runtime_budgets.budgets),
                    "candidate_budgets": dict(engine.runtime_budgets.budgets),
                    "budget_after": dict(engine.runtime_budgets.budgets),
                    "check_safety": agent_config.check_safety,
                    "safety_checked": False,
                }
            )
        else:
            budget_before = dict(engine.runtime_budgets.budgets)
            updates = apply_budget_action_candidate(
                action=action,
                budget_state=engine.runtime_budgets,
                ordered_tasks=ordered_tasks,
            )
            merged = merge_budget_candidate(engine.runtime_budgets, updates)

            accepted = True
            reject_reason: str | None = None
            reject_diagnostics: tuple[dict[str, str | int | float], ...] = ()
            safety_checked = False
            if agent_config.check_safety:
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
    if agent_config.reward_mode == "mendes":
        total_reward += (
            0.1 * final_delta_job_start
            - 1.0 * final_delta_lo_overrun
            - 2.0 * final_delta_hi_overrun
        )
    elif agent_config.reward_mode == "event_delta":
        total_reward += (
            -5.0 * final_delta_mode_changes
            - 2.0 * final_delta_lo_cancellations
            - 20.0 * final_delta_deadline_misses
            + 0.05 * final_delta_job_start
        )
    else:
        total_reward += (
            -5.0 * final_delta_mode_changes
            - 2.0 * final_delta_lo_cancellations
            - 20.0 * final_delta_deadline_misses
        )
    _ = monitor.consume_reward()

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
