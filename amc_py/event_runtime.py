"""事件驱动运行时（阶段三、四、五实现）。

本版本新增能力：
1. 固定优先级抢占；
2. completion/overrun 事件 token 失效机制；
3. BudgetOverrunEvent 的 AMC / AMC+ 语义；
4. HI 模式下 LO release 抑制；
5. 系统空闲时 HI -> LO 恢复。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .budget_runtime import BudgetState, BudgetUpdate
from .event_models import Event, EventQueue, EventType
from .models import Criticality, Task
from .runtime import compute_default_end_time
from .runtime_models import (
    BudgetUpdateEvent,
    DeadlineMiss,
    Job,
    JobCancellationEvent,
    ModeRecoveryEvent,
    ModeSwitchEvent,
    RuntimeConfig,
    RuntimeSemantics,
    SimulationResult,
    SystemMode,
)
from .runtime_scenarios import ExecutionScenario


# 事件驱动 runtime bridge 当前只支持 AMC family methods，口径与 tick runtime 保持一致。
_EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS = {"amc_rtb", "amc_max"}


def _validate_event_runtime_bridge_method(method: str) -> None:
    """校验事件驱动 bridge API 的 method 是否属于当前支持集合。"""

    if method not in _EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS:
        supported = ", ".join(sorted(_EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS))
        raise ValueError(
            "当前 event runtime bridge 仅支持 AMC family methods "
            f"({supported})；收到 method={method!r}。"
        )


def _build_job(task: Task, release_index: int, scenario: ExecutionScenario) -> Job:
    """按现有 tick runtime 的同等逻辑构建 job。"""

    release_time = release_index * task.period
    absolute_deadline = release_time + task.deadline
    actual_cost = scenario.actual_cost_for(task, release_index)
    return Job(
        task=task,
        release_index=release_index,
        release_time=release_time,
        absolute_deadline=absolute_deadline,
        actual_cost=actual_cost,
    )


def _job_key(task_name: str, release_index: int) -> tuple[str, int]:
    """统一构造 job 主键。"""

    return (task_name, release_index)


def _select_highest_priority_ready_job(
    active_jobs: Sequence[Job], priority_map: dict[str, int]
) -> Job | None:
    """从 active_jobs 中选出优先级最高且未完成的 job。"""

    best: Job | None = None
    best_priority = -1
    for job in active_jobs:
        if job.finished():
            continue
        prio = priority_map[job.task.name]
        if best is None or prio < best_priority:
            best = job
            best_priority = prio
    return best


@dataclass(slots=True)
class _RuntimeState:
    """事件循环运行态。"""

    current_time: int
    mode: SystemMode
    active_jobs: list[Job]
    running_job: Job | None
    run_started_at: int | None
    event_token_counter: int
    valid_completion_tokens: dict[tuple[str, int], int]
    valid_overrun_tokens: dict[tuple[str, int], int]

    def next_token(self) -> int:
        """生成新 token。"""

        self.event_token_counter += 1
        return self.event_token_counter


def _update_running_progress(state: _RuntimeState, now: int) -> None:
    """把 running_job 在 [run_started_at, now) 的执行量结算到 executed_time。"""

    if state.running_job is None or state.run_started_at is None:
        return
    elapsed = now - state.run_started_at
    if elapsed <= 0:
        state.run_started_at = now
        return
    state.running_job.executed_time += elapsed
    state.run_started_at = now


def _invalidate_job_events(state: _RuntimeState, job: Job) -> None:
    """使某个 job 的 completion/overrun 事件全部失效。"""

    key = _job_key(job.task.name, job.release_index)
    state.valid_completion_tokens.pop(key, None)
    state.valid_overrun_tokens.pop(key, None)


def _maybe_recover_to_lo(state: _RuntimeState, result: SimulationResult, now: int) -> None:
    """HI 模式下若系统空闲，则恢复到 LO 模式。"""

    if state.mode is SystemMode.HI and not state.active_jobs and state.running_job is None:
        state.mode = SystemMode.LO
        result.mode_recoveries.append(ModeRecoveryEvent(recovery_time=now))


def _schedule_running_job_events(
    state: _RuntimeState,
    queue: EventQueue,
    runtime_budgets: BudgetState,
    now: int,
    cfg: RuntimeConfig,
) -> None:
    """为当前 running job 安排 completion 与 overrun 事件。"""

    if state.running_job is None:
        return
    job = state.running_job
    key = _job_key(job.task.name, job.release_index)

    completion_token = state.next_token()
    state.valid_completion_tokens[key] = completion_token
    queue.push(
        Event(
            time=now + job.remaining(),
            event_type=EventType.JOB_COMPLETION,
            task_name=job.task.name,
            release_index=job.release_index,
            token=completion_token,
        )
    )

    if state.mode is not SystemMode.LO:
        return

    budget = runtime_budgets.budget_of(job.task)
    remaining_to_completion = job.actual_cost - job.executed_time
    remaining_to_overrun = budget + 1 - job.executed_time
    if remaining_to_overrun <= 0 and remaining_to_completion > 0:
        # 已执行量在预算更新后立刻越界：同一时刻触发 overrun 事件。
        overrun_token = state.next_token()
        state.valid_overrun_tokens[key] = overrun_token
        queue.push(
            Event(
                time=now,
                event_type=EventType.BUDGET_OVERRUN,
                task_name=job.task.name,
                release_index=job.release_index,
                token=overrun_token,
            )
        )
        return

    if remaining_to_overrun > 0 and remaining_to_overrun < remaining_to_completion:
        overrun_token = state.next_token()
        state.valid_overrun_tokens[key] = overrun_token
        queue.push(
            Event(
                time=now + remaining_to_overrun,
                event_type=EventType.BUDGET_OVERRUN,
                task_name=job.task.name,
                release_index=job.release_index,
                token=overrun_token,
            )
        )
    else:
        state.valid_overrun_tokens.pop(key, None)


def _reschedule(
    state: _RuntimeState,
    queue: EventQueue,
    runtime_budgets: BudgetState,
    priority_map: dict[str, int],
    now: int,
    cfg: RuntimeConfig,
    force: bool = False,
) -> None:
    """按固定优先级重调度，必要时触发抢占并重排事件。"""

    selected = _select_highest_priority_ready_job(state.active_jobs, priority_map)
    if selected is state.running_job and not force:
        return

    if state.running_job is not None:
        _invalidate_job_events(state, state.running_job)

    state.running_job = selected
    if selected is None:
        state.run_started_at = None
        return

    state.run_started_at = now
    _schedule_running_job_events(state, queue, runtime_budgets, now, cfg)


def _drop_active_lo_jobs(active_jobs: list[Job], now: int) -> None:
    """切入 HI 模式时，丢弃所有未完成 LO job。"""

    for job in list(active_jobs):
        if job.task.criticality is not Criticality.LO or job.finished():
            continue
        job.dropped = True
        job.drop_time = now
        active_jobs.remove(job)


def simulate_ordered_taskset_event_driven(
    ordered_tasks: Sequence[Task],
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> SimulationResult:
    """事件驱动仿真入口（阶段三、四、五）。"""

    if not ordered_tasks:
        raise ValueError("ordered_tasks 不能为空")

    task_names = [task.name for task in ordered_tasks]
    if len(task_names) != len(set(task_names)):
        raise ValueError("ordered_tasks 中存在重复任务名")

    cfg = config or RuntimeConfig()
    runtime_budgets = (
        budget_state.copy() if budget_state is not None else BudgetState.from_tasks(ordered_tasks)
    )
    updates = budget_updates or []

    end_time = (
        cfg.end_time
        if cfg.end_time is not None
        else compute_default_end_time(
            ordered_tasks,
            jobs_per_task=cfg.jobs_per_task,
            hyperperiod_limit=cfg.hyperperiod_limit,
        )
    )
    priority_map = {task.name: idx for idx, task in enumerate(ordered_tasks)}
    task_map = {task.name: task for task in ordered_tasks}

    queue = EventQueue()
    for task in ordered_tasks:
        queue.push(
            Event(
                time=0,
                event_type=EventType.JOB_ARRIVAL,
                task_name=task.name,
                release_index=0,
            )
        )
    for update in updates:
        if update.time < 0:
            raise ValueError("budget update time 必须为非负整数")
        queue.push(
            Event(
                time=update.time,
                event_type=EventType.BUDGET_UPDATE,
                payload={"updates": dict(update.updates)},
            )
        )

    result = SimulationResult()
    state = _RuntimeState(
        current_time=0,
        mode=SystemMode.LO,
        active_jobs=[],
        running_job=None,
        run_started_at=None,
        event_token_counter=0,
        valid_completion_tokens={},
        valid_overrun_tokens={},
    )
    all_jobs: list[Job] = []
    jobs_by_key: dict[tuple[str, int], Job] = {}

    while not queue.empty():
        event = queue.pop()
        if event.time >= end_time:
            break
        state.current_time = event.time
        _update_running_progress(state, event.time)

        if event.event_type is EventType.BUDGET_UPDATE:
            update_payload = dict(event.payload.get("updates", {}))
            runtime_budgets.apply_updates(update_payload)
            result.budget_update_events.append(
                BudgetUpdateEvent(time=event.time, updates=dict(update_payload))
            )
            _reschedule(
                state, queue, runtime_budgets, priority_map, event.time, cfg, force=True
            )
            continue

        if event.event_type is EventType.JOB_ARRIVAL:
            assert event.task_name is not None
            assert event.release_index is not None
            task = task_map[event.task_name]

            if state.mode is SystemMode.HI and task.criticality is Criticality.LO:
                next_release_index = event.release_index + 1
                next_release_time = next_release_index * task.period
                if next_release_time < end_time:
                    queue.push(
                        Event(
                            time=next_release_time,
                            event_type=EventType.JOB_ARRIVAL,
                            task_name=task.name,
                            release_index=next_release_index,
                        )
                    )
                _reschedule(state, queue, runtime_budgets, priority_map, event.time, cfg)
                continue

            job = _build_job(task, event.release_index, scenario)
            state.active_jobs.append(job)
            all_jobs.append(job)
            jobs_by_key[_job_key(task.name, event.release_index)] = job
            queue.push(
                Event(
                    time=job.absolute_deadline,
                    event_type=EventType.DEADLINE_CHECK,
                    task_name=job.task.name,
                    release_index=job.release_index,
                )
            )

            next_release_index = event.release_index + 1
            next_release_time = next_release_index * task.period
            if next_release_time < end_time:
                queue.push(
                    Event(
                        time=next_release_time,
                        event_type=EventType.JOB_ARRIVAL,
                        task_name=task.name,
                        release_index=next_release_index,
                    )
                )

            _reschedule(state, queue, runtime_budgets, priority_map, event.time, cfg)
            continue

        if event.event_type is EventType.DEADLINE_CHECK:
            assert event.task_name is not None
            assert event.release_index is not None
            key = _job_key(event.task_name, event.release_index)
            job = jobs_by_key.get(key)
            if job is None:
                continue
            if not job.finished():
                result.deadline_misses.append(
                    DeadlineMiss(
                        task=job.task.name,
                        release_index=job.release_index,
                        release_time=job.release_time,
                        absolute_deadline=job.absolute_deadline,
                        mode_at_miss=state.mode,
                        executed_at_miss=job.executed_time,
                    )
                )
                if cfg.stop_at_first_miss:
                    break
            continue

        if event.event_type is EventType.JOB_COMPLETION:
            assert event.task_name is not None
            assert event.release_index is not None
            assert event.token is not None
            key = _job_key(event.task_name, event.release_index)
            if state.valid_completion_tokens.get(key) != event.token:
                continue

            job = jobs_by_key.get(key)
            if job is None or state.running_job is not job:
                continue

            job.executed_time = job.actual_cost
            job.completion_time = event.time
            if job in state.active_jobs:
                state.active_jobs.remove(job)
            _invalidate_job_events(state, job)
            state.running_job = None
            state.run_started_at = None
            _maybe_recover_to_lo(state, result, event.time)
            _reschedule(state, queue, runtime_budgets, priority_map, event.time, cfg)
            continue

        if event.event_type is EventType.BUDGET_OVERRUN:
            assert event.task_name is not None
            assert event.release_index is not None
            assert event.token is not None
            key = _job_key(event.task_name, event.release_index)
            if state.valid_overrun_tokens.get(key) != event.token:
                continue

            job = jobs_by_key.get(key)
            if job is None or state.running_job is not job:
                continue

            budget = runtime_budgets.budget_of(job.task)
            if job.executed_time <= budget:
                # 理论上不应命中；若因重排导致事件边界变化，直接忽略旧事件。
                continue

            if cfg.semantics is RuntimeSemantics.AMC_PLUS and job.task.criticality is Criticality.LO:
                job.dropped = True
                job.drop_time = event.time
                if job in state.active_jobs:
                    state.active_jobs.remove(job)
                result.job_cancellations.append(
                    JobCancellationEvent(
                        cancel_time=event.time,
                        task=job.task.name,
                        release_index=job.release_index,
                        executed_at_cancel=job.executed_time,
                        budget_at_cancel=budget,
                    )
                )
                _invalidate_job_events(state, job)
                state.running_job = None
                state.run_started_at = None
                _maybe_recover_to_lo(state, result, event.time)
                _reschedule(state, queue, runtime_budgets, priority_map, event.time, cfg)
                continue

            reason = (
                "hi_budget_overrun"
                if job.task.criticality is Criticality.HI
                else "lo_budget_overrun_standard_amc"
            )
            state.mode = SystemMode.HI
            result.mode_switches.append(
                ModeSwitchEvent(
                    switch_time=event.time,
                    triggering_task=job.task.name,
                    triggering_release_index=job.release_index,
                    executed_at_switch=job.executed_time,
                    budget_at_switch=budget,
                    reason=reason,
                )
            )

            if cfg.drop_lo_jobs_on_hi_switch:
                _drop_active_lo_jobs(state.active_jobs, event.time)

            if state.running_job is not None and (
                state.running_job.dropped or state.running_job.finished()
            ):
                _invalidate_job_events(state, state.running_job)
                state.running_job = None
                state.run_started_at = None

            _maybe_recover_to_lo(state, result, event.time)
            _reschedule(state, queue, runtime_budgets, priority_map, event.time, cfg)
            continue

    result.jobs = all_jobs
    result.end_time = min(state.current_time, end_time)
    result.final_mode = state.mode
    return result


def simulate_taskset_with_policy_event_driven(
    tasks: Sequence[Task],
    method: str,
    priority_policy: str,
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> SimulationResult:
    """事件驱动 runtime 的策略桥接入口。"""

    _validate_event_runtime_bridge_method(method)

    # 局部导入，避免模块初始化时把 experiments 的较大依赖链提前拉起。
    from .experiments import resolve_ordering

    ordered_tasks = resolve_ordering(tasks, priority_policy, method)
    return simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=config,
        budget_state=budget_state,
        budget_updates=budget_updates,
    )


__all__ = [
    "simulate_ordered_taskset_event_driven",
    "simulate_taskset_with_policy_event_driven",
]
