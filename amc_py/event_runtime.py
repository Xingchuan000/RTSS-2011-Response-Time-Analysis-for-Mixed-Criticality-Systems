"""事件驱动运行时（阶段三、四、五实现）。

本版本在原有函数式仿真入口基础上新增 `EventRuntimeEngine`，用于：
1. 支持分段推进（run_until）；
2. 支持运行中应用预算更新（apply_budget_updates）；
3. 保持原 `simulate_ordered_taskset_event_driven` API 兼容。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .budget_runtime import BudgetState, BudgetUpdate
from .event_models import Event, EventQueue, EventType
from .models import Criticality, Task
from .rl.monitor import RuntimeMonitor
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
    ScheduleTick,
    SimulationResult,
    SystemMode,
)
from .runtime_scenarios import ExecutionScenario


_EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS = {"amc_rtb", "amc_max"}


def _validate_event_runtime_bridge_method(method: str) -> None:
    """校验事件驱动 bridge API 的 method 是否属于当前支持集合。"""

    if method not in _EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS:
        supported = ", ".join(sorted(_EVENT_RUNTIME_BRIDGE_SUPPORTED_METHODS))
        raise ValueError(
            "当前 event runtime bridge 仅支持 AMC family methods "
            f"({supported})；收到 method={method!r}。"
        )


def _build_job(
    task: Task,
    release_index: int,
    scenario: ExecutionScenario,
    runtime_budget_at_release: int,
) -> Job:
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
        runtime_budget_at_release=runtime_budget_at_release,
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
    started_jobs: set[tuple[str, int]]

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

    budget = job.runtime_budget_at_release
    if budget is None:
        budget = runtime_budgets.budget_of(job.task)
    remaining_to_completion = job.actual_cost - job.executed_time
    remaining_to_overrun = budget + 1 - job.executed_time
    if remaining_to_overrun <= 0 and remaining_to_completion > 0:
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


def _drop_active_lo_jobs(active_jobs: list[Job], now: int) -> None:
    """切入 HI 模式时，丢弃所有未完成 LO job。"""

    for job in list(active_jobs):
        if job.task.criticality is not Criticality.LO or job.finished():
            continue
        job.dropped = True
        job.drop_time = now
        active_jobs.remove(job)


def _reschedule(
    state: _RuntimeState,
    queue: EventQueue,
    runtime_budgets: BudgetState,
    priority_map: dict[str, int],
    now: int,
    cfg: RuntimeConfig,
    monitor: RuntimeMonitor | None = None,
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

    selected_key = _job_key(selected.task.name, selected.release_index)
    if selected_key not in state.started_jobs:
        state.started_jobs.add(selected_key)
        if monitor is not None:
            monitor.record_job_start(selected.task.name)

    state.run_started_at = now
    _schedule_running_job_events(state, queue, runtime_budgets, now, cfg)


@dataclass(slots=True)
class EventRuntimeEngine:
    """可分段推进的事件驱动 runtime 引擎。"""

    ordered_tasks: Sequence[Task]
    scenario: ExecutionScenario
    config: RuntimeConfig
    budget_state: BudgetState
    end_time: int
    monitor: RuntimeMonitor | None = None
    task_names: list[str] = None  # type: ignore[assignment]
    priority_map: dict[str, int] = None  # type: ignore[assignment]
    task_map: dict[str, Task] = None  # type: ignore[assignment]
    queue: EventQueue = None  # type: ignore[assignment]
    result: SimulationResult = None  # type: ignore[assignment]
    state: _RuntimeState = None  # type: ignore[assignment]
    all_jobs: list[Job] = None  # type: ignore[assignment]
    jobs_by_key: dict[tuple[str, int], Job] = None  # type: ignore[assignment]
    halted: bool = False

    def __post_init__(self) -> None:
        """初始化事件队列与内部状态。"""

        self.task_names = [task.name for task in self.ordered_tasks]
        self.priority_map = {task.name: idx for idx, task in enumerate(self.ordered_tasks)}
        self.task_map = {task.name: task for task in self.ordered_tasks}
        self.queue = EventQueue()
        self.result = SimulationResult()
        self.state = _RuntimeState(
            current_time=0,
            mode=SystemMode.LO,
            active_jobs=[],
            running_job=None,
            run_started_at=None,
            event_token_counter=0,
            valid_completion_tokens={},
            valid_overrun_tokens={},
            started_jobs=set(),
        )
        self.all_jobs: list[Job] = []
        self.jobs_by_key: dict[tuple[str, int], Job] = {}

        if self.monitor is not None:
            self.monitor.ensure_tasks(self.task_names)

        for task in self.ordered_tasks:
            self.queue.push(
                Event(
                    time=0,
                    event_type=EventType.JOB_ARRIVAL,
                    task_name=task.name,
                    release_index=0,
                )
            )

    @classmethod
    def build(
        cls,
        *,
        ordered_tasks: Sequence[Task],
        scenario: ExecutionScenario,
        config: RuntimeConfig | None = None,
        budget_state: BudgetState | None = None,
        budget_updates: Sequence[BudgetUpdate] | None = None,
        monitor: RuntimeMonitor | None = None,
    ) -> "EventRuntimeEngine":
        """按旧入口参数构建可运行引擎。"""

        if not ordered_tasks:
            raise ValueError("ordered_tasks 不能为空")

        task_names = [task.name for task in ordered_tasks]
        if len(task_names) != len(set(task_names)):
            raise ValueError("ordered_tasks 中存在重复任务名")

        cfg = config or RuntimeConfig()
        runtime_budgets = (
            budget_state.copy() if budget_state is not None else BudgetState.from_tasks(ordered_tasks)
        )
        end_time = (
            cfg.end_time
            if cfg.end_time is not None
            else compute_default_end_time(
                ordered_tasks,
                jobs_per_task=cfg.jobs_per_task,
                hyperperiod_limit=cfg.hyperperiod_limit,
            )
        )

        engine = cls(
            ordered_tasks=ordered_tasks,
            scenario=scenario,
            config=cfg,
            budget_state=runtime_budgets,
            end_time=end_time,
            monitor=monitor,
        )
        for update in budget_updates or []:
            if update.time < 0:
                raise ValueError("budget update time 必须为非负整数")
            engine.queue.push(
                Event(
                    time=update.time,
                    event_type=EventType.BUDGET_UPDATE,
                    payload={"updates": dict(update.updates)},
                )
            )
        return engine

    @property
    def current_time(self) -> int:
        """返回当前引擎时间。"""

        return self.state.current_time

    @property
    def runtime_budgets(self) -> BudgetState:
        """返回当前运行时预算状态。"""

        return self.budget_state

    def _append_debug_event(self, event: str, **payload: object) -> None:
        """追加一条调试事件。

        这里统一补齐 runtime 上下文，避免各事件分支分别拼装同一批字段，
        也保证导出的 debug 日志在结构上稳定可解析。
        """

        running_job = self.state.running_job
        current_budget: dict[str, int] = dict(self.budget_state.budgets)
        self.result.debug_events.append(
            {
                "time": self.state.current_time,
                "event": event,
                "mode": self.state.mode.name,
                "running_task": None if running_job is None else running_job.task.name,
                "running_release_index": None if running_job is None else running_job.release_index,
                "running_executed_time": None if running_job is None else running_job.executed_time,
                "running_actual_cost": None if running_job is None else running_job.actual_cost,
                "running_budget_at_release": None if running_job is None else running_job.runtime_budget_at_release,
                "current_global_budget": current_budget,
                "active_jobs_count": len(self.state.active_jobs),
                **payload,
            }
        )

    def _reschedule(self, now: int, force: bool = False) -> None:
        """实例级重调度封装。

        包一层实例方法的原因有两个：
        - 统一在调度切换前后补齐 debug 日志，记录是谁被抢占、谁开始运行；
        - 避免各事件分支重复拼装 `_reschedule(...)` 的长参数列表。
        """

        previous = self.state.running_job
        selected = _select_highest_priority_ready_job(self.state.active_jobs, self.priority_map)
        if previous is selected and not force:
            return

        if previous is not None and previous is not selected:
            self._append_debug_event(
                "preempt",
                task=previous.task.name,
                release_index=previous.release_index,
                executed_time=previous.executed_time,
            )

        _reschedule(
            self.state,
            self.queue,
            self.budget_state,
            self.priority_map,
            now,
            self.config,
            monitor=self.monitor,
            force=force,
        )

        current = self.state.running_job
        if current is not None and current is not previous:
            self._append_debug_event(
                "job_start",
                task=current.task.name,
                release_index=current.release_index,
                executed_time=current.executed_time,
                actual_cost=current.actual_cost,
                runtime_budget_at_release=current.runtime_budget_at_release,
            )

    def _advance_time(self, now: int) -> None:
        """统一推进 runtime 时间，并结算 running job 的执行量。

        这是本轮修复最关键的语义收口点：
        - 只要 runtime 时间从 old_time 前进到 now，就必须先把 `[old_time, now)`
          区间内 running job 已经实际获得的 CPU 时间结算进 executed_time；
        - 如果开启 `capture_trace`，还要把这段区间按 tick 展开到 `result.trace`，
          这样后续导出的 runtime trace 才能真实反映 CPU 在每个整数 tick 上
          的执行任务，而不是只在有事件的时刻留下稀疏快照。
        """

        old_time = self.state.current_time
        if now < old_time:
            raise ValueError("cannot move runtime time backwards")

        if self.config.capture_trace:
            running_job = self.state.running_job
            for tick in range(old_time, now):
                self.result.trace.append(
                    ScheduleTick(
                        time=tick,
                        executing_task=None if running_job is None else running_job.task.name,
                        executing_release_index=None if running_job is None else running_job.release_index,
                        mode=self.state.mode,
                    )
                )

        _update_running_progress(self.state, now)
        self.state.current_time = now

    def apply_budget_updates(self, updates: Mapping[str, int]) -> None:
        """在当前时刻应用预算更新，并强制触发重调度。

        这里显式调用 `_advance_time(self.state.current_time)` 不是多余操作。
        `run_until()` 在 agent 决策边界返回后，当前时刻的 running job 可能刚好
        持续执行到了该边界；如果这里直接改预算并强制重排，分段推进路径就会把
        边界前最后一段执行量“吞掉”。统一走 `_advance_time()` 可以保证：
        只要预算更新发生在某个逻辑时间点，更新前 CPU 已经执行过的时间一定先
        结算完成。
        """

        self._advance_time(self.state.current_time)
        update_payload = dict(updates)
        self.budget_state.apply_updates(update_payload)
        self.result.budget_update_events.append(
            BudgetUpdateEvent(time=self.state.current_time, updates=update_payload)
        )
        self._append_debug_event("budget_update", updates=update_payload)
        self._reschedule(self.state.current_time, force=True)

    def _process_event(self, event: Event) -> bool:
        """处理单个事件。返回 False 表示应终止运行。"""

        self._advance_time(event.time)

        if event.event_type is EventType.BUDGET_UPDATE:
            update_payload = dict(event.payload.get("updates", {}))
            self.budget_state.apply_updates(update_payload)
            self.result.budget_update_events.append(
                BudgetUpdateEvent(time=event.time, updates=dict(update_payload))
            )
            self._append_debug_event("budget_update", updates=dict(update_payload))
            self._reschedule(event.time, force=True)
            return True

        if event.event_type is EventType.JOB_ARRIVAL:
            assert event.task_name is not None
            assert event.release_index is not None
            task = self.task_map[event.task_name]

            if self.state.mode is SystemMode.HI and task.criticality is Criticality.LO:
                next_release_index = event.release_index + 1
                next_release_time = next_release_index * task.period
                if next_release_time < self.end_time:
                    self.queue.push(
                        Event(
                            time=next_release_time,
                            event_type=EventType.JOB_ARRIVAL,
                            task_name=task.name,
                            release_index=next_release_index,
                        )
                    )
                self._reschedule(event.time)
                return True

            job = _build_job(
                task,
                event.release_index,
                self.scenario,
                runtime_budget_at_release=self.budget_state.budget_of(task),
            )
            self.state.active_jobs.append(job)
            self.all_jobs.append(job)
            self.jobs_by_key[_job_key(task.name, event.release_index)] = job
            self._append_debug_event(
                "job_arrival",
                task=job.task.name,
                criticality=job.task.criticality.value,
                release_index=job.release_index,
                release_time=job.release_time,
                absolute_deadline=job.absolute_deadline,
                actual_cost=job.actual_cost,
                runtime_budget_at_release=job.runtime_budget_at_release,
            )
            self.queue.push(
                Event(
                    time=job.absolute_deadline,
                    event_type=EventType.DEADLINE_CHECK,
                    task_name=job.task.name,
                    release_index=job.release_index,
                )
            )

            next_release_index = event.release_index + 1
            next_release_time = next_release_index * task.period
            if next_release_time < self.end_time:
                self.queue.push(
                    Event(
                        time=next_release_time,
                        event_type=EventType.JOB_ARRIVAL,
                        task_name=task.name,
                        release_index=next_release_index,
                    )
                )

            self._reschedule(event.time)
            return True

        if event.event_type is EventType.DEADLINE_CHECK:
            assert event.task_name is not None
            assert event.release_index is not None
            key = _job_key(event.task_name, event.release_index)
            job = self.jobs_by_key.get(key)
            if job is None:
                return True
            if not job.finished():
                self._append_debug_event(
                    "deadline_miss",
                    task=job.task.name,
                    criticality=job.task.criticality.value,
                    release_index=job.release_index,
                    release_time=job.release_time,
                    absolute_deadline=job.absolute_deadline,
                    actual_cost=job.actual_cost,
                    executed_at_miss=job.executed_time,
                    runtime_budget_at_release=job.runtime_budget_at_release,
                    completion_time=job.completion_time,
                    dropped=job.dropped,
                    drop_time=job.drop_time,
                )
                self.result.deadline_misses.append(
                    DeadlineMiss(
                        task=job.task.name,
                        release_index=job.release_index,
                        release_time=job.release_time,
                        absolute_deadline=job.absolute_deadline,
                        mode_at_miss=self.state.mode,
                        executed_at_miss=job.executed_time,
                    )
                )
                if self.config.stop_at_first_miss:
                    return False
            return True

        if event.event_type is EventType.JOB_COMPLETION:
            assert event.task_name is not None
            assert event.release_index is not None
            assert event.token is not None
            key = _job_key(event.task_name, event.release_index)
            if self.state.valid_completion_tokens.get(key) != event.token:
                return True

            job = self.jobs_by_key.get(key)
            if job is None or self.state.running_job is not job:
                return True

            job.executed_time = job.actual_cost
            job.completion_time = event.time
            self._append_debug_event(
                "job_completion",
                task=job.task.name,
                release_index=job.release_index,
                completion_time=job.completion_time,
                actual_cost=job.actual_cost,
            )
            if self.monitor is not None:
                self.monitor.record_job_completion(job.task.name, job.executed_time)
            if job in self.state.active_jobs:
                self.state.active_jobs.remove(job)
            _invalidate_job_events(self.state, job)
            self.state.running_job = None
            self.state.run_started_at = None
            _maybe_recover_to_lo(self.state, self.result, event.time)
            self._reschedule(event.time)
            return True

        if event.event_type is EventType.BUDGET_OVERRUN:
            assert event.task_name is not None
            assert event.release_index is not None
            assert event.token is not None
            key = _job_key(event.task_name, event.release_index)
            if self.state.valid_overrun_tokens.get(key) != event.token:
                return True

            job = self.jobs_by_key.get(key)
            if job is None or self.state.running_job is not job:
                return True

            budget = job.runtime_budget_at_release
            if budget is None:
                budget = self.budget_state.budget_of(job.task)
            if job.executed_time <= budget:
                return True

            if (
                self.config.semantics is RuntimeSemantics.AMC_PLUS
                and job.task.criticality is Criticality.LO
            ):
                self._append_debug_event(
                    "budget_overrun",
                    task=job.task.name,
                    criticality=job.task.criticality.value,
                    release_index=job.release_index,
                    executed_time=job.executed_time,
                    runtime_budget_at_release=budget,
                    overrun_semantics=self.config.semantics.value,
                )
                if self.monitor is not None:
                    self.monitor.record_lo_budget_overrun(job.task.name, job.executed_time)
                job.dropped = True
                job.drop_time = event.time
                if job in self.state.active_jobs:
                    self.state.active_jobs.remove(job)
                self.result.job_cancellations.append(
                    JobCancellationEvent(
                        cancel_time=event.time,
                        task=job.task.name,
                        release_index=job.release_index,
                        executed_at_cancel=job.executed_time,
                        budget_at_cancel=budget,
                    )
                )
                _invalidate_job_events(self.state, job)
                self.state.running_job = None
                self.state.run_started_at = None
                _maybe_recover_to_lo(self.state, self.result, event.time)
                self._reschedule(event.time)
                return True

            if self.monitor is not None:
                if job.task.criticality is Criticality.HI:
                    self.monitor.record_hi_budget_overrun(job.task.name, job.executed_time)
                else:
                    self.monitor.record_lo_budget_overrun(job.task.name, job.executed_time)

            reason = (
                "hi_budget_overrun"
                if job.task.criticality is Criticality.HI
                else "lo_budget_overrun_standard_amc"
            )
            self._append_debug_event(
                "budget_overrun",
                task=job.task.name,
                criticality=job.task.criticality.value,
                release_index=job.release_index,
                executed_time=job.executed_time,
                runtime_budget_at_release=budget,
                overrun_semantics=self.config.semantics.value,
                reason=reason,
            )
            self.state.mode = SystemMode.HI
            self.result.mode_switches.append(
                ModeSwitchEvent(
                    switch_time=event.time,
                    triggering_task=job.task.name,
                    triggering_release_index=job.release_index,
                    executed_at_switch=job.executed_time,
                    budget_at_switch=budget,
                    reason=reason,
                )
            )

            if self.config.drop_lo_jobs_on_hi_switch:
                _drop_active_lo_jobs(self.state.active_jobs, event.time)

            if self.state.running_job is not None and (
                self.state.running_job.dropped or self.state.running_job.finished()
            ):
                _invalidate_job_events(self.state, self.state.running_job)
                self.state.running_job = None
                self.state.run_started_at = None

            _maybe_recover_to_lo(self.state, self.result, event.time)
            self._reschedule(event.time)
            return True

        return True

    def run_until(self, stop_time: int, include_boundary: bool = False) -> None:
        """推进仿真直到 stop_time。

        - `include_boundary=False`：不处理 `time == stop_time` 的事件；
        - `include_boundary=True`：处理到并包含 `time == stop_time` 的事件。
        """

        target_time = min(stop_time, self.end_time)
        halted_now = False
        while not self.queue.empty():
            event = self.queue.pop()
            if include_boundary:
                boundary_hit = event.time > target_time
            else:
                boundary_hit = event.time >= target_time
            if boundary_hit:
                self.queue.push(event)
                break
            if event.time >= self.end_time:
                break
            if not self._process_event(event):
                self.halted = True
                halted_now = True
                break
        if self.state.current_time < target_time and not halted_now and not self.halted:
            self._advance_time(target_time)

    def finish(self) -> SimulationResult:
        """收敛并输出最终仿真结果。"""

        self.result.jobs = self.all_jobs
        self.result.end_time = min(self.state.current_time, self.end_time)
        self.result.final_mode = self.state.mode
        return self.result


def simulate_ordered_taskset_event_driven(
    ordered_tasks: Sequence[Task],
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
    monitor: RuntimeMonitor | None = None,
) -> SimulationResult:
    """事件驱动仿真入口（保持原 API，内部转发到 EventRuntimeEngine）。"""

    engine = EventRuntimeEngine.build(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=config,
        budget_state=budget_state,
        budget_updates=budget_updates,
        monitor=monitor,
    )
    engine.run_until(engine.end_time)
    return engine.finish()


def simulate_taskset_with_policy_event_driven(
    tasks: Sequence[Task],
    method: str,
    priority_policy: str,
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
    monitor: RuntimeMonitor | None = None,
) -> SimulationResult:
    """事件驱动 runtime 的策略桥接入口。"""

    _validate_event_runtime_bridge_method(method)

    from .experiments import resolve_ordering

    ordered_tasks = resolve_ordering(tasks, priority_policy, method)
    return simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=config,
        budget_state=budget_state,
        budget_updates=budget_updates,
        monitor=monitor,
    )


__all__ = [
    "EventRuntimeEngine",
    "simulate_ordered_taskset_event_driven",
    "simulate_taskset_with_policy_event_driven",
]
