"""事件驱动运行时（阶段三、四、五实现）。

本版本在原有函数式仿真入口基础上新增 `EventRuntimeEngine`，用于：
1. 支持分段推进（run_until）；
2. 支持运行中应用预算更新（apply_budget_updates）；
3. 保持原 `simulate_ordered_taskset_event_driven` API 兼容。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .amc import build_design_r_lo_map
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
    LoJobLossEvent,
    LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
    LO_LOSS_BUDGET_CANCELLATION,
    LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
    ModeRecoveryEvent,
    ModeSwitchEvent,
    RuntimeConfig,
    budget_overrun_guard_units,
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


def _is_response_based_semantics(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否使用 `s_i + R_i(LO)` 作为 degraded mode 入口。"""

    return semantics in {RuntimeSemantics.AMC_RA, RuntimeSemantics.AMC_RH}


def _is_c_amc_semantics(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否为 C-AMC-sem without DVFS。"""

    return semantics is RuntimeSemantics.C_AMC_SEM


def _uses_idle_recovery(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否使用“系统空闲即恢复”的规则。"""

    return semantics in {
        RuntimeSemantics.AMC,
        RuntimeSemantics.AMC_PLUS,
        RuntimeSemantics.AMC_RA,
        RuntimeSemantics.C_AMC_SEM,
    }


def _build_job(
    task: Task,
    release_index: int,
    scenario: ExecutionScenario,
    runtime_budget_at_release: int,
    actual_cost_override: int | None = None,
    *,
    released_in_mode: SystemMode = SystemMode.LO,
    is_degraded: bool = False,
    service_quality_if_completed: float = 1.0,
    original_actual_cost: int | None = None,
    original_runtime_budget_at_release: int | None = None,
) -> Job:
    """按现有 tick runtime 的同等逻辑构建 job。"""

    release_time = release_index * task.period
    absolute_deadline = release_time + task.deadline
    raw_actual_cost = scenario.actual_cost_for(task, release_index)
    actual_cost = actual_cost_override if actual_cost_override is not None else raw_actual_cost
    return Job(
        task=task,
        release_index=release_index,
        release_time=release_time,
        absolute_deadline=absolute_deadline,
        actual_cost=actual_cost,
        runtime_budget_at_release=runtime_budget_at_release,
        released_in_mode=released_in_mode,
        is_degraded=is_degraded,
        service_quality_if_completed=float(service_quality_if_completed),
        original_actual_cost=raw_actual_cost if original_actual_cost is None else original_actual_cost,
        original_runtime_budget_at_release=(
            runtime_budget_at_release
            if original_runtime_budget_at_release is None
            else original_runtime_budget_at_release
        ),
    )


def _c_amc_sem_degraded_lo_budget(task: Task, cfg: RuntimeConfig) -> int:
    """计算 C-AMC-sem 中 LO task 在 HI mode 释放时的 degraded budget。"""

    budget = int(round(task.c_lo * cfg.c_amc_sem_lo_degradation_ratio))
    return max(1, min(task.c_lo, budget))


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
    valid_response_expiry_tokens: dict[tuple[str, int], int]
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
    """使某个 job 的 completion/overrun/response-expiry 事件全部失效。"""

    key = _job_key(job.task.name, job.release_index)
    state.valid_completion_tokens.pop(key, None)
    state.valid_overrun_tokens.pop(key, None)
    state.valid_response_expiry_tokens.pop(key, None)

def _has_expired_active_hi_job(active_jobs: Sequence[Job], now: int) -> bool:
    """判断当前是否仍存在已经达到自身 response expiry 的 active HI job。"""

    for job in active_jobs:
        if job.task.criticality is not Criticality.HI:
            continue
        if job.finished():
            continue
        if job.response_time_expiry is not None and job.response_time_expiry <= now:
            return True
    return False


def _maybe_recover_to_lo(
    state: _RuntimeState,
    result: SimulationResult,
    now: int,
    cfg: RuntimeConfig,
) -> None:
    """对使用 idle recovery 的语义，在系统空闲时恢复到 LO 模式。"""

    if not _uses_idle_recovery(cfg.semantics):
        return
    if state.mode is SystemMode.HI and not state.active_jobs and state.running_job is None:
        state.mode = SystemMode.LO
        result.mode_recoveries.append(ModeRecoveryEvent(recovery_time=now, reason="idle"))


def _maybe_recover_rh_to_lo(state: _RuntimeState, result: SimulationResult, now: int) -> None:
    """AMC-RH：当某个 HI job 完成后，若不存在 expired active HI job，则恢复到 LO。"""

    if state.mode is not SystemMode.HI:
        return
    if _has_expired_active_hi_job(state.active_jobs, now):
        return
    state.mode = SystemMode.LO
    result.mode_recoveries.append(
        ModeRecoveryEvent(recovery_time=now, reason="rh_no_expired_hi_job")
    )


def _compute_busy_period_start_for_new_job(
    *,
    active_jobs: Sequence[Job],
    new_task: Task,
    priority_map: dict[str, int],
    now: int,
) -> int:
    """计算新 job 所属 priority-level busy period 的起点。"""

    new_prio = priority_map[new_task.name]
    candidates: list[Job] = []
    for job in active_jobs:
        if job.finished():
            continue
        if priority_map[job.task.name] <= new_prio:
            candidates.append(job)
    if not candidates:
        return now
    predecessor = max(candidates, key=lambda job: priority_map[job.task.name])
    if predecessor.busy_period_start is None:
        return predecessor.release_time
    return predecessor.busy_period_start


def _schedule_response_time_expiry_for_hi_job(
    state: _RuntimeState,
    queue: EventQueue,
    job: Job,
    now: int,
) -> None:
    """为 RA/RH 语义下的 HI job 安排 response-time expiry 事件。"""

    if job.task.criticality is not Criticality.HI:
        return
    if job.response_time_expiry is None:
        return
    token = state.next_token()
    key = _job_key(job.task.name, job.release_index)
    state.valid_response_expiry_tokens[key] = token
    queue.push(
        Event(
            time=max(now, job.response_time_expiry),
            event_type=EventType.RESPONSE_TIME_EXPIRY,
            task_name=job.task.name,
            release_index=job.release_index,
            token=token,
        )
    )


def _enter_degraded_mode_due_to_response_expiry(
    *,
    state: _RuntimeState,
    result: SimulationResult,
    job: Job,
    now: int,
    cfg: RuntimeConfig,
) -> None:
    """RA/RH：由 HI job 的 response-time expiry 触发 degraded mode。"""

    if state.mode is SystemMode.HI:
        return
    state.mode = SystemMode.HI
    result.mode_switches.append(
        ModeSwitchEvent(
            switch_time=now,
            triggering_task=job.task.name,
            triggering_release_index=job.release_index,
            executed_at_switch=job.executed_time,
            budget_at_switch=job.runtime_budget_at_release,
            reason="hi_response_time_expiry",
        )
    )
    if cfg.drop_lo_jobs_on_hi_switch:
        dropped_lo_jobs = _drop_active_lo_jobs(state.active_jobs, now)
        _record_active_lo_drops_on_mode_switch(result, dropped_lo_jobs, now)


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
        # C-AMC-sem 的计划边界要求：HI mode 中新释放的 LO job 仍然会运行，
        # 因此必须继续安排预算超限事件，才能复用既有的取消统计口径。
        # 除了这一种情况之外，其余语义在 HI mode 下保持原实现不变。
        if not (
            cfg.semantics is RuntimeSemantics.C_AMC_SEM
            and job.task.criticality is Criticality.LO
        ):
            return

    if _is_response_based_semantics(cfg.semantics) and job.task.criticality is Criticality.HI:
        # AMC-RA / AMC-RH 中，HI 任务不能通过 budget overrun 触发 degraded mode；
        # 它们统一由 RESPONSE_TIME_EXPIRY 事件负责切换。
        state.valid_overrun_tokens.pop(key, None)
        return

    budget = job.runtime_budget_at_release
    if budget is None:
        budget = runtime_budgets.budget_of(job.task)
    remaining_to_completion = job.actual_cost - job.executed_time
    remaining_to_overrun = budget + budget_overrun_guard_units(cfg) - job.executed_time
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


def _drop_active_lo_jobs(active_jobs: list[Job], now: int) -> list[Job]:
    """切入 HI 模式时，丢弃所有未完成 LO job，并返回被丢弃的 job 列表。"""

    dropped_jobs: list[Job] = []
    for job in list(active_jobs):
        if job.task.criticality is not Criticality.LO or job.finished():
            continue
        job.dropped = True
        job.drop_time = now
        active_jobs.remove(job)
        dropped_jobs.append(job)
    return dropped_jobs


def _record_lo_job_loss(
    result: SimulationResult,
    *,
    loss_time: int,
    job: Job,
    reason: str,
    budget_at_loss: int | None = None,
) -> None:
    """把单个 LO job 的未正常完成事件追加到 reason-level 结果列表。"""

    result.lo_job_losses.append(
        LoJobLossEvent(
            loss_time=loss_time,
            task=job.task.name,
            release_index=job.release_index,
            release_time=job.release_time,
            executed_at_loss=job.executed_time,
            budget_at_loss=budget_at_loss,
            reason=reason,
        )
    )


def _record_active_lo_drops_on_mode_switch(
    result: SimulationResult,
    dropped_jobs: list[Job],
    now: int,
) -> None:
    """把模式切换时被直接丢弃的 active LO job 逐个写入 reason-level 统计。"""

    for job in dropped_jobs:
        _record_lo_job_loss(
            result,
            loss_time=now,
            job=job,
            reason=LO_LOSS_ACTIVE_DROPPED_ON_MODE_SWITCH,
            budget_at_loss=job.runtime_budget_at_release,
        )


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
    design_r_lo: dict[str, int] = None  # type: ignore[assignment]
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
        self.design_r_lo = (
            build_design_r_lo_map(self.ordered_tasks)
            if _is_response_based_semantics(self.config.semantics)
            else {}
        )
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
            valid_response_expiry_tokens={},
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

        if not self.config.capture_debug_events:
            return

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
                "running_released_in_mode": None if running_job is None else running_job.released_in_mode.name,
                "running_is_degraded": None if running_job is None else running_job.is_degraded,
                "running_service_quality_if_completed": (
                    None if running_job is None else running_job.service_quality_if_completed
                ),
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

    def _record_or_suppress_dropped_lo_release(
        self,
        task: Task,
        release_index: int,
        now: int,
    ) -> None:
        """在 degraded mode 中处理 LO release：默认 suppress，可选记录为 dropped job。"""

        if not self.config.record_dropped_lo_releases:
            self._append_debug_event(
                "lo_release_suppressed",
                task=task.name,
                release_index=release_index,
            )
            return

        job = _build_job(
            task,
            release_index,
            self.scenario,
            runtime_budget_at_release=self.budget_state.budget_of(task),
            released_in_mode=self.state.mode,
            is_degraded=False,
            service_quality_if_completed=1.0,
            original_runtime_budget_at_release=self.budget_state.budget_of(task),
        )
        job.dropped = True
        job.drop_time = now
        self.all_jobs.append(job)
        self.jobs_by_key[_job_key(task.name, release_index)] = job
        self.result.job_cancellations.append(
            JobCancellationEvent(
                cancel_time=now,
                task=task.name,
                release_index=release_index,
                executed_at_cancel=0,
                budget_at_cancel=job.runtime_budget_at_release or task.c_lo,
                reason=LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
            )
        )
        _record_lo_job_loss(
            self.result,
            loss_time=now,
            job=job,
            reason=LO_LOSS_RELEASE_DROPPED_IN_DEGRADED_MODE,
            budget_at_loss=job.runtime_budget_at_release or task.c_lo,
        )
        self._append_debug_event(
            "lo_release_dropped",
            task=task.name,
            release_index=release_index,
        )

    def _maybe_enter_c_amc_sem_hi_mode_at_arrival(
        self,
        events: Sequence[Event],
        now: int,
    ) -> bool:
        """C-AMC-sem：HI abnormal job 在 release 时刻触发 LO->HI。"""

        if not _is_c_amc_semantics(self.config.semantics):
            return False
        if self.state.mode is not SystemMode.LO:
            return False

        abnormal_arrivals: list[tuple[Event, Task, int]] = []
        for event in events:
            assert event.task_name is not None
            assert event.release_index is not None
            task = self.task_map[event.task_name]
            if task.criticality is not Criticality.HI:
                continue
            actual_cost = self.scenario.actual_cost_for(task, event.release_index)
            if actual_cost > task.c_lo:
                abnormal_arrivals.append((event, task, actual_cost))

        if not abnormal_arrivals:
            return False

        # 同一时刻如果有多个 HI abnormal arrival，严格沿用当前固定优先级
        # 顺序，选出优先级最高的触发者写入唯一一次 mode switch 记录。
        trigger_event, trigger_task, trigger_actual_cost = min(
            abnormal_arrivals,
            key=lambda item: self.priority_map[item[1].name],
        )
        assert trigger_event.release_index is not None

        self.state.mode = SystemMode.HI
        self.result.mode_switches.append(
            ModeSwitchEvent(
                switch_time=now,
                triggering_task=trigger_task.name,
                triggering_release_index=trigger_event.release_index,
                executed_at_switch=0,
                budget_at_switch=trigger_task.c_lo,
                reason="semi_clairvoyant_hi_abnormal_arrival",
            )
        )
        self._append_debug_event(
            "c_amc_sem_mode_switch",
            triggering_task=trigger_task.name,
            triggering_release_index=trigger_event.release_index,
            trigger_actual_cost=trigger_actual_cost,
            trigger_c_lo=trigger_task.c_lo,
        )
        return True

    def _schedule_next_release(self, task: Task, release_index: int) -> None:
        """安排某个任务的下一次周期释放。"""

        next_release_index = release_index + 1
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

    def _process_single_arrival_in_priority_order(
        self,
        event: Event,
        *,
        release_mode: SystemMode | None = None,
    ) -> None:
        """按优先级顺序处理同一时刻的单个 arrival。"""

        assert event.task_name is not None
        assert event.release_index is not None
        task = self.task_map[event.task_name]
        now = event.time
        if release_mode is None:
            release_mode = self.state.mode

        # 先安排下一次 release，确保 degraded mode 中被 dropped 的 release
        # 也会继续推进 release_index，不会阻断未来周期行为。
        self._schedule_next_release(task, event.release_index)

        runtime_budget_at_release = self.budget_state.budget_of(task)
        actual_cost_override: int | None = None
        # 下面这组元数据会直接写入 Job，供后处理严格区分：
        # - 这个 job 是按 LO 还是 HI 语义释放的；
        # - 是否属于计划定义下的 degraded LO release；
        # - 若完成时应计入多少服务质量；
        # - 降级前的原始 cost / budget 分别是多少。
        is_degraded = False
        service_quality_if_completed = 1.0
        original_actual_cost = None
        original_runtime_budget_at_release = runtime_budget_at_release

        if release_mode is SystemMode.HI and task.criticality is Criticality.LO:
            if _is_c_amc_semantics(self.config.semantics):
                # C-AMC-sem 下 HI mode 中的 LO release 不能被 suppress，而是要
                # 用 XF 缩放后的 degraded budget 重新构造 job，并把实际执行需求
                # 截断到该 budget，避免引入计划外的额外运行量。
                is_degraded = True
                service_quality_if_completed = float(self.config.c_amc_sem_lo_degradation_ratio)
                original_actual_cost = self.scenario.actual_cost_for(task, event.release_index)
                original_runtime_budget_at_release = self.budget_state.budget_of(task)
                runtime_budget_at_release = _c_amc_sem_degraded_lo_budget(
                    task,
                    self.config,
                )
                actual_cost_override = min(original_actual_cost, runtime_budget_at_release)
                self._append_debug_event(
                    "c_amc_sem_degraded_lo_release",
                    task=task.name,
                    release_index=event.release_index,
                    original_actual_cost=original_actual_cost,
                    degraded_actual_cost=actual_cost_override,
                    degraded_budget=runtime_budget_at_release,
                    xf=self.config.c_amc_sem_lo_degradation_ratio,
                )
            else:
                self._record_or_suppress_dropped_lo_release(task, event.release_index, now)
                return

        job = _build_job(
            task,
            event.release_index,
            self.scenario,
            runtime_budget_at_release=runtime_budget_at_release,
            actual_cost_override=actual_cost_override,
            released_in_mode=release_mode,
            is_degraded=is_degraded,
            service_quality_if_completed=service_quality_if_completed,
            original_actual_cost=original_actual_cost,
            original_runtime_budget_at_release=original_runtime_budget_at_release,
        )
        job.busy_period_start = _compute_busy_period_start_for_new_job(
            active_jobs=self.state.active_jobs,
            new_task=task,
            priority_map=self.priority_map,
            now=now,
        )
        if _is_response_based_semantics(self.config.semantics) and task.criticality is Criticality.HI:
            job.response_time_expiry = job.busy_period_start + self.design_r_lo[task.name]

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
            released_in_mode=job.released_in_mode.name,
            is_degraded=job.is_degraded,
            service_quality_if_completed=job.service_quality_if_completed,
            original_actual_cost=job.original_actual_cost,
            original_runtime_budget_at_release=job.original_runtime_budget_at_release,
            busy_period_start=job.busy_period_start,
            response_time_expiry=job.response_time_expiry,
        )
        self.queue.push(
            Event(
                time=job.absolute_deadline,
                event_type=EventType.DEADLINE_CHECK,
                task_name=job.task.name,
                release_index=job.release_index,
            )
        )
        if _is_response_based_semantics(self.config.semantics) and task.criticality is Criticality.HI:
            _schedule_response_time_expiry_for_hi_job(self.state, self.queue, job, now)

    def _process_job_arrival_batch(self, first_event: Event) -> bool:
        """把同一时刻的全部 arrival 取出后按优先级顺序处理。"""

        now = first_event.time
        events = [first_event]
        events.extend(
            self.queue.pop_all_matching(time=now, event_type=EventType.JOB_ARRIVAL)
        )
        events.sort(
            key=lambda event: self.priority_map[event.task_name]  # type: ignore[index]
        )
        mode_before_batch = self.state.mode
        switched_by_c_amc_sem_batch = self._maybe_enter_c_amc_sem_hi_mode_at_arrival(events, now)
        for arrival in events:
            release_mode = self.state.mode
            # 计划要求的严格边界：如果同一 arrival batch 内是由 HI abnormal arrival
            # 触发的 C-AMC-sem 切换，并且显式打开了 primary_on_switch_time，
            # 那么这个 batch 里同一时刻的 LO release 仍按切换前 LO mode primary
            # 语义创建；只有严格晚于 switch_time 的 LO release 才 degraded。
            if (
                switched_by_c_amc_sem_batch
                and self.config.c_amc_sem_primary_on_switch_time
                and mode_before_batch is SystemMode.LO
                and arrival.time == now
            ):
                release_mode = SystemMode.LO
            self._process_single_arrival_in_priority_order(arrival, release_mode=release_mode)
        self._reschedule(now)
        return True

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
            return self._process_job_arrival_batch(event)

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
                    released_in_mode=job.released_in_mode.name,
                    is_degraded=job.is_degraded,
                    service_quality_if_completed=job.service_quality_if_completed,
                    original_actual_cost=job.original_actual_cost,
                    original_runtime_budget_at_release=job.original_runtime_budget_at_release,
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
                released_in_mode=job.released_in_mode.name,
                is_degraded=job.is_degraded,
                service_quality_if_completed=job.service_quality_if_completed,
                original_actual_cost=job.original_actual_cost,
                original_runtime_budget_at_release=job.original_runtime_budget_at_release,
            )
            if self.monitor is not None:
                self.monitor.record_job_completion(job.task.name, job.executed_time)
            completed_job_was_hi = job.task.criticality is Criticality.HI
            if job in self.state.active_jobs:
                self.state.active_jobs.remove(job)
            _invalidate_job_events(self.state, job)
            self.state.running_job = None
            self.state.run_started_at = None
            if self.config.semantics is RuntimeSemantics.AMC_RH and completed_job_was_hi:
                _maybe_recover_rh_to_lo(self.state, self.result, event.time)
            else:
                _maybe_recover_to_lo(self.state, self.result, event.time, self.config)
            self._reschedule(event.time)
            return True

        if event.event_type is EventType.RESPONSE_TIME_EXPIRY:
            assert event.task_name is not None
            assert event.release_index is not None
            assert event.token is not None
            key = _job_key(event.task_name, event.release_index)
            if self.state.valid_response_expiry_tokens.get(key) != event.token:
                return True

            job = self.jobs_by_key.get(key)
            if job is None:
                return True
            if job.finished():
                return True
            if job.task.criticality is not Criticality.HI:
                return True

            self._append_debug_event(
                "response_time_expiry",
                task=job.task.name,
                release_index=job.release_index,
                busy_period_start=job.busy_period_start,
                response_time_expiry=job.response_time_expiry,
                executed_time=job.executed_time,
            )

            if self.state.mode is SystemMode.LO:
                _enter_degraded_mode_due_to_response_expiry(
                    state=self.state,
                    result=self.result,
                    job=job,
                    now=event.time,
                    cfg=self.config,
                )
                if self.state.running_job is not None and self.state.running_job.dropped:
                    _invalidate_job_events(self.state, self.state.running_job)
                    self.state.running_job = None
                    self.state.run_started_at = None
                self._reschedule(event.time, force=True)
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

            if _is_response_based_semantics(self.config.semantics) and job.task.criticality is Criticality.HI:
                return True

            budget = job.runtime_budget_at_release
            if budget is None:
                budget = self.budget_state.budget_of(job.task)
            if job.executed_time <= budget:
                return True

            if (
                self.config.semantics in {
                    RuntimeSemantics.AMC_PLUS,
                    RuntimeSemantics.AMC_RA,
                    RuntimeSemantics.AMC_RH,
                    RuntimeSemantics.C_AMC_SEM,
                }
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
                        reason=LO_LOSS_BUDGET_CANCELLATION,
                    )
                )
                _record_lo_job_loss(
                    self.result,
                    loss_time=event.time,
                    job=job,
                    reason=LO_LOSS_BUDGET_CANCELLATION,
                    budget_at_loss=budget,
                )
                _invalidate_job_events(self.state, job)
                self.state.running_job = None
                self.state.run_started_at = None
                _maybe_recover_to_lo(self.state, self.result, event.time, self.config)
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
                dropped_lo_jobs = _drop_active_lo_jobs(self.state.active_jobs, event.time)
                _record_active_lo_drops_on_mode_switch(
                    self.result,
                    dropped_lo_jobs,
                    event.time,
                )

            if self.state.running_job is not None and (
                self.state.running_job.dropped or self.state.running_job.finished()
            ):
                _invalidate_job_events(self.state, self.state.running_job)
                self.state.running_job = None
                self.state.run_started_at = None

            _maybe_recover_to_lo(self.state, self.result, event.time, self.config)
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
    cfg = config or RuntimeConfig()
    if _is_response_based_semantics(cfg.semantics) and method != "amc_rtb":
        raise ValueError("AMC_RA/AMC_RH runtime semantics must be used with method='amc_rtb'")

    from .experiments import resolve_ordering

    ordered_tasks = resolve_ordering(tasks, priority_policy, method)
    return simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=cfg,
        budget_state=budget_state,
        budget_updates=budget_updates,
        monitor=monitor,
    )


__all__ = [
    "EventRuntimeEngine",
    "simulate_ordered_taskset_event_driven",
    "simulate_taskset_with_policy_event_driven",
]
