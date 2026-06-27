"""AMC 运行时仿真器（阶段运行时模拟 · 第 4 轮）。

本模块实现一个“基于 tick 的固定优先级抢占式仿真器”。它消费由上游
`resolve_ordering()` 得到的 **已排序任务列表** 与 `ExecutionScenario`
注入的 job 实际执行时间，逐 tick 推进系统状态，并把最终的 job 队列、
trace、deadline miss 等信息写入 `SimulationResult`。

本轮（第 4 轮）的范围约定：
1. 主循环完整接入 AMC 的 `LO -> HI` 切换语义；
2. 支持切换事件记录（`ModeSwitchEvent`）；
3. 支持切换后丢弃活动中的 LO jobs（可通过配置开关关闭）；
4. 支持切换后抑制未来 LO job 释放；
5. `simulate_taskset_with_policy()` 与 `compare_static_and_runtime()`
   仍是第 5 轮任务，本轮不实现。

仿真约定（全局语义）：
- 时间是非负整数 tick。每个 tick 覆盖半开区间 `[t, t+1)`。
- 在 tick `t` 开始时：先释放 release_time == t 的 job，再检查
  absolute_deadline == t 的 miss；随后从就绪集合中挑最高优先级运行。
- 若一个 job 在 tick `t` 执行后 executed_time 达到 actual_cost，
  其 completion_time 记为 `t + 1`（即下一 tick 的起点）。
- 主循环外再补一次 `t == end_time` 的 miss 扫描，确保刚好踩在
  仿真终点的 deadline 也会被记录。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from math import lcm

from .amc import build_design_r_lo_map
from .models import Criticality, Task
from .budget_runtime import BudgetState, BudgetUpdate
from .runtime_models import (
    BudgetUpdateEvent,
    DeadlineMiss,
    Job,
    JobCancellationEvent,
    ModeRecoveryEvent,
    ModeSwitchEvent,
    RuntimeComparisonResult,
    RuntimeConfig,
    RuntimeSemantics,
    ScheduleTick,
    SimulationResult,
    SystemMode,
)
from .runtime_scenarios import ExecutionScenario


# ---------------------------------------------------------------------------
# runtime bridge 约束：当前仅支持 AMC family methods
# ---------------------------------------------------------------------------

# 说明：
# - simulate_taskset_with_policy() / compare_static_and_runtime() 是“桥接入口”；
# - 它们底层复用的是 AMC 风格运行时语义（LO->HI 切换、LO 降级处理）；
# - 因此本轮 hotfix 仅允许 AMC family methods，避免把其它静态方法误导为
#   “已有对应 runtime 语义实现”。
_RUNTIME_BRIDGE_SUPPORTED_METHODS = {"amc_rtb", "amc_max"}


def _validate_runtime_bridge_method(method: str) -> None:
    """校验 runtime bridge API 的 method 是否属于当前支持集合。"""

    if method not in _RUNTIME_BRIDGE_SUPPORTED_METHODS:
        supported = ", ".join(sorted(_RUNTIME_BRIDGE_SUPPORTED_METHODS))
        raise ValueError(
            "当前 runtime bridge 仅支持 AMC family methods "
            f"({supported})；收到 method={method!r}。"
            "simulate_taskset_with_policy() / compare_static_and_runtime() "
            "当前采用 AMC 运行时语义，不应用于 SMC/SMC-NO/其它方法的‘对应 runtime’解释。"
        )


def _is_response_based_semantics(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否以 response expiry 作为 degraded mode 入口。"""

    return semantics in {RuntimeSemantics.AMC_RA, RuntimeSemantics.AMC_RH}


def _is_c_amc_semantics(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否为 C-AMC-sem without DVFS。"""

    return semantics is RuntimeSemantics.C_AMC_SEM


def _uses_idle_recovery(semantics: RuntimeSemantics) -> bool:
    """判断当前语义是否使用 idle recovery。"""

    return semantics in {
        RuntimeSemantics.AMC,
        RuntimeSemantics.AMC_PLUS,
        RuntimeSemantics.AMC_RA,
        RuntimeSemantics.C_AMC_SEM,
    }


# ---------------------------------------------------------------------------
# 时间窗辅助：hyperperiod 与默认 end_time
# ---------------------------------------------------------------------------


def compute_hyperperiod(tasks: Sequence[Task]) -> int:
    """计算任务集的 hyperperiod（所有任务周期的最小公倍数）。

    - 对周期相同的多任务，结果等于该周期；
    - 对互素周期，结果等于周期乘积；
    - 空任务集视为非法输入（没有可定义的 hyperperiod）。
    """

    if not tasks:
        raise ValueError("tasks 不能为空，无法定义 hyperperiod")

    # functools.reduce + math.lcm 支持任意多个周期；从第一个周期开始折叠累乘最小公倍数。
    return reduce(lcm, (task.period for task in tasks))


def compute_default_end_time(
    tasks: Sequence[Task],
    jobs_per_task: int = 5,
    hyperperiod_limit: int = 100_000,
) -> int:
    """估计一个“够用”的默认仿真终止时刻。

    策略：
    - 若 hyperperiod <= hyperperiod_limit，则以 hyperperiod 为基准，同时保证
      至少能给每个任务释放 `jobs_per_task` 次——取两者的 max；
    - 若 hyperperiod 过大（例如参数含大素数周期），则回退到
      `jobs_per_task * max_period`，避免仿真时间爆炸。

    这样默认值在“小且规整”的任务集上可覆盖完整 hyperperiod，而在“大周期”
    任务集上也不会让仿真无限拖长。
    """

    if not tasks:
        raise ValueError("tasks 不能为空")
    if jobs_per_task <= 0:
        raise ValueError("jobs_per_task 必须为正整数")
    if hyperperiod_limit <= 0:
        raise ValueError("hyperperiod_limit 必须为正整数")

    hyperperiod = compute_hyperperiod(tasks)
    max_period = max(task.period for task in tasks)
    # 每任务至少释放 jobs_per_task 次所需的最小 end_time：用最大周期的倍数近似。
    jobs_based = jobs_per_task * max_period

    if hyperperiod <= hyperperiod_limit:
        # 小 hyperperiod：优先覆盖完整周期；同时保证 jobs_per_task 下限。
        return max(hyperperiod, jobs_based)

    # 大 hyperperiod：只保证每任务有足够释放次数，不强行展开完整周期。
    return jobs_based


# ---------------------------------------------------------------------------
# 单步辅助函数
# ---------------------------------------------------------------------------


def _build_job(
    task: Task,
    release_index: int,
    scenario: ExecutionScenario,
    actual_cost_override: int | None = None,
    *,
    runtime_budget_at_release: int | None = None,
    released_in_mode: SystemMode = SystemMode.LO,
    is_degraded: bool = False,
    service_quality_if_completed: float = 1.0,
    original_actual_cost: int | None = None,
    original_runtime_budget_at_release: int | None = None,
) -> Job:
    """根据任务定义 + release_index + 场景构造出一个 Job 实例。

    - release_time 按“绝对时刻 = release_index * period”推导，保证周期释放；
    - absolute_deadline = release_time + task.deadline；
    - actual_cost 由场景的 `actual_cost_for()` 负责计算，并在其中执行关键级校验。
    """

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
    """计算 C-AMC-sem 中 HI mode 新释放 LO job 的 degraded budget。"""

    budget = int(round(task.c_lo * cfg.c_amc_sem_lo_degradation_ratio))
    return max(1, min(task.c_lo, budget))


def _compute_busy_period_start_for_new_job(
    *,
    active_jobs: Sequence[Job],
    new_task: Task,
    priority_map: dict[str, int],
    now: int,
) -> int:
    """计算 tick runtime 中新 job 的 busy period 起点。"""

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


def _release_jobs_at_time_with_busy_periods(
    current_time: int,
    ordered_tasks: Sequence[Task],
    next_release_index: dict[str, int],
    scenario: ExecutionScenario,
    active_jobs: Sequence[Job],
    priority_map: dict[str, int],
    semantics: RuntimeSemantics,
    design_r_lo: dict[str, int],
    record_dropped_lo_releases: bool = False,
    suppress_lo_releases: bool = False,
    mode: SystemMode = SystemMode.LO,
    c_amc_sem_lo_degradation_ratio: float = 0.5,
    release_mode: SystemMode | None = None,
    runtime_budgets: BudgetState | None = None,
) -> tuple[list[Job], list[JobCancellationEvent]]:
    """在 `current_time` 释放所有应当释放的 job，并按优先级填充 busy period 信息。

    - 仅当 `release_index * task.period == current_time` 时才释放；
    - 返回本次新释放的 job 列表，由调用方统一并入 active / all 集合；
    - `next_release_index` 会在本函数内被原地更新。

    参数 `suppress_lo_releases` 用于表达“当前系统已经在 HI 模式，后续不再释放 LO 任务”：
    - 为 False：按常规周期释放 HI/LO 全部任务；
    - 为 True：仅允许 HI 任务继续释放，LO 任务在到达时刻会被抑制，但
      release_index 仍推进，避免恢复 LO 模式后补发 HI 期间错过的历史 LO jobs。

    这里特意采用“在释放器层过滤”的方式，而不是在外层先过滤任务列表，目的是：
    1. 保持 release_index 的推进规则集中在一处；
    2. 避免外层遗漏某些任务导致状态不一致；
    3. 让单元测试可以直接验证 suppress 语义。
    """

    if release_mode is None:
        release_mode = mode

    released: list[Job] = []
    dropped_release_cancellations: list[JobCancellationEvent] = []
    due_tasks: list[tuple[int, Task]] = []
    for task in ordered_tasks:
        idx = next_release_index[task.name]
        release_time = idx * task.period
        if release_time == current_time:
            next_release_index[task.name] = idx + 1
            due_tasks.append((idx, task))

    due_tasks.sort(key=lambda item: priority_map[item[1].name])
    for release_index, task in due_tasks:
        current_runtime_budget = (
            runtime_budgets.budget_of(task) if runtime_budgets is not None else task.c_lo
        )
        if release_mode is SystemMode.HI and _is_c_amc_semantics(semantics) and task.criticality is Criticality.LO:
            # 只有 C-AMC-sem 会在 HI mode 中继续释放 LO job。该分支用
            # XF 计算 imprecise/degraded budget，并把本次实际执行需求截断到
            # degraded budget，避免把旧 runtime 改成计划外的“原始成本继续执行”。
            degraded_budget = max(
                1,
                min(task.c_lo, int(round(task.c_lo * c_amc_sem_lo_degradation_ratio))),
            )
            original_actual_cost = scenario.actual_cost_for(task, release_index)
            original_budget = current_runtime_budget
            job = _build_job(
                task,
                release_index,
                scenario,
                actual_cost_override=min(original_actual_cost, degraded_budget),
                runtime_budget_at_release=degraded_budget,
                released_in_mode=release_mode,
                is_degraded=True,
                service_quality_if_completed=float(c_amc_sem_lo_degradation_ratio),
                original_actual_cost=original_actual_cost,
                original_runtime_budget_at_release=original_budget,
            )
            released.append(job)
            continue

        if suppress_lo_releases and task.criticality is Criticality.LO:
            if not record_dropped_lo_releases:
                continue
            job = _build_job(
                task,
                release_index,
                scenario,
                runtime_budget_at_release=current_runtime_budget,
                released_in_mode=release_mode,
            )
            job.dropped = True
            job.drop_time = current_time
            released.append(job)
            # degraded mode 中被直接 dropped 的 LO release 不是“执行超预算取消”，
            # 但为了与 event runtime 结果口径一致，这里仍需记录一条专门的
            # JobCancellationEvent，reason 固定为 lo_release_dropped_in_degraded_mode。
            dropped_release_cancellations.append(
                JobCancellationEvent(
                    cancel_time=current_time,
                    task=job.task.name,
                    release_index=job.release_index,
                    executed_at_cancel=0,
                budget_at_cancel=job.runtime_budget_at_release or current_runtime_budget,
                reason="lo_release_dropped_in_degraded_mode",
            )
            )
            continue

        job = _build_job(
            task,
            release_index,
            scenario,
            runtime_budget_at_release=current_runtime_budget,
            released_in_mode=release_mode,
        )
        job.busy_period_start = _compute_busy_period_start_for_new_job(
            active_jobs=[*active_jobs, *released],
            new_task=task,
            priority_map=priority_map,
            now=current_time,
        )
        if _is_response_based_semantics(semantics) and task.criticality is Criticality.HI:
            job.response_time_expiry = job.busy_period_start + design_r_lo[task.name]
        released.append(job)
    return released, dropped_release_cancellations


def _release_jobs_at_time(
    current_time: int,
    ordered_tasks: Sequence[Task],
    next_release_index: dict[str, int],
    scenario: ExecutionScenario,
    suppress_lo_releases: bool = False,
) -> list[Job]:
    """兼容旧测试的简单 release helper，不计算 busy period / response expiry。"""

    priority_map = {task.name: idx for idx, task in enumerate(ordered_tasks)}
    released, _ = _release_jobs_at_time_with_busy_periods(
        current_time=current_time,
        ordered_tasks=ordered_tasks,
        next_release_index=next_release_index,
        scenario=scenario,
        active_jobs=[],
        priority_map=priority_map,
        semantics=RuntimeSemantics.AMC_PLUS,
        design_r_lo={},
        record_dropped_lo_releases=False,
        suppress_lo_releases=suppress_lo_releases,
        runtime_budgets=None,
    )
    return released


def _select_highest_priority_ready_job(
    active_jobs: Sequence[Job],
    priority_map: dict[str, int],
) -> Job | None:
    """从 `active_jobs` 中选出优先级最高、尚未完成的 job。

    - 约定：`priority_map[name]` 越小代表优先级越高（0 为最高）；
    - 已完成（含被丢弃）的 job 一律跳过；
    - 若没有任何可运行 job，返回 None，对应 CPU 空闲。
    """

    best_job: Job | None = None
    best_priority: int = -1

    for job in active_jobs:
        if job.finished():
            # 已完成或已丢弃的 job 不再参与调度选择。
            continue

        priority = priority_map[job.task.name]
        if best_job is None or priority < best_priority:
            best_job = job
            best_priority = priority

    return best_job


def _check_deadline_misses(
    current_time: int,
    active_jobs: Sequence[Job],
    mode: SystemMode,
) -> list[DeadlineMiss]:
    """扫描所有活动 job，记录 absolute_deadline 正好等于 current_time 的 miss。

    - 只在 “deadline 时刻” 检查一次，避免重复记录；
    - 已完成 / 已丢弃的 job 不算 miss；
    - miss 记录包含模式信息，便于后续分析 HI 模式下的 miss（第 4 轮及以后）。
    """

    misses: list[DeadlineMiss] = []
    for job in active_jobs:
        if job.absolute_deadline != current_time:
            continue
        if job.finished():
            continue

        misses.append(
            DeadlineMiss(
                task=job.task.name,
                release_index=job.release_index,
                release_time=job.release_time,
                absolute_deadline=job.absolute_deadline,
                mode_at_miss=mode,
                executed_at_miss=job.executed_time,
            )
        )
    return misses


def _should_switch_to_hi(job: Job, mode: SystemMode, budget: int) -> bool:
    """判断 HI job 是否在 LO 模式下超过其运行时预算并触发模式切换。"""

    if mode is not SystemMode.LO:
        return False
    if job.task.criticality is not Criticality.HI:
        return False
    return job.executed_time > budget


def _should_cancel_lo_job(
    job: Job,
    mode: SystemMode,
    budget: int,
    semantics: RuntimeSemantics = RuntimeSemantics.AMC_PLUS,
) -> bool:
    """判断 LO job 是否超过其运行时预算并应被局部取消。

    兼容性说明：
    - AMC+/RA/RH 保持旧行为：只在 LO mode 下检查 LO budget cancellation；
    - C-AMC-sem 额外允许 HI mode 中的 degraded LO job 继续沿用同一套取消口径，
      这样 JNE/LDM/HDM/NID/TID 统计可与 event runtime 保持一致。
    """

    if not (
        mode is SystemMode.LO
        or (_is_c_amc_semantics(semantics) and mode is SystemMode.HI)
    ):
        return False
    if job.task.criticality is not Criticality.LO:
        return False
    return job.executed_time > budget


def _cancel_lo_job(
    active_jobs: list[Job],
    job: Job,
    current_time: int,
    budget: int,
) -> JobCancellationEvent:
    """取消一个超预算 LO job，并返回对应取消事件。"""

    job.dropped = True
    job.drop_time = current_time
    if job in active_jobs:
        active_jobs.remove(job)
    return JobCancellationEvent(
        cancel_time=current_time,
        task=job.task.name,
        release_index=job.release_index,
        executed_at_cancel=job.executed_time,
        budget_at_cancel=budget,
    )


def _drop_active_lo_jobs(active_jobs: list[Job], current_time: int) -> list[Job]:
    """在进入 HI 模式时丢弃所有仍处于活动态的 LO jobs。

    AMC 的标准语义是：一旦某个 HI job 越过 `c_lo`，系统切换到 HI 模式后，
    为保障 HI 任务预算，LO 任务会被牺牲。这里对“需要丢弃”的判断口径是：
    - 任务关键级为 LO；
    - job 尚未完成（`finished() == False`）。

    函数行为：
    - 对每个命中的 LO job 标记 `dropped=True`、`drop_time=current_time`；
    - 从 `active_jobs` 中原地移除这些 job，避免后续继续被调度；
    - 返回被丢弃的 job 列表，便于调用方测试或统计。
    """

    dropped: list[Job] = []
    for job in list(active_jobs):
        if job.task.criticality is not Criticality.LO:
            continue
        if job.finished():
            continue

        job.dropped = True
        job.drop_time = current_time
        active_jobs.remove(job)
        dropped.append(job)
    return dropped


def _find_expired_active_hi_job(
    active_jobs: Sequence[Job],
    now: int,
    priority_map: dict[str, int],
) -> Job | None:
    """找出当前已经达到 response expiry 的 active HI job。"""

    expired = [
        job
        for job in active_jobs
        if job.task.criticality is Criticality.HI
        and not job.finished()
        and job.response_time_expiry is not None
        and job.response_time_expiry <= now
    ]
    if not expired:
        return None
    return min(expired, key=lambda job: (job.response_time_expiry, priority_map[job.task.name]))


def _has_expired_active_hi_job(active_jobs: Sequence[Job], now: int) -> bool:
    """判断 RH 恢复检查时，是否仍有 expired active HI job。"""

    for job in active_jobs:
        if job.task.criticality is not Criticality.HI:
            continue
        if job.finished():
            continue
        if job.response_time_expiry is not None and job.response_time_expiry <= now:
            return True
    return False


def _peek_due_releases_at_time(
    current_time: int,
    ordered_tasks: Sequence[Task],
    next_release_index: dict[str, int],
) -> list[tuple[int, Task]]:
    """查看当前 tick 将要释放的任务，但不推进 release_index。"""

    due_tasks: list[tuple[int, Task]] = []
    for task in ordered_tasks:
        release_index = next_release_index[task.name]
        if release_index * task.period == current_time:
            due_tasks.append((release_index, task))
    return due_tasks


def _find_c_amc_sem_hi_abnormal_arrival(
    *,
    current_time: int,
    ordered_tasks: Sequence[Task],
    next_release_index: dict[str, int],
    scenario: ExecutionScenario,
    priority_map: dict[str, int],
) -> tuple[int, Task] | None:
    """找出当前 release time 会触发 C-AMC-sem 切换的最高优先级 HI arrival。"""

    abnormal_arrivals: list[tuple[int, Task]] = []
    for release_index, task in _peek_due_releases_at_time(
        current_time,
        ordered_tasks,
        next_release_index,
    ):
        if task.criticality is not Criticality.HI:
            continue
        if scenario.actual_cost_for(task, release_index) > task.c_lo:
            abnormal_arrivals.append((release_index, task))
    if not abnormal_arrivals:
        return None
    return min(abnormal_arrivals, key=lambda item: priority_map[item[1].name])


# ---------------------------------------------------------------------------
# 主仿真循环
# ---------------------------------------------------------------------------


def simulate_ordered_taskset(
    ordered_tasks: Sequence[Task],
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> SimulationResult:
    """按 tick 推进的固定优先级抢占式仿真器（第 4 轮：含 HI 切换语义）。

    参数：
    - `ordered_tasks`: 已按优先级从高到低排好序的任务列表；
    - `scenario`:      执行场景，负责给出每个 job 的实际执行时间；
    - `config`:        运行时配置；None 表示使用默认 `RuntimeConfig()`。

    返回：
    - `SimulationResult`：包含 job 历史、trace、deadline miss 等信息；
      当出现 HI 任务越过 `c_lo` 时会记录 `mode_switches`，并把 `final_mode`
      更新为 `SystemMode.HI`。

    边界校验：
    - 空 `ordered_tasks` 视为非法输入；
    - 任务名重复会导致优先级映射不稳定，同样视为非法。
    """

    # -------- 输入校验 --------
    if not ordered_tasks:
        raise ValueError("ordered_tasks 不能为空")

    task_names = [task.name for task in ordered_tasks]
    if len(set(task_names)) != len(task_names):
        raise ValueError("ordered_tasks 存在重复任务名，无法建立稳定的优先级映射")

    # -------- 配置装配 --------
    cfg = config if config is not None else RuntimeConfig()
    runtime_budgets = (
        budget_state.copy()
        if budget_state is not None
        else BudgetState.from_tasks(ordered_tasks)
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
    response_based = _is_response_based_semantics(cfg.semantics)
    design_r_lo = build_design_r_lo_map(ordered_tasks) if response_based else {}

    # 任务名 -> 优先级索引（0 最高）。由输入顺序直接推导，不再重新排序。
    priority_map: dict[str, int] = {task.name: idx for idx, task in enumerate(ordered_tasks)}
    # 每个任务下一次应释放的 release_index；按周期推进。
    next_release_index: dict[str, int] = {task.name: 0 for task in ordered_tasks}

    # 活动集合：持有尚未完成/尚未丢弃的 job。
    # 全部集合：按释放顺序追加，返回给调用方做历史分析。
    active_jobs: list[Job] = []
    all_jobs: list[Job] = []

    # trace / miss 容器；trace 可能因 capture_trace=False 而保持为空。
    trace: list[ScheduleTick] = []
    deadline_misses: list[DeadlineMiss] = []

    # 系统初始模式总是 LO；若发生切换会更新为 HI。
    mode = SystemMode.LO
    mode_switches: list[ModeSwitchEvent] = []
    mode_recoveries: list[ModeRecoveryEvent] = []
    budget_update_events: list[BudgetUpdateEvent] = []
    job_cancellations: list[JobCancellationEvent] = []

    # 仿真实际终止时刻：默认等于配置 end_time，若 stop_at_first_miss 命中则提前。
    actual_end_time = end_time
    stopped_early = False
    updates_by_time: dict[int, list[BudgetUpdate]] = {}
    if budget_updates is not None:
        for update in budget_updates:
            if update.time < 0:
                raise ValueError("budget update time 必须 >= 0")
            updates_by_time.setdefault(update.time, []).append(update)

    for t in range(end_time):
        # ---- 0. 在 tick 起点应用预算更新（先于 release 生效） ----
        for update in updates_by_time.get(t, []):
            runtime_budgets.apply_updates(update.updates)
            budget_update_events.append(BudgetUpdateEvent(time=t, updates=dict(update.updates)))

        # ---- 1. RA/RH 先检查“已存在 active HI job 是否在本时刻达到 response expiry” ----
        if response_based and mode is SystemMode.LO:
            expired_job = _find_expired_active_hi_job(active_jobs, t, priority_map)
            if expired_job is not None:
                mode = SystemMode.HI
                mode_switches.append(
                    ModeSwitchEvent(
                        switch_time=t,
                        triggering_task=expired_job.task.name,
                        triggering_release_index=expired_job.release_index,
                        executed_at_switch=expired_job.executed_time,
                        budget_at_switch=expired_job.runtime_budget_at_release,
                        reason="hi_response_time_expiry",
                    )
                )
                if cfg.drop_lo_jobs_on_hi_switch:
                    _drop_active_lo_jobs(active_jobs, current_time=t)

        # ---- 2. C-AMC-sem 在 release time 先做 semi-clairvoyant 模式切换 ----
        mode_before_release_batch = mode
        switched_by_c_amc_sem_batch = False

        if _is_c_amc_semantics(cfg.semantics) and mode is SystemMode.LO:
            abnormal_arrival = _find_c_amc_sem_hi_abnormal_arrival(
                current_time=t,
                ordered_tasks=ordered_tasks,
                next_release_index=next_release_index,
                scenario=scenario,
                priority_map=priority_map,
            )
            if abnormal_arrival is not None:
                trigger_release_index, trigger_task = abnormal_arrival
                mode = SystemMode.HI
                switched_by_c_amc_sem_batch = True
                mode_switches.append(
                    ModeSwitchEvent(
                        switch_time=t,
                        triggering_task=trigger_task.name,
                        triggering_release_index=trigger_release_index,
                        executed_at_switch=0,
                        budget_at_switch=trigger_task.c_lo,
                        reason="semi_clairvoyant_hi_abnormal_arrival",
                    )
                )

        # ---- 3. 在 tick 起点释放到期的 job ----
        release_mode = mode
        if (
            switched_by_c_amc_sem_batch
            and cfg.c_amc_sem_primary_on_switch_time
            and mode_before_release_batch is SystemMode.LO
        ):
            release_mode = SystemMode.LO

        newly_released, dropped_release_cancellations = _release_jobs_at_time_with_busy_periods(
            current_time=t,
            ordered_tasks=ordered_tasks,
            next_release_index=next_release_index,
            scenario=scenario,
            active_jobs=active_jobs,
            priority_map=priority_map,
            semantics=cfg.semantics,
            design_r_lo=design_r_lo,
            record_dropped_lo_releases=cfg.record_dropped_lo_releases,
            # 只有 AMC/AMC+/RA/RH 在 HI mode 中抑制未来 LO release；
            # C-AMC-sem 的 LO release 会继续进入调度，但用 degraded budget/cost。
            suppress_lo_releases=(mode is SystemMode.HI and not _is_c_amc_semantics(cfg.semantics)),
            mode=mode,
            c_amc_sem_lo_degradation_ratio=cfg.c_amc_sem_lo_degradation_ratio,
            release_mode=release_mode,
            runtime_budgets=runtime_budgets,
        )
        job_cancellations.extend(dropped_release_cancellations)
        # 新释放的 normal jobs 进入活动队列；若是 degraded mode 中被直接 dropped 的
        # LO release，则只进入历史集合，不进入 active_jobs。
        active_jobs.extend([job for job in newly_released if not job.dropped])
        all_jobs.extend(newly_released)

        # ---- 4. RA/RH 在完成同刻 arrival 后再次检查 expiry，覆盖 inherited busy period 已过期的边界 ----
        if response_based and mode is SystemMode.LO:
            expired_job = _find_expired_active_hi_job(active_jobs, t, priority_map)
            if expired_job is not None:
                mode = SystemMode.HI
                mode_switches.append(
                    ModeSwitchEvent(
                        switch_time=t,
                        triggering_task=expired_job.task.name,
                        triggering_release_index=expired_job.release_index,
                        executed_at_switch=expired_job.executed_time,
                        budget_at_switch=expired_job.runtime_budget_at_release,
                        reason="hi_response_time_expiry",
                    )
                )
                if cfg.drop_lo_jobs_on_hi_switch:
                    _drop_active_lo_jobs(active_jobs, current_time=t)

        # ---- 5. 检查当前 tick 是否有 job 正好踩到 deadline 但未完成 ----
        misses_at_t = _check_deadline_misses(t, active_jobs, mode)
        deadline_misses.extend(misses_at_t)
        if misses_at_t and cfg.stop_at_first_miss:
            # 命中首个 miss 后立即停止：不再执行本 tick，end_time 设为 t。
            actual_end_time = t
            stopped_early = True
            break

        # ---- 6. 选出最高优先级可运行 job ----
        running = _select_highest_priority_ready_job(active_jobs, priority_map)

        # ---- 7. 记录本 tick 的执行快照（可选） ----
        if cfg.capture_trace:
            trace.append(
                ScheduleTick(
                    time=t,
                    executing_task=running.task.name if running is not None else None,
                    executing_release_index=(
                        running.release_index if running is not None else None
                    ),
                    mode=mode,
                )
            )

        # ---- 8. 推进被选中 job 一个 tick，如完成则登记 completion_time ----
        completed_hi_job = False
        if running is not None:
            running.executed_time += 1

            if running.executed_time >= running.actual_cost:
                # 完成时刻记作“tick 结束的那一刻”，即 t + 1。
                running.completion_time = t + 1
                completed_hi_job = running.task.criticality is Criticality.HI
                # 从活动集合移除，减少后续选择 / miss 扫描的范围。
                active_jobs.remove(running)

        # ---- 9. 检查是否触发 LO -> HI 切换，并执行切换后动作 ----
        if running is not None:
            current_budget = runtime_budgets.budget_of(running.task)
            if running.runtime_budget_at_release is not None:
                current_budget = running.runtime_budget_at_release

            if _should_cancel_lo_job(running, mode, current_budget, cfg.semantics):
                if cfg.semantics in {
                    RuntimeSemantics.AMC_PLUS,
                    RuntimeSemantics.AMC_RA,
                    RuntimeSemantics.AMC_RH,
                    RuntimeSemantics.C_AMC_SEM,
                }:
                    cancel_time = t + 1
                    event = _cancel_lo_job(
                        active_jobs=active_jobs,
                        job=running,
                        current_time=cancel_time,
                        budget=current_budget,
                    )
                    job_cancellations.append(event)
                    running = None
                else:
                    switch_time = t + 1
                    mode = SystemMode.HI
                    mode_switches.append(
                        ModeSwitchEvent(
                            switch_time=switch_time,
                            triggering_task=running.task.name,
                            triggering_release_index=running.release_index,
                            executed_at_switch=running.executed_time,
                            budget_at_switch=current_budget,
                            reason="lo_budget_overrun_standard_amc",
                        )
                    )
                    if cfg.drop_lo_jobs_on_hi_switch:
                        _drop_active_lo_jobs(active_jobs, current_time=switch_time)

            elif (
                not response_based
                and _should_switch_to_hi(running, mode, current_budget)
            ):
                # 切换时刻定义为“触发 tick 结束边界”，即 t + 1。
                switch_time = t + 1
                mode = SystemMode.HI
                mode_switches.append(
                    ModeSwitchEvent(
                        switch_time=switch_time,
                        triggering_task=running.task.name,
                        triggering_release_index=running.release_index,
                        executed_at_switch=running.executed_time,
                        budget_at_switch=current_budget,
                    )
                )

                # 可配置是否在切换瞬间丢弃所有活动 LO jobs。
                if cfg.drop_lo_jobs_on_hi_switch:
                    _drop_active_lo_jobs(active_jobs, current_time=switch_time)

        # ---- 10. 恢复规则：AMC/A+/RA/C-AMC-sem 用 idle，RH 仅在 HI completion 后检查 response-aware recovery ----
        if cfg.semantics is RuntimeSemantics.AMC_RH:
            if completed_hi_job and mode is SystemMode.HI and not _has_expired_active_hi_job(active_jobs, t + 1):
                mode = SystemMode.LO
                mode_recoveries.append(
                    ModeRecoveryEvent(recovery_time=t + 1, reason="rh_no_expired_hi_job")
                )
        elif _uses_idle_recovery(cfg.semantics) and mode is SystemMode.HI and not active_jobs:
            recovery_time = t + 1
            mode = SystemMode.LO
            mode_recoveries.append(ModeRecoveryEvent(recovery_time=recovery_time, reason="idle"))

    # ---- 8. 主循环外的“终点 miss 扫描” ----
    # 主循环 range(end_time) 只迭代到 t = end_time - 1，这里补一次检查，
    # 确保 absolute_deadline 恰好等于 end_time 的 job 也会被标记为 miss。
    if not stopped_early:
        final_misses = _check_deadline_misses(end_time, active_jobs, mode)
        deadline_misses.extend(final_misses)

    return SimulationResult(
        jobs=all_jobs,
        trace=trace,
        mode_switches=mode_switches,
        mode_recoveries=mode_recoveries,
        budget_update_events=budget_update_events,
        job_cancellations=job_cancellations,
        deadline_misses=deadline_misses,
        end_time=actual_end_time,
        final_mode=mode,
    )


def simulate_taskset_with_policy(
    tasks: Sequence[Task],
    method: str,
    priority_policy: str,
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> SimulationResult:
    """AMC runtime bridge：按 `method + priority_policy` 自动解析顺序并执行仿真。

    这是“输入桥接层”而非新的调度器：
    - 底层执行仍由 `simulate_ordered_taskset()` 完成；
    - 本函数用于复用静态分析侧的优先级解析逻辑，让调用方可直接传入
      `method + priority_policy`，无需先手工排序。

    参数：
    - `tasks`: 原始任务集（未排序）；
    - `method`: 当前仅支持 AMC family method（`amc_rtb`、`amc_max`）；
    - `priority_policy`: 优先级策略（`dm` / `crmpo` / `opa`）；
    - `scenario`: 运行时实际执行时间注入场景；
    - `config`: 仿真配置，不传则使用默认配置。

    实现要点：
    - 在入口先校验 `method`，对非 AMC 方法立即 fail-fast；
    - **必须** 复用 `experiments.resolve_ordering()`，确保 runtime 与静态分析
      在 DM / CrMPO / OPA 下看到完全一致的优先级顺序；
    - 本函数不在 runtime 里重造排序逻辑，避免两套策略实现漂移；
    - OPA 失败时会沿用 `resolve_ordering()` 的 RuntimeError 语义向上抛出。

    说明：
    - 若你要研究非 AMC 方法的运行时行为，请改用更底层的
      `simulate_ordered_taskset()`，并自行说明排序与运行时语义的对应关系。
    """

    cfg = config if config is not None else RuntimeConfig()

    # 入口先做 method 语义限制，避免误导性调用继续执行。
    _validate_runtime_bridge_method(method)
    if _is_response_based_semantics(cfg.semantics) and method != "amc_rtb":
        raise ValueError("AMC_RA/AMC_RH runtime semantics must be used with method='amc_rtb'")

    # 懒加载导入：避免在导入 runtime 模块时就触发 experiments 里的重型依赖初始化。
    from .experiments import resolve_ordering

    # 通过第 1 轮暴露的公共接口统一解析顺序，保证策略口径一致。
    ordered_tasks = resolve_ordering(tasks, priority_policy=priority_policy, method=method)
    return simulate_ordered_taskset(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=cfg,
        budget_state=budget_state,
        budget_updates=budget_updates,
    )


def compare_static_and_runtime(
    tasks: Sequence[Task],
    method: str,
    priority_policy: str,
    scenario: ExecutionScenario,
    config: RuntimeConfig | None = None,
    budget_state: BudgetState | None = None,
    budget_updates: Sequence[BudgetUpdate] | None = None,
) -> RuntimeComparisonResult:
    """AMC runtime bridge：执行“静态分析 vs 运行时仿真”的统一对照。

    对照流程分三步：
    1. 先调用 `evaluate_taskset()` 得到静态分析结论；
    2. 再调用 `resolve_ordering()` 取得最终优先级顺序（与静态入口一致）；
    3. 在该顺序上调用 `simulate_ordered_taskset()` 得到运行时结果。

    为什么先做静态分析：
    - 方便即使运行时路径后续抛错，也能保留静态侧结果用于诊断；
    - 与“先看理论判定，再看运行时行为”的使用习惯一致。

    参数补充说明：
    - `method` 当前仅支持 AMC family method（`amc_rtb`、`amc_max`）；
    - 非 AMC 方法会在入口直接报错，防止形成“静态方法可传入就等于已有对应
      runtime 语义”的错误理解。

    返回：
    - `RuntimeComparisonResult`，同时携带静态结果、运行时结果、顺序信息与
      method/policy 元数据，便于脚本和实验层直接消费。
    """

    cfg = config if config is not None else RuntimeConfig()

    # 在任何静态分析或排序解析前先 fail-fast，避免语义误导。
    _validate_runtime_bridge_method(method)
    if _is_response_based_semantics(cfg.semantics) and method != "amc_rtb":
        raise ValueError("AMC_RA/AMC_RH runtime semantics must be used with method='amc_rtb'")

    # 懒加载导入：保持 runtime 核心仿真模块在导入阶段尽量轻量。
    from .experiments import evaluate_taskset, resolve_ordering

    # 第一步：静态分析结果（可能可调度，也可能不可调度）。
    static_result = evaluate_taskset(
        tasks=tasks,
        method=method,
        priority_policy=priority_policy,
    )

    # 第二步：统一顺序解析；OPA 失败会抛 RuntimeError，这里按原语义透传。
    ordered_tasks = resolve_ordering(tasks, priority_policy=priority_policy, method=method)

    # 第三步：在同一顺序上执行 runtime 仿真，确保“可比性”。
    runtime_result = simulate_ordered_taskset(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=cfg,
        budget_state=budget_state,
        budget_updates=budget_updates,
    )

    return RuntimeComparisonResult(
        static_result=static_result,
        runtime_result=runtime_result,
        ordered_task_names=[task.name for task in ordered_tasks],
        method=method,
        priority_policy=priority_policy,
    )


__all__ = [
    "compute_hyperperiod",
    "compute_default_end_time",
    "simulate_ordered_taskset",
    "simulate_taskset_with_policy",
    "compare_static_and_runtime",
    # 以下为内部辅助函数，但暴露出来便于测试与第 4 轮扩展。
    "_build_job",
    "_release_jobs_at_time",
    "_release_jobs_at_time_with_busy_periods",
    "_select_highest_priority_ready_job",
    "_check_deadline_misses",
    "_should_switch_to_hi",
    "_should_cancel_lo_job",
    "_cancel_lo_job",
    "_drop_active_lo_jobs",
]
