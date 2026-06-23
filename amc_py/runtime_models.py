"""运行时仿真的数据模型（阶段运行时模拟 · 第 2 轮）。

本模块仅负责“数据容器”这一层——把 AMC 运行时仿真器会用到的概念
（系统模式、配置、Job、事件、trace、仿真结果）集中定义成类型安全的
dataclass，方便后续 `runtime.py` 主循环与 `runtime_scenarios.py` 场景
层共用，也方便测试层直接拼装/断言。

设计要点：
- 只引入“描述状态”的结构，不包含任何调度/仿真循环逻辑；
- 可变类（如 `Job`、`SimulationResult`）反映仿真过程中会被渐进更新的
  对象；其余事件与快照类一律 frozen，避免被意外篡改；
- 与 `amc_py.models` 中的 `Task`/`Criticality`/`SchedulabilityResult`
  解耦：本模块只依赖它们的公开字段，不侵入修改原有类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import Criticality, SchedulabilityResult, Task


class SystemMode(str, Enum):
    """运行时系统模式。

    AMC 语义中系统有两种运行模式：
    - `LO`: 初始模式；所有 HI/LO 任务都以 C(LO) 为预算执行；
    - `HI`: 当某个 HI 任务越过 C(LO) 边界时进入；此后仅保留 HI 任务，
      LO 任务根据配置可能被丢弃并停止后续释放。

    继承 `str` 是为了序列化/打印时显示 "LO"/"HI"，与 `Criticality`
    的写法保持一致。
    """

    LO = "LO"
    HI = "HI"


class RuntimeSemantics(str, Enum):
    """运行时语义开关。"""

    AMC = "AMC"
    AMC_PLUS = "AMC_PLUS"
    AMC_RA = "AMC_RA"
    AMC_RH = "AMC_RH"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """运行时仿真器的配置参数集合。

    字段说明：
    - `end_time`: 指定仿真终止时刻（闭区间右端）。为 `None` 时交给
      仿真器基于 hyperperiod 与 `jobs_per_task` 自动决定。
    - `jobs_per_task`: 自动估算 end_time 时的“每任务最少释放次数”基线。
    - `hyperperiod_limit`: 自动估算 end_time 时允许的最大 hyperperiod
      上限，避免输入任务周期过大导致仿真范围爆炸。
    - `capture_trace`: 是否记录每个 tick 的执行快照（便于调试/可视化，
      但会增加内存开销）。
    - `capture_debug_events`: 是否记录事件级 debug 日志。它通常比逐 tick trace
      更稀疏，但在长时域 HOUT 中仍可能累计出较大内存占用，因此默认关闭。
    - `stop_at_first_miss`: 是否在首次出现 deadline miss 时立即停止仿真。
    - `drop_lo_jobs_on_hi_switch`: 进入 HI 模式时是否丢弃所有未完成的 LO job。
    - `semantics`: 运行时语义开关：
      `AMC_PLUS` 表示 LO 超预算仅局部取消，HI 超预算触发模式切换；
      `AMC` 表示任意任务超 LO 预算都触发模式切换；
      `AMC_RA` / `AMC_RH` 表示 HI 任务通过 `s_i + R_i(LO)` 触发退化模式。
    - `record_dropped_lo_releases`: 是否把 degraded mode 中被直接 dropped 的
      LO release 也记入 `SimulationResult.jobs` 与 `job_cancellations`，用于
      论文里的 JNE（jobs not executed）统计。

    设计上使用 frozen dataclass，是为了强调“配置对象一经创建不再可变”，
    避免仿真器内部无意篡改用户传入的配置。
    """

    end_time: int | None = None
    jobs_per_task: int = 5
    hyperperiod_limit: int = 100_000
    capture_trace: bool = False
    capture_debug_events: bool = False
    stop_at_first_miss: bool = False
    drop_lo_jobs_on_hi_switch: bool = True
    semantics: RuntimeSemantics = RuntimeSemantics.AMC_PLUS
    record_dropped_lo_releases: bool = False

    def __post_init__(self) -> None:
        # 基本的范围校验：避免在仿真过程中才报错，尽量把问题前置。
        if self.end_time is not None and self.end_time <= 0:
            raise ValueError("end_time 必须为正整数或 None")
        if self.jobs_per_task <= 0:
            raise ValueError("jobs_per_task 必须为正整数")
        if self.hyperperiod_limit <= 0:
            raise ValueError("hyperperiod_limit 必须为正整数")


@dataclass(slots=True)
class Job:
    """运行时的一次任务实例（job）。

    Job 表示某个 `Task` 在具体 release_index 下的一次释放，携带它在
    仿真过程中累积起来的执行时间、完成时刻与丢弃状态等可变信息。

    字段含义：
    - `task`: 所属的周期任务定义；
    - `release_index`: 第几次释放（从 0 起），对应的绝对释放时刻为
      `release_index * task.period`；
    - `release_time`: 绝对释放时刻，显式保存避免到处重新计算；
    - `absolute_deadline`: 绝对截止时刻 = `release_time + task.deadline`；
    - `actual_cost`: 由 `ExecutionScenario` 注入的本次实际执行时间；
    - `runtime_budget_at_release`: 该 job 释放当刻看到的运行时预算。后续全局预算
      即使变化，也不允许追溯修改已经释放 job 的预算判定语义；
    - `executed_time`: 已经累计执行的 tick 数，随着仿真推进递增；
    - `completion_time`: 完成时刻（job 首次执行满 actual_cost 的那一刻
      的下一 tick 起点），未完成时为 None；
    - `dropped`: 是否因 HI 模式切换被丢弃；
    - `drop_time`: 被丢弃的绝对时刻（None 表示没被丢弃）。
    - `busy_period_start`: 该 job 所属 priority-level busy period 的起点 `s_i`；
    - `response_time_expiry`: `s_i + R_i(LO)`，仅 RA/RH 语义下的 HI job 有意义。

    该类是可变的，因为 job 状态必须随仿真 tick 演化。
    """

    task: Task
    release_index: int
    release_time: int
    absolute_deadline: int
    actual_cost: int
    runtime_budget_at_release: int | None = None
    executed_time: int = 0
    completion_time: int | None = None
    dropped: bool = False
    drop_time: int | None = None
    busy_period_start: int | None = None
    response_time_expiry: int | None = None

    def remaining(self) -> int:
        """返回本 job 还需执行的时间，下限为 0。

        一旦 `dropped=True`，语义上 job 已失效，剩余执行量视为 0；
        否则返回 `actual_cost - executed_time`（取 max(., 0) 防溢出）。
        """

        if self.dropped:
            return 0
        return max(0, self.actual_cost - self.executed_time)

    def finished(self) -> bool:
        """判断 job 是否已经不再需要继续执行。

        包含两种收敛情况：
        - 已执行足够长时间（`remaining() == 0`），或
        - 被 HI 模式切换丢弃。
        """

        return self.dropped or self.remaining() == 0


@dataclass(frozen=True, slots=True)
class ModeSwitchEvent:
    """记录一次 `LO -> HI` 模式切换事件。

    字段含义：
    - `switch_time`: 切换生效的绝对时刻（仿真约定为“触发 tick 结束后的
      下一 tick 起点”，由 `runtime.py` 主循环负责赋值）；
    - `triggering_task`: 触发切换的 HI 任务名；
    - `triggering_release_index`: 触发切换的具体 release 索引；
    - `executed_at_switch`: 该 job 在切换时已经执行过的时间量，通常
      等于 `task.c_lo + 1`（即首次越过 C(LO) 的那一刻）。

    frozen 是因为该事件一旦发生便不应再修改。
    """

    switch_time: int
    triggering_task: str
    triggering_release_index: int
    executed_at_switch: int
    budget_at_switch: int | None = None
    reason: str = "hi_budget_overrun"


@dataclass(frozen=True, slots=True)
class JobCancellationEvent:
    """记录一次 LO job 因运行时预算超限而被局部取消的事件。"""

    cancel_time: int
    task: str
    release_index: int
    executed_at_cancel: int
    budget_at_cancel: int
    reason: str = "lo_budget_overrun"


@dataclass(frozen=True, slots=True)
class ModeRecoveryEvent:
    """记录一次 HI 模式在空闲后恢复到 LO 模式的事件。"""

    recovery_time: int
    reason: str = "idle"


@dataclass(frozen=True, slots=True)
class BudgetUpdateEvent:
    """记录一次运行时预算更新事件。"""

    time: int
    updates: dict[str, int]


@dataclass(frozen=True, slots=True)
class DeadlineMiss:
    """记录一次 deadline miss 事件，便于后续报告与可视化。

    字段含义：
    - `task`: 错过 deadline 的任务名；
    - `release_index`: 对应的 release 索引；
    - `release_time`: 释放时刻；
    - `absolute_deadline`: 绝对截止时刻；
    - `mode_at_miss`: miss 被检测到时的系统模式；
    - `executed_at_miss`: miss 被检测到时该 job 已累计执行的时间。
    """

    task: str
    release_index: int
    release_time: int
    absolute_deadline: int
    mode_at_miss: SystemMode
    executed_at_miss: int


@dataclass(frozen=True, slots=True)
class ScheduleTick:
    """单个调度 tick 的快照。

    仿真器按离散整数时间步推进。每一个 tick 覆盖半开区间 `[time, time+1)`，
    其间最多有一个 job 在 CPU 上执行。该类为此时刻的现场快照。

    - `time`: tick 起点的绝对时刻；
    - `executing_task`: 本 tick 在 CPU 上执行的任务名；None 表示空闲；
    - `executing_release_index`: 对应的 release 索引；idle 时为 None；
    - `mode`: tick 起点时刻的系统模式。
    """

    time: int
    executing_task: str | None
    executing_release_index: int | None
    mode: SystemMode


@dataclass(slots=True)
class SimulationResult:
    """整次仿真的最终结果。

    字段含义：
    - `jobs`: 仿真过程中曾被释放出来的所有 job（含已完成/已丢弃/未完成）；
    - `trace`: 逐 tick 的调度快照，可能为空（当 `capture_trace=False`）；
    - `debug_events`: 面向排障的事件级调试日志。该字段不会参与调度语义，
      仅用于导出 trace、定位 accepted action 与 deadline miss 之间的因果链；
    - `mode_switches`: 本次仿真发生过的 `LO -> HI` 切换事件列表；
    - `mode_recoveries`: 本次仿真发生过的 `HI -> LO` 恢复事件列表；
    - `budget_update_events`: 仿真期间应用的预算更新事件列表；
    - `job_cancellations`: 记录 LO job 因预算超限被局部取消的事件列表；
    - `deadline_misses`: 本次仿真观察到的所有 deadline miss 记录；
    - `end_time`: 仿真实际终止时刻（不一定等于配置里的 end_time，
      例如 `stop_at_first_miss=True` 时可能提前结束）；
    - `final_mode`: 仿真结束时的系统模式。

    作为结果容器使用可变 dataclass，便于 `runtime.py` 主循环边跑边填，
    跑完交给调用方只读使用。
    """

    jobs: list[Job] = field(default_factory=list)
    trace: list[ScheduleTick] = field(default_factory=list)
    debug_events: list[dict[str, object]] = field(default_factory=list)
    mode_switches: list[ModeSwitchEvent] = field(default_factory=list)
    mode_recoveries: list[ModeRecoveryEvent] = field(default_factory=list)
    budget_update_events: list[BudgetUpdateEvent] = field(default_factory=list)
    job_cancellations: list[JobCancellationEvent] = field(default_factory=list)
    deadline_misses: list[DeadlineMiss] = field(default_factory=list)
    end_time: int = 0
    final_mode: SystemMode = SystemMode.LO

    # ---- 便捷视图（只做简单过滤，避免调用方重复写列表推导） ----
    @property
    def mode_switch(self) -> ModeSwitchEvent | None:
        """兼容旧接口：返回第一条模式切换事件，不存在时返回 None。"""

        return self.mode_switches[0] if self.mode_switches else None

    def mode_switched(self) -> bool:
        """本次仿真是否触发过 LO -> HI 切换。"""

        return bool(self.mode_switches)

    def mode_change_count(self) -> int:
        """返回模式切换事件总次数。"""

        return len(self.mode_switches)

    def lo_job_cancellation_count(self) -> int:
        """返回 LO job 局部取消事件总次数。"""

        return len(self.job_cancellations)

    def mode_recovery_count(self) -> int:
        """返回 HI->LO 模式恢复事件总次数。"""

        return len(self.mode_recoveries)

    def deadline_missed(self) -> bool:
        """本次仿真是否发生过任何 deadline miss。"""

        return bool(self.deadline_misses)

    def dropped_jobs(self) -> list[Job]:
        """被 HI 切换丢弃的 job 列表。"""

        return [job for job in self.jobs if job.dropped]

    def completed_jobs(self) -> list[Job]:
        """成功完成的 job 列表（不含被丢弃的）。"""

        return [
            job
            for job in self.jobs
            if job.completion_time is not None and not job.dropped
        ]

    def jobs_of(self, task_name: str) -> list[Job]:
        """返回属于指定任务名的所有 job（按 release_index 升序）。"""

        return sorted(
            (job for job in self.jobs if job.task.name == task_name),
            key=lambda job: job.release_index,
        )


@dataclass(frozen=True, slots=True)
class RuntimeComparisonResult:
    """静态分析与运行时仿真的联合对比结果。

    - `static_result`: 静态分析（RTA/UB-H&L 等）给出的可调度性结论；
    - `runtime_result`: 在同一优先级顺序下、同一 scenario 下的运行时仿真结果；
    - `ordered_task_names`: 解析后的最终优先级顺序（由高到低），便于
      结果使用者核对两侧“看到的”是否是同一组任务顺序；
    - `method`/`priority_policy`: 本次对比对应的方法与优先级策略名，
      便于批量跑多组对比时识别来源。

    这是一个只读的汇总对象，因此用 frozen。
    """

    static_result: SchedulabilityResult
    runtime_result: SimulationResult
    ordered_task_names: list[str]
    method: str
    priority_policy: str

    # ---- 常用布尔视图 ----

    def static_schedulable(self) -> bool:
        """静态分析是否判定可调度。"""

        return self.static_result.schedulable

    def mode_switched(self) -> bool:
        """运行时仿真是否触发过 HI 切换。"""

        return self.runtime_result.mode_switched()

    def deadline_missed(self) -> bool:
        """运行时仿真是否观察到任何 deadline miss。"""

        return self.runtime_result.deadline_missed()


__all__ = [
    "SystemMode",
    "RuntimeSemantics",
    "RuntimeConfig",
    "Job",
    "ModeSwitchEvent",
    "ModeRecoveryEvent",
    "BudgetUpdateEvent",
    "JobCancellationEvent",
    "DeadlineMiss",
    "ScheduleTick",
    "SimulationResult",
    "RuntimeComparisonResult",
    # 以下两个是外部常用到的枚举/类型，重新导出一次便于 runtime 模块共享引用。
    "Criticality",
    "Task",
]
