"""运行时数据模型 + scenario 层单元测试（阶段运行时模拟 · 第 2 轮）。

本测试文件覆盖两件事：
1. `amc_py.runtime_models` 中数据容器的行为是否符合预期（Job / SimulationResult 等便捷视图）；
2. `amc_py.runtime_scenarios` 中各类 scenario 工厂及其 `actual_cost_for`
   入口是否能正确：
     - 给出符合关键级约束的 actual_cost；
     - 在非法输入（LO 任务超 c_lo、HI 任务超 c_hi、非法 default_hi 等）时抛错。

注意：本轮尚未实现 runtime 主循环，所以测试全部围绕“静态构造对象 +
直接调用数据模型/ scenario 接口”进行，不依赖 `runtime.py`。
"""

from __future__ import annotations

import pytest

from amc_py.models import Criticality, SchedulabilityResult, Task
from amc_py.runtime_models import (
    DeadlineMiss,
    Job,
    ModeSwitchEvent,
    RuntimeSemantics,
    RuntimeComparisonResult,
    RuntimeConfig,
    ScheduleTick,
    SimulationResult,
    SystemMode,
)
from amc_py.runtime_scenarios import (
    ExecutionScenario,
    _validate_actual_cost,
    make_all_hi_jobs_hi_budget_scenario,
    make_nominal_scenario,
    make_single_hi_overrun_scenario,
    make_single_lo_overrun_scenario,
    make_table_scenario,
)


# ---------------------------------------------------------------------------
# 参考任务集工具
# ---------------------------------------------------------------------------


def _reference_tasks() -> list[Task]:
    """构造一个覆盖 HI / LO / `c_hi == c_lo` 的小型参考任务集。

    - `tau_hi`: 标准 HI 任务，c_hi > c_lo，可被 single HI overrun 触发；
    - `tau_hi_tight`: 特殊 HI 任务，c_hi == c_lo，用于测试 `c_lo_plus_one` 不可用；
    - `tau_lo`: 普通 LO 任务，用于验证 LO 场景约束。
    """

    return [
        Task("tau_hi", period=20, deadline=20, c_lo=2, c_hi=5, criticality=Criticality.HI),
        Task("tau_hi_tight", period=30, deadline=30, c_lo=3, c_hi=3, criticality=Criticality.HI),
        Task("tau_lo", period=10, deadline=10, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]


def _by_name(tasks: list[Task]) -> dict[str, Task]:
    """按任务名建立索引，测试里多次通过名字拿 Task 对象。"""

    return {task.name: task for task in tasks}


# ---------------------------------------------------------------------------
# _validate_actual_cost 直接单测
# ---------------------------------------------------------------------------


def test_validate_actual_cost_accepts_valid_hi_and_lo_ranges() -> None:
    """合法范围内的 actual_cost 不应该抛任何异常。"""

    tasks = _by_name(_reference_tasks())

    # HI 任务：允许任何介于 [1, c_hi] 的值（这里直接取 c_hi 作为上界）。
    _validate_actual_cost(tasks["tau_hi"], actual_cost=1)
    _validate_actual_cost(tasks["tau_hi"], actual_cost=tasks["tau_hi"].c_lo)
    _validate_actual_cost(tasks["tau_hi"], actual_cost=tasks["tau_hi"].c_hi)

    # LO 任务：允许 [1, c_lo]；c_hi 对 LO 任务没有意义，这里校验 c_lo 边界即可。
    _validate_actual_cost(tasks["tau_lo"], actual_cost=1)
    _validate_actual_cost(tasks["tau_lo"], actual_cost=tasks["tau_lo"].c_lo)


def test_validate_actual_cost_rejects_hi_task_above_c_hi() -> None:
    """HI 任务 actual_cost 严格超过 c_hi 时必须报错。"""

    tasks = _by_name(_reference_tasks())

    with pytest.raises(ValueError, match="HI 任务 tau_hi .* 超过 c_hi=5"):
        _validate_actual_cost(tasks["tau_hi"], actual_cost=6)


def test_validate_actual_cost_allows_lo_task_above_c_lo() -> None:
    """AMC+ 语义下 LO 任务允许超过 c_lo，取消逻辑由 runtime 层处理。"""

    tasks = _by_name(_reference_tasks())
    _validate_actual_cost(tasks["tau_lo"], actual_cost=3)


def test_validate_actual_cost_rejects_nonpositive_values() -> None:
    """零/负数不合法：仿真器按整数 tick 推进，至少需要执行 1 个 tick。"""

    tasks = _by_name(_reference_tasks())

    with pytest.raises(ValueError, match="必须 >= 1"):
        _validate_actual_cost(tasks["tau_lo"], actual_cost=0)

    with pytest.raises(ValueError, match="必须 >= 1"):
        _validate_actual_cost(tasks["tau_hi"], actual_cost=-1)


# ---------------------------------------------------------------------------
# nominal scenario
# ---------------------------------------------------------------------------


def test_make_nominal_scenario_returns_c_lo_for_every_task() -> None:
    """标称场景下，所有任务（无论 HI/LO）都应该跑 c_lo。"""

    scenario = make_nominal_scenario()
    assert scenario.name == "nominal"

    for task in _reference_tasks():
        # 连续几次 release 都应返回同样的 c_lo，验证场景的无状态性。
        for release_index in range(3):
            assert scenario.actual_cost_for(task, release_index) == task.c_lo


# ---------------------------------------------------------------------------
# single HI overrun scenario
# ---------------------------------------------------------------------------


def test_single_hi_overrun_only_targets_specified_release() -> None:
    """只有“指定任务 + 指定 release”才超限，其余 job 仍按 c_lo。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_hi_overrun_scenario("tau_hi", release_index=1, overrun_to="c_hi")

    # 被钦定的那次 release：跑到 c_hi。
    assert scenario.actual_cost_for(tasks["tau_hi"], release_index=1) == tasks["tau_hi"].c_hi

    # 同一任务的其他 release：仍然是 c_lo。
    assert scenario.actual_cost_for(tasks["tau_hi"], release_index=0) == tasks["tau_hi"].c_lo
    assert scenario.actual_cost_for(tasks["tau_hi"], release_index=2) == tasks["tau_hi"].c_lo

    # 其它任务，任何 release，都保持 c_lo。
    assert scenario.actual_cost_for(tasks["tau_hi_tight"], release_index=1) == tasks["tau_hi_tight"].c_lo
    assert scenario.actual_cost_for(tasks["tau_lo"], release_index=1) == tasks["tau_lo"].c_lo


def test_single_hi_overrun_supports_c_lo_plus_one_when_c_hi_strictly_gt_c_lo() -> None:
    """`c_lo_plus_one` 能精确触发“刚刚越过 LO 边界”的测试场景。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_hi_overrun_scenario(
        "tau_hi", release_index=0, overrun_to="c_lo_plus_one"
    )

    expected = tasks["tau_hi"].c_lo + 1
    assert scenario.actual_cost_for(tasks["tau_hi"], release_index=0) == expected


def test_single_hi_overrun_rejects_c_lo_plus_one_when_c_hi_equals_c_lo() -> None:
    """HI 任务若 c_hi == c_lo，则 `c_lo_plus_one` 在语义上不合法。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_hi_overrun_scenario(
        "tau_hi_tight", release_index=0, overrun_to="c_lo_plus_one"
    )

    with pytest.raises(ValueError, match="无法应用 c_lo_plus_one 场景"):
        scenario.actual_cost_for(tasks["tau_hi_tight"], release_index=0)


def test_single_hi_overrun_rejects_lo_task_target() -> None:
    """尝试让 LO 任务“超限”在 AMC 语义下是错误的实验设置。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_hi_overrun_scenario("tau_lo", release_index=0, overrun_to="c_hi")

    with pytest.raises(ValueError, match="仅支持 HI 任务"):
        scenario.actual_cost_for(tasks["tau_lo"], release_index=0)


def test_single_hi_overrun_rejects_unknown_overrun_to() -> None:
    """非法 overrun_to 关键字应在工厂构造时直接抛错（fail-fast）。"""

    with pytest.raises(ValueError, match="不支持的 overrun_to"):
        make_single_hi_overrun_scenario("tau_hi", release_index=0, overrun_to="bogus")


def test_single_hi_overrun_rejects_negative_release_index() -> None:
    """构造时给出负数 release_index 应直接报错。"""

    with pytest.raises(ValueError, match="release_index=-1 非法"):
        make_single_hi_overrun_scenario("tau_hi", release_index=-1)


def test_single_lo_overrun_returns_c_lo_plus_one_for_target_release() -> None:
    """single LO overrun 场景应仅对命中 job 返回 c_lo+1。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_lo_overrun_scenario("tau_lo", release_index=0)
    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo + 1
    assert scenario.actual_cost_for(tasks["tau_lo"], 1) == tasks["tau_lo"].c_lo
    assert scenario.actual_cost_for(tasks["tau_hi"], 0) == tasks["tau_hi"].c_lo


def test_single_lo_overrun_rejects_hi_task_target() -> None:
    """single LO overrun 命中 HI 任务时应抛错。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_single_lo_overrun_scenario("tau_hi", release_index=0)
    with pytest.raises(ValueError, match="single LO overrun scenario only supports LO tasks"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)


# ---------------------------------------------------------------------------
# all HI jobs HI budget scenario
# ---------------------------------------------------------------------------


def test_all_hi_jobs_hi_budget_defaults_to_every_hi_task() -> None:
    """不传 task_names 时，所有 HI 任务都跑 c_hi，LO 任务仍跑 c_lo。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_all_hi_jobs_hi_budget_scenario()

    assert scenario.actual_cost_for(tasks["tau_hi"], 0) == tasks["tau_hi"].c_hi
    assert scenario.actual_cost_for(tasks["tau_hi_tight"], 0) == tasks["tau_hi_tight"].c_hi
    # LO 任务不受影响。
    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo


def test_all_hi_jobs_hi_budget_respects_task_names_filter() -> None:
    """task_names 限定后，只有命中的 HI 任务跑 c_hi。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_all_hi_jobs_hi_budget_scenario(task_names=["tau_hi"])

    # 命中：跑 c_hi。
    assert scenario.actual_cost_for(tasks["tau_hi"], 0) == tasks["tau_hi"].c_hi
    # 未命中的 HI 任务：回退到 c_lo。
    assert scenario.actual_cost_for(tasks["tau_hi_tight"], 0) == tasks["tau_hi_tight"].c_lo


def test_all_hi_jobs_hi_budget_never_bumps_lo_tasks_even_if_named() -> None:
    """即使用户把 LO 任务名写进 task_names，也不会让 LO 任务跑超 c_lo。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_all_hi_jobs_hi_budget_scenario(task_names=["tau_hi", "tau_lo"])

    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo


# ---------------------------------------------------------------------------
# table scenario
# ---------------------------------------------------------------------------


def test_make_table_scenario_explicit_entry_overrides_default() -> None:
    """显式 (task_name, release_index) 表项应覆盖默认策略。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_table_scenario(
        actual_costs={
            ("tau_hi", 0): tasks["tau_hi"].c_hi,   # 明确给第 0 次 release 指派 c_hi
            ("tau_lo", 1): 1,                     # LO 任务第 1 次 release 跑 1 个 tick
        },
        default_hi="c_lo",
        default_lo="c_lo",
    )

    # 显式指定：使用表中的值。
    assert scenario.actual_cost_for(tasks["tau_hi"], 0) == tasks["tau_hi"].c_hi
    assert scenario.actual_cost_for(tasks["tau_lo"], 1) == 1

    # 未命中表：按 default 回退。
    assert scenario.actual_cost_for(tasks["tau_hi"], 1) == tasks["tau_hi"].c_lo
    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo


def test_make_table_scenario_default_hi_c_hi_pushes_all_hi_to_c_hi() -> None:
    """default_hi=c_hi 时，未显式列出的 HI 任务也跑 c_hi。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_table_scenario(actual_costs={}, default_hi="c_hi")

    assert scenario.actual_cost_for(tasks["tau_hi"], 0) == tasks["tau_hi"].c_hi
    assert scenario.actual_cost_for(tasks["tau_hi_tight"], 2) == tasks["tau_hi_tight"].c_hi
    # LO 任务仍然是 c_lo。
    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo


def test_make_table_scenario_rejects_invalid_default_hi() -> None:
    """default_hi 非受支持值应立即 fail-fast，不能等到调用时才暴露。"""

    with pytest.raises(ValueError, match="不支持的 default_hi"):
        make_table_scenario(actual_costs={}, default_hi="c_lo_plus_one")


def test_make_table_scenario_rejects_invalid_default_lo() -> None:
    """default_lo 目前仅允许 c_lo；其它关键字必须在工厂阶段拦下。"""

    with pytest.raises(ValueError, match="不支持的 default_lo"):
        make_table_scenario(actual_costs={}, default_lo="c_hi")


def test_make_table_scenario_rejects_malformed_keys() -> None:
    """表的键必须是 (str, int) 形式；其它结构必须在工厂阶段报错。"""

    with pytest.raises(ValueError, match="actual_costs 的键"):
        # 刻意传一个非 tuple 的键以触发 schema 校验。
        make_table_scenario(actual_costs={"tau_hi": 3})  # type: ignore[dict-item]


def test_make_table_scenario_raises_when_lo_task_assigned_above_c_lo() -> None:
    """AMC+ 语义下，对 LO 任务显式指定大于 c_lo 的值应允许通过。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_table_scenario(
        actual_costs={("tau_lo", 0): tasks["tau_lo"].c_lo + 1},
    )

    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo + 1


def test_make_table_scenario_raises_when_hi_task_assigned_above_c_hi() -> None:
    """对 HI 任务显式指定了大于 c_hi 的值时，调用 actual_cost_for 必须报错。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_table_scenario(
        actual_costs={("tau_hi", 0): tasks["tau_hi"].c_hi + 1},
    )

    with pytest.raises(ValueError, match="HI 任务 tau_hi"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)


# ---------------------------------------------------------------------------
# ExecutionScenario 公共约束
# ---------------------------------------------------------------------------


def test_execution_scenario_rejects_negative_release_index() -> None:
    """actual_cost_for 的 release_index 必须 >= 0。"""

    tasks = _by_name(_reference_tasks())
    scenario = make_nominal_scenario()

    with pytest.raises(ValueError, match="release_index=-1 非法"):
        scenario.actual_cost_for(tasks["tau_hi"], -1)


def test_custom_resolver_can_be_wrapped_in_execution_scenario() -> None:
    """用户自定义 resolver 仍需经过关键级校验：HI 超 c_hi 会被拦下。"""

    tasks = _by_name(_reference_tasks())

    def custom_resolver(task: Task, release_index: int) -> int:
        if task.criticality is Criticality.HI:
            return task.c_hi + 1
        return task.c_lo + 5

    scenario = ExecutionScenario(name="custom", resolver=custom_resolver)

    # HI 任务触发校验失败。
    with pytest.raises(ValueError, match="HI 任务 tau_hi"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)

    # LO 任务超过 c_lo 在 AMC+ 场景层允许通过。
    assert scenario.actual_cost_for(tasks["tau_lo"], 0) == tasks["tau_lo"].c_lo + 5


def test_execution_scenario_rejects_float_actual_cost() -> None:
    """resolver 返回 float 时必须报 TypeError，不能再被静默截断为 int。"""

    tasks = _by_name(_reference_tasks())

    def float_resolver(task: Task, release_index: int) -> float:  # noqa: ARG001
        # 故意返回浮点数，验证 actual_cost_for 的严格整数类型约束。
        return 1.5

    scenario = ExecutionScenario(name="float_cost", resolver=float_resolver)

    with pytest.raises(TypeError, match="actual_cost.*float"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)


def test_execution_scenario_rejects_string_actual_cost() -> None:
    """resolver 返回 str 时必须报 TypeError，不能再被 int(\"2\") 接受。"""

    tasks = _by_name(_reference_tasks())

    def string_resolver(task: Task, release_index: int) -> str:  # noqa: ARG001
        # 故意返回字符串，确保不会被隐式解析为整数。
        return "2"

    scenario = ExecutionScenario(name="string_cost", resolver=string_resolver)

    with pytest.raises(TypeError, match="actual_cost.*str"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)


def test_execution_scenario_rejects_bool_actual_cost() -> None:
    """resolver 返回 bool 时必须报 TypeError，避免 True/False 被当成 1/0。"""

    tasks = _by_name(_reference_tasks())

    def bool_resolver(task: Task, release_index: int) -> bool:  # noqa: ARG001
        # 故意返回布尔值；即便 bool 是 int 子类，也必须被显式拒绝。
        return True

    scenario = ExecutionScenario(name="bool_cost", resolver=bool_resolver)

    with pytest.raises(TypeError, match="actual_cost.*bool"):
        scenario.actual_cost_for(tasks["tau_hi"], 0)


# ---------------------------------------------------------------------------
# runtime_models 数据容器便捷视图
# ---------------------------------------------------------------------------


def _make_job(task: Task, release_index: int = 0, actual_cost: int | None = None) -> Job:
    """测试辅助：基于 Task 快速构造一个 Job（复用 c_lo 作为默认 actual_cost）。"""

    cost = task.c_lo if actual_cost is None else actual_cost
    return Job(
        task=task,
        release_index=release_index,
        release_time=release_index * task.period,
        absolute_deadline=release_index * task.period + task.deadline,
        actual_cost=cost,
    )


def test_runtime_config_defaults_and_validation() -> None:
    """RuntimeConfig 默认值合理，且非法参数会 fail-fast。"""

    default_cfg = RuntimeConfig()
    assert default_cfg.end_time is None
    assert default_cfg.jobs_per_task == 5
    assert default_cfg.capture_trace is False
    assert default_cfg.capture_debug_events is False
    assert default_cfg.stop_at_first_miss is False
    assert default_cfg.drop_lo_jobs_on_hi_switch is True
    assert default_cfg.semantics is RuntimeSemantics.AMC_PLUS
    assert default_cfg.record_dropped_lo_releases is False

    # end_time 允许为 None 或正整数。
    RuntimeConfig(end_time=100)
    with pytest.raises(ValueError, match="end_time"):
        RuntimeConfig(end_time=0)
    with pytest.raises(ValueError, match="jobs_per_task"):
        RuntimeConfig(jobs_per_task=0)
    with pytest.raises(ValueError, match="hyperperiod_limit"):
        RuntimeConfig(hyperperiod_limit=-1)


def test_job_remaining_and_finished_helpers() -> None:
    """Job.remaining / finished 应正确反映 executed / dropped 的组合情况。"""

    tasks = _by_name(_reference_tasks())
    job = _make_job(tasks["tau_hi"], release_index=0, actual_cost=3)

    # 初始：尚未执行，remaining=3，未完成。
    assert job.remaining() == 3
    assert job.finished() is False

    # 执行 2 tick 后：remaining=1。
    job.executed_time = 2
    assert job.remaining() == 1
    assert job.finished() is False

    # 执行满：remaining=0 且 finished。
    job.executed_time = 3
    assert job.remaining() == 0
    assert job.finished() is True

    # 即使实际执行超过 actual_cost，remaining() 也不会返回负数。
    job.executed_time = 5
    assert job.remaining() == 0
    assert job.finished() is True


def test_job_finished_when_dropped_even_without_execution() -> None:
    """被 HI 切换 drop 的 job 应立即被视为 finished，无论是否执行过。"""

    tasks = _by_name(_reference_tasks())
    job = _make_job(tasks["tau_lo"], release_index=0, actual_cost=2)

    job.dropped = True
    job.drop_time = 7
    # 被丢弃后 remaining 归零、finished 为真。
    assert job.remaining() == 0
    assert job.finished() is True


def test_simulation_result_helpers_group_jobs_and_modes() -> None:
    """SimulationResult 的便捷视图应正确分类 job 并反映模式切换/错失状态。"""

    tasks = _by_name(_reference_tasks())

    completed = _make_job(tasks["tau_hi"], release_index=0, actual_cost=2)
    completed.executed_time = 2
    completed.completion_time = 5  # 任意合法时刻

    running = _make_job(tasks["tau_hi"], release_index=1, actual_cost=2)
    running.executed_time = 1  # 仿真结束时仍未跑完

    dropped = _make_job(tasks["tau_lo"], release_index=0, actual_cost=2)
    dropped.dropped = True
    dropped.drop_time = 12

    miss = DeadlineMiss(
        task="tau_lo",
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        mode_at_miss=SystemMode.HI,
        executed_at_miss=1,
    )

    result = SimulationResult(
        jobs=[completed, running, dropped],
        trace=[ScheduleTick(time=0, executing_task="tau_hi", executing_release_index=0, mode=SystemMode.LO)],
        mode_switches=[
            ModeSwitchEvent(
                switch_time=6,
                triggering_task="tau_hi",
                triggering_release_index=0,
                executed_at_switch=3,
            )
        ],
        deadline_misses=[miss],
        end_time=20,
        final_mode=SystemMode.HI,
    )

    # 已完成 job 只包含 completed；被 dropped 的不算 completed。
    completed_jobs = result.completed_jobs()
    assert [job.release_index for job in completed_jobs] == [0]
    assert completed_jobs[0].task.name == "tau_hi"

    # 被丢弃的 job 通过 dropped_jobs 暴露。
    dropped_jobs = result.dropped_jobs()
    assert len(dropped_jobs) == 1
    assert dropped_jobs[0].task.name == "tau_lo"

    # mode_switched 与 deadline_missed 的布尔视图。
    assert result.mode_switched() is True
    assert result.deadline_missed() is True

    # 按任务名分组：应按 release_index 升序返回。
    tau_hi_jobs = result.jobs_of("tau_hi")
    assert [job.release_index for job in tau_hi_jobs] == [0, 1]
    assert result.jobs_of("tau_unknown") == []


def test_runtime_comparison_result_exposes_helpers() -> None:
    """RuntimeComparisonResult 的便捷方法应正确聚合两侧结果。"""

    static = SchedulabilityResult(
        schedulable=True,
        method="amc_rtb",
        response_times={"tau_hi": 3},
        details="ok",
    )
    runtime = SimulationResult(end_time=10, final_mode=SystemMode.LO)
    comparison = RuntimeComparisonResult(
        static_result=static,
        runtime_result=runtime,
        ordered_task_names=["tau_hi", "tau_lo"],
        method="amc_rtb",
        priority_policy="dm",
    )

    assert comparison.static_schedulable() is True
    assert comparison.mode_switched() is False
    assert comparison.deadline_missed() is False
    assert comparison.ordered_task_names == ["tau_hi", "tau_lo"]
    assert comparison.method == "amc_rtb"
    assert comparison.priority_policy == "dm"
