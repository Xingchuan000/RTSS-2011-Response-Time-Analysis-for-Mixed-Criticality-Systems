"""运行时仿真器单元测试（阶段运行时模拟 · 第 3 轮）。

本文件覆盖 `amc_py.runtime` 的基础 tick 仿真器，重点放在：
1. `compute_hyperperiod` / `compute_default_end_time` 的数学正确性；
2. `_build_job` / `_release_jobs_at_time` / `_select_highest_priority_ready_job`
   / `_check_deadline_misses` / `_should_switch_to_hi` 的细粒度单测；
3. `simulate_ordered_taskset` 在 nominal 场景下：job 能按时完成、trace 正确、
   capture_trace 开关生效；
4. 两任务固定优先级抢占路径逐 tick 验证；
5. 蓄意过载任务集下记录 deadline miss；
6. `stop_at_first_miss=True / False` 两种行为都符合预期。

本轮不包含 HI 模式切换、LO job 丢弃、future LO release 抑制等测试，
这些留给第 4 轮。`_should_switch_to_hi` 只作为纯 predicate 单独单测。
"""

from __future__ import annotations

import pytest

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.runtime import (
    _build_job,
    _cancel_lo_job,
    _check_deadline_misses,
    _drop_active_lo_jobs,
    _release_jobs_at_time,
    _select_highest_priority_ready_job,
    _should_cancel_lo_job,
    _should_switch_to_hi,
    compute_default_end_time,
    compute_hyperperiod,
    simulate_ordered_taskset,
)
from amc_py.runtime_models import (
    Job,
    RuntimeConfig,
    SystemMode,
)
from amc_py.runtime_scenarios import (
    ExecutionScenario,
    make_all_hi_jobs_hi_budget_scenario,
    make_nominal_scenario,
    make_single_hi_overrun_scenario,
    make_single_lo_overrun_scenario,
)


# ---------------------------------------------------------------------------
# 通用测试工具
# ---------------------------------------------------------------------------


def _lo(name: str, period: int, c_lo: int, *, deadline: int | None = None) -> Task:
    """构造一个 LO 任务，deadline 默认等于 period。"""

    return Task(
        name=name,
        period=period,
        deadline=deadline if deadline is not None else period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def _hi(name: str, period: int, c_lo: int, c_hi: int, *, deadline: int | None = None) -> Task:
    """构造一个 HI 任务，deadline 默认等于 period。"""

    return Task(
        name=name,
        period=period,
        deadline=deadline if deadline is not None else period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


def _trace_task_at(result, time: int) -> str | None:
    """从 trace 里取出指定 tick 正在运行的任务名，找不到返回 None。"""

    for tick in result.trace:
        if tick.time == time:
            return tick.executing_task
    return None


# ---------------------------------------------------------------------------
# compute_hyperperiod
# ---------------------------------------------------------------------------


def test_compute_hyperperiod_single_task_equals_its_period() -> None:
    """单任务的 hyperperiod 就是自身周期。"""

    tasks = [_lo("a", period=7, c_lo=1)]
    assert compute_hyperperiod(tasks) == 7


def test_compute_hyperperiod_coprime_periods_is_product() -> None:
    """互素周期的 hyperperiod 等于它们的乘积。"""

    tasks = [_lo("a", period=3, c_lo=1), _lo("b", period=5, c_lo=1)]
    assert compute_hyperperiod(tasks) == 15


def test_compute_hyperperiod_with_common_factor_uses_lcm() -> None:
    """有公因子的周期，hyperperiod 是最小公倍数。"""

    tasks = [_lo("a", period=4, c_lo=1), _lo("b", period=6, c_lo=1)]
    assert compute_hyperperiod(tasks) == 12


def test_compute_hyperperiod_three_tasks_applies_lcm_reduction() -> None:
    """多任务场景验证 functools.reduce 按顺序做 lcm 折叠。"""

    tasks = [
        _lo("a", period=4, c_lo=1),
        _lo("b", period=6, c_lo=1),
        _lo("c", period=10, c_lo=1),
    ]
    # lcm(4, 6) = 12；lcm(12, 10) = 60。
    assert compute_hyperperiod(tasks) == 60


def test_compute_hyperperiod_rejects_empty_taskset() -> None:
    """空任务集不存在合法 hyperperiod 定义。"""

    with pytest.raises(ValueError, match="tasks 不能为空"):
        compute_hyperperiod([])


# ---------------------------------------------------------------------------
# compute_default_end_time
# ---------------------------------------------------------------------------


def test_default_end_time_small_hyperperiod_dominated_by_jobs_based() -> None:
    """小 hyperperiod 时，jobs_based 更大就取 jobs_based。"""

    tasks = [_lo("a", period=4, c_lo=1), _lo("b", period=6, c_lo=1)]
    # hyperperiod = 12；jobs_based = 5 * 6 = 30；min(.., limit=1000) 不触发。
    assert compute_default_end_time(tasks, jobs_per_task=5, hyperperiod_limit=1000) == 30


def test_default_end_time_small_hyperperiod_larger_than_jobs_based() -> None:
    """小 hyperperiod 本身比 jobs_based 更大时，取 hyperperiod。"""

    tasks = [_lo("a", period=100, c_lo=1), _lo("b", period=101, c_lo=1)]
    # hyperperiod = 10100；jobs_based = 5 * 101 = 505；hyperperiod_limit 足够大。
    assert (
        compute_default_end_time(tasks, jobs_per_task=5, hyperperiod_limit=20000) == 10100
    )


def test_default_end_time_large_hyperperiod_falls_back_to_jobs_based() -> None:
    """hyperperiod 超出 limit 时退化为 jobs_based。"""

    tasks = [_lo("a", period=997, c_lo=1), _lo("b", period=1009, c_lo=1)]
    # hyperperiod = 997 * 1009 = 1_005_973，远超默认 limit=100_000；
    # jobs_based = 5 * 1009 = 5045。
    assert (
        compute_default_end_time(tasks, jobs_per_task=5, hyperperiod_limit=100_000) == 5045
    )


def test_default_end_time_rejects_invalid_params() -> None:
    """非法参数应立即 fail-fast。"""

    tasks = [_lo("a", period=5, c_lo=1)]
    with pytest.raises(ValueError, match="jobs_per_task"):
        compute_default_end_time(tasks, jobs_per_task=0)
    with pytest.raises(ValueError, match="hyperperiod_limit"):
        compute_default_end_time(tasks, hyperperiod_limit=-1)
    with pytest.raises(ValueError, match="tasks 不能为空"):
        compute_default_end_time([])


# ---------------------------------------------------------------------------
# _build_job / _release_jobs_at_time
# ---------------------------------------------------------------------------


def test_build_job_sets_release_deadline_and_actual_cost() -> None:
    """_build_job 按周期/截止期/场景推导出合法的 Job。"""

    task = _lo("a", period=10, c_lo=3)
    scenario = make_nominal_scenario()
    job = _build_job(task, release_index=2, scenario=scenario)

    assert job.task is task
    assert job.release_index == 2
    assert job.release_time == 20
    assert job.absolute_deadline == 30
    assert job.actual_cost == 3  # nominal 场景下 LO 任务 = c_lo
    assert job.executed_time == 0
    assert job.completion_time is None
    assert job.dropped is False


def test_release_jobs_at_time_releases_only_matching_tasks() -> None:
    """释放器只会释放 release_time 等于 current_time 的 job，并推进计数。"""

    tasks = [_lo("a", period=5, c_lo=1), _lo("b", period=3, c_lo=1)]
    scenario = make_nominal_scenario()
    counters = {task.name: 0 for task in tasks}

    # t=0：两个任务同时释放。
    released_0 = _release_jobs_at_time(0, tasks, counters, scenario)
    assert sorted(j.task.name for j in released_0) == ["a", "b"]
    assert counters == {"a": 1, "b": 1}

    # t=1：没有任务到期。
    released_1 = _release_jobs_at_time(1, tasks, counters, scenario)
    assert released_1 == []
    assert counters == {"a": 1, "b": 1}

    # t=3：b 第 2 次释放。
    released_3 = _release_jobs_at_time(3, tasks, counters, scenario)
    assert [j.task.name for j in released_3] == ["b"]
    assert released_3[0].release_index == 1
    assert counters == {"a": 1, "b": 2}

    # t=5：a 第 2 次释放；b 无动作（6 才到期）。
    released_5 = _release_jobs_at_time(5, tasks, counters, scenario)
    assert [j.task.name for j in released_5] == ["a"]
    assert released_5[0].release_index == 1
    assert counters == {"a": 2, "b": 2}


# ---------------------------------------------------------------------------
# _select_highest_priority_ready_job
# ---------------------------------------------------------------------------


def test_select_highest_priority_prefers_lower_priority_index() -> None:
    """priority_map 值越小越优先被选。"""

    tasks = [_lo("hi", period=5, c_lo=1), _lo("lo", period=5, c_lo=1)]
    priority_map = {"hi": 0, "lo": 1}
    scenario = make_nominal_scenario()
    counters = {task.name: 0 for task in tasks}
    active = _release_jobs_at_time(0, tasks, counters, scenario)

    selected = _select_highest_priority_ready_job(active, priority_map)
    assert selected is not None
    assert selected.task.name == "hi"


def test_select_highest_priority_returns_none_on_empty_ready_set() -> None:
    """没有可运行 job 时返回 None，对应 CPU 空闲。"""

    assert _select_highest_priority_ready_job([], {"a": 0}) is None


def test_select_highest_priority_skips_finished_jobs() -> None:
    """已完成 / 已丢弃的 job 不应被重新选中。"""

    task = _lo("a", period=5, c_lo=2)
    done_job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=5,
        actual_cost=2,
        executed_time=2,
        completion_time=2,
    )
    dropped_job = Job(
        task=task,
        release_index=1,
        release_time=5,
        absolute_deadline=10,
        actual_cost=2,
        dropped=True,
    )

    assert _select_highest_priority_ready_job([done_job, dropped_job], {"a": 0}) is None


# ---------------------------------------------------------------------------
# _check_deadline_misses
# ---------------------------------------------------------------------------


def test_check_deadline_misses_records_only_at_exact_deadline() -> None:
    """miss 仅在 absolute_deadline == current_time 的那一 tick 被记录。"""

    task = _lo("a", period=5, c_lo=3)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=5,
        actual_cost=3,
        executed_time=1,  # 未完成
    )

    # deadline 之前不会记录 miss。
    assert _check_deadline_misses(4, [job], SystemMode.LO) == []

    # 正好在 deadline 上且未完成：记录 miss。
    misses = _check_deadline_misses(5, [job], SystemMode.LO)
    assert len(misses) == 1
    assert misses[0].task == "a"
    assert misses[0].mode_at_miss is SystemMode.LO
    assert misses[0].executed_at_miss == 1


def test_check_deadline_misses_ignores_finished_or_dropped_jobs() -> None:
    """已完成或被丢弃的 job 不应再被计为 miss。"""

    task = _lo("a", period=5, c_lo=3)
    finished = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=5,
        actual_cost=3,
        executed_time=3,
        completion_time=5,
    )
    dropped = Job(
        task=task,
        release_index=1,
        release_time=5,
        absolute_deadline=10,
        actual_cost=3,
        dropped=True,
    )

    assert _check_deadline_misses(5, [finished], SystemMode.LO) == []
    assert _check_deadline_misses(10, [dropped], SystemMode.LO) == []


# ---------------------------------------------------------------------------
# _should_switch_to_hi（第 3 轮只做独立 predicate 单测）
# ---------------------------------------------------------------------------


def test_should_switch_to_hi_true_when_hi_task_exceeds_c_lo_in_lo_mode() -> None:
    """HI 任务在 LO 模式下，executed_time 越过 c_lo 就应判为需要切换。"""

    task = _hi("h", period=10, c_lo=2, c_hi=5)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=5,
        executed_time=3,  # > c_lo=2
    )
    assert _should_switch_to_hi(job, SystemMode.LO, budget=2) is True


def test_should_switch_to_hi_false_at_or_below_c_lo() -> None:
    """executed_time 恰好等于 c_lo 或更小时，尚未越界，不应切换。"""

    task = _hi("h", period=10, c_lo=2, c_hi=5)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=5,
        executed_time=2,  # == c_lo
    )
    assert _should_switch_to_hi(job, SystemMode.LO, budget=2) is False


def test_should_switch_to_hi_false_for_lo_tasks() -> None:
    """LO 任务无论执行多久都不会触发 HI 切换。"""

    task = _lo("l", period=10, c_lo=2)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=2,
        executed_time=10,  # 蓄意给个很大的值，仍不该触发
    )
    assert _should_switch_to_hi(job, SystemMode.LO, budget=2) is False


def test_should_switch_to_hi_false_when_mode_already_hi() -> None:
    """模式已经是 HI 时，不需要再次触发切换。"""

    task = _hi("h", period=10, c_lo=2, c_hi=5)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=5,
        executed_time=4,
    )
    assert _should_switch_to_hi(job, SystemMode.HI, budget=2) is False


def test_should_cancel_lo_job_true_when_lo_task_exceeds_runtime_budget() -> None:
    """LO 任务在 LO 模式下超过运行时 budget 时应被取消。"""

    task = _lo("l", period=10, c_lo=2)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=4,
        executed_time=3,
    )
    assert _should_cancel_lo_job(job, SystemMode.LO, budget=2) is True


def test_cancel_lo_job_marks_drop_and_returns_event() -> None:
    """取消 LO job 后应从活动队列移除并记录事件信息。"""

    task = _lo("l", period=10, c_lo=2)
    job = Job(
        task=task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=4,
        executed_time=3,
    )
    active = [job]
    event = _cancel_lo_job(active, job, current_time=4, budget=2)
    assert job.dropped is True
    assert job.drop_time == 4
    assert active == []
    assert event.task == "l"
    assert event.budget_at_cancel == 2


# ---------------------------------------------------------------------------
# _drop_active_lo_jobs（第 4 轮新增）
# ---------------------------------------------------------------------------


def test_drop_active_lo_jobs_marks_and_removes_only_unfinished_lo_jobs() -> None:
    """进入 HI 模式时，只应丢弃“未完成”的 LO job，HI job 与已完成 LO job 不受影响。"""

    lo_task = _lo("lo", period=10, c_lo=2)
    hi_task = _hi("hi", period=10, c_lo=1, c_hi=3)

    lo_unfinished = Job(
        task=lo_task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=2,
        executed_time=1,  # 尚未完成，应被丢弃
    )
    lo_finished = Job(
        task=lo_task,
        release_index=1,
        release_time=10,
        absolute_deadline=20,
        actual_cost=2,
        executed_time=2,
        completion_time=12,  # 已完成，不应被丢弃
    )
    hi_unfinished = Job(
        task=hi_task,
        release_index=0,
        release_time=0,
        absolute_deadline=10,
        actual_cost=3,
        executed_time=1,  # HI job 不属于 drop 目标
    )

    active_jobs = [lo_unfinished, lo_finished, hi_unfinished]
    dropped = _drop_active_lo_jobs(active_jobs, current_time=7)

    assert dropped == [lo_unfinished]
    assert lo_unfinished.dropped is True
    assert lo_unfinished.drop_time == 7
    # 被丢弃的 job 会从 active 队列中移除，防止后续被调度。
    assert lo_unfinished not in active_jobs
    # 其余 job 仍在 active 队列中。
    assert lo_finished in active_jobs
    assert hi_unfinished in active_jobs


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：nominal 场景
# ---------------------------------------------------------------------------


def test_simulate_single_task_nominal_completes_all_releases() -> None:
    """单任务在 nominal 场景下，所有释放都按时完成，且不会 miss 或 switch。"""

    tasks = [_lo("a", period=5, c_lo=2)]
    cfg = RuntimeConfig(end_time=20, capture_trace=True)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    # end_time=20，周期 5，理应有 4 次释放（t=0,5,10,15）。
    assert len(result.jobs) == 4
    for job in result.jobs:
        assert job.completion_time is not None
        # nominal 场景：LO 任务实际跑 c_lo=2 个 tick。
        assert job.actual_cost == 2
        assert job.completion_time == job.release_time + 2

    # 不应有 miss 或 mode switch。
    assert result.deadline_missed() is False
    assert result.mode_switched() is False
    assert result.final_mode is SystemMode.LO
    assert result.end_time == 20
    # 这里显式开启 trace：tick 数应等于 end_time。
    assert len(result.trace) == 20


def test_simulate_nominal_two_task_hyperperiod_all_jobs_complete() -> None:
    """两任务 nominal 场景：按 hyperperiod 仿真，所有 job 都完成。"""

    tasks = [
        _lo("a", period=5, c_lo=2),
        _lo("b", period=7, c_lo=2),
    ]
    # hyperperiod = 35；整数倍释放下每任务都能完整跑若干次。
    cfg = RuntimeConfig(end_time=35)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    # a: 35/5=7 次释放；b: 35/7=5 次释放。
    assert len(result.jobs_of("a")) == 7
    assert len(result.jobs_of("b")) == 5
    for job in result.jobs:
        assert job.completion_time is not None

    assert result.deadline_missed() is False
    assert result.mode_switched() is False


def test_simulate_uses_default_end_time_when_none_specified() -> None:
    """不指定 end_time 时，应用 compute_default_end_time 的结果。"""

    tasks = [_lo("a", period=5, c_lo=1)]
    # compute_default_end_time：hp=5；jobs_based = 5 * 5 = 25；max=25。
    result = simulate_ordered_taskset(tasks, make_nominal_scenario())

    assert result.end_time == 25
    assert len(result.jobs) == 5  # t=0,5,10,15,20 各释放一次


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：抢占路径
# ---------------------------------------------------------------------------


def test_simulate_two_task_preemption_trace_matches_expected_pattern() -> None:
    """两任务固定优先级抢占：高优先级短任务打断低优先级长任务。"""

    tasks = [
        _lo("hi", period=4, c_lo=1),   # 优先级 0：周期短、任务轻
        _lo("lo", period=10, c_lo=5),  # 优先级 1：周期长、任务重
    ]
    cfg = RuntimeConfig(end_time=10, capture_trace=True)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    # 按推理：
    # t=0: hi_0 运行并完成（c_lo=1）；
    # t=1..3: lo_0 连跑三 tick；
    # t=4: hi_1 抢占并完成；
    # t=5..6: lo_0 恢复，t=6 时完成（总共 5 tick）；
    # t=7: 空闲；
    # t=8: hi_2 完成；
    # t=9: 空闲。
    assert _trace_task_at(result, 0) == "hi"
    assert _trace_task_at(result, 1) == "lo"
    assert _trace_task_at(result, 2) == "lo"
    assert _trace_task_at(result, 3) == "lo"
    assert _trace_task_at(result, 4) == "hi"  # 抢占点
    assert _trace_task_at(result, 5) == "lo"
    assert _trace_task_at(result, 6) == "lo"
    assert _trace_task_at(result, 7) is None  # idle
    assert _trace_task_at(result, 8) == "hi"
    assert _trace_task_at(result, 9) is None  # idle

    # 所有 hi / lo 的 job 都应按时完成，不产生 miss。
    assert result.deadline_missed() is False
    for job in result.jobs:
        assert job.completion_time is not None


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：trace 开关
# ---------------------------------------------------------------------------


def test_simulate_capture_trace_false_yields_empty_trace() -> None:
    """capture_trace=False 时，trace 为空，但 jobs / misses 仍完整。"""

    tasks = [_lo("a", period=5, c_lo=2)]
    cfg = RuntimeConfig(end_time=20, capture_trace=False)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    assert result.trace == []
    assert len(result.jobs) == 4
    assert result.deadline_missed() is False


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：deadline miss
# ---------------------------------------------------------------------------


def test_simulate_overloaded_taskset_records_deadline_miss() -> None:
    """总利用率 > 1 的 LO 任务集必然在短时间内出现 miss。"""

    # util(a) = 2/3, util(b) = 3/4；合计 17/12 > 1。
    tasks = [
        _lo("a", period=3, c_lo=2),
        _lo("b", period=4, c_lo=3),
    ]
    cfg = RuntimeConfig(end_time=20)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    assert result.deadline_missed() is True
    # 至少应包含一个 miss；overloaded 场景通常会出现多次。
    assert len(result.deadline_misses) >= 1
    first = result.deadline_misses[0]
    assert first.mode_at_miss is SystemMode.LO
    # miss 出现在 absolute_deadline 时刻，并且未完成。
    assert first.executed_at_miss < first.absolute_deadline - first.release_time


def test_simulate_deadline_miss_exactly_at_end_time_is_captured() -> None:
    """absolute_deadline 正好等于 end_time 的 miss 由“终点扫描”兜底。"""

    # 任务 a：周期=10，deadline=10，c_lo=11 —— 物理上不可能在 deadline 前完成。
    # 但 Task 允许 c_lo <= period 且 deadline == period，所以此处构造一个会超时的 job。
    # 为避免受 Task 约束影响，这里直接造一个 deadline == end_time 的单一释放场景：
    # 让 end_time = 10、任务 a 的 absolute_deadline = 10。
    tasks = [_lo("a", period=10, c_lo=11)]  # actual_cost=c_lo=11 > deadline=10 → 必 miss
    cfg = RuntimeConfig(end_time=10)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    # absolute_deadline=10 的 miss 只会在主循环结束后的终点扫描中被发现。
    assert result.deadline_missed() is True
    assert any(miss.absolute_deadline == 10 for miss in result.deadline_misses)


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：stop_at_first_miss
# ---------------------------------------------------------------------------


def test_simulate_stop_at_first_miss_true_ends_at_first_miss_time() -> None:
    """stop_at_first_miss=True 时，end_time 等于首个 miss 被检测到的那一 tick。"""

    tasks = [
        _lo("a", period=3, c_lo=2),
        _lo("b", period=4, c_lo=3),
    ]
    cfg = RuntimeConfig(end_time=50, stop_at_first_miss=True, capture_trace=True)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    # 必须命中 miss；end_time 应该提前（小于配置的 50）。
    assert result.deadline_missed() is True
    assert result.end_time < 50
    # 首个 miss 记录的 absolute_deadline 应当等于 actual end_time。
    assert result.deadline_misses[0].absolute_deadline == result.end_time
    # trace 长度应当等于实际执行过的 tick 数（不超过 end_time）。
    assert len(result.trace) == result.end_time


def test_simulate_stop_at_first_miss_false_continues_until_end() -> None:
    """stop_at_first_miss=False 时，仿真一路跑到配置的 end_time。"""

    tasks = [
        _lo("a", period=3, c_lo=2),
        _lo("b", period=4, c_lo=3),
    ]
    cfg = RuntimeConfig(end_time=30, stop_at_first_miss=False)
    result = simulate_ordered_taskset(tasks, make_nominal_scenario(), cfg)

    assert result.end_time == 30
    # 过载任务集在 30 tick 内至少应出现 2 次以上 miss。
    assert len(result.deadline_misses) >= 2


# ---------------------------------------------------------------------------
# simulate_ordered_taskset：输入校验
# ---------------------------------------------------------------------------


def test_simulate_rejects_empty_task_list() -> None:
    """空任务集视为非法输入。"""

    with pytest.raises(ValueError, match="ordered_tasks 不能为空"):
        simulate_ordered_taskset([], make_nominal_scenario())


def test_simulate_rejects_duplicate_task_names() -> None:
    """重复任务名会让优先级映射不稳定，必须拦下。"""

    tasks = [_lo("a", period=5, c_lo=1), _lo("a", period=7, c_lo=1)]
    with pytest.raises(ValueError, match="重复任务名"):
        simulate_ordered_taskset(tasks, make_nominal_scenario())


# ---------------------------------------------------------------------------
# 第 4 轮：HI 模式切换语义
# ---------------------------------------------------------------------------


def test_single_hi_overrun_triggers_mode_switch_and_records_event() -> None:
    """single HI overrun 场景应触发一次 LO->HI 切换，并写入 ModeSwitchEvent。"""

    tasks = [
        _hi("h", period=6, c_lo=1, c_hi=3),
        _lo("l", period=8, c_lo=2),
    ]
    scenario = make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi")
    cfg = RuntimeConfig(end_time=12)
    result = simulate_ordered_taskset(tasks, scenario, cfg)

    assert result.mode_switched() is True
    assert result.mode_recovery_count() >= 1
    assert result.final_mode is SystemMode.LO
    assert result.mode_switch is not None
    # h_0 在 t=0 与 t=1 连续执行，第二次执行后越过 c_lo=1，故切换时刻是 2。
    assert result.mode_switch.switch_time == 2
    assert result.mode_switch.triggering_task == "h"
    assert result.mode_switch.triggering_release_index == 0
    assert result.mode_switch.executed_at_switch == 2


def test_hi_switch_drops_active_lo_jobs_when_drop_enabled() -> None:
    """默认配置下，切换到 HI 后应丢弃活动中的 LO job。"""

    tasks = [
        _hi("h", period=6, c_lo=1, c_hi=3),
        _lo("l", period=8, c_lo=3),
    ]
    scenario = make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi")
    result = simulate_ordered_taskset(tasks, scenario, RuntimeConfig(end_time=12))

    lo_jobs = result.jobs_of("l")
    assert len(lo_jobs) >= 1
    # l_0 在 t=0 已释放，切换发生时仍未完成，因此应被 drop。
    assert lo_jobs[0].release_index == 0
    assert lo_jobs[0].dropped is True
    assert lo_jobs[0].drop_time == 2
    assert lo_jobs[0].completion_time is None


def test_hi_switch_keeps_active_lo_jobs_when_drop_disabled() -> None:
    """当 drop_lo_jobs_on_hi_switch=False 时，活动 LO job 应被保留并可继续执行。"""

    tasks = [
        _hi("h", period=20, c_lo=1, c_hi=2),
        _lo("l", period=20, c_lo=2),
    ]
    scenario = make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi")
    cfg = RuntimeConfig(end_time=8, drop_lo_jobs_on_hi_switch=False)
    result = simulate_ordered_taskset(tasks, scenario, cfg)

    lo_jobs = result.jobs_of("l")
    assert len(lo_jobs) == 1
    assert lo_jobs[0].dropped is False
    # HI job在 t=0,1 用完预算后，LO job 应有机会完成。
    assert lo_jobs[0].completion_time is not None


def test_hi_mode_suppresses_future_lo_releases_after_switch() -> None:
    """HI 模式下应抑制 LO release，恢复 LO 后只继续新周期 release，不补发历史周期。"""

    tasks = [
        _hi("h", period=6, c_lo=1, c_hi=3),
        _lo("l", period=2, c_lo=1),
    ]
    scenario = make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi")
    result = simulate_ordered_taskset(tasks, scenario, RuntimeConfig(end_time=10))

    # l 的理论释放时刻是 0,2,4,6,8；切换时刻为 2，HI 期间（2,3）抑制 release，
    # 恢复 LO 后从新周期继续，不补发 2 和 3 期间错过的 release。
    lo_jobs = result.jobs_of("l")
    release_indexes = [job.release_index for job in lo_jobs]
    assert 0 in release_indexes
    assert 1 not in release_indexes
    assert 2 in release_indexes


def test_mode_switch_happens_only_once_even_when_multiple_hi_jobs_overrun() -> None:
    """系统最多发生一次 LO->HI 切换；进入 HI 后不会重复记录切换事件。"""

    tasks = [
        _hi("h1", period=5, c_lo=1, c_hi=3),
        _hi("h2", period=7, c_lo=1, c_hi=3),
        _lo("l", period=9, c_lo=1),
    ]
    scenario = make_all_hi_jobs_hi_budget_scenario()
    result = simulate_ordered_taskset(tasks, scenario, RuntimeConfig(end_time=20))

    assert result.mode_switched() is True
    assert result.mode_switch is not None
    # 由于 h1 优先级最高且 release 在 t=0，首次触发者应为 h1_0。
    assert result.mode_switch.triggering_task == "h1"
    assert result.mode_switch.triggering_release_index == 0
    # 结果模型只保存单个 mode_switch，天然保证“只记录一次”。
    assert result.final_mode is SystemMode.HI


def test_custom_scenario_errors_propagate_out_of_simulator() -> None:
    """自定义 scenario 若返回非法 actual_cost，异常应直接透传给调用方。"""

    # 故意构造一个会让 HI 任务“越过 c_hi”的 resolver，触发 _validate_actual_cost。
    def bad_resolver(task: Task, release_index: int) -> int:  # noqa: ARG001
        return task.c_hi + 5

    bad_scenario = ExecutionScenario(name="bad", resolver=bad_resolver)
    tasks = [_hi("a", period=5, c_lo=2, c_hi=3)]

    with pytest.raises(ValueError, match="HI 任务"):
        simulate_ordered_taskset(tasks, bad_scenario, RuntimeConfig(end_time=10))


def test_lo_overrun_cancels_only_that_job_and_stays_lo_mode() -> None:
    """LO overrun 只取消该 LO job，不触发 HI 切换。"""

    tasks = [_hi("h", period=8, c_lo=2, c_hi=4), _lo("l", period=8, c_lo=2)]
    scenario = make_single_lo_overrun_scenario("l", release_index=0, actual_cost=3)
    result = simulate_ordered_taskset(tasks, scenario, RuntimeConfig(end_time=10))

    assert result.lo_job_cancellation_count() == 1
    assert result.mode_change_count() == 0
    assert result.final_mode is SystemMode.LO


def test_hi_overrun_uses_runtime_budget_not_task_c_lo() -> None:
    """HI overrun 判定应使用 runtime budget，而不是 task.c_lo。"""

    task = _hi("h", period=10, c_lo=3, c_hi=5)
    tasks = [task]

    no_switch = simulate_ordered_taskset(
        tasks,
        make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        RuntimeConfig(end_time=6),
        budget_state=BudgetState(budgets={"h": 5}, initial_budgets={"h": 5}),
    )
    assert no_switch.mode_change_count() == 0

    switch = simulate_ordered_taskset(
        tasks,
        make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        RuntimeConfig(end_time=6),
        budget_state=BudgetState(budgets={"h": 4}, initial_budgets={"h": 4}),
    )
    assert switch.mode_change_count() == 1
    assert switch.mode_switch is not None
    assert switch.mode_switch.executed_at_switch == 5
    assert switch.mode_switch.budget_at_switch == 4


def test_default_budget_state_matches_old_c_lo_behavior() -> None:
    """不传 budget_state 时应退回旧行为（budget=c_lo）。"""

    tasks = [_hi("h", period=10, c_lo=3, c_hi=5)]
    result = simulate_ordered_taskset(
        tasks,
        make_single_hi_overrun_scenario("h", release_index=0, overrun_to="c_hi"),
        RuntimeConfig(end_time=6),
    )
    assert result.mode_change_count() == 1
    assert result.mode_switch is not None
    assert result.mode_switch.budget_at_switch == 3


def test_runtime_budget_state_is_copied() -> None:
    """仿真器应复制传入 budget_state，避免污染调用方对象。"""

    tasks = [_hi("h", period=10, c_lo=3, c_hi=5)]
    state = BudgetState.from_tasks(tasks)
    before = state.copy()
    simulate_ordered_taskset(
        tasks,
        make_nominal_scenario(),
        RuntimeConfig(end_time=4),
        budget_state=state,
    )
    assert state.budgets == before.budgets
    assert state.initial_budgets == before.initial_budgets
