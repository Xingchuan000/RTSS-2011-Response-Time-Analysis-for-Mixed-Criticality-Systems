"""事件驱动 runtime 分段推进回归测试。"""

from __future__ import annotations

from amc_py.dqn import resolve_experiment_bundle
from amc_py.dqn.experiment import build_rtss11_experiment_config
from amc_py.event_runtime import EventRuntimeEngine, simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.agents import RandomBudgetAgent
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _single_lo_task() -> Task:
    """构造一个中间长时间没有事件的单任务场景。"""

    return Task("lo", period=200, deadline=200, c_lo=150, c_hi=150, criticality=Criticality.LO)


def test_run_until_without_boundary_event_settles_running_job_progress() -> None:
    """推进到无事件边界时，也必须结算 running job 的执行量。"""

    task = _single_lo_task()
    engine = EventRuntimeEngine.build(
        ordered_tasks=[task],
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_PLUS),
    )

    engine.run_until(100, include_boundary=True)

    assert engine.state.running_job is not None
    assert engine.state.running_job.task.name == "lo"
    assert engine.state.running_job.executed_time == 100


def test_apply_budget_updates_does_not_drop_executed_time_of_running_job() -> None:
    """在决策边界应用预算更新时，不应丢失当前 job 已执行时间。"""

    task = _single_lo_task()
    engine = EventRuntimeEngine.build(
        ordered_tasks=[task],
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=200, semantics=RuntimeSemantics.AMC_PLUS),
    )

    engine.run_until(100, include_boundary=True)
    assert engine.state.running_job is not None
    before = engine.state.running_job.executed_time

    engine.apply_budget_updates({"lo": 140})

    assert engine.state.running_job is not None
    assert engine.state.running_job.executed_time == before


def test_segmented_run_matches_one_shot_run_for_trace_and_misses() -> None:
    """分段推进后的 miss / 模式切换 / 局部取消应与一次性运行一致。"""

    tasks = [
        Task("h", 10, 10, 2, 3, Criticality.HI),
        Task("l", 15, 15, 2, 2, Criticality.LO),
    ]
    cfg = RuntimeConfig(end_time=120, semantics=RuntimeSemantics.AMC_PLUS)
    scenario = make_nominal_scenario()

    one_shot = simulate_ordered_taskset_event_driven(tasks, scenario, cfg)

    engine = EventRuntimeEngine.build(ordered_tasks=tasks, scenario=scenario, config=cfg)
    for t in range(0, 120, 20):
        engine.run_until(t, include_boundary=True)
    engine.run_until(120, include_boundary=True)
    segmented = engine.finish()

    assert len(segmented.deadline_misses) == len(one_shot.deadline_misses)
    assert segmented.mode_change_count() == one_shot.mode_change_count()
    assert segmented.lo_job_cancellation_count() == one_shot.lo_job_cancellation_count()
    assert len(segmented.trace) == len(one_shot.trace)


def test_rtss11_regression_seed_keeps_zero_deadline_miss_under_safety() -> None:
    """已知问题 seed 在修复后不应再出现 deadline miss。"""

    config = build_rtss11_experiment_config(
        total_util=0.55,
        num_tasks=20,
        cf=2.0,
        cp=0.5,
        require_schedulable=True,
    )
    bundle = resolve_experiment_bundle(config, seed=112)
    actions = build_budget_action_space(bundle.ordered_tasks)
    result = simulate_ordered_taskset_with_agent(
        ordered_tasks=bundle.ordered_tasks,
        scenario=bundle.scenario,
        agent=RandomBudgetAgent(actions=actions, seed=112),
        runtime_config=RuntimeConfig(end_time=10000, semantics=RuntimeSemantics.AMC_PLUS),
        agent_config=AgentRuntimeConfig(agent_period=1000, end_time=10000, check_safety=True),
        bounds=bundle.normalization_bounds,
    )

    assert len(result.runtime_result.deadline_misses) == 0
