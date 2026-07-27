"""LO service/JNE 增量 tracker 测试。"""

from __future__ import annotations

import math

import pytest

from amc_py.metrics import compute_lo_quality_weighted_metrics
from amc_py.models import Criticality, Task
from amc_py.rl.agents import NoOpBudgetAgent
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.lo_service_reward import LoServiceRewardTracker
from amc_py.rl.runtime_wrapper import AgentRuntimeConfig, simulate_ordered_taskset_with_agent
from amc_py.runtime_models import DeadlineMiss, Job, SimulationResult, SystemMode
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _task(name: str = "LO1") -> Task:
    return Task(name=name, period=20, deadline=20, c_lo=4, c_hi=4, criticality=Criticality.LO)


def _job(
    task: Task,
    release_index: int,
    *,
    service: float = 1.0,
    completion_time: int | None = None,
    dropped: bool = False,
) -> Job:
    release_time = release_index * task.period
    return Job(
        task=task,
        release_index=release_index,
        release_time=release_time,
        absolute_deadline=release_time + task.deadline,
        actual_cost=task.c_lo,
        completion_time=completion_time,
        dropped=dropped,
        service_quality_if_completed=service,
    )


def test_tracker_full_completion_and_drop() -> None:
    task = _task()
    result = SimulationResult()
    tracker = LoServiceRewardTracker()

    result.jobs.extend([_job(task, 0, completion_time=4), _job(task, 1, dropped=True)])
    delta = tracker.consume(result, terminal=False)

    assert delta.released_jobs == 2
    assert delta.finalized_jobs == 2
    assert delta.service_quality_sum == 1.0
    assert delta.equiv_jne == 1.0
    assert delta.zero_service_jobs == 1
    assert delta.partial_service_jobs == 0


@pytest.mark.parametrize("service", [0.5, 0.75])
def test_tracker_counts_partial_quality_completion(service: float) -> None:
    task = _task()
    result = SimulationResult(jobs=[_job(task, 0, service=service, completion_time=4)])

    delta = LoServiceRewardTracker().consume(result, terminal=False)

    assert delta.finalized_jobs == 1
    assert delta.service_quality_sum == service
    assert delta.equiv_jne == 1.0 - service
    assert delta.partial_service_jobs == 1


def test_tracker_deadline_miss_late_completion_is_zero() -> None:
    task = _task()
    job = _job(task, 0)
    result = SimulationResult(jobs=[job])
    tracker = LoServiceRewardTracker()

    first = tracker.consume(result, terminal=False)
    result.deadline_misses.append(
        DeadlineMiss(
            task=task.name,
            release_index=0,
            release_time=0,
            absolute_deadline=task.deadline,
            mode_at_miss=SystemMode.LO,
            executed_at_miss=task.c_lo,
        )
    )
    job.completion_time = 10
    second = tracker.consume(result, terminal=False)

    assert first.finalized_jobs == 0
    assert second.finalized_jobs == 1
    assert second.service_quality_sum == 0.0


def test_tracker_does_not_double_account_and_terminal_flushes_active_job() -> None:
    task = _task()
    active = _job(task, 0)
    result = SimulationResult(jobs=[active])
    tracker = LoServiceRewardTracker()

    first = tracker.consume(result, terminal=False)
    second = tracker.consume(result, terminal=False)
    terminal = tracker.consume(result, terminal=True)
    repeated_terminal = tracker.consume(result, terminal=True)

    assert first.finalized_jobs == 0
    assert second.finalized_jobs == 0
    assert terminal.finalized_jobs == 1
    assert terminal.service_quality_sum == 0.0
    assert terminal.equiv_jne == 1.0
    assert repeated_terminal.finalized_jobs == 0


def test_tracker_prime_has_no_reward_for_time_zero_release() -> None:
    task = _task()
    job = _job(task, 0)
    result = SimulationResult(jobs=[job])
    tracker = LoServiceRewardTracker()
    tracker.prime(result)

    delta = tracker.consume(result, terminal=False)
    assert delta.released_jobs == 0
    assert delta.finalized_jobs == 0
    assert tracker.cumulative_released_jobs == 1

    job.completion_time = 4
    completed = tracker.consume(result, terminal=False)
    assert completed.finalized_jobs == 1
    assert completed.service_quality_sum == 1.0


@pytest.mark.parametrize(
    ("semantic_name", "service"),
    [
        ("AMC", 1.0),
        ("AMC_RA", 1.0),
        ("AMC_RH", 1.0),
        ("C_AMC_SEM", 0.5),
        ("Q_AMC", 0.75),
    ],
)
def test_tracker_conserves_metrics_across_amc_family(semantic_name: str, service: float) -> None:
    del semantic_name
    task = _task()
    result = SimulationResult(
        jobs=[
            _job(task, 0, service=service, completion_time=4),
            _job(task, 1, service=1.0, completion_time=24),
            _job(task, 2, dropped=True),
        ]
    )
    tracker = LoServiceRewardTracker()
    delta = tracker.consume(result, terminal=True)
    metrics = compute_lo_quality_weighted_metrics(result)

    assert math.isclose(delta.service_quality_sum, metrics.lo_total_service_sum, abs_tol=1e-9)
    assert math.isclose(delta.equiv_jne, metrics.lo_equiv_jne, abs_tol=1e-9)


def test_env_and_runtime_wrapper_reward_parity_for_fixed_noop_sequence() -> None:
    tasks = [
        Task("h", 20, 20, 2, 3, Criticality.HI),
        Task("l", 20, 20, 2, 2, Criticality.LO),
    ]
    runtime_config = RuntimeConfig(end_time=40, semantics=RuntimeSemantics.AMC_PLUS)
    reward_mode = "interval_lo_equiv_jne_v1_balanced"
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        runtime_config=runtime_config,
        agent_period=10,
        check_safety=False,
        action_space="single",
        reward_mode=reward_mode,
    )
    env.reset(seed=0)
    env_total_reward = 0.0
    while True:
        step = env.step(None)
        env_total_reward += step.reward
        if step.done:
            break

    wrapper_result = simulate_ordered_taskset_with_agent(
        ordered_tasks=tasks,
        scenario=make_nominal_scenario(),
        agent=NoOpBudgetAgent(),
        runtime_config=runtime_config,
        agent_config=AgentRuntimeConfig(
            agent_period=10,
            end_time=40,
            reward_mode=reward_mode,
        ),
    )

    assert math.isclose(env_total_reward, wrapper_result.total_reward, abs_tol=1e-9)
