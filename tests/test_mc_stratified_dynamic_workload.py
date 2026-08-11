"""Tests for the independent MC-Stratified-Dynamic workload family."""

from __future__ import annotations

import random

from amc_py.models import Criticality
from amc_py.workloads.mc_stratified_dynamic import (
    MCStratifiedDynamicWorkloadConfig,
    MCStratifiedDynamicWorkloadProvider,
    _mixture_quantile_budget,
    build_mc_stratified_dynamic_execution_scenario,
    build_mc_stratified_dynamic_workload,
)


def _task_fingerprint(workload):
    return tuple(
        (task.name, task.period, task.deadline, task.c_lo, task.c_hi, task.criticality.value)
        for task in workload.tasks
    )


def test_same_taskset_seed_reproduces_tasks_and_metadata() -> None:
    config = MCStratifiedDynamicWorkloadConfig(seed=42)
    first = build_mc_stratified_dynamic_workload(config)
    second = build_mc_stratified_dynamic_workload(config)

    assert first.tasks == second.tasks
    assert first.task_meta == second.task_meta
    assert repr(first.metadata) == repr(second.metadata)


def test_seed_paired_period_family_switches_on_parity() -> None:
    even = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=10)
    )
    odd = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=11)
    )

    assert even.metadata["period_family"] == "semi_harmonic"
    assert odd.metadata["period_family"] == "log_uniform"
    assert set(meta.period_ms for meta in even.task_meta).issubset({10, 20, 25, 50, 100, 200})
    assert all(10 <= meta.period_ms <= 200 for meta in odd.task_meta)


def test_utilization_cap_and_criticality_invariants() -> None:
    workload = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=7)
    )
    assert abs(sum(meta.generated_u_lo for meta in workload.task_meta) - workload.metadata["total_util_target"]) < 1e-12
    assert all(meta.generated_u_lo <= workload.config.max_task_util + 1e-12 for meta in workload.task_meta)
    assert workload.metadata["total_util_actual"] == sum(
        meta.base_demand_lo / meta.period for meta in workload.task_meta
    )

    for task, meta in zip(workload.tasks, workload.task_meta, strict=True):
        assert task.deadline == task.period
        assert task.c_lo == meta.initial_budget
        assert task.c_hi >= task.c_lo
        assert task.c_hi <= task.deadline
        if task.criticality is Criticality.LO:
            assert task.c_hi == task.c_lo
        else:
            assert meta.base_demand_hi >= meta.base_demand_lo


def test_initial_budget_is_the_execution_mixture_quantile() -> None:
    workload = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=9)
    )
    for task, meta in zip(workload.tasks, workload.task_meta, strict=True):
        expected = _mixture_quantile_budget(
            normal_min=meta.normal_cost_min,
            normal_max=meta.normal_cost_max,
            stress_min=meta.stress_cost_min,
            stress_max=meta.stress_cost_max,
            stress_stationary_prob=meta.stress_stationary_prob,
            quantile=meta.budget_quantile,
        )
        assert task.c_lo == expected


def test_scenario_is_random_access_reproducible_and_taskset_independent() -> None:
    workload = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=12)
    )
    scenario_a = build_mc_stratified_dynamic_execution_scenario(
        workload,
        scenario_seed=101,
    )
    scenario_b = build_mc_stratified_dynamic_execution_scenario(
        workload,
        scenario_seed=101,
    )

    task = workload.tasks[0]
    forward = {index: scenario_a.actual_cost_for(task, index) for index in range(80)}
    reverse = {index: scenario_b.actual_cost_for(task, index) for index in reversed(range(80))}
    assert forward == reverse

    scenario_c = build_mc_stratified_dynamic_execution_scenario(
        workload,
        scenario_seed=102,
    )
    changed = [scenario_c.actual_cost_for(task, index) for index in range(80)]
    assert changed != [forward[index] for index in range(80)]
    assert _task_fingerprint(workload) == _task_fingerprint(
        build_mc_stratified_dynamic_workload(
            MCStratifiedDynamicWorkloadConfig(seed=12)
        )
    )


def test_actual_cost_bounds_and_lo_stress_can_exceed_initial_budget() -> None:
    workload = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=0)
    )
    scenario = build_mc_stratified_dynamic_execution_scenario(workload, scenario_seed=77)
    lo_over_budget = False
    for task in workload.tasks:
        meta = workload.task_meta[workload.tasks.index(task)]
        costs = [scenario.actual_cost_for(task, index) for index in range(200)]
        assert all(meta.normal_cost_min <= cost <= meta.normal_cost_max or meta.stress_cost_min <= cost <= meta.stress_cost_max for cost in costs)
        if task.criticality is Criticality.HI:
            assert max(costs) <= task.c_hi
        else:
            lo_over_budget = lo_over_budget or any(cost > task.c_lo for cost in costs)
    assert lo_over_budget


def test_persistent_stress_has_positive_lag_one_autocorrelation() -> None:
    workload = build_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=14)
    )
    task_index = next(
        index for index, task in enumerate(workload.tasks) if task.criticality is Criticality.HI
    )
    task = workload.tasks[task_index]
    meta = workload.task_meta[task_index]
    scenario = build_mc_stratified_dynamic_execution_scenario(workload, scenario_seed=314)
    costs = [scenario.actual_cost_for(task, index) for index in range(1000)]
    states = [cost >= meta.stress_cost_min for cost in costs]
    mean = sum(states) / len(states)
    numerator = sum((left - mean) * (right - mean) for left, right in zip(states, states[1:]))
    denominator = sum((state - mean) ** 2 for state in states)
    autocorrelation = numerator / denominator

    iid_rng = random.Random(314)
    iid_states = [iid_rng.random() < meta.stress_stationary_prob for _ in states]
    iid_mean = sum(iid_states) / len(iid_states)
    iid_numerator = sum(
        (left - iid_mean) * (right - iid_mean)
        for left, right in zip(iid_states, iid_states[1:])
    )
    iid_denominator = sum((state - iid_mean) ** 2 for state in iid_states)
    iid_autocorrelation = iid_numerator / iid_denominator

    assert autocorrelation > 0.05
    assert autocorrelation > iid_autocorrelation


def test_provider_returns_required_bundle_schema_without_legacy_import() -> None:
    provider = MCStratifiedDynamicWorkloadProvider(
        MCStratifiedDynamicWorkloadConfig(seed=0)
    )
    bundle = provider.build(0)
    assert provider.name == "mc_stratified_dynamic"
    assert bundle.metadata is not None
    assert bundle.metadata["workload_family"] == "mc_stratified_dynamic"
    assert bundle.metadata["schema_version"] == "mc_stratified_dynamic_workload_v1"
    for field in (
        "taskset_seed",
        "scenario_seed",
        "period_family",
        "total_util_target",
        "total_util_actual",
        "criticality_factor_target",
        "criticality_factor_actual",
        "initial_budget_util_total",
        "initial_budget_util_hi",
        "initial_budget_util_lo",
        "task_meta",
    ):
        assert field in bundle.metadata
