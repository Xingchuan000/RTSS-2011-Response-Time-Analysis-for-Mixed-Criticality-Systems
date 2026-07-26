from __future__ import annotations

import pytest
import json

from amc_py.budget_runtime import BudgetUpdate
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.qamc.demand import map_full_cost_to_quality_cost
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
    partition_design_budget,
)
from amc_py.qamc.metrics_support import compute_qamc_metrics
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario
from amc_py.rl.env import AmcBudgetEnv


def _bundle(task: Task):
    return build_qamc_profile_bundle(
        [task],
        taskset_fingerprint=compute_taskset_fingerprint([task]),
        spec=QAmcProfileSpec(),
    )


def test_partition_uses_isolated_to_interference_ratio() -> None:
    assert partition_design_budget(7, isolated_to_interference_ratio=0.5) == (2, 5)
    assert partition_design_budget(1, isolated_to_interference_ratio=0.5) == (1, 0)


def test_demand_mapping_preserves_identity_and_interference() -> None:
    task = Task("L", 20, 20, 9, 9, Criticality.LO)
    profile = _bundle(task).profiles["L"]
    full = map_full_cost_to_quality_cost(full_quality_actual_cost=4, profile=profile, runtime_level=profile.initial_runtime_level)
    low = map_full_cost_to_quality_cost(full_quality_actual_cost=10, profile=profile, runtime_level=profile.threshold_runtime_level)
    assert full.quality_specific_actual_cost == 4
    assert low.observed_interference_component == 7
    assert low.quality_specific_actual_cost <= 10


@pytest.mark.parametrize("cost,overrun", [(6, False), (7, True)])
def test_qamc_strict_budget_plus_one_boundary(cost: int, overrun: bool) -> None:
    task = Task("L", 20, 20, 6, 6, Criticality.LO)
    bundle = _bundle(task)
    scenario = make_table_scenario(actual_costs={("L", 0): cost}, default_hi="c_lo", default_lo="c_lo")
    result = simulate_ordered_taskset_event_driven(
        [task], scenario, RuntimeConfig(end_time=10, semantics=RuntimeSemantics.Q_AMC), qamc_profile_bundle=bundle
    )
    job = result.jobs[0]
    assert job.qamc_stopped_by_overrun is overrun
    assert (job.completion_time is None) is overrun
    assert result.final_mode.value == "LO"


def test_qamc_quality_change_applies_only_to_next_release() -> None:
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = _bundle(task)
    scenario = make_table_scenario(
        actual_costs={("L", 0): 7, ("L", 1): 2}, default_hi="c_lo", default_lo="c_lo"
    )
    result = simulate_ordered_taskset_event_driven(
        [task], scenario, RuntimeConfig(end_time=20, semantics=RuntimeSemantics.Q_AMC), qamc_profile_bundle=bundle
    )
    first, second = result.jobs[:2]
    assert first.qamc_target_runtime_level_at_release == bundle.profiles["L"].initial_runtime_level
    assert second.qamc_target_runtime_level_at_release == bundle.profiles["L"].threshold_runtime_level
    assert first.runtime_budget_at_release == second.runtime_budget_at_release == task.c_lo
    assert result.budget_update_events == []


def test_qamc_rejects_nonvacuity_mutations() -> None:
    with pytest.raises(ValueError, match="QAMC_NONVACUITY"):
        RuntimeConfig(semantics=RuntimeSemantics.Q_AMC, nonvacuity_profile="c3_retroactive_release_budget")


def test_qamc_metrics_report_real_time_at_rank_distribution() -> None:
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = _bundle(task)
    scenario = make_table_scenario(
        actual_costs={("L", 0): 7, ("L", 1): 2},
        default_hi="c_lo",
        default_lo="c_lo",
    )
    result = simulate_ordered_taskset_event_driven(
        [task],
        scenario,
        RuntimeConfig(end_time=20, semantics=RuntimeSemantics.Q_AMC),
        qamc_profile_bundle=bundle,
    )
    metrics = compute_qamc_metrics(result, bundle)
    occupancy = json.loads(metrics.task_time_at_raw_rank_ratio_json)
    assert set(occupancy) == {"2", "4"}
    assert sum(occupancy.values()) == pytest.approx(1.0)
    assert occupancy["4"] == pytest.approx(7 / 20)


def test_would_overrun_design_uses_release_demand_not_early_stop_progress() -> None:
    task = Task("L", 20, 20, 6, 6, Criticality.LO)
    bundle = _bundle(task)
    scenario = make_table_scenario(
        actual_costs={("L", 0): 7},
        default_hi="c_lo",
        default_lo="c_lo",
    )
    result = simulate_ordered_taskset_event_driven(
        [task],
        scenario,
        RuntimeConfig(end_time=10, semantics=RuntimeSemantics.Q_AMC),
        budget_updates=[BudgetUpdate(time=0, updates={"L": 4})],
        qamc_profile_bundle=bundle,
    )
    job = result.jobs[0]
    assert job.executed_time == 5
    assert job.actual_cost == 7
    assert job.qamc_would_overrun_design_c_lo is True
    metrics = compute_qamc_metrics(result, bundle)
    assert metrics.offline_budget_update_event_count == 1
    assert metrics.offline_budget_update_task_count == 1
    assert metrics.unspecified_budget_update_event_count == 0


def test_qamc_hi_overrun_keeps_triggering_hi_job_and_uses_idle_recovery() -> None:
    task = Task("H", 20, 20, 2, 4, Criticality.HI)
    bundle = build_qamc_profile_bundle(
        [task],
        taskset_fingerprint=compute_taskset_fingerprint([task]),
        spec=QAmcProfileSpec(),
    )
    result = simulate_ordered_taskset_event_driven(
        [task],
        make_table_scenario(
            actual_costs={("H", 0): 3},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        RuntimeConfig(end_time=10, semantics=RuntimeSemantics.Q_AMC),
        qamc_profile_bundle=bundle,
    )
    assert len(result.mode_switches) == 1
    assert result.jobs[0].completion_time == 3
    assert result.jobs[0].dropped is False
    assert result.qamc_quality_changes == []
    assert len(result.mode_recoveries) == 1
    assert result.final_mode.value == "LO"


def test_min_quality_fallback_does_not_restore_quality_or_double_count_trigger() -> None:
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = _bundle(task)
    scenario = make_table_scenario(
        actual_costs={("L", 0): 7, ("L", 1): 8, ("L", 2): 2},
        default_hi="c_lo",
        default_lo="c_lo",
    )
    result = simulate_ordered_taskset_event_driven(
        [task],
        scenario,
        RuntimeConfig(end_time=30, semantics=RuntimeSemantics.Q_AMC),
        qamc_profile_bundle=bundle,
    )
    assert len(result.qamc_quality_changes) == 1
    assert len(result.qamc_min_quality_exhaustions) == 1
    trigger_losses = [
        loss
        for loss in result.lo_job_losses
        if loss.task == "L" and loss.release_index == 1
    ]
    assert len(trigger_losses) == 1
    assert trigger_losses[0].reason == "lo_loss_qamc_min_quality_exhausted"
    assert result.jobs[2].qamc_target_runtime_level_at_release == 0


def test_qamc_fixed_wmax_floor_applies_even_when_reference_ratio_is_zero() -> None:
    task = Task("L", 20, 20, 9, 9, Criticality.LO)
    bundle = _bundle(task)
    env = AmcBudgetEnv(
        ordered_tasks=[task],
        scenario=make_table_scenario(
            actual_costs={},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        runtime_config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.Q_AMC),
        agent_period=1,
        action_space="single",
        budget_decrease_ratio=0.9,
        budget_floor_ratio=0.0,
        check_safety=True,
        qamc_profile_bundle=bundle,
    )
    env.reset()
    decrease = next(
        action for action in env._actions if action.decrease_tasks == ("L",)
    )
    evaluation = env.evaluate_budget_candidate(action=decrease)
    assert evaluation.accepted is False
    assert evaluation.reject_reason == "budget_floor_violation:L"
