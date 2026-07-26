from __future__ import annotations

import pytest

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.qamc.demand import map_full_cost_to_quality_cost
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import build_qamc_profile_bundle, partition_design_budget
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


def _bundle(task: Task):
    return build_qamc_profile_bundle([task], taskset_fingerprint="test-taskset", spec=QAmcProfileSpec())


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
