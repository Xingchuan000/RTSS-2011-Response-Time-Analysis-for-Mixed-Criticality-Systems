from __future__ import annotations

import pytest

from amc_py.dqn.experiment import (
    build_env_from_experiment_config,
    build_mc_stratified_dynamic_experiment_config,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.mc_stratified_dynamic_target import build_target


FEATURES = {
    "observation_mode": "v11_full_10d", "ema_alpha": 0.2,
    "overrun_ema_alpha": 0.1, "history_k": 8, "event_window": 10,
    "max_cost_weight": 0.7, "risk_max_scale": 3.0,
    "include_safety_margin": True,
}
RUNTIME = {
    "runtime_semantics": "C_AMC_SEM", "c_amc_sem_xf": 0.5,
    "end_time": 8_000_000, "agent_period": 25_000, "action_space": "single",
    "include_explicit_noop": True, "budget_increase_ratio": 0.02,
    "budget_decrease_ratio": 0.02, "budget_floor_ratio": 0.9,
    "forbid_decreasing_hi_budgets": True, "mask_detail_mode": "full",
    "enable_deploy_cap_mask": True, "deploy_cap_mask_ratio": 4.0,
    "deploy_cap_mask_criticality": "lo", "capture_trace": True,
    "capture_debug_events": False, "processor_overhead": 0,
}


def _workload(seed: int) -> dict[str, object]:
    return {"num_tasks": 12, "hi_ratio": 0.5, "period_family": "seed_paired",
            "period_scale": 500, "fixed_taskset_seed": seed,
            "scenario_seed_offset": 100000, "require_schedulable": True,
            "check_safety": True}


@pytest.mark.parametrize("seed", [1775, 1776])
def test_dynamic_target_is_deterministic_and_matches_artifact_schema(seed: int) -> None:
    workload = _workload(seed)
    config = build_mc_stratified_dynamic_experiment_config(**workload)
    env = build_env_from_experiment_config(
        config, seed=seed, end_time=8_000_000, agent_period=25_000,
        semantics=RuntimeSemantics.C_AMC_SEM, reward_mode="mendes",
        action_space="single", include_explicit_noop=True,
        budget_increase_ratio=0.02, budget_decrease_ratio=0.02,
        budget_floor_ratio=0.9, forbid_decreasing_hi_budgets=True,
        mask_detail_mode="full", enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0, deploy_cap_mask_criticality="lo",
        capture_trace=True, capture_debug_events=False, c_amc_sem_xf=0.5,
        feature_config=FeatureConfig(**FEATURES),
    )
    target = build_target(
        seed=seed, workload_args=workload, runtime_args=RUNTIME,
        feature_config=FEATURES,
        expected_feature_names=env.get_observation_feature_names(),
        expected_action_definitions=env.get_action_definitions(),
    )
    assert target.provenance["workload_family"] == "mc_stratified_dynamic"
    assert target.provenance["taskset_seed"] == seed
    assert len(target.ordered_tasks) == 12
    assert len(target.feature_names) == 128
    assert len(target.action_definitions) == 25
    assert target.action_definitions[24]["is_noop"] is True
    assert target.runtime_adapter.export_mask_contract()["explicit_noop_action_ids"] == [24]


def test_dynamic_target_rejects_seed_mismatch() -> None:
    with pytest.raises(ValueError, match="DYNAMIC_FIXED_TASKSET_SEED_MISMATCH"):
        build_target(
            seed=1775, workload_args=_workload(1776), runtime_args=RUNTIME,
            feature_config=FEATURES, expected_feature_names=[],
            expected_action_definitions=[],
        )
