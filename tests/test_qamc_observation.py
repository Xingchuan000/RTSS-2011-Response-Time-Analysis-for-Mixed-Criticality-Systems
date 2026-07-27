from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import build_qamc_profile_bundle, compute_taskset_fingerprint
from amc_py.rl.env import AmcBudgetEnv
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


def _qamc_env(mode: str) -> AmcBudgetEnv:
    tasks = [
        Task("L", 20, 20, 9, 9, Criticality.LO),
        Task("H", 20, 20, 3, 4, Criticality.HI),
    ]
    bundle = build_qamc_profile_bundle(
        tasks,
        taskset_fingerprint=compute_taskset_fingerprint(tasks),
        spec=QAmcProfileSpec(),
    )
    return AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=make_table_scenario(actual_costs={}, default_hi="c_lo", default_lo="c_lo"),
        runtime_config=RuntimeConfig(end_time=40, semantics=RuntimeSemantics.Q_AMC),
        agent_period=10,
        action_space="single",
        qamc_profile_bundle=bundle,
        feature_config=FeatureConfig(observation_mode=mode),
    )


@pytest.mark.parametrize(
    ("mode", "per_task_dim"),
    [("v14_qamc_quality_11d", 11), ("v14_qamc_full_12d", 12)],
)
def test_qamc_observation_schema_and_range(mode: str, per_task_dim: int) -> None:
    env = _qamc_env(mode)
    observation = env.reset()
    assert len(observation.state_vector) == per_task_dim * len(env.ordered_tasks) + 8
    assert len(observation.state_vector) == FeatureConfig(observation_mode=mode).expected_state_dim(
        len(env.ordered_tasks)
    )
    assert all(0.0 <= value <= 1.0 for value in observation.state_vector)


def test_qamc_observation_fails_closed_for_non_qamc_runtime() -> None:
    tasks = [
        Task("L", 20, 20, 9, 9, Criticality.LO),
        Task("H", 20, 20, 3, 4, Criticality.HI),
    ]
    with pytest.raises(
        ValueError,
        match="QAMC_OBSERVATION_REQUIRES_QAMC_RUNTIME",
    ):
        AmcBudgetEnv(
            ordered_tasks=tasks,
            scenario=make_table_scenario(
                actual_costs={},
                default_hi="c_lo",
                default_lo="c_lo",
            ),
            runtime_config=RuntimeConfig(
                end_time=40,
                semantics=RuntimeSemantics.AMC_PLUS,
            ),
            agent_period=10,
            action_space="single",
            feature_config=FeatureConfig(
                observation_mode="v14_qamc_full_12d"
            ),
        )


def test_o2_preserves_every_o0_feature_exactly() -> None:
    o0 = _qamc_env("v11_full_10d").reset()
    o2_env = _qamc_env("v14_qamc_full_12d")
    o2 = o2_env.reset()
    task_count = len(o2_env.ordered_tasks)

    for task_index in range(task_count):
        o0_start = task_index * 10
        o2_start = task_index * 12
        assert o2.state_vector[o2_start : o2_start + 10] == (
            o0.state_vector[o0_start : o0_start + 10]
        )
    assert o2.state_vector[-8:] == o0.state_vector[-8:]


def test_o2_feature_name_order_matches_state_layout() -> None:
    env = _qamc_env("v14_qamc_full_12d")
    names = env.get_observation_feature_names()
    assert names[10].endswith("qamc_target_quality_normalized")
    assert names[11].endswith("qamc_target_demand_ratio")
    assert names[22].endswith("qamc_target_quality_normalized")
    assert names[-8:] == (
        "global.total_budget_util",
        "global.hi_budget_util",
        "global.lo_budget_util",
        "global.recent_mode_change_rate",
        "global.recent_lo_cancel_rate",
        "global.recent_hi_overrun_rate",
        "global.recent_lo_overrun_rate",
        "global.safety_margin_min",
    )


def test_quality_transition_is_visible_only_for_next_release() -> None:
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = build_qamc_profile_bundle(
        [task],
        taskset_fingerprint=compute_taskset_fingerprint([task]),
        spec=QAmcProfileSpec(),
    )
    env = AmcBudgetEnv(
        ordered_tasks=[task],
        scenario=make_table_scenario(
            actual_costs={("L", 0): 7, ("L", 1): 2},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        runtime_config=RuntimeConfig(
            end_time=20,
            semantics=RuntimeSemantics.Q_AMC,
        ),
        agent_period=10,
        action_space="single",
        include_explicit_noop=True,
        qamc_profile_bundle=bundle,
        feature_config=FeatureConfig(
            observation_mode="v14_qamc_full_12d"
        ),
    )
    before = env.reset()
    noop_action = next(action for action in env.actions if action.is_noop)
    after = env.step(noop_action.action_id).observation

    assert after.state_vector[10] < before.state_vector[10]
    assert after.state_vector[11] < before.state_vector[11]
    jobs = env.runtime_result.jobs_of("L")
    assert jobs[0].qamc_target_runtime_level_at_release == (
        bundle.profiles["L"].initial_runtime_level
    )
    assert jobs[1].qamc_target_runtime_level_at_release < (
        jobs[0].qamc_target_runtime_level_at_release
    )
