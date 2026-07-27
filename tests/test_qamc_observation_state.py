from __future__ import annotations

import pytest

from amc_py.models import Criticality, Task
from amc_py.qamc.observation_state import build_qamc_task_observation_state
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
)


def test_demand_ratio_uses_total_design_demand_not_isolated_ratio() -> None:
    tasks = [
        Task("L", 20, 20, 9, 9, Criticality.LO),
        Task("H", 20, 20, 3, 4, Criticality.HI),
    ]
    bundle = build_qamc_profile_bundle(
        tasks,
        taskset_fingerprint=compute_taskset_fingerprint(tasks),
        spec=QAmcProfileSpec(),
    )
    state = build_qamc_task_observation_state(
        ordered_tasks=tasks,
        profile_bundle=bundle,
        current_level_by_task={"L": 0},
    )
    assert state["L"].target_demand_ratio == pytest.approx(7.0 / 9.0)
    assert state["L"].target_demand_ratio != pytest.approx(1.0 / 3.0)
    assert state["H"].target_quality_normalized == 1.0
    assert state["H"].target_demand_ratio == 1.0
