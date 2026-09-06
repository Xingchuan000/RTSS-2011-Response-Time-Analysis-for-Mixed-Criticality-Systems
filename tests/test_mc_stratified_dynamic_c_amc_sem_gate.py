"""Integration checks for C-AMC-sem admission in mc_stratified_dynamic."""

from __future__ import annotations

import pytest

from amc_py.experiments import evaluate_taskset
from amc_py.workloads.mc_stratified_dynamic import (
    MCStratifiedDynamicWorkloadConfig,
    build_mc_stratified_dynamic_workload,
    generate_mc_stratified_dynamic_workload,
)


def test_existing_seed_715_passes_legacy_gate_but_fails_c_amc_sem_dm() -> None:
    raw = generate_mc_stratified_dynamic_workload(
        MCStratifiedDynamicWorkloadConfig(seed=715, require_schedulable=False)
    )
    legacy = evaluate_taskset(list(raw.tasks), method="amc_rtb", priority_policy="dm")
    csem_dm = evaluate_taskset(
        list(raw.tasks),
        method="c_amc_sem",
        priority_policy="dm",
        c_amc_sem_xf=0.5,
    )

    assert legacy.schedulable is True
    assert csem_dm.schedulable is False


def test_workload_gate_can_reject_seed_with_c_amc_sem_dm() -> None:
    config = MCStratifiedDynamicWorkloadConfig(
        seed=715,
        require_schedulable=True,
        sched_method="c_amc_sem",
        priority_policy="dm",
        c_amc_sem_xf=0.5,
    )
    with pytest.raises(RuntimeError, match="not schedulable"):
        build_mc_stratified_dynamic_workload(config)


def test_c_amc_sem_xf_is_validated_by_workload_config() -> None:
    with pytest.raises(ValueError, match="c_amc_sem_xf"):
        MCStratifiedDynamicWorkloadConfig(c_amc_sem_xf=0.0)
