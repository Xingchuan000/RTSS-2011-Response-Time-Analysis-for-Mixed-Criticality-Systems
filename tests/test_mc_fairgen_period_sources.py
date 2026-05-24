from __future__ import annotations

import pytest

from amc_py.workloads.mc_fairgen import MCFairGenWorkloadConfig, build_mc_fairgen_workload


def _build(period_source: str):
    cfg = MCFairGenWorkloadConfig(
        seed=0,
        num_tasks=12,
        hi_ratio=0.5,
        period_source=period_source,  # type: ignore[arg-type]
        period_scale=100,
        u_hi_lo_min=0.20,
        u_hi_lo_max=0.35,
        u_hi_hi_min=0.45,
        u_hi_hi_max=0.70,
        u_lo_lo_min=0.25,
        u_lo_lo_max=0.45,
        hi_budget_rho_min=0.55,
        hi_budget_rho_max=0.75,
        lo_budget_rho_min=0.20,
        lo_budget_rho_max=0.40,
        hi_overrun_prob=0.08,
        lo_overrun_prob=0.12,
        hi_overrun_factor_min=1.02,
        hi_overrun_factor_max=1.25,
        lo_overrun_factor_min=1.02,
        lo_overrun_factor_max=1.25,
    )
    return build_mc_fairgen_workload(cfg)


def test_automotive_can_generate_12_tasks() -> None:
    assert len(_build("automotive").tasks) == 12


def test_controlled_sparse_can_generate_12_tasks() -> None:
    assert len(_build("controlled_sparse").tasks) == 12


def test_controlled_medium_periods_from_expected_set() -> None:
    workload = _build("controlled_medium")
    allowed = {10, 20, 50, 100, 200}
    period_scale = 100
    assert {task.period // period_scale for task in workload.tasks}.issubset(allowed)


def test_invalid_period_source_raises() -> None:
    with pytest.raises(ValueError):
        _build("invalid_source")
