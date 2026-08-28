from __future__ import annotations

from types import SimpleNamespace

from formal_toolchain.semantics.release_fixed_demand import derive_release_fixed_demand
from formal_toolchain.v9_2.kernel import (
    effective_demand_hi, effective_demand_lo_degraded, effective_demand_lo_primary,
)


class _Crit:
    def __init__(self, value: str) -> None:
        self.value = value


def _job(*, criticality: str, raw: int, budget: int | None, degraded: bool = False):
    task = SimpleNamespace(name=f"{criticality.lower()}_task", criticality=_Crit(criticality))
    return SimpleNamespace(
        task=task,
        criticality=criticality,
        actual_cost=raw,
        original_actual_cost=raw,
        runtime_budget_at_release=budget,
        is_degraded=degraded,
    )


def test_primary_lo_kernel_matches_frozen_release_fixed_semantics() -> None:
    for raw, budget in ((14, 10), (7, 10), (1, 3)):
        concrete = derive_release_fixed_demand(_job(
            criticality="LO", raw=raw, budget=budget, degraded=False,
        ))
        assert concrete.removal_demand == effective_demand_lo_primary(raw, budget)


def test_degraded_lo_kernel_matches_frozen_release_fixed_semantics() -> None:
    for raw, degraded_cost in ((14, 5), (4, 5), (1, 1)):
        concrete = derive_release_fixed_demand(
            _job(criticality="LO", raw=raw, budget=degraded_cost, degraded=True),
            task_reference={"degraded_cost": degraded_cost},
        )
        assert concrete.removal_demand == effective_demand_lo_degraded(raw, degraded_cost)


def test_hi_kernel_matches_frozen_release_fixed_semantics() -> None:
    for raw in (1, 5, 12):
        concrete = derive_release_fixed_demand(_job(
            criticality="HI", raw=raw, budget=None,
        ))
        assert concrete.removal_demand == effective_demand_hi(raw)
