from __future__ import annotations

from formal_toolchain.v9_2.symbolic_state import TaskBound


def test_lo_history_domain_covers_raw_demand_above_c_lo() -> None:
    task = TaskBound(
        name="lo", priority=0, period=100, deadline=100, criticality="LO",
        c_lo=10, c_hi=10, initial_budget=10,
        actual_demand_min=4, actual_demand_max=14,
    )
    assert task.history_cost_upper == 14


def test_history_domain_still_covers_initial_c_lo_when_raw_support_is_lower() -> None:
    task = TaskBound(
        name="lo", priority=0, period=100, deadline=100, criticality="LO",
        c_lo=10, c_hi=10, initial_budget=10,
        actual_demand_min=2, actual_demand_max=8,
    )
    assert task.history_cost_upper == 10
