from __future__ import annotations

from types import SimpleNamespace

from formal_toolchain.adapters.mc_stratified_dynamic_target import _actual_demand_metadata
from formal_toolchain.v9_1.bindings import _environment_domain


class _Crit:
    def __init__(self, value: str) -> None:
        self.value = value


def test_dynamic_lo_raw_demand_can_exceed_c_lo() -> None:
    lo_task = SimpleNamespace(name="lo", c_lo=10, c_hi=10)
    hi_task = SimpleNamespace(name="hi", c_lo=5, c_hi=12)
    meta = (
        SimpleNamespace(name="lo", criticality=_Crit("LO"), normal_cost_min=4,
                        normal_cost_max=9, stress_cost_min=8, stress_cost_max=14),
        SimpleNamespace(name="hi", criticality=_Crit("HI"), normal_cost_min=2,
                        normal_cost_max=5, stress_cost_min=9, stress_cost_max=15),
    )
    bundle = SimpleNamespace(
        ordered_tasks=(lo_task, hi_task), metadata={"task_meta": meta}
    )
    bounds = _actual_demand_metadata(bundle)
    assert bounds["lo"]["max"] == 14
    # Production HI resolver clamps the scenario to C_HI.
    assert bounds["hi"]["max"] == 12


def test_environment_domain_uses_frozen_raw_bounds_not_c_lo() -> None:
    taskset = {"ordered_tasks": [{
        "name": "lo", "criticality": "LO", "code_c_lo": 10, "code_c_hi": 10,
        "period": 100, "deadline": 100,
    }]}
    domain = _environment_domain(taskset, {
        "lo": {"min": 4, "max": 14, "kind": "RAW_EXECUTION_COST_INTEGER_ENVELOPE"}
    })
    row = domain["tasks"][0]["actual_demand_domain"]
    assert row["min"] == 4
    assert row["max"] == 14
