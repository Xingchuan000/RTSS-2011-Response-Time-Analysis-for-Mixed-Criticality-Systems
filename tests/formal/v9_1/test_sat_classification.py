from formal_toolchain.v9_1.counterexample_replay import replay_concrete_counterexample
from formal_toolchain.v9_1.safe_prefix_reachability import check_witness_boot_safe_prefix
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound


def _model():
    return BoundModel((TaskBound("hi", 0, 3, 3, "HI", 1, 2, 1, 1, 2),), 2)


def test_sat_without_boot_reachability_remains_unresolved():
    result = check_witness_boot_safe_prefix({}, _model())
    assert result.status == "UNRESOLVED"
    assert result.code == "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE"


def test_replay_requires_independent_concrete_replay():
    witness = {"z0_reachable_from_boot": True, "z0_no_prior_hi_miss": True,
               "z0_budgets": {"hi": 1}, "safe_prefix_trace": [{"time": 0}],
               "exact_periodic_demand_prefix": {"hi:0": 2}}
    result = replay_concrete_counterexample(witness, _model(), target_task="hi")
    assert result.status == "UNRESOLVED"
    assert result.code == "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE"
