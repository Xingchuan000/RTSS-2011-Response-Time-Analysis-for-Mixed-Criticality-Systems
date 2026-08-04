from nonvacuity_lab.hout.normalizer import normalize_event
from nonvacuity_lab.hout.aggregate import compare_paired_hout
from nonvacuity_lab.runners.run_plan import RunKind, build_run_plan


def test_run_plan_keeps_gradient_out_of_single_mutation_path():
    assert build_run_plan({"mutation_id": "D1", "mutation_class": "ENVELOPE_GRADIENT"}).run_kind is RunKind.ENVELOPE_GRADIENT


def test_hout_pair_requires_same_scenario_and_decision_keys():
    base = [normalize_event({"time": 0, "scenario_seed": 101555, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": False})]
    mutated = [normalize_event({"time": 0, "scenario_seed": 101555, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": True})]
    result = compare_paired_hout(base, mutated)
    assert result["delta"]["implicit_noop_rate"] == 1
