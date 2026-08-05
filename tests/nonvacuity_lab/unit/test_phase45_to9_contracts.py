from pathlib import Path

from nonvacuity_lab.hout.normalizer import normalize_event
from nonvacuity_lab.hout.aggregate import compare_paired_hout
from nonvacuity_lab.runners.run_plan import RunKind, build_run_plan
from nonvacuity_lab.mutators.ast_guard_delete import MutationError, delete_reject_return
from amc_py.rl.actions import round_budget_product
from formal_toolchain.semantics.frozen_c_amc_sem_action_runtime import (
    round_budget_product as frozen_round_budget_product,
)
from nonvacuity_lab.mutators.catalog.selection_mutations import build_selection_catalog
from nonvacuity_lab.mutators.retroactive_release_budget import insert_retroactive_release_rewrite
from amc_py.evaluation.ordinary_tree_hout import build_ordinary_tree_hout_runner
import types
import sys


def test_run_plan_keeps_gradient_out_of_single_mutation_path():
    assert build_run_plan({"mutation_id": "D1", "mutation_class": "ENVELOPE_GRADIENT"}).run_kind is RunKind.ENVELOPE_GRADIENT


def test_hout_pair_requires_same_scenario_and_decision_keys():
    base = [normalize_event({"time": 0, "scenario_seed": 101555, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": False})]
    mutated = [normalize_event({"time": 0, "scenario_seed": 101555, "controller_decision_index": 0, "all_invalid": False, "implicit_noop": True})]
    result = compare_paired_hout(base, mutated)
    assert result["delta"]["implicit_noop_rate"] == 1


def test_deployed_and_frozen_rounding_are_coherent():
    for mode in ("ceil_floor", "nearest"):
        for direction in ("increase", "decrease"):
            assert round_budget_product(10.5, direction=direction, mode=mode) == frozen_round_budget_product(10.5, direction=direction, mode=mode)


def test_guard_selector_requires_one_current_source_match():
    source = """
def evaluate(x):
    if x == 1:
        return {\"accepted\": False, \"reason\": \"floor_reject\"}
    return {\"accepted\": True}
"""
    updated = delete_reject_return(source, "floor_reject")
    assert "floor_reject" not in updated
    try:
        delete_reject_return(source, "missing_reason")
    except MutationError as exc:
        assert "GUARD_SELECTOR_NOT_UNIQUE" in str(exc)
    else:
        raise AssertionError("unmatched selector must fail closed")


def test_selection_catalog_binds_current_integer_policy_source():
    catalog = build_selection_catalog(Path.cwd())
    assert set(catalog) == {"B1", "B2", "B3"}
    assert all(item[0]["before_ast_hash"] for item in catalog.values())
    assert all("nonvacuity_selection_semantics" not in item[0]["before_snippet"] for item in catalog.values())


def test_c3_inserts_once_in_each_runtime_mirror():
    source = """
class Runtime:
    def apply_budget_updates(self, updates):
        self.budget_state.apply_updates(updates)
"""
    updated, count = insert_retroactive_release_rewrite(source)
    assert count == 1
    assert "runtime_budget_at_release" in updated


def test_ordinary_hout_factory_runs_each_scenario(monkeypatch, tmp_path: Path):
    module = types.ModuleType("test_hout_callback")
    module.run = lambda **kwargs: ({"scenario_seed": kwargs["scenario_seed"], "score": 2}, [])
    monkeypatch.setitem(sys.modules, "test_hout_callback", module)
    seed = tmp_path / "seed"
    seed.mkdir()
    tree = seed / "tree.json"
    tree.write_text("{}", encoding="utf-8")
    runner = build_ordinary_tree_hout_runner(
        seed_dir=seed, tree_path=tree,
        runtime_config={"scenario_runner_factory": "test_hout_callback:run"},
    )
    summary, events = runner.run(scenarios=[1, 2])
    assert summary["scenario_count"] == 2
    assert summary["score"] == 2
    assert events == []
