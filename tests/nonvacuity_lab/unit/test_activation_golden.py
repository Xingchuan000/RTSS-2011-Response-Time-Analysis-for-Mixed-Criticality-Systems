from __future__ import annotations

import pytest

from nonvacuity_lab.activation.policy_witness import evaluate_policy_witness
from nonvacuity_lab.activation.symbolic_activation import solve_symbolic_activation


def _b1_witness():
    return {
        "leaf_id": 7,
        "state": {"hi_budget": 5, "lo_budget": 5},
        "lower": {"hi_budget": 4, "lo_budget": 0},
        "upper": {"hi_budget": 10, "lo_budget": 10},
        "leaf_guard": [{"field": "lo_budget", "op": ">=", "value": 0}],
        "ranking": [13, 2],
        "actions": {
            "13": {"delta": {"hi_budget": -2}},
            "2": {"delta": {"lo_budget": -1}},
        },
    }


def test_a1_b1_same_witness_is_mask_contained_then_unsafe_when_bypassed():
    result = evaluate_policy_witness(_b1_witness())
    assert result["leaf_guard_satisfied"] is True
    assert result["baseline_top1_valid"] is False
    assert result["baseline_first_valid_action"] == 2
    assert result["baseline_selected_rank"] == 1
    assert result["mutated_raw_top1_action"] == 13
    assert result["mutated_post_invariant_violation"] is True
    assert result["activated_b1"] is True


def test_symbolic_b1_activation_uses_witness_not_profile_name():
    activation = solve_symbolic_activation(
        mutation_id="B1_same_tree",
        rule={"activation_kind": "b1", "candidate_witnesses": [_b1_witness()]},
    )
    assert activation.status.value == "ACTIVATED"
    assert activation.action_id == 13


def test_z3_symbolic_b1_finds_guard_and_illegal_post_state():
    pytest.importorskip("z3")
    activation = solve_symbolic_activation(
        mutation_id="B1_z3",
        rule={
            "activation_kind": "b1",
            "symbolic_model": {
                "leaf_id": 7,
                "variables": {
                    "hi_budget": {"min": 4, "max": 10},
                    "lo_budget": {"min": 0, "max": 10},
                },
                "lower": {"hi_budget": 4, "lo_budget": 0},
                "upper": {"hi_budget": 10, "lo_budget": 10},
                "leaf_guard": [
                    {"field": "hi_budget", "op": "==", "value": 4}
                ],
                "ranking": [13, 2],
                "actions": {
                    "13": {"delta": {"hi_budget": -1}},
                    "2": {"delta": {"lo_budget": 0}},
                },
            },
        },
    )
    assert activation.status.value == "ACTIVATED"
    assert activation.details["solver"] == "Z3"
    assert activation.post_invariant_violation is True


def test_b3_sat_and_unsat_are_distinguished():
    sat = _b1_witness()
    sat["actions"]["2"] = {"delta": {"lo_budget": -6}}
    activated = solve_symbolic_activation(
        mutation_id="B3_sat",
        rule={"activation_kind": "b3", "candidate_witnesses": [sat]},
    )
    not_activated = solve_symbolic_activation(
        mutation_id="B3_unsat",
        rule={"activation_kind": "b3", "candidate_witnesses": [_b1_witness()]},
    )
    assert activated.status.value == "ACTIVATED"
    assert not_activated.status.value == "NOT_ACTIVATED"


def test_b4_requires_target_guard_to_be_the_unique_blocker():
    unique = {
        "leaf_id": 4,
        "state": {"budget": 5, "risk": 8},
        "lower": {"budget": 0},
        "upper": {"budget": 10},
        "leaf_guard": [],
        "ranking": [3],
        "disabled_guard": "RISK_CAP",
        "actions": {
            "3": {
                "delta": {"budget": 0},
                "guards": [
                    {
                        "name": "RISK_CAP",
                        "field": "risk",
                        "op": "<=",
                        "value": 7,
                    }
                ],
            }
        },
    }
    redundant = {
        **unique,
        "actions": {
            "3": {
                "delta": {"budget": 10},
                "guards": unique["actions"]["3"]["guards"],
            }
        },
    }
    assert evaluate_policy_witness(unique)["activated_b4"] is True
    assert evaluate_policy_witness(redundant)["activated_b4"] is False


def test_symbolic_builder_accepts_list_action_definitions(tmp_path):
    pytest.importorskip("z3")
    import json
    from nonvacuity_lab.activation.symbolic_model_builder import build_symbolic_problem

    tree = {
        "root_node_id": 0,
        "nodes": [],
        "leaves": [{"node_id": 0, "action_ranking": [0, 1]}],
        "feature_names": ["x"],
    }
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    features = tmp_path / "features.json"
    features.write_text(json.dumps(["x"]), encoding="utf-8")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [
        {"name": "L", "criticality": "LO", "initial_runtime_budget": 2,
         "budget_floor": 1, "action_hard_upper": 3}
    ]}), encoding="utf-8")
    actions = tmp_path / "actions.json"
    actions.write_text(json.dumps([
        {"action_id": 0, "increase_task": "L", "increase_ratio": 0.5,
         "minimum_increment": 1},
        {"action_id": 1, "decrease_tasks": ["L"], "decrease_ratio": 0.5,
         "minimum_increment": 1},
    ]), encoding="utf-8")
    from nonvacuity_lab.canonical import file_hash
    problem = build_symbolic_problem(
        {"tree_path": str(tree_path), "tree_sha256": file_hash(tree_path),
         "leaf_id": 0, "action_id": 0, "original_ranking": [0, 1]},
        {"feature_schema_path": str(features), "taskset_path": str(tasks),
         "action_definitions_path": str(actions), "feature_scale": 1000},
        "B2_NO_FIRST_VALID_DIFFERENCE",
    )
    assert problem.metadata["action_id"] == 0
