from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.analysis.expectations import classify_experiment
from nonvacuity_lab.config_resolver import _bind_symbolic_activation
from nonvacuity_lab.schema import ExpectedResult
from nonvacuity_lab.v2_runner import _v2_mutation_to_v1


def test_integrity_expectation_accepts_both_fail_closed_statuses():
    expected = ExpectedResult.from_mapping({
        "integrity_result_statuses": [
            "PROOF_BUNDLE_INVALID",
            "MODEL_CONFORMANCE_FAILED",
        ],
        "require_activation": False,
    })
    for status in ("PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED"):
        result = classify_experiment(
            expected=expected,
            proof_result={"result_status": status},
            activation_result=None,
            integrity=True,
        )
        assert result["status"] == "INTEGRITY_REJECTION_EXPECTED"

    proved = classify_experiment(
        expected=expected,
        proof_result={"result_status": "DEPLOYED_TREE_PROVED"},
        activation_result=None,
        integrity=True,
    )
    assert proved["status"] == "INTEGRITY_REJECTION_MISSING"

    timeout = classify_experiment(
        expected=expected,
        proof_result={"result_status": "VERIFIER_TIMEOUT"},
        activation_result=None,
        integrity=True,
    )
    assert timeout["status"] == "VERIFIER_TIMEOUT"


def test_v2_translation_preserves_upstream_and_integrity_contract(tmp_path: Path):
    mutation = {
        "mutation_id": "F1_demo",
        "mutation_class": "BUNDLE_TREE_TAMPER",
        "mutator": {"kind": "bundle_tamper", "parameters": {}},
        "activation": {"mode": "none"},
        "expected": {
            "allowed_result_statuses": [
                "PROOF_BUNDLE_INVALID",
                "MODEL_CONFORMANCE_FAILED",
            ],
            "integrity_result_statuses": [
                "PROOF_BUNDLE_INVALID",
                "MODEL_CONFORMANCE_FAILED",
            ],
            "allowed_upstream_obligations": ["SOURCE_TREE_INTEGRITY"],
            "allow_strict_upstream_failure": True,
            "require_activation": False,
        },
    }
    translated = _v2_mutation_to_v1(mutation, config={}, base_dir=tmp_path)
    expected = translated["expected"]
    assert expected["integrity_result_statuses"] == [
        "PROOF_BUNDLE_INVALID",
        "MODEL_CONFORMANCE_FAILED",
    ]
    assert expected["allowed_upstream_obligations"] == ["SOURCE_TREE_INTEGRITY"]
    assert expected["allow_strict_upstream_failure"] is True


def test_symbolic_binding_is_retained_in_combined_activation_mode(monkeypatch):
    monkeypatch.setattr(
        "nonvacuity_lab.config_resolver._discover_symbolic_binding",
        lambda target: {"taskset_path": "taskset.json"},
    )
    mutation = {
        "resolved_target": {"tree_path": "tree.json", "leaf_id": 1, "action_id": 2},
        "activation": {"mode": "symbolic_auto_or_hout"},
    }
    _bind_symbolic_activation(mutation, "B2")
    assert mutation["activation"]["formula_kind"] == "B2_NO_FIRST_VALID_DIFFERENCE"
    assert mutation["activation"]["binding"]["overlay_semantics"] == "top1_valid_else_noop"


def test_templates_publish_current_c3_and_integrity_contracts():
    root = Path("configs/nonvacuity/templates")
    for name in ("ppp_full_campaign.template.json", "ppp_minimal_campaign.template.json"):
        data = json.loads((root / name).read_text(encoding="utf-8"))
        by_id = {row["mutation_id"]: row for row in data["mutations"]}
        for mutation_id, row in by_id.items():
            canonical = mutation_id.split("_", 1)[0]
            if canonical in {"B2", "B3", "B4"}:
                assert row["activation"]["mode"] == "symbolic_auto_or_hout"
            if canonical in {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}:
                assert set(row["expected"]["integrity_result_statuses"]) == {
                    "PROOF_BUNDLE_INVALID",
                    "MODEL_CONFORMANCE_FAILED",
                }

    full = json.loads((root / "ppp_full_campaign.template.json").read_text(encoding="utf-8"))
    c3 = next(row for row in full["mutations"] if row["mutation_id"] == "C3_retroactive_release")
    assert c3["expected"]["allowed_result_statuses"] == ["MODEL_CONFORMANCE_FAILED"]
    assert c3["expected"]["allowed_first_failing_obligations"] == [
        "ACTIVE_RELEASE_BUDGET_INVARIANT"
    ]
    assert c3["expected"]["require_failure"] is True
    assert c3["mutator"]["parameters"]["patches"] == [
        {"target_file": "amc_py/event_runtime.py"}
    ]
