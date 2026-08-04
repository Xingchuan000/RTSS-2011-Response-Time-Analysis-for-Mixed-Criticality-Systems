from __future__ import annotations

import json
from pathlib import Path

import pytest

from nonvacuity_lab.analysis.expectations import classify_experiment
from nonvacuity_lab.paper_acceptance import evaluate_paper_acceptance
from nonvacuity_lab.preflight import audit_campaign
from nonvacuity_lab.runners import campaign as campaign_runner
from nonvacuity_lab.schema import (
    CampaignConfig,
    ExpectedResult,
    ExperimentStatus,
    MutationClass,
    MutationManifest,
)
from nonvacuity_lab.v2_runner import _v2_mutation_to_v1


def _manifest(tmp_path: Path, *, mutation_id: str = "D1_dynamic_envelope_gradient") -> MutationManifest:
    seed = tmp_path / "s1"
    (seed / "best_overall").mkdir(parents=True)
    (seed / "best_overall" / "integer_tree.json").write_text("{}\n", encoding="utf-8")
    return MutationManifest(
        schema_version="nonvacuity_mutation_v1",
        enabled=True,
        mutation_id=mutation_id,
        mutation_class=MutationClass.ENVELOPE,
        seed_dir=seed,
        base_seed=1,
        tree_variant="best_overall",
        paired_with=None,
        single_semantic_change=True,
        mutator={
            "kind": "envelope",
            "parameters": {
                "target_file": "formal_inputs/envelope.json",
                "json_pointer": "/task/upper",
                "delta": 1,
            },
        },
        activation={"mode": "none"},
        expected=ExpectedResult(
            allowed_result_statuses=("REFERENCE_CERTIFICATE_FAILED",),
            allowed_first_failing_obligations=("RTA_HI_BOUND",),
            require_failure=True,
            require_activation=False,
        ),
        reuse_source_bundle=None,
        metadata={"initial_step": 1, "maximum_delta": 4},
    )


def test_expectation_classifier_honours_multiple_allowed_statuses():
    expected = ExpectedResult(
        allowed_result_statuses=("DEPLOYED_TREE_PROVED", "REFERENCE_CERTIFICATE_FAILED"),
        require_activation=False,
    )
    result = classify_experiment(
        expected=expected,
        proof_result={"result_status": "DEPLOYED_TREE_PROVED"},
        activation_result=None,
    )
    assert result["status"] == ExperimentStatus.PASS_EXPECTED.value


def test_v2_translation_preserves_allowed_sets_and_hout_profile(tmp_path: Path):
    profile = {
        "profile_id": "h5",
        "taskset_path": str(tmp_path / "taskset.json"),
        "scenario_seeds": [1, 2],
        "runtime_config_path": str(tmp_path / "runtime.json"),
        "horizon": 5,
        "controller_release_times": [0],
        "worker_count": 1,
        "random_seed": 7,
        "base_command": ["python", "run.py"],
        "mutated_command": ["python", "run.py"],
    }
    mutation = {
        "mutation_id": "B2_demo",
        "mutation_class": "NO_FIRST_VALID",
        "enabled": True,
        "seed_dir": str(tmp_path / "s1"),
        "tree_variant": "best_overall",
        "mutator": {"kind": "coherent_source_patch", "parameters": {}},
        "activation": {"mode": "hout"},
        "expected": {
            "allowed_result_statuses": ["DEPLOYED_TREE_PROVED", "POLICY_CONTRACT_VIOLATION"],
            "allowed_first_failing_obligations": ["DEPLOYED_POLICY_PRESERVATION"],
            "allowed_failure_routes": ["POLICY"],
            "require_activation": False,
        },
        "hout_profile_id": "h5",
    }
    translated = _v2_mutation_to_v1(mutation, config={"hout_profiles": {"h5": profile}}, base_dir=tmp_path)
    assert translated["expected"]["allowed_result_statuses"] == [
        "DEPLOYED_TREE_PROVED",
        "POLICY_CONTRACT_VIOLATION",
    ]
    assert translated["metadata"]["hout"]["scenario_seeds"] == [1, 2]


def test_dynamic_d1_preflight_allows_resolver_owned_target(tmp_path: Path):
    root = tmp_path / "bundles"
    root.mkdir()
    manifest = _manifest(tmp_path)
    manifest = MutationManifest(
        **{
            **manifest.__dict__,
            "seed_dir": None,
            "mutator": {"kind": "envelope", "parameters": {"delta": 1}},
            "metadata": {
                "dynamic_minimum_slack_selection": True,
                "bundle_roots": [str(root)],
            },
        }
    )
    config = CampaignConfig(
        schema_version="ppp_nonvacuity_campaign_v1",
        enabled=False,
        campaign_id="ready",
        proof_route="protected_prefix",
        output_root=tmp_path / "out",
        source_root=tmp_path,
        preserve_workspaces=True,
        run_baselines=True,
        run_semantic_recompile=True,
        run_integrity_reuse=True,
        run_hout=False,
        fail_on_not_activated=True,
        mutations=(manifest,),
    )
    report = audit_campaign(config, include_disabled=True)
    assert report["status"] == "PASS", report


def test_gradient_campaign_runs_fresh_delta_proofs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = _manifest(tmp_path)
    calls: list[tuple[int, MutationClass]] = []

    def fake_run_one(delta_manifest, **kwargs):
        if delta_manifest.mutation_class is MutationClass.BASELINE:
            delta = 0
            proof = {"result_status": "DEPLOYED_TREE_PROVED", "minimum_slack": 2}
            result = {"status": "PASS_EXPECTED", "baseline": {"proof_result": proof}}
        else:
            delta = int(delta_manifest.mutator["parameters"]["delta"])
            if delta < 2:
                proof = {"result_status": "DEPLOYED_TREE_PROVED", "minimum_slack": 2 - delta}
            else:
                proof = {
                    "result_status": "REFERENCE_CERTIFICATE_FAILED",
                    "violated_obligation_id": "RTA_HI_BOUND",
                    "minimum_slack": 2 - delta,
                }
            result = {"status": "PASS_EXPECTED", "semantic_recompile": {"proof_result": proof}}
        calls.append((delta, delta_manifest.mutation_class))
        return result

    monkeypatch.setattr(campaign_runner, "run_one", fake_run_one)
    result = campaign_runner._run_envelope_gradient_campaign(
        manifest,
        campaign_id="gradient",
        output_root=tmp_path / "out",
        source_root=tmp_path,
        enabled_by_cli=True,
        timeout_seconds=10,
    )
    assert result["status"] == ExperimentStatus.GRADIENT_EXPECTED_FAILURE_FOUND.value
    assert result["gradient"]["delta_star"] == 2
    assert calls[0] == (0, MutationClass.BASELINE)
    assert {delta for delta, _ in calls} >= {0, 1, 2}


def test_paper_acceptance_canonicalizes_real_mutation_ids(tmp_path: Path):
    result = {
        "mutation_results": [
            {"mutation_id": "P0_s185_compact", "status": "PASS_EXPECTED", "baseline": {"proof_result": {"result_status": "DEPLOYED_TREE_PROVED"}}},
            {"mutation_id": "A1_s185_compact", "status": "PASS_EXPECTED", "semantic_recompile": {"proof_result": {"result_status": "DEPLOYED_TREE_PROVED"}}},
            {"mutation_id": "B1_s185_compact", "status": "FAIL_EXPECTED", "semantic_recompile": {"proof_result": {"result_status": "POLICY_CONTRACT_VIOLATION"}}},
        ]
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    acceptance = evaluate_paper_acceptance(result)
    findings = {item["finding_id"]: item for item in acceptance["findings"]}
    assert findings["present:P0"]["passed"] is True
    assert findings["present:A1"]["passed"] is True
    assert findings["present:B1"]["passed"] is True
