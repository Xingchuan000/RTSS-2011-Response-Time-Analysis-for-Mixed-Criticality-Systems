from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.mutators.factory import build_mutator
from nonvacuity_lab.mutators.retroactive_release_budget import RetroactiveReleaseBudgetMutation
from nonvacuity_lab.preflight import audit_mutation, audit_v2_campaign_path
from nonvacuity_lab.schema import ExpectedResult, MutationClass, MutationManifest
from nonvacuity_lab.v2_runner import _v2_mutation_to_v1


def _seed(tmp_path: Path) -> Path:
    seed = tmp_path / "s185"
    variant = seed / "best_overall"
    variant.mkdir(parents=True)
    (variant / "integer_tree.json").write_text("{}\n", encoding="utf-8")
    return seed


def test_c3_factory_uses_ast_mutator():
    assert isinstance(build_mutator("RETROACTIVE_RELEASE_BUDGET"), RetroactiveReleaseBudgetMutation)


def test_v2_translation_preserves_resolved_target_for_symbolic_auto(tmp_path: Path):
    target = {
        "seed_dir": str(_seed(tmp_path)),
        "tree_path": str(tmp_path / "s185" / "best_overall" / "integer_tree.json"),
        "tree_sha256": "0" * 64,
        "leaf_id": 1,
        "action_id": 2,
    }
    translated = _v2_mutation_to_v1(
        {
            "mutation_id": "B3_demo",
            "mutation_class": "ALL_INVALID_FORCE_TOP1",
            "tree_variant": "best_overall",
            "resolved_target": target,
            "mutator": {"kind": "coherent_source_patch", "parameters": {}},
            "activation": {"mode": "symbolic_auto"},
            "expected": {"allowed_result_statuses": ["POLICY_CONTRACT_VIOLATION"]},
        },
        config={"hout_profiles": {}},
        base_dir=tmp_path,
    )
    assert translated["metadata"]["resolved_target"] == target


def test_symbolic_auto_missing_bindings_fail_before_workspace(tmp_path: Path):
    seed = _seed(tmp_path)
    manifest = MutationManifest(
        schema_version="nonvacuity_mutation_v1",
        enabled=True,
        mutation_id="B3_demo",
        mutation_class=MutationClass.ALL_INVALID_FORCE_TOP1,
        seed_dir=seed,
        base_seed=185,
        tree_variant="best_overall",
        paired_with=None,
        single_semantic_change=True,
        mutator={"kind": "coherent_source_patch", "parameters": {"semantic_change_id": "all_invalid_force_top1", "patches": []}},
        activation={"mode": "symbolic_auto"},
        expected=ExpectedResult(allowed_result_statuses=("POLICY_CONTRACT_VIOLATION",)),
        reuse_source_bundle=None,
        metadata={},
    )
    result = audit_mutation(manifest, source_root=tmp_path)
    codes = {item["code"] for item in result["issues"]}
    assert "SYMBOLIC_AUTO_BINDING_MISSING" in codes
    assert "SYMBOLIC_AUTO_FORMULA_MISSING" in codes
    assert "SYMBOLIC_AUTO_RUNTIME_FACTORY_MISSING" in codes
    assert "SYMBOLIC_AUTO_TARGET_MISSING" in codes


def test_v2_preflight_rejects_template_without_workspace(tmp_path: Path):
    path = tmp_path / "template.json"
    path.write_text(json.dumps({
        "schema_version": "nonvacuity_campaign_v2",
        "config_kind": "TEMPLATE",
        "campaign_id": "demo",
        "proof_route": "protected_prefix",
        "enabled": False,
        "source_binding": {"clean_source_root": ".", "clean_source_root_sha256": ""},
        "mutations": [],
    }), encoding="utf-8")
    result = audit_v2_campaign_path(path, include_disabled=True)
    assert result["status"] == "CAMPAIGN_PREFLIGHT_FAILED"
    assert result["issues"][0]["code"] == "CONFIG_NOT_RESOLVED"


def test_templates_match_planned_seed_roles():
    full = json.loads(Path("configs/nonvacuity/templates/ppp_full_campaign.template.json").read_text(encoding="utf-8"))
    rows = {item["mutation_id"]: item for item in full["mutations"]}
    assert rows["A2_s397_dangerous_top1_masked"]["tree_variant"] == "best_balanced"
    assert rows["B5_s397_mask_bypass"]["pair_with"] == "A2_s397_dangerous_top1_masked"
    assert rows["B2_s1264_no_first_valid"]["tree_variant"] == "best_balanced"
    assert rows["B3_s1264_all_invalid_force_top1"]["tree_variant"] == "best_balanced"
    assert rows["B4_s1264_guard_ablation"]["tree_variant"] == "best_overall"


def test_cli_resolve_success_has_zero_exit(monkeypatch, tmp_path: Path):
    from nonvacuity_lab.cli import main

    source = Path.cwd()
    seed = tmp_path / "s185"
    variant = seed / "best_overall"
    variant.mkdir(parents=True)
    tree = variant / "integer_tree.json"
    tree.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "leaf_audit.json").write_text(json.dumps({"rows": [{
        "seed": 185, "tree_variant": "best_overall", "seed_dir": str(seed),
        "tree_path": str(tree), "leaf_id": 0, "action_ranking": [0],
    }]}), encoding="utf-8")
    template = tmp_path / "template.json"
    template.write_text(json.dumps({
        "schema_version": "nonvacuity_campaign_v2", "config_kind": "TEMPLATE",
        "campaign_id": "resolve_exit", "proof_route": "protected_prefix", "enabled": False,
        "source_binding": {"clean_source_root": ".", "clean_source_root_sha256": ""},
        "mutations": [{
            "mutation_id": "P0_s185", "mutation_class": "BASELINE", "enabled": False,
            "seed": 185, "tree_variant": "best_overall",
            "mutator": {"kind": "baseline", "parameters": {}},
            "expected": {"allowed_result_statuses": ["DEPLOYED_TREE_PROVED"], "require_activation": False},
        }],
    }), encoding="utf-8")
    output = tmp_path / "nested" / "resolved.json"
    assert main([
        "resolve", "--template", str(template), "--audit-root", str(audit),
        "--source-root", str(source), "--output", str(output), "--require-all-resolved",
    ]) == 0
    assert output.is_file()
    resolved = json.loads(output.read_text(encoding="utf-8"))
    assert resolved["mutations"][0]["resolution_status"] == "RESOLVED"


def test_doctor_validates_only_enabled_experiment_capabilities():
    from nonvacuity_lab.doctor.runner import (
        _active_config_view,
        _requires_z3,
    )

    config = {
        "mutations": [
            {
                "mutation_id": "P0",
                "enabled": True,
                "activation": {"mode": "none"},
            },
            {
                "mutation_id": "B2_disabled",
                "enabled": False,
                "activation": {"mode": "symbolic_auto"},
                "hout_profile_id": "missing_profile",
                "mutator": {
                    "parameters": {
                        "patches": [{"target_file": "missing.py", "before_snippet": "x"}]
                    }
                },
            },
        ],
        "hout_profiles": {
            "missing_profile": {"taskset_path": "missing.json"},
            "unused_profile": {"taskset_path": "also_missing.json"},
        },
    }
    enabled = [item for item in config["mutations"] if item["enabled"]]
    active = _active_config_view(config, enabled)

    assert _requires_z3(enabled) is False
    assert [item["mutation_id"] for item in active["mutations"]] == ["P0"]
    assert active["hout_profiles"] == {}


def test_doctor_requires_z3_for_enabled_symbolic_mutation():
    from nonvacuity_lab.doctor.runner import _requires_z3

    assert _requires_z3([
        {"enabled": True, "activation": {"mode": "symbolic_auto"}}
    ]) is True
