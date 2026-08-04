from __future__ import annotations

import json
import shutil
from pathlib import Path

from nonvacuity_lab.audit_cli import main as audit_main
from nonvacuity_lab.manifest import load_campaign
from nonvacuity_lab.resolve_cli import main as resolve_main


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_phase3_audit_is_seed_variant_scoped_and_resolver_output_is_reloadable(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    seed_root = tmp_path / "seeds"
    seed = seed_root / "s185"
    shutil.copytree(
        root / "tests" / "formal" / "fixtures" / "synthetic_p0" / "best_overall",
        seed / "best_overall",
    )
    shutil.copytree(
        root / "tests" / "formal" / "fixtures" / "synthetic_p0" / "formal_inputs",
        seed / "formal_inputs",
    )

    hout_root = tmp_path / "hout"
    _write(
        hout_root / "s185" / "best_overall" / "events.json",
        {
            "events": [
                {
                    "seed": 185,
                    "tree_variant": "best_overall",
                    "scenario_id": 7,
                    "leaf_id": 1,
                    "raw_top1_action_id": 5,
                    "raw_top1_invalid": True,
                    "rejected_action_id": 5,
                    "selected_action_id": 3,
                    "selected_rank": 2,
                    "reject_reason": "decrease_hi_forbidden",
                }
            ]
        },
    )
    # Same numeric leaf id in another seed must not contaminate s185.
    _write(
        hout_root / "s999" / "best_overall" / "events.json",
        {"events": [{"seed": 999, "tree_variant": "best_overall", "leaf_id": 1}]},
    )

    proof_root = tmp_path / "proofs"
    bundle = proof_root / "s185_best_overall"
    _write(
        bundle / "proof_summary.json",
        {
            "taskset_seed": 185,
            "tree_variant": "best_overall",
            "rta": {
                "task_id": "SYN_HI_1",
                "criticality": "HI",
                "deadline": 50,
                "R_LO": 45,
                "R_HI": 49,
                "limiting_lo_task": "SYN_LO",
                "envelope_target_file": "formal_inputs/code_taskset_canonical.json",
                "envelope_json_pointer": "/ordered_tasks/1/action_hard_upper",
                "witness": {"value": 9},
            },
        },
    )
    _write(bundle / "candidate" / "obligation.json", {"obligation_id": "RTA_HI_BOUND"})

    audit_root = tmp_path / "audit"
    assert audit_main(
        [
            "--seed-root", str(seed_root),
            "--proof-bundle-root", str(proof_root),
            "--hout-root", str(hout_root),
            "--source-root", str(root),
            "--out", str(audit_root),
        ]
    ) == 0
    rows = json.loads((audit_root / "leaf_audit.json").read_text(encoding="utf-8"))["rows"]
    leaf = next(row for row in rows if row["seed"] == 185 and row["leaf_id"] == 1)
    assert leaf["hout_hit_count"] == 1
    assert leaf["tree_hash"]
    assert leaf["guard"] == [{
        "feature_index": 0,
        "feature_name": "T00.SYN_HI_0.budget_norm",
        "op": "<=",
        "threshold_int": 500,
    }]
    risk = next(item for item in leaf["action_risks"] if item["action_id"] == 5)
    assert risk["risk_class"] == "HI_BUDGET_DECREASE"
    assert risk["observed_reject_count"] == 1

    template = tmp_path / "campaign.json"
    output_root = tmp_path / "outputs"
    mutations = [
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": False,
            "mutation_id": "A1",
            "mutation_class": "DANGEROUS_TOP1",
            "base_seed": 185,
            "seed_dir": str(seed),
            "tree_variant": "best_overall",
            "mutator": {"kind": "dangerous_top1", "parameters": {"leaf_candidates": [], "dangerous_actions": []}},
            "activation": {"mode": "hout"},
        },
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": False,
            "mutation_id": "B4",
            "mutation_class": "GUARD_ABLATION",
            "base_seed": 185,
            "seed_dir": str(seed),
            "tree_variant": "best_overall",
            "mutator": {"kind": "source_overlay", "parameters": {}},
            "activation": {"mode": "hout"},
        },
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": False,
            "mutation_id": "D1",
            "mutation_class": "ENVELOPE",
            "tree_variant": "best_overall",
            "mutator": {"kind": "envelope", "parameters": {"delta": 1}},
            "activation": {"mode": "hout"},
        },
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": False,
            "mutation_id": "F5",
            "mutation_class": "BUNDLE_INTEGRITY",
            "reuse_source_bundle": str(bundle),
            "tree_variant": "best_overall",
            "mutator": {"kind": "bundle_tamper", "parameters": {"tamper_kind": "json_pointer", "value": 0}},
        },
        {
            "schema_version": "nonvacuity_mutation_v1",
            "enabled": False,
            "mutation_id": "F6",
            "mutation_class": "BUNDLE_INTEGRITY",
            "reuse_source_bundle": str(bundle),
            "tree_variant": "best_overall",
            "mutator": {"kind": "bundle_tamper", "parameters": {"tamper_kind": "delete_artifact"}},
        },
    ]
    _write(
        template,
        {
            "schema_version": "ppp_nonvacuity_campaign_v1",
            "enabled": False,
            "campaign_id": "phase3_test",
            "proof_route": "protected_prefix",
            "output_root": str(output_root),
            "source_root": str(root),
            "mutations": mutations,
        },
    )
    resolved_path = tmp_path / "resolved.json"
    assert resolve_main([
        "--template", str(template),
        "--audit-root", str(audit_root),
        "--out", str(resolved_path),
        "--require-all-resolved",
    ]) == 0
    resolved = load_campaign(resolved_path)
    assert resolved.enabled is False
    assert all(not item.enabled for item in resolved.mutations)
    by_id = {item.mutation_id: item for item in resolved.mutations}
    assert by_id["A1"].mutator["parameters"]["leaf_id"] == 1
    assert by_id["A1"].mutator["parameters"]["action_id"] == 5
    assert by_id["D1"].mutator["parameters"]["json_pointer"] == "/ordered_tasks/1/action_hard_upper"
    assert by_id["F5"].mutator["parameters"]["target_file"] == "proof_summary.json"
    assert by_id["F6"].mutator["parameters"]["target_file"] == "candidate/obligation.json"
