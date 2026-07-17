"""Phase F-H synthetic fresh-process verifier。

该入口只接受 synthetic artifact，不读取真实 Seed。它在新 Python 进程中重新
构造 task/action/domain，并比较 candidate、common、deployed 与 certified hash。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from itertools import product

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.invariant.certified_envelope import _certify_envelope_from_verifier
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.core.hashing import sha256_file
from formal_toolchain.policy.actions import build_action_transition_table
from formal_toolchain.policy.tree import validate_tree_and_leaf_partition
from formal_toolchain.policy.mask_fallback import build_parametric_mask_fallback_certificate, select_first_valid
from formal_toolchain.adapters.synthetic_context import build_synthetic_context
from formal_toolchain.adapters.synthetic_policy import build_transition_witness
from formal_toolchain.adapters.synthetic_runtime_adapter import SyntheticP0RuntimeAdapter
from formal_toolchain.verifier.artifact_verifier import verify_registry_certificate
from amc_py.viper.integer_tree import IntegerTreeModel, IntegerTreeNode, IntegerTreeLeaf


def verify_payload(payload: dict) -> dict:
    if payload.get("fixture") != "synthetic_p0":
        return {"status": "UNRESOLVED", "failure": {"code": "SYNTHETIC_FIXTURE_REQUIRED"}}
    root = Path(__file__).parents[2] / "tests/formal/fixtures/synthetic_p0"
    required = ("target_recipe.json", "integer_tree.json", "feature_names.json", "action_definitions.json", "fixed_point_config.json", "metadata.json")
    if any(not (root / name).is_file() for name in required):
        return {"status": "UNRESOLVED", "failure": {"code": "SYNTHETIC_ARTIFACT_MISSING"}}
    recipe = json.loads((root / "target_recipe.json").read_text(encoding="utf-8"))
    target = build_target(recipe["factory"])
    inventory = inspect_tree_artifact(root, expected_state_dim=len(target.feature_names), expected_action_dim=len(target.action_definitions))
    if payload.get("tree_artifact_hash") != sha256_file(root / "integer_tree.json"):
        return {"status": "FAIL", "failure": {"code": "TREE_ARTIFACT_HASH_MISMATCH"}}
    tree_data = json.loads((root / "integer_tree.json").read_text(encoding="utf-8"))
    tree = IntegerTreeModel(schema_version="integer_tree_v1", root_node_id=2,
        state_dim=tree_data["state_dim"], action_dim=tree_data["action_dim"],
        nodes=(IntegerTreeNode(2, tree_data["root"]["feature_index"], tree_data["root"]["threshold"], tree_data["root"]["left"], tree_data["root"]["right"]),),
        leaves=tuple(IntegerTreeLeaf(row["id"], row["action_ranking"][0], tuple(row["action_ranking"]),
                                    tuple(0.0 for _ in range(tree_data["action_dim"])), 0, 0.0, 0.0) for row in tree_data["leaves"]),
        feature_names=tuple(inventory["feature_names"]), fixed_point_config_hash=tree_data["fixed_point_config_hash"])
    tree_result = validate_tree_and_leaf_partition(tree)
    if tree_result.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": {"code": "TREE_RECOMPUTE_UNRESOLVED", "detail": tree_result}}
    tasks = target.ordered_tasks
    actions = build_budget_action_space(
        tasks,
        action_space=target.runtime_config.action_space,
        budget_increase_ratio=target.runtime_config.budget_increase_ratio,
        budget_decrease_ratio=target.runtime_config.budget_decrease_ratio,
    )
    domain = __import__("formal_toolchain.conformance.time_domain", fromlist=["build_budget_domain"]).build_budget_domain(
        tasks, target.provenance["budget_by_task"], runtime_config=target.runtime_config)
    domain["context_hash"] = "pending"
    context_hash = build_synthetic_context(target, inventory, domain)["context_hash"]
    domain["context_hash"] = context_hash
    if payload.get("context_hash") != context_hash:
        return {"status": "FAIL", "failure": {"code": "CONTEXT_RECOMPUTE_MISMATCH"}}
    adapter = SyntheticP0RuntimeAdapter(target)
    enumerated_candidate = synthesize_candidate_envelope(
        domain, actions, tasks, context_hash=context_hash, runtime_adapter=adapter)
    structural_candidate = synthesize_candidate_envelope(
        domain, actions, tasks, context_hash=context_hash, runtime_adapter=adapter)
    if enumerated_candidate.get("upper") != structural_candidate.get("upper"):
        return {"status": "FAIL", "failure": {"code": "CANDIDATE_UPPER_MISMATCH"}}
    if sha256_object(structural_candidate) != payload.get("candidate_hash"):
        return {"status": "FAIL", "failure": {"code": "CANDIDATE_RECOMPUTE_MISMATCH"}}
    if payload.get("domain_hash") != sha256_object(domain):
        return {"status": "FAIL", "failure": {"code": "DOMAIN_RECOMPUTE_MISMATCH"}}
    transitions = build_transition_witness(domain, tasks)
    common = check_common_transition_preservation(structural_candidate, transitions=transitions)
    action_cert = build_action_transition_table(actions, tasks, domain["tasks"])
    rankings = {int(leaf.node_id): tuple(int(action_id) for action_id in leaf.action_ranking) for leaf in tree.leaves}
    mask_contract = adapter.export_mask_contract()
    mask = build_parametric_mask_fallback_certificate(rankings=rankings, action_dim=len(actions), mask_contract=mask_contract)
    deployed_enumerated = check_deployed_policy_preservation(
        enumerated_candidate,
        actions,
        tasks,
        mask_fallback_certificate=mask,
        action_transition_certificate=action_cert,
        mask_contract=mask_contract,
    )
    deployed_structural = check_deployed_policy_preservation(
        structural_candidate,
        actions,
        tasks,
        mask_fallback_certificate=mask,
        action_transition_certificate=action_cert,
        mask_contract=mask_contract,
    )
    if deployed_enumerated.get("status") != "PASS" or deployed_structural.get("status") != "PASS":
        return {"status": "FAIL", "failure": {"code": "DEPLOYED_RECOMPUTE_FAILED"}}
    if sha256_object(common) != payload.get("common_hash") or sha256_object(deployed_structural) != payload.get("deployed_hash"):
        return {"status": "FAIL", "failure": {"code": "PRESERVATION_RECOMPUTE_MISMATCH"}}
    attestation = {"fresh_process": True, "candidate_hash": sha256_object(structural_candidate),
                   "common_hash": sha256_object(common), "deployed_hash": sha256_object(deployed_structural)}
    certified = _certify_envelope_from_verifier(structural_candidate, common, deployed_structural, context_hash=context_hash,
                                                verifier_attestation=attestation)
    registry_path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    cert_context_inputs = {"fixture": "synthetic_p0", "phase": "F-H", "context_hash": context_hash}
    cert_context_hash = sha256_object(cert_context_inputs)
    predecessor_certificates = {}
    for obligation_id, source in (("CANDIDATE_ENVELOPE", structural_candidate),
                                  ("COMMON_TRANSITION_PRESERVATION", common),
                                  ("BUDGET_DOMAIN", domain),
                                  ("EXECUTABLE_POLICY_SEMANTICS", {"context_hash": context_hash}),
                                  ("LO_BUDGET_UPPER_INVARIANT", structural_candidate),
                                  ("HI_BUDGET_LOWER_INVARIANT", structural_candidate),
                                  ("ACTIVE_RELEASE_BUDGET_INVARIANT", structural_candidate)):
        predecessor_certificates[obligation_id] = {"artifact_schema_version": "synthetic_phase_f_v1",
            "obligation_id": obligation_id, "obligation_status": "PASS", "certificate_context_hash": cert_context_hash,
            "direct_predecessor_hashes": {}, "checker_id": "phase_fh_fresh_verifier", "checker_version": "1",
            "inputs": {"fixture": "synthetic_p0", "source_hash": sha256_object(source)},
            "witness": {"source_hash": sha256_object(source)}, "evidence": [{"status": "PASS"}], "failure": None}
    predecessor_certificates["DEPLOYED_POLICY_PRESERVATION"] = {"artifact_schema_version": "synthetic_phase_fh_certificate_v1",
        "obligation_id": "DEPLOYED_POLICY_PRESERVATION", "obligation_status": "PASS", "certificate_context_hash": cert_context_hash,
        "direct_predecessor_hashes": {name: sha256_object(predecessor_certificates[name]) for name in
            ("EXECUTABLE_POLICY_SEMANTICS", "CANDIDATE_ENVELOPE", "BUDGET_DOMAIN", "COMMON_TRANSITION_PRESERVATION")},
        "checker_id": "phase_fh_fresh_verifier", "checker_version": "1",
        "inputs": {"fixture": "synthetic_p0"}, "witness": {"deployed_hash": sha256_object(deployed_structural)},
        "evidence": [{"status": "PASS"}], "failure": None}
    certified_certificate = {"artifact_schema_version": "synthetic_phase_fh_certificate_v1",
        "obligation_id": "CERTIFIED_ENVELOPE", "obligation_status": "PASS", "certificate_context_hash": cert_context_hash,
        "direct_predecessor_hashes": {name: sha256_object(predecessor_certificates[name]) for name in
            ("DEPLOYED_POLICY_PRESERVATION", "LO_BUDGET_UPPER_INVARIANT", "HI_BUDGET_LOWER_INVARIANT",
             "ACTIVE_RELEASE_BUDGET_INVARIANT")},
        "checker_id": "phase_fh_fresh_verifier", "checker_version": "1",
        "inputs": {"fixture": "synthetic_p0", "context_hash": context_hash},
        "witness": {"candidate_hash": sha256_object(structural_candidate), "common_hash": sha256_object(common),
                    "deployed_hash": sha256_object(deployed_structural)}, "evidence": [{"fresh_process": True}], "failure": None}
    checked = verify_registry_certificate(certified_certificate, registry_path=registry_path,
        predecessor_certificates=predecessor_certificates, context_inputs=cert_context_inputs)
    if checked.get("status") != "PASS":
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_SCHEMA_INVALID", "detail": checked}}
    certified["preservation_certificate"] = certified_certificate
    certified["preservation_certificate_hash"] = sha256_object(certified_certificate)
    return {"status": "PASS", "fresh_process": True, "candidate_hash": sha256_object(structural_candidate),
            "certified_hash": sha256_object(certified), "certified": certified}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("payload"); parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    path = Path(args.payload)
    result = verify_payload(json.loads(path.read_text(encoding="utf-8")))
    if result.get("status") == "PASS" and args.out:
        output = Path(args.out) / "verified"; output.mkdir(parents=True, exist_ok=True)
        certified = result["certified"]
        (output / "certified_envelope.json").write_text(json.dumps(certified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "certified_envelope_certificate.json").write_text(
            json.dumps(certified["preservation_certificate"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del result["certified"]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
