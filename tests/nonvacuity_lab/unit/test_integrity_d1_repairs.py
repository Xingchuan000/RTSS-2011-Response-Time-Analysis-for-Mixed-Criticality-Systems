from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.config_resolver import (
    _bind_integrity_mutation,
    _seed_token,
    refresh_resolved_runtime_bindings,
)


def _write_proof_run(root: Path, *, seed: int, variant: str, threshold: int) -> Path:
    run = root / f"r0_s{seed}" / f"{variant}_protected_prefix"
    request = run / "request"
    candidate = run / "candidate"
    (request / "inputs" / "tree_artifact").mkdir(parents=True)
    (candidate / "artifacts").mkdir(parents=True)
    (request / "proof_request.json").write_text(
        json.dumps({"taskset_seed": seed, "tree_variant": variant}), encoding="utf-8"
    )
    (request / "inputs" / "tree_artifact" / "integer_tree.json").write_text(
        json.dumps({"leaves": [], "nodes": [{"threshold_int": threshold}]}),
        encoding="utf-8",
    )
    (candidate / "artifacts" / "proof_request.json").write_text(
        json.dumps({"seed": seed, "variant": variant}), encoding="utf-8"
    )
    (candidate / "component_contexts.json").write_text("{}", encoding="utf-8")
    return run


def test_seed_token_ignores_experiment_range_and_uses_run_seed(tmp_path: Path):
    path = (
        tmp_path
        / "formalv1_csem_t10_s1550_1599"
        / "r0_s397"
        / "best_balanced_protected_prefix"
    )
    assert _seed_token(path) == "397"


def test_integrity_binding_uses_request_tree_and_cross_seed_candidate(tmp_path: Path):
    proof_root = tmp_path / "formalv1_csem_t10_s1550_1599"
    base = _write_proof_run(proof_root, seed=185, variant="best_overall", threshold=10)
    other = _write_proof_run(proof_root, seed=397, variant="best_balanced", threshold=20)

    f1 = {"mutator": {"parameters": {"tamper_kind": "json_pointer"}}}
    _bind_integrity_mutation(f1, canonical="F1", proof_root=proof_root, source_root=tmp_path)
    assert f1["reuse_source_bundle"] == str(base)
    assert f1["mutator"]["parameters"]["target_file"].startswith("request/")
    assert f1["mutator"]["parameters"]["json_pointer"].endswith("/threshold_int")

    f2 = {"mutator": {"parameters": {"tamper_kind": "replace_from"}}}
    _bind_integrity_mutation(f2, canonical="F2", proof_root=proof_root, source_root=tmp_path)
    parameters = f2["mutator"]["parameters"]
    assert parameters["target_file"] == "candidate/artifacts/proof_request.json"
    assert Path(parameters["source_file"]) == other / "candidate" / "artifacts" / "proof_request.json"


def test_runtime_refresh_repairs_stale_f1_f2_and_d1_contract(tmp_path: Path):
    proof_root = tmp_path / "proofs"
    _write_proof_run(proof_root, seed=185, variant="best_overall", threshold=10)
    _write_proof_run(proof_root, seed=397, variant="best_balanced", threshold=20)
    config = {
        "mutations": [
            {
                "mutation_id": "F1_tree_tamper",
                "mutator": {"parameters": {"tamper_kind": "json_pointer"}},
            },
            {
                "mutation_id": "F2_cross_seed",
                "mutator": {"parameters": {"tamper_kind": "replace_from"}},
            },
            {
                "mutation_id": "D1_dynamic_envelope_gradient",
                "metadata": {"bundle_roots": [str(proof_root)]},
                "expected": {"allowed_first_failing_obligations": ["RTA_HI_BOUND"]},
            },
        ]
    }
    refresh_resolved_runtime_bindings(
        config,
        source_root=tmp_path,
        mutation_ids={"F1_tree_tamper", "F2_cross_seed", "D1_dynamic_envelope_gradient"},
    )
    by_id = {row["mutation_id"]: row for row in config["mutations"]}
    assert by_id["F1_tree_tamper"]["mutator"]["parameters"]["target_file"].startswith("request/")
    assert by_id["F2_cross_seed"]["mutator"]["parameters"]["source_file"]
    assert by_id["D1_dynamic_envelope_gradient"]["expected"]["allowed_first_failing_obligations"] == [
        "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"
    ]
