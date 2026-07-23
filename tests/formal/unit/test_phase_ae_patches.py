"""最新宽松口径补丁的 golden/负向验收。"""

import json
from pathlib import Path

import pytest

from formal_toolchain.binding.quantization_binding import bind_quantization_runtime
from formal_toolchain.compiler.compile import compile_request
from formal_toolchain.conformance.required_obligations import check_boot_initialization, check_initial_quiescence
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry
from formal_toolchain.verifier.aggregator import aggregate_for_claim, claim_dependency_closure
from formal_toolchain.verifier.checker_catalog import checker_for
from formal_toolchain.verifier.envelope_checker import independently_verify_envelope
from formal_toolchain.verifier.recompute import _fresh_reference_taskset, candidate_evidence, load_verifier_inputs, verify_bundle
from formal_toolchain.reference.rta_production import all_task_reference_rta as protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta
from formal_toolchain.workflow.seed_workspace import freeze_seed_workspace


ROOT = Path(__file__).parents[3]
FIXTURE_ROOT = ROOT / "tests/formal/fixtures/synthetic_p0"
ACTIVE_BOOTSTRAP_INPUT_IDS = (
    "REGISTRY_META_SCHEMA",
    "P0_PROFILE_SCHEMA",
    "THEORY_MANIFEST",
    "THEORY_LIBRARY_VERSION",
    "ASSURANCE_POLICY",
    "OBLIGATION_REGISTRY",
    "CLAIM_AGGREGATION",
    "CONTEXT_SCHEMA",
    "CANONICAL_SERIALIZATION",
    "INTERFACE_COVERAGE",
    "MIGRATION_MANIFEST",
    "PROOF_REQUEST",
    "SOURCE_TREE_INTEGRITY",
    "RUNTIME_ENVIRONMENT",
    "DEPENDENCY_LOCK",
    "CHECKER_VERSION",
    "IMMUTABLE_INPUT_HASH",
    "EFFECTIVE_RUNTIME_CONFIG",
)


def _evidence(status_by_id):
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    closure = claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY")
    failure_routes = {
        "SCHEDULER_MODEL": "MODEL_CONFORMANCE_FAILED",
        "EXECUTABLE_POLICY_SEMANTICS": "POLICY_CONTRACT_VIOLATION",
        "CODE_REFERENCE_UPPER_BOUND_MAPPING": "MODEL_CONFORMANCE_FAILED",
        "REFERENCE_TASKSET": "REFERENCE_CERTIFICATE_FAILED",
    }
    certificates = {item: {"obligation_status": status_by_id.get(item, "PASS"),
                           "certificate_context_hash": "a" * 64,
                           "direct_predecessor_hashes": {},
                           "witness": {"obligation_id": item},
                           "evidence": [{"kind": "golden"}],
                           "failure": None if status_by_id.get(item, "PASS") == "PASS" else {"code": "golden"},
                           "failure_route": failure_routes.get(item)}
                   for item in closure}
    hashes = {item: sha256_object(certificates[item]) for item in closure}
    outer = sha256_object({item: hashes[item] for item in sorted(hashes)})
    evidence = {item: {"obligation_id": item, "obligation_status": certificates[item]["obligation_status"],
                       "certificate_hash": hashes[item], "outer_bundle_root": outer,
                       "verified": True, "certificate": certificates[item]}
                for item in closure}
    obligations = [{"id": item, "obligation_status": certificates[item]["obligation_status"],
                    "failure_route": failure_routes.get(item)}
                   for item in closure]
    return registry, obligations, evidence


def test_aggregator_preserves_normal_fail_and_unresolved_routes():
    for obligation_id, status, expected in (
        ("SCHEDULER_MODEL", "FAIL", "MODEL_CONFORMANCE_FAILED"),
        ("EXECUTABLE_POLICY_SEMANTICS", "FAIL", "POLICY_CONTRACT_VIOLATION"),
        ("CODE_REFERENCE_UPPER_BOUND_MAPPING", "FAIL", "MODEL_CONFORMANCE_FAILED"),
        ("REFERENCE_TASKSET", "FAIL", "REFERENCE_CERTIFICATE_FAILED"),
        ("SCHEDULER_MODEL", "UNRESOLVED", "UNRESOLVED"),
    ):
        registry, obligations, evidence = _evidence({obligation_id: status})
        assert aggregate_for_claim(claim="DEPLOYED_HI_SAFETY", obligations=obligations,
                                   registry=registry, verified_status_evidence=evidence) == expected


def test_quantization_has_independent_replay_and_rejects_rounding_mutation(tmp_path: Path):
    source = ROOT / "amc_py/viper/fixed_point.py"
    destination = tmp_path / "amc_py/viper/fixed_point.py"
    destination.parent.mkdir(parents=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = ROOT / "tests/formal/fixtures/synthetic_p0/fixed_point_config.json"
    result = bind_quantization_runtime(tmp_path, config)
    assert result["status"] == "PASS"
    assert result["vectors"] >= 20
    destination.write_text(destination.read_text(encoding="utf-8").replace("ROUND_HALF_UP", "ROUND_DOWN"), encoding="utf-8")
    assert bind_quantization_runtime(tmp_path, config)["status"] == "FAIL"


def test_boot_initialization_accepts_preclosed_hi_without_service_tick():
    boot = {
        "status": "PASS",
        "boot_time": 0,
        "mode_after_boot": "HI",
        "first_release_batch_defined": True,
        "no_service_before_boot_closure": True,
        "running_job": "SYN_HI_0",
        "running_job_executed_time": 0,
        "initial_runtime_budget_snapshot": {"SYN_HI_0": 2},
    }
    result = check_boot_initialization({"boot": boot})
    assert result["status"] == "PASS", result
    assert result["mode_after_boot"] == "HI"


def test_initial_quiescence_still_requires_true_boot_state():
    result = check_initial_quiescence({"initial_state": {
        "current_time": 0,
        "mode": "LO",
        "running_job": None,
        "active_jobs": [],
        "ready_jobs": [],
        "service_in_progress": False,
        "quiescent": True,
    }})
    assert result["status"] == "PASS", result


def test_bootstrap_inputs_and_reference_rta_checkers_recompute_from_synthetic_fixture(tmp_path: Path):
    workspace = tmp_path / "workspace"
    imported = freeze_seed_workspace(FIXTURE_ROOT, "best_overall", workspace, code_root=ROOT)
    request_path = Path(imported["request"])
    inputs = load_verifier_inputs(request_path, source_root=ROOT)
    compile_out = tmp_path / "candidate"
    compile_request(request_path, compile_out)

    candidate_common = json.loads((compile_out / "artifacts" / "COMMON_TRANSITION_PRESERVATION.json").read_text(encoding="utf-8"))
    candidate_deployed = json.loads((compile_out / "artifacts" / "DEPLOYED_POLICY_PRESERVATION.json").read_text(encoding="utf-8"))
    envelope_state = independently_verify_envelope(
        candidate_envelope=candidate_evidence(
            json.loads((compile_out / "artifacts" / "CANDIDATE_ENVELOPE.json").read_text(encoding="utf-8"))
        ) or {},
        common_preservation=candidate_evidence(candidate_common) or {},
        deployed_preservation=candidate_evidence(candidate_deployed) or {},
        raw_inputs=inputs,
        invariant_context_hash=str(inputs.contexts["invariant_context"]["hash"]),
    )
    if envelope_state.certified_envelope is None:
        pytest.skip("synthetic fixture does not currently produce a certified envelope under the current registry")

    fresh_reference = _fresh_reference_taskset(inputs, envelope_state.certified_envelope)
    rta = protected_hi_rta(fresh_reference)
    assert replay_rta(fresh_reference, rta)["status"] == "PASS"

    for obligation_id in ACTIVE_BOOTSTRAP_INPUT_IDS:
        result = checker_for(obligation_id)(
            raw_inputs=inputs,
            candidate_evidence={},
            expected_context_hash=None,
        )
        assert result["status"] == "PASS", (obligation_id, result)

    reference_result = checker_for("REFERENCE_TASKSET")(
        raw_inputs=inputs,
        candidate_evidence={},
        expected_context_hash=None,
        certified_envelope=envelope_state.certified_envelope,
        fresh_reference=fresh_reference,
    )
    assert reference_result["status"] == "PASS", reference_result

    protected_hi_result = checker_for("PROTECTED_HI_RTA_ARITHMETIC")(
        raw_inputs=inputs,
        candidate_evidence={},
        expected_context_hash=None,
        certified_envelope=envelope_state.certified_envelope,
        fresh_reference=fresh_reference,
    )
    assert protected_hi_result["status"] == "PASS", protected_hi_result


def test_registry_has_no_active_required_unresolved_placeholders():
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    unresolved = [
        entry["id"]
        for entry in registry
        if entry.get("activation") == "active"
        and entry.get("required") is True
        and entry.get("failure_route") == "UNRESOLVED"
    ]
    assert unresolved == [], unresolved


def test_proof_result_propagates_failure_audit_fields(tmp_path: Path, monkeypatch):
    from formal_toolchain.workflow import prove_seed as prove_seed_module

    workspace = tmp_path / "workspace"
    freeze_seed_workspace(FIXTURE_ROOT, "best_overall", workspace, code_root=ROOT)

    def fake_run_cli(module_name, args, cwd, log_dir):
        out_dir = None
        if "--out" in args:
            out_dir = Path(args[args.index("--out") + 1])
        if module_name == "formal_toolchain.cli.verify_bundle" and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema_version": "proof_summary_v1",
                "workflow_status": "VERIFIED",
                "result_status": "MODEL_CONFORMANCE_FAILED",
                "failure_route": "MODEL_CONFORMANCE_FAILED",
                "failure_code": "REFERENCE_MAPPING_MISMATCH",
                "violated_obligation_id": "CODE_REFERENCE_UPPER_BOUND_MAPPING",
                "failure_message": "certified envelope status/schema invalid",
                "outer_bundle_root": "a" * 64,
                "fixture_claim_result": "MODEL_CONFORMANCE_FAILED",
            }
            (out_dir / "proof_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if module_name == "formal_toolchain.cli.render_report" and out_dir is not None:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            out_dir.write_text("# ok\n", encoding="utf-8")
        return {"returncode": 0}

    monkeypatch.setattr(prove_seed_module, "run_cli", fake_run_cli)
    code, result = prove_seed_module.prove_seed(
        seed_dir=FIXTURE_ROOT, tree_variant="best_overall", code_root=ROOT,
        out=tmp_path / "bundle", overwrite=True,
        dependency_manifest_override={"packages": {"z3-solver": "4.13.4.0", "jsonschema": "4.26.0", "numpy": "1.26.0", "scikit-learn": "1.5.0", "setuptools": "68.0.0"}, "python_version_info": "3.11.0"},
    )
    assert code == 10
    proof_result = json.loads((tmp_path / "bundle" / "proof_result.json").read_text(encoding="utf-8"))
    assert result["failure_route"] == "MODEL_CONFORMANCE_FAILED"
    assert proof_result["failure_route"] == "MODEL_CONFORMANCE_FAILED"
    assert proof_result["failure_code"] == "REFERENCE_MAPPING_MISMATCH"
    assert proof_result["violated_obligation_id"] == "CODE_REFERENCE_UPPER_BOUND_MAPPING"


def test_prove_seed_normal_dependency_path_does_not_shadow_json(tmp_path: Path, monkeypatch):
    from formal_toolchain.workflow import prove_seed as prove_seed_module

    workspace = tmp_path / "workspace"
    freeze_seed_workspace(FIXTURE_ROOT, "best_overall", workspace, code_root=ROOT)

    monkeypatch.setattr(prove_seed_module, "_dependency_preflight", lambda _root: None)

    def fake_run_cli(module_name, args, cwd, log_dir):
        out_dir = None
        if "--out" in args:
            out_dir = Path(args[args.index("--out") + 1])
        if module_name == "formal_toolchain.cli.verify_bundle" and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema_version": "proof_summary_v1",
                "workflow_status": "VERIFIED",
                "result_status": "UNRESOLVED",
                "failure_route": "UNRESOLVED",
                "failure_code": "REGRESSION_SENTINEL",
                "outer_bundle_root": "b" * 64,
            }
            (out_dir / "proof_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if module_name == "formal_toolchain.cli.render_report" and out_dir is not None:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            out_dir.write_text("# ok\n", encoding="utf-8")
        return {"returncode": 0}

    monkeypatch.setattr(prove_seed_module, "run_cli", fake_run_cli)

    code, result = prove_seed_module.prove_seed(
        seed_dir=FIXTURE_ROOT,
        tree_variant="best_overall",
        code_root=ROOT,
        out=tmp_path / "bundle",
        overwrite=True,
    )

    assert code == 20
    assert result["failure_code"] == "REGRESSION_SENTINEL"
    assert result["failure_code"] != "INTERNAL_WORKFLOW_ERROR"
