"""最新宽松口径补丁的 golden/负向验收。"""

import json
from pathlib import Path

from formal_toolchain.binding.quantization_binding import bind_quantization_runtime
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry
from formal_toolchain.verifier.aggregator import aggregate_for_claim, claim_dependency_closure


ROOT = Path(__file__).parents[3]


def _evidence(status_by_id):
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    closure = claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY")
    certificates = {item: {"obligation_status": status_by_id.get(item, "PASS"),
                           "certificate_context_hash": "a" * 64,
                           "direct_predecessor_hashes": {},
                           "witness": {"obligation_id": item},
                           "evidence": [{"kind": "golden"}],
                           "failure": None if status_by_id.get(item, "PASS") == "PASS" else {"code": "golden"}}
                   for item in closure}
    hashes = {item: sha256_object(certificates[item]) for item in closure}
    outer = sha256_object({item: hashes[item] for item in sorted(hashes)})
    evidence = {item: {"obligation_id": item, "obligation_status": certificates[item]["obligation_status"],
                       "certificate_hash": hashes[item], "outer_bundle_root": outer,
                       "verified": True, "certificate": certificates[item]}
                for item in closure}
    obligations = [{"id": item, "obligation_status": certificates[item]["obligation_status"]} for item in closure]
    return registry, obligations, evidence


def test_aggregator_preserves_normal_fail_and_unresolved_routes():
    for obligation_id, status, expected in (
        ("SCHEDULER_MODEL", "FAIL", "MODEL_CONFORMANCE_FAILED"),
        ("EXECUTABLE_POLICY_SEMANTICS", "FAIL", "POLICY_CONTRACT_VIOLATION"),
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
