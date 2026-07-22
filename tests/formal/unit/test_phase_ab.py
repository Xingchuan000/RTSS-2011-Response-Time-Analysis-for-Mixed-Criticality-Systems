"""Phase A/B 的最小验收测试；不接触 amc_py 运行语义。"""

import math

import pytest

from formal_toolchain.core.canonical_json import canonical_dumps, canonical_path
from formal_toolchain.core.contexts import build_contexts
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import validate_registry
from formal_toolchain.verifier.aggregator import aggregate


def test_canonical_json_is_order_independent_and_rejects_float():
    assert canonical_dumps({"b": 2, "a": 1}) == '{"a":1,"b":2}\n'
    with pytest.raises(TypeError):
        canonical_dumps({"x": 1.0})
    with pytest.raises(ValueError):
        canonical_dumps({"x": math.nan})
    assert canonical_path(r"a\\b/c") == "a/b/c"


def test_contexts_are_layered():
    contexts = build_contexts({"bootstrap": {"v": 1}, "implementation": {"v": 2},
                               "semantic": {"v": 3}, "policy": {"tree": "x"},
                               "candidate_envelope": {"value": 1},
                               "certified_envelope": {"candidate_envelope_hash": sha256_object({"value": 1}),
                                                      "preservation_certificate": {"obligation_status": "PASS", "witness": {"ok": True}},
                                                      "preservation_certificate_hash": sha256_object({"obligation_status": "PASS", "witness": {"ok": True}})},
                               "bridge": {}, "bundle_inputs": {}})
    assert set(contexts) == {"bootstrap_context", "implementation_context", "semantic_context",
                             "policy_context", "invariant_context", "reference_context",
                             "bridge_context", "composition_context", "bundle_context"}
    assert len(contexts["bundle_context"]["hash"]) == 64


def test_claim_aggregation_fail_closed_and_order_independent():
    gates = {"A", "B"}
    assert aggregate([{"id": "A", "status": "PASS"}, {"id": "B", "status": "PASS"}], gates) == "DEPLOYED_TREE_PROVED"
    assert aggregate([{"id": "A", "status": "PASS"}, {"id": "B", "status": "UNRESOLVED"}], gates) == "UNRESOLVED"
    assert aggregate([{"id": "A", "status": "MODEL_CONFORMANCE_FAILED"}, {"id": "B", "status": "PROOF_BUNDLE_INVALID"}], gates) == "PROOF_BUNDLE_INVALID"


def test_registry_dag_has_no_cycle():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parents[3] / "formal_toolchain/specs/obligation_registry.json").read_text())
    validate_registry(data["entries"])


def test_registry_rejects_path_traversal():
    with pytest.raises(ValueError):
        validate_registry([{
            "id": "X", "profile": "P0", "kind": "obligation", "activation": "active",
            "required": True, "depends_on": [], "artifact": "../x.json",
            "artifact_schema": "certificates/x.json", "summary_path": "x",
            "failure_route": "PROOF_BUNDLE_INVALID", "gates_claims": [],
            "status_evidence_rule": "certificate_status",
        }])
