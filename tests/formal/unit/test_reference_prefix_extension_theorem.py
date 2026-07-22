from __future__ import annotations

import json
from pathlib import Path
import pytest

from formal_toolchain.theory.backends.reference_prefix_extension import ReferencePrefixExtensionBackend
from formal_toolchain.theory.backends.reference_prefix_extension import (
    _verify_periodic_arithmetic,
    _verify_case_partition,
    _verify_closure_rank_decrease,
    _prove_unsat,
    verify_reference_prefix_extension_math,
)
from formal_toolchain.core.hashing import sha256_file


THEORY_DIR = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory"


def _load_theorem():
    stmt_path = THEORY_DIR / "statements" / "REFERENCE_PREFIX_EXTENSION.json"
    return json.loads(stmt_path.read_text(encoding="utf-8"))


def test_proof_object_hashes_match():
    theorem = _load_theorem()
    proof_obj = theorem.get("proof_object", {})
    assert proof_obj.get("path") == "proofs/REFERENCE_PREFIX_EXTENSION.proof.json"
    proof_path = THEORY_DIR / proof_obj["path"]
    assert proof_path.is_file()
    actual_hash = sha256_file(proof_path)
    assert actual_hash == proof_obj["sha256"], "proof object hash mismatch"


def test_backend_accepts_valid_proof():
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    backend = ReferencePrefixExtensionBackend()
    result = backend.verify(proof_path, theorem=theorem)
    assert result["status"] == "PASS", f"Backend rejected valid proof: {result}"


def test_backend_rejects_wrong_proof_hash():
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    bad_theorem = dict(theorem)
    bad_theorem["statement_hash"] = "0" * 64
    backend = ReferencePrefixExtensionBackend()
    result = backend.verify(proof_path, theorem=bad_theorem)
    assert result["status"] == "FAIL", "Should reject wrong statement hash"


def test_backend_rejects_missing_source_hash():
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    old_bindings = dict(proof["source_bindings"])
    proof["source_bindings"] = {
        "formal_toolchain/reference/reference_state.py": "0" * 64,
        "formal_toolchain/reference/executable_semantics.py": "0" * 64,
    }
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(proof, f)
        tmp_path = Path(f.name)
    try:
        backend = ReferencePrefixExtensionBackend()
        result = backend.verify(tmp_path, theorem=theorem)
        assert result["status"] == "FAIL", "Should reject wrong source hash"
    finally:
        tmp_path.unlink()


def test_backend_rejects_declared_axiom():
    stmt_path = THEORY_DIR / "statements" / "REFERENCE_PREFIX_EXTENSION.json"
    theorem = json.loads(stmt_path.read_text(encoding="utf-8"))
    assert theorem.get("assurance_level") != "DECLARED_AXIOM_TCB", \
        "Theorem must not remain declared axiom"


def test_backend_has_exact_three_case_ids():
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert len(proof.get("case_ids", [])) == 3
    expected = {
        "SAME_TIMESTAMP_CLOSURE",
        "READY_SERVICE_OR_EARLIER_BOUNDARY",
        "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
    }
    assert set(proof["case_ids"]) == expected


def test_backend_uses_z3_unsat():
    result = _verify_periodic_arithmetic()
    assert result["status"] == "PASS"
    assert {row["result"] for row in result["obligations"].values()} == {"UNSAT"}


def test_case_partition_is_machine_proved():
    result = _verify_case_partition()
    assert result["status"] == "PASS"
    assert set(result["obligations"]) == {
        "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE",
        "CLOSED_STATE_CASE_PARTITION_EXCLUSIVE",
    }


def test_case_partition_mutation_is_rejected():
    import z3
    context = z3.Context()
    result = _prove_unsat(
        z3=z3,
        obligation_id="CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE",
        proposition=z3.BoolVal(False, ctx=context),
        context=context,
    )
    assert result["status"] == "FAIL"


def test_closure_rank_is_machine_proved():
    result = _verify_closure_rank_decrease()
    assert result["status"] == "PASS"
    assert "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE" in result["obligations"]


def test_proof_object_contains_all_six_obligations():
    result = verify_reference_prefix_extension_math()
    assert result["status"] == "PASS"
    assert set(result["obligations"]) == {
        "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE",
        "CLOSED_STATE_CASE_PARTITION_EXCLUSIVE",
        "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE",
        "LEAST_FUTURE_RELEASE_STRICT",
        "LEAST_FUTURE_RELEASE_CONGRUENT",
        "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE",
    }


def test_backend_rejects_modified_arithmetic():
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["solver_obligation_receipts"]["LEAST_FUTURE_RELEASE_STRICT"]["smt2_hash"] = "0" * 64
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(proof, f)
        tmp_path = Path(f.name)
    try:
        result = ReferencePrefixExtensionBackend().verify(tmp_path, theorem=theorem)
        assert result["status"] == "FAIL"
    finally:
        tmp_path.unlink()


@pytest.mark.parametrize("mutation", [
    ["SAME_TIMESTAMP_CLOSURE", "SAME_TIMESTAMP_CLOSURE", "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT"],
    ["SAME_TIMESTAMP_CLOSURE", "READY_SERVICE_OR_EARLIER_BOUNDARY", "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT"],
    ["SAME_TIMESTAMP_CLOSURE", "READY_SERVICE_OR_EARLIER_BOUNDARY", "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT", "EXTRA"],
])
def test_case_ids_mutations_rejected(mutation):
    theorem = _load_theorem()
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["case_ids"] = mutation
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(proof, f)
        tmp_path = Path(f.name)
    try:
        assert ReferencePrefixExtensionBackend().verify(tmp_path, theorem=theorem)["status"] == "FAIL"
    finally:
        tmp_path.unlink()
