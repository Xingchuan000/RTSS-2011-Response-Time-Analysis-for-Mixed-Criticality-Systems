from __future__ import annotations

import json
from pathlib import Path
import pytest

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.theory.loader import TCB_BACKENDS
from formal_toolchain.verifier.bridge_proof_checker import verify_prefix_extension_proof_object


THEORY_DIR = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory"


def _load_theorem():
    stmt_path = THEORY_DIR / "statements" / "REFERENCE_PREFIX_EXTENSION.json"
    return json.loads(stmt_path.read_text(encoding="utf-8"))


def test_theorem_loaded_by_loader():
    theorem = _load_theorem()
    assert "reference-prefix-extension-z3-v3" in TCB_BACKENDS
    backend = TCB_BACKENDS["reference-prefix-extension-z3-v3"]
    proof_path = THEORY_DIR / theorem["proof_object"]["path"]
    result = backend.verify(proof_path, theorem=theorem)
    assert result["status"] == "PASS"


def test_legacy_v2_schema_rejected():
    legacy = {
        "artifact_schema_version": "synthetic_v2",
        "obligation_id": "REFERENCE_PREFIX_EXTENSION",
        "obligation_status": "PASS",
        "certificate_context_hash": "c" * 64,
        "direct_predecessor_hashes": {
            "REFERENCE_TASKSET": "0" * 64,
            "TIME_PROGRESS": "1" * 64,
            "EFFECTIVE_EVENT_ORDER": "2" * 64,
        },
        "checker_id": "test",
        "checker_version": "v2",
        "inputs": {
            "theorem_id": "REFERENCE_PREFIX_EXTENSION",
        },
        "witness": {
            "schema_version": "reference_prefix_extension_v2",
            "quantification": "FOR_ALL_FINITE_VALID_REFERENCE_PREFIXES",
            "multiple_pending_jobs_supported": True,
            "finite_prefix_job_map_total": True,
            "horizon_independent": True,
        },
        "evidence": [],
        "failure": None,
    }
    from formal_toolchain.core.hashing import sha256_object
    legacy["artifact_hash"] = sha256_object({k: legacy[k] for k in (
        "artifact_schema_version", "obligation_id", "obligation_status",
        "certificate_context_hash", "direct_predecessor_hashes", "checker_id",
        "checker_version", "inputs", "witness", "evidence", "failure"
    )})

    result = verify_prefix_extension_proof_object(
        candidate=legacy,
        bridge_context_hash="c" * 64,
        contexts={}, predecessors={}, reference_taskset={},
    )
    assert result["status"] == "FAIL"
    assert "LEGACY" in result.get("code", "")


def test_missing_predecessor_rejected():
    theorem = _load_theorem()
    stmt_hash = theorem.get("statement_hash", "0" * 64)
    assump_hash = theorem.get("assumption_hash", "0" * 64)
    proof_obj_hash = theorem.get("proof_object", {}).get("sha256", "0" * 64)

    candidate = {
        "artifact_schema_version": "certificate_envelope_v2",
        "obligation_id": "REFERENCE_PREFIX_EXTENSION",
        "obligation_status": "PASS",
        "certificate_context_hash": "c" * 64,
        "direct_predecessor_hashes": {
            "TIME_PROGRESS": "1" * 64,
            "EFFECTIVE_EVENT_ORDER": "2" * 64,
        },
        "checker_id": "test",
        "checker_version": "v4",
        "inputs": {
            "theorem_id": "REFERENCE_PREFIX_EXTENSION",
            "reference_taskset_fingerprint": "0" * 64,
            "theorem_statement_hash": stmt_hash,
            "theorem_assumption_hash": assump_hash,
            "theorem_proof_object_hash": proof_obj_hash,
            "reference_state_source_hash": "0" * 64,
            "executable_semantics_source_hash": "0" * 64,
        },
        "witness": {
            "schema_version": "reference_prefix_extension_v4",
            "closed_state_predicate": "is_closed_reference_state",
            "case_ids": [
                "SAME_TIMESTAMP_CLOSURE",
                "READY_SERVICE_OR_EARLIER_BOUNDARY",
                "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
            ],
            "backend_receipt_hash": "0" * 64,
            "theorem_proof_receipt": {"verified": True},
        },
        "evidence": [],
        "failure": None,
    }
    from formal_toolchain.core.hashing import sha256_object
    candidate["artifact_hash"] = sha256_object({k: candidate[k] for k in (
        "artifact_schema_version", "obligation_id", "obligation_status",
        "certificate_context_hash", "direct_predecessor_hashes", "checker_id",
        "checker_version", "inputs", "witness", "evidence", "failure"
    )})

    import logging
    logging.basicConfig(level=logging.DEBUG)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("formal_toolchain.verifier.bridge_proof_checker.verify_obligation_certificate", lambda _: True)
        result = verify_prefix_extension_proof_object(
            candidate=candidate,
            bridge_context_hash="c" * 64,
            contexts={}, predecessors={}, reference_taskset={},
        )
    assert result["status"] == "FAIL", f"Expected FAIL, got {result}"
    assert "PREDECESSOR" in result.get("code", "")
