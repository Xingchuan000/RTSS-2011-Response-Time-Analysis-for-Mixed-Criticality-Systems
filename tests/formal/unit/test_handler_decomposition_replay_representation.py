from __future__ import annotations

from formal_toolchain.verifier.bridge_replay import (
    _handler_decomposition_replay_inputs,
)


def test_fresh_handler_replay_uses_raw_semantic_proof_rows() -> None:
    raw = [{"case_id": "BOOT_TO_PRECLOSED_0", "z3_proof_result": "PASS"}]
    envelope = [{
        "artifact_schema_version": "certificate_envelope_v2",
        "artifact_hash": "a" * 64,
        "witness": dict(raw[0]),
    }]
    compiled = {"proofs": raw, "proof_certificates": envelope}

    assert _handler_decomposition_replay_inputs(compiled) == raw


def test_fresh_handler_replay_rejects_missing_raw_proofs() -> None:
    compiled = {
        "proofs": [],
        "proof_certificates": [{
            "artifact_schema_version": "certificate_envelope_v2",
            "artifact_hash": "b" * 64,
            "witness": {"case_id": "BOOT_TO_PRECLOSED_0"},
        }],
    }

    assert _handler_decomposition_replay_inputs(compiled) == []
