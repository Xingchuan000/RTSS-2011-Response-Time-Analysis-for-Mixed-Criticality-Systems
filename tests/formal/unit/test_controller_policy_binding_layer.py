from __future__ import annotations

import pytest

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.verifier.bridge_proof_checker import (
    _verified_controller_policy_binding_hash,
)


def _verified_policy_certificate() -> dict:
    return obligation_certificate(
        obligation_id="DEPLOYED_POLICY_PRESERVATION",
        status="PASS",
        context_hash="a" * 64,
        inputs={"fresh_process": True},
        witness={"semantic_preservation": True},
        checker_id="test.deployed_policy",
        checker_version="test-v1",
    )


def test_controller_replay_uses_verified_policy_certificate_artifact_hash() -> None:
    verified_policy = _verified_policy_certificate()
    semantic_payload_hash = sha256_object({"semantic": "pre-envelope"})
    assert semantic_payload_hash != verified_policy["artifact_hash"]

    selected = _verified_controller_policy_binding_hash(
        {"deployed_preservation_certificate_hash": semantic_payload_hash},
        verified_policy,
    )

    assert selected == verified_policy["artifact_hash"]
    assert selected != semantic_payload_hash


def test_controller_policy_binding_rejects_tampered_verified_certificate() -> None:
    verified_policy = _verified_policy_certificate()
    verified_policy["witness"]["semantic_preservation"] = False

    with pytest.raises(ValueError, match="VERIFIED_DEPLOYED_POLICY_CERTIFICATE_REQUIRED"):
        _verified_controller_policy_binding_hash(
            {"deployed_preservation_certificate_hash": "b" * 64},
            verified_policy,
        )
