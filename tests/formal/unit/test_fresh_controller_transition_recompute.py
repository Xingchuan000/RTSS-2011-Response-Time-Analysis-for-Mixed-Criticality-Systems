from pathlib import Path
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.verifier.recompute import recompute_controller_transition_certificate


def test_fresh_verifier_rebuilds_controller_certificate_from_source() -> None:
    certificate = recompute_controller_transition_certificate(
        source_root=Path("."),
        verified_action_binding={
            "status": "PASS",
            "action_dim": 25,
            "explicit_noop": True,
            "action_space_type": "single",
        },
        verified_policy_binding={
            "status": "PASS",
            "artifact_hash": sha256_object({"policy": "fresh"}),
        },
        verified_controller_postclosure={
            "obligation_status": "PASS",
            "artifact_hash": sha256_object({"controller_postclosure": "fresh"}),
        },
        context_hash="0" * 64,
    )
    assert certificate["obligation_status"] == "PASS"
    assert certificate["witness"]["source_kind"] == "CONTROLLER_SYNCHRONOUS"
