from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.verifier.bridge_proof_checker import verify_closed_prefix_proof_object


def test_candidate_boolean_cannot_authorize_n5():
    candidate = obligation_certificate(
        obligation_id="CLOSED_PREFIX_REFINEMENT", status="PASS", context_hash="a" * 64,
        inputs={}, witness={"pointwise_closed_prefix_relation": True},
        checker_id="test", checker_version="1")
    result = verify_closed_prefix_proof_object(candidate=candidate, bridge_context_hash="a" * 64)
    assert result["status"] == "FAIL"
    assert result["code"] == "CLOSED_PREFIX_LEGACY_SCHEMA_REJECTED"
