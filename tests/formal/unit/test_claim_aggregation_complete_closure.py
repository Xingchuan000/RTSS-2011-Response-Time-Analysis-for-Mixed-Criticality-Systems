from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.registry import build_claim_closure
from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.verifier.aggregator import (
    aggregate_for_claim,
    claim_dependency_closure,
)


def test_all_pass_verified_surface_reaches_deployed_tree_proved():
    registry = list(resolve_registry("protected_prefix").entries)
    closure = build_claim_closure(registry, "DEPLOYED_HI_SAFETY")

    assert "CLAIM_AGGREGATION_RESULT" in closure.structural
    assert "CLAIM_AGGREGATION_RESULT" not in closure.verified_artifacts
    assert claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY") == set(
        closure.verified_artifacts
    )

    certificates = {
        obligation_id: obligation_certificate(
            obligation_id=obligation_id,
            status="PASS",
            context_hash="a" * 64,
            inputs={},
            witness={"test": True},
            checker_id="test",
            checker_version="test-v1",
        )
        for obligation_id in closure.verified_artifacts
    }
    outer_root = "b" * 64
    status_evidence = {
        obligation_id: {
            "obligation_id": obligation_id,
            "obligation_status": certificate["obligation_status"],
            "certificate_hash": certificate["artifact_hash"],
            "verified": True,
            "outer_bundle_root": outer_root,
        }
        for obligation_id, certificate in certificates.items()
    }

    assert aggregate_for_claim(
        claim="DEPLOYED_HI_SAFETY",
        registry=registry,
        verified_certificates=certificates,
        verified_status_evidence=status_evidence,
        verified_outer_root=outer_root,
    ) == "DEPLOYED_TREE_PROVED"
