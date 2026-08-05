from __future__ import annotations

from formal_toolchain.core.hashing import sha256_object, sha256_proof_object
from formal_toolchain.verifier import envelope_checker


def test_deployed_failure_keeps_untrusted_candidate_reference_view(monkeypatch):
    candidate = {
        "status": "PASS",
        "schema_version": "candidate_envelope_v2",
        "method": "single_action_safety_polytope_projection",
        "context_hash": "a" * 64,
        "lower": {"task": 1},
        "candidate_positive_lower": {"task": 1},
        "upper": {"task": 3},
        "active_release_budget_upper": {"task": 3},
        "action_hard_upper": {"task": 5},
        "safety_polytope_hash": "b" * 64,
        "safety_polytope_rows": [],
        "coordinate_upper_witnesses": {},
        "initial_budget": {"task": 2},
        "initial_budget_in_polytope": True,
        "proof_rule": "test-rule",
    }
    common = {
        "status": "PASS",
        "schema_version": "common_transition_preservation_v1",
        "active_release_budget_immutable": True,
        "controller_budget_write": False,
        "invariant_checked": True,
        "candidate_envelope_hash": sha256_object(candidate),
        "safety_polytope_hash": candidate["safety_polytope_hash"],
    }
    deployed = {
        "status": "FAIL",
        "schema_version": "deployed_policy_preservation_v2",
        "failure": {
            "route": "POLICY_CONTRACT_VIOLATION",
            "code": "DEPLOYED_POLICY_PRESERVATION_FAILED",
        },
    }

    monkeypatch.setattr(
        envelope_checker,
        "_expected_candidate_from_math",
        lambda raw_inputs, invariant_context_hash: candidate,
    )

    result = envelope_checker.independently_verify_envelope(
        candidate_envelope=candidate,
        common_preservation=common,
        deployed_preservation=deployed,
        raw_inputs=object(),
        invariant_context_hash="a" * 64,
    )

    assert result.candidate_status == "PASS"
    assert result.common_status == "PASS"
    assert result.deployed_status == "FAIL"
    assert result.certified_envelope is None

    reference_view = result.candidate_reference_envelope
    assert reference_view is not None
    assert reference_view["trust_level"] == "CANDIDATE_UNVERIFIED"
    assert reference_view["not_a_certified_envelope"] is True
    assert reference_view["candidate_envelope_hash"] == sha256_proof_object(candidate)
    assert reference_view["common_candidate_hash"] == sha256_proof_object(common)
    assert reference_view["deployed_candidate_hash"] == sha256_proof_object(deployed)
