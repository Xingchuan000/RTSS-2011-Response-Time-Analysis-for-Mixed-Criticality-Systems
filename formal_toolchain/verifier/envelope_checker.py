"""fresh verifier 的 candidate/common/deployed envelope 认证边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.invariant.safety_polytope import (
    derive_componentwise_upper,
    verify_production_rows,
    vector_satisfies_rows,
)


@dataclass(frozen=True)
class VerifiedEnvelopeResult:
    candidate_status: str
    common_status: str
    deployed_status: str
    certified_envelope: dict[str, Any] | None


def _task_names(raw_inputs: Any) -> list[str]:
    return [str(task.name) for task in raw_inputs.target.ordered_tasks]


def _expected_candidate_from_math(raw_inputs: Any, invariant_context_hash: str) -> dict[str, Any]:
    domain = build_budget_domain(
        raw_inputs.target.ordered_tasks,
        raw_inputs.target.provenance.get("budget_by_task"),
        runtime_config=raw_inputs.target.runtime_config,
    )
    domain["context_hash"] = invariant_context_hash
    adapter = raw_inputs.target.runtime_adapter
    if adapter is None:
        raise ValueError("FORMAL_RUNTIME_ADAPTER_MISSING")
    production_polytope = adapter.export_budget_safety_polytope()
    verified = verify_production_rows(production_polytope, raw_inputs.target.ordered_tasks)
    if verified.get("status") != "PASS":
        raise ValueError("SAFETY_POLYTOPE_PRODUCTION_MISMATCH")

    names = _task_names(raw_inputs)
    formal_lower = {name: int(domain["tasks"][name]["formal_lower"]) for name in names}
    candidate_lower = {
        name: int(domain["tasks"][name]["candidate_positive_lower"])
        for name in names
    }
    hard_upper = {
        name: int(domain["tasks"][name]["action_hard_upper"])
        for name in names
    }
    derived = derive_componentwise_upper(
        rows=verified["rows"],
        task_order=names,
        candidate_lower=candidate_lower,
        action_hard_upper=hard_upper,
    )
    if derived.get("status") != "PASS":
        raise ValueError("SAFETY_POLYTOPE_COMPONENTWISE_UPPER_FAILED")

    initial = {name: int(domain["tasks"][name]["initial"]) for name in names}
    if not vector_satisfies_rows(initial, verified["rows"]):
        raise ValueError("INITIAL_BUDGET_OUTSIDE_SAFETY_POLYTOPE")

    upper = dict(derived["upper"])
    return {
        "status": "PASS",
        "schema_version": "candidate_envelope_v2",
        "method": "single_action_safety_polytope_projection",
        "context_hash": invariant_context_hash,
        "lower": formal_lower,
        "candidate_positive_lower": candidate_lower,
        "upper": upper,
        "active_release_budget_upper": dict(upper),
        "action_hard_upper": hard_upper,
        "safety_polytope_hash": verified["row_hash"],
        "safety_polytope_rows": verified["rows"],
        "coordinate_upper_witnesses": derived["witnesses"],
        "initial_budget": initial,
        "initial_budget_in_polytope": True,
        "proof_rule": (
            "mask_valid_non_noop_implies_candidate_in_polytope; "
            "polytope_implies_componentwise_envelope"
        ),
    }


def independently_verify_envelope(*, candidate_envelope: Mapping[str, Any], common_preservation: Mapping[str, Any],
                                  deployed_preservation: Mapping[str, Any], raw_inputs: Any,
                                  policy_regions: Any = None, common_transition_ir: Any = None,
                                  invariant_context_hash: str) -> VerifiedEnvelopeResult:
    """重新检查 candidate 与两项 preservation，成功后才生成 trusted envelope."""
    try:
        expected_candidate = _expected_candidate_from_math(raw_inputs, invariant_context_hash)
    except (KeyError, TypeError, ValueError):
        return VerifiedEnvelopeResult("FAIL", "UNRESOLVED", "UNRESOLVED", None)

    if sha256_object(dict(candidate_envelope)) != sha256_object(expected_candidate):
        return VerifiedEnvelopeResult("FAIL", "UNRESOLVED", "UNRESOLVED", None)

    if (
        common_preservation.get("status") != "PASS"
        or common_preservation.get("schema_version") != "common_transition_preservation_v1"
        or common_preservation.get("active_release_budget_immutable") is not True
        or common_preservation.get("controller_budget_write") is not False
        or common_preservation.get("invariant_checked") is not True
        or common_preservation.get("candidate_envelope_hash") != sha256_object(expected_candidate)
        or common_preservation.get("safety_polytope_hash") != expected_candidate.get("safety_polytope_hash")
    ):
        return VerifiedEnvelopeResult("PASS", "FAIL", "UNRESOLVED", None)

    deployed_fields = {
        "universal_state_quantification",
        "state_enumeration_used",
        "proof_rule",
        "candidate_envelope_hash",
        "mask_fallback_hash",
        "action_transition_hash",
        "safety_polytope_hash",
        "permanently_masked_action_ids",
        "action_witnesses",
        "implicit_noop_checked",
    }
    if (
        deployed_preservation.get("status") != "PASS"
        or deployed_preservation.get("schema_version") != "deployed_policy_preservation_v2"
        or not deployed_fields <= set(deployed_preservation)
        or deployed_preservation.get("universal_state_quantification") is not True
        or deployed_preservation.get("state_enumeration_used") is not False
        or deployed_preservation.get("implicit_noop_checked") is not True
        or deployed_preservation.get("candidate_envelope_hash") != sha256_object(expected_candidate)
        or deployed_preservation.get("safety_polytope_hash") != expected_candidate.get("safety_polytope_hash")
    ):
        return VerifiedEnvelopeResult("PASS", "PASS", "FAIL", None)

    preservation = {
        "obligation_status": "PASS",
        "candidate_hash": sha256_object(candidate_envelope),
        "common_hash": sha256_object(common_preservation),
        "deployed_hash": sha256_object(deployed_preservation),
        "fresh_process": True,
    }
    certified = {
        "schema_version": "certified_envelope_v3",
        "status": "PASS",
        "trust_level": "VERIFIED",
        "method": expected_candidate.get("method"),
        "safety_polytope_hash": expected_candidate.get("safety_polytope_hash"),
        "coordinate_upper_witness_hash": sha256_object(expected_candidate.get("coordinate_upper_witnesses", {})),
        "candidate_envelope_hash": sha256_object(candidate_envelope),
        "action_transition_hash": deployed_preservation.get("action_transition_hash"),
        "mask_fallback_hash": deployed_preservation.get("mask_fallback_hash"),
        "common_preservation_certificate_hash": sha256_object(common_preservation),
        "deployed_preservation_certificate_hash": sha256_object(deployed_preservation),
        "preservation_certificate_hash": sha256_object(preservation),
        "preservation_certificate": preservation,
        "certificate_context_hash": invariant_context_hash,
        "lower": dict(expected_candidate["lower"]),
        "upper": dict(expected_candidate["upper"]),
        "active_release_budget_upper": dict(expected_candidate["active_release_budget_upper"]),
        "verified_by": "fresh_verifier",
    }
    certified["artifact_hash"] = sha256_object(certified)
    return VerifiedEnvelopeResult("PASS", "PASS", "PASS", certified)
