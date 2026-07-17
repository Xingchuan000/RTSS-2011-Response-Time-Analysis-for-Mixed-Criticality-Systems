"""fresh verifier 的 candidate/common/deployed envelope 认证边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True)
class VerifiedEnvelopeResult:
    candidate_status: str
    common_status: str
    deployed_status: str
    certified_envelope: dict[str, Any] | None


def independently_verify_envelope(*, candidate_envelope: Mapping[str, Any], common_preservation: Mapping[str, Any],
                                  deployed_preservation: Mapping[str, Any], raw_inputs: Any,
                                  policy_regions: Any = None, common_transition_ir: Any = None,
                                  invariant_context_hash: str) -> VerifiedEnvelopeResult:
    """重新检查 candidate 与两项 preservation，成功后才生成 trusted envelope。"""
    # 先重算有限预算域，确认 candidate envelope 的数值确实来自当前
    # verifier target；仅检查 status=PASS 会让任意 upper/lower 被放行。
    try:
        from formal_toolchain.conformance.time_domain import build_budget_domain
        domain = build_budget_domain(
            raw_inputs.target.ordered_tasks,
            raw_inputs.target.provenance.get("budget_by_task"),
            runtime_config=raw_inputs.target.runtime_config,
        )
        expected_upper = {name: int(row["runtime_deploy_cap"])
                          for name, row in domain["tasks"].items()}
        expected_lower = {
            name: (int(row["code_lower"])
                   if str(getattr(next(task for task in raw_inputs.target.ordered_tasks
                                     if str(task.name) == name).criticality,
                              "value", next(task for task in raw_inputs.target.ordered_tasks
                                             if str(task.name) == name).criticality)) == "HI"
                   else 0)
            for name, row in domain["tasks"].items()
        }
    except (KeyError, TypeError, ValueError, StopIteration):
        return VerifiedEnvelopeResult("FAIL", "UNRESOLVED", "UNRESOLVED", None)
    if (candidate_envelope.get("schema_version") != "candidate_envelope_v1"
            or candidate_envelope.get("status") != "PASS"
            or dict(candidate_envelope.get("upper", {})) != expected_upper
            or dict(candidate_envelope.get("active_release_budget_upper", {})) != expected_upper
            or dict(candidate_envelope.get("lower", {})) != expected_lower):
        return VerifiedEnvelopeResult("FAIL", "UNRESOLVED", "UNRESOLVED", None)

    # preservation 证据还必须携带其约定的语义字段；status 本身不是
    # preservation proof。字段缺失或被篡改时，fresh verifier 不生成 envelope。
    if (common_preservation.get("status") != "PASS"
            or common_preservation.get("schema_version") != "common_transition_preservation_v1"
            or common_preservation.get("active_release_budget_immutable") is not True
            or common_preservation.get("controller_budget_write") is not False
            or common_preservation.get("invariant_checked") is not True):
        return VerifiedEnvelopeResult("PASS", "FAIL", "UNRESOLVED", None)
    deployed_fields = ("leaf_count", "budget_state_count", "selected_action_ids",
                       "first_valid_positions", "noop_state_count", "witnesses")
    if (deployed_preservation.get("status") != "PASS"
            or deployed_preservation.get("schema_version") != "deployed_policy_preservation_v1"
            or any(field not in deployed_preservation for field in deployed_fields)
            or not isinstance(deployed_preservation.get("leaf_count"), int)
            or not isinstance(deployed_preservation.get("budget_state_count"), int)
            or not isinstance(deployed_preservation.get("selected_action_ids"), list)
            or not isinstance(deployed_preservation.get("first_valid_positions"), list)
            or len(deployed_preservation["selected_action_ids"]) != len(deployed_preservation["first_valid_positions"])
            or not isinstance(deployed_preservation.get("noop_state_count"), int)
            or not isinstance(deployed_preservation.get("witnesses"), list)
            or deployed_preservation.get("implicit_noop_checked") is not True):
        return VerifiedEnvelopeResult("PASS", "PASS", "FAIL", None)
    upper = candidate_envelope.get("upper")
    lower = candidate_envelope.get("lower")
    active = candidate_envelope.get("active_release_budget_upper")
    if not isinstance(upper, Mapping) or not isinstance(lower, Mapping) or not isinstance(active, Mapping) or dict(upper) != dict(active):
        return VerifiedEnvelopeResult("FAIL", "UNRESOLVED", "UNRESOLVED", None)
    preservation = {"obligation_status": "PASS",
                    "candidate_hash": sha256_object(dict(candidate_envelope)),
                    "common_hash": sha256_object(dict(common_preservation)),
                    "deployed_hash": sha256_object(dict(deployed_preservation)),
                    "verified_by": "fresh_verifier"}
    certified = {
        "schema_version": "certified_envelope_v2", "status": "PASS",
        "trust_level": "VERIFIED", "candidate_envelope_hash": preservation["candidate_hash"],
        "common_preservation_certificate_hash": preservation["common_hash"],
        "deployed_preservation_certificate_hash": preservation["deployed_hash"],
        "preservation_certificate_hash": sha256_object(preservation),
        "preservation_certificate": preservation, "certificate_context_hash": invariant_context_hash,
        "lower": dict(lower), "upper": dict(upper), "active_release_budget_upper": dict(active),
        "verified_by": "fresh_verifier",
    }
    certified["artifact_hash"] = sha256_object(certified)
    return VerifiedEnvelopeResult("PASS", "PASS", "PASS", certified)
