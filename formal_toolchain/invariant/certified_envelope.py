"""Phase H05：candidate 只有在全部 preservation certificate PASS 后才能认证。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_object, sha256_proof_object


def build_candidate_envelope_view(
    candidate: Mapping[str, Any],
    common: Mapping[str, Any],
    deployed: Mapping[str, Any],
) -> dict[str, Any]:
    """构造 compiler/verifier 共享的未认证 reference envelope view。

    该对象只允许用于候选 reference/route 上下文的确定性重算；它明确携带
    ``CANDIDATE_UNVERIFIED``，不能越过 Phase H 认证边界充当 certified envelope。
    """

    return {
        "status": "CANDIDATE",
        "schema_version": "candidate_envelope_view_v1",
        "trust_level": "CANDIDATE_UNVERIFIED",
        "not_a_certified_envelope": True,
        "candidate_envelope_hash": sha256_proof_object(candidate),
        "common_candidate_hash": sha256_proof_object(common),
        "deployed_candidate_hash": sha256_proof_object(deployed),
        "lower": dict(candidate.get("lower", {})),
        "upper": dict(candidate.get("upper", {})),
        "active_release_budget_upper": dict(
            candidate.get("active_release_budget_upper", {})
        ),
    }


def certify_envelope(candidate: Mapping[str, Any], common: Mapping[str, Any], deployed: Mapping[str, Any], *, context_hash: str | None = None,
                     verifier_attestation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """保留兼容入口，但 compiler/test caller 不得直接认证。"""
    raise ValueError("certified envelope 只能由 formal verifier 内部接口生成")


def _certify_envelope_from_verifier(candidate: Mapping[str, Any], common: Mapping[str, Any], deployed: Mapping[str, Any], *, context_hash: str,
                                    verifier_attestation: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("status") != "PASS" or common.get("status") != "PASS" or deployed.get("status") != "PASS":
        raise ValueError("candidate/common/deployed 任一未通过，不能生成 certified envelope")
    if context_hash is None:
        raise ValueError("certified envelope 必须绑定 context_hash")
    if verifier_attestation is None or verifier_attestation.get("fresh_process") is not True:
        raise ValueError("certified envelope 只能由 fresh-process verifier attestation 生成")
    required_hashes = {"candidate_hash": sha256_proof_object(candidate), "common_hash": sha256_proof_object(common),
                       "deployed_hash": sha256_proof_object(deployed)}
    if any(verifier_attestation.get(key) != value for key, value in required_hashes.items()):
        raise ValueError("verifier attestation 与 preservation 输入 hash 不一致")
    preservation = {"obligation_status": "PASS", "candidate_hash": sha256_proof_object(candidate),
                    "common_hash": sha256_proof_object(common), "deployed_hash": sha256_proof_object(deployed),
                    "fresh_process": True}
    candidate_hash = sha256_proof_object(candidate)
    coordinate_upper_witness_hash = sha256_object(candidate.get("coordinate_upper_witnesses", {}))
    return {
        "status": "PASS",
        "schema_version": "certified_envelope_v3",
        "method": candidate.get("method"),
        "safety_polytope_hash": candidate.get("safety_polytope_hash"),
        "coordinate_upper_witness_hash": coordinate_upper_witness_hash,
        "action_transition_hash": deployed.get("action_transition_hash"),
        "mask_fallback_hash": deployed.get("mask_fallback_hash"),
        "candidate_envelope_hash": candidate_hash,
        "context_hash": context_hash,
        "preservation_certificate_hash": sha256_object(preservation),
        "preservation_certificate": preservation,
        "lower": dict(candidate["lower"]),
        "upper": dict(candidate["upper"]),
        "active_release_budget_upper": dict(candidate["active_release_budget_upper"]),
    }
