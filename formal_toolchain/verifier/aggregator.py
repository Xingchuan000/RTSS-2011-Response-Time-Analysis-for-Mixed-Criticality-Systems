"""严格、与 JSON 顺序无关的 claim aggregation。"""

from __future__ import annotations

import json
from pathlib import Path
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.obligation_ids import LEGACY_PROTECTED_HI_IDS
from formal_toolchain.core.registry import build_claim_closure


PRIORITY = (
    "PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED",
    "CONCRETE_TIMING_COUNTEREXAMPLE", "POLICY_CONTRACT_VIOLATION",
    "REFERENCE_COUNTEREXAMPLE", "REFERENCE_CERTIFICATE_FAILED",
    "UNRESOLVED", "DEPLOYED_TREE_PROVED",
)


def reject_legacy_core_for_current_claim(
    certificates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    present = sorted(
        obligation_id
        for obligation_id in LEGACY_PROTECTED_HI_IDS
        if obligation_id in certificates
    )
    if not present:
        return None
    return {
        "status": "UNRESOLVED",
        "route": "UNRESOLVED",
        "code": "LEGACY_PROTECTED_HI_CORE_NOT_ELIGIBLE",
        "legacy_obligations": present,
    }


def aggregate_phase_ae_local(statuses: list[dict[str, object]]) -> str:
    """聚合已独立执行的 Phase A-E 局部检查，不伪造最终安全 claim。"""
    if any(item.get("status") == "FAIL" and item.get("failure", {}).get("route") == "PROOF_BUNDLE_INVALID"
           for item in statuses if isinstance(item.get("failure"), dict)):
        return "PROOF_BUNDLE_INVALID"
    if any(item.get("status") == "FAIL" for item in statuses):
        return "PHASE_AE_REJECTED"
    if any(item.get("status") == "UNRESOLVED" for item in statuses):
        return "UNRESOLVED"
    return "PHASE_AE_ACCEPTED"


def claim_dependency_closure(registry: list[dict[str, object]], claim: str) -> set[str]:
    """Return the canonical verified acceptance surface for ``claim``.

    The compiler and fresh verifier already partition the claim closure into
    mathematical, authorization, and structural certificates through
    :func:`build_claim_closure`.  Aggregation must consume that same set rather
    than reconstructing only the mathematical dependency chain.  The derived
    ``CLAIM_AGGREGATION_RESULT`` summary is intentionally excluded by the
    canonical closure because it is produced only after aggregation.
    """
    return set(build_claim_closure(registry, claim).verified_artifacts)


def mathematical_claim_closure(registry: list[dict[str, object]], claim: str) -> set[str]:
    """只从 mathematical_root 出发的闭包。"""
    entries = {str(item["id"]): item for item in registry
               if item.get("activation") == "active" and item.get("required") is True}
    roots = [item["id"] for item in registry
             if item.get("proof_role") == "mathematical_root"
             and claim in item.get("gates_claims", [])]
    if len(roots) != 1 or roots[0] != "FINAL_CLAIM_COMPOSITION":
        raise ValueError("EXACTLY_ONE_MATHEMATICAL_ROOT_REQUIRED")
    closure = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for dependency in entries[current].get("depends_on", []):
            dependency = str(dependency)
            if dependency not in entries:
                raise ValueError(f"claim 依赖 inactive/unknown obligation: {dependency}")
            if dependency not in closure:
                closure.add(dependency)
                stack.append(dependency)
    return closure


def authorization_gates(registry: list[dict[str, object]], claim: str) -> set[str]:
    """返回 claim 所需的所有 authorization gate ID。"""
    return {
        str(row["id"]) for row in registry
        if row.get("proof_role") == "authorization_gate"
        and claim in row.get("authorization_for", [])
    }


def _check_proof_role_invariants(
    registry: list[dict[str, object]],
    certificates: dict[str, dict[str, object]],
    claim: str = "DEPLOYED_HI_SAFETY",
) -> dict[str, object] | None:
    """验证 proof_role 不变量：唯一数学根、授权门全 PASS。"""
    by_id = {str(item["id"]): item for item in registry}
    math_roots = [item["id"] for item in registry if item.get("proof_role") == "mathematical_root"]
    if math_roots != ["FINAL_CLAIM_COMPOSITION"]:
        return {"code": "PROOF_ROLE_INVARIANT_VIOLATED",
                "math_roots": math_roots}

    # 数学根必须 PASS
    root_cert = certificates.get("FINAL_CLAIM_COMPOSITION")
    if not isinstance(root_cert, dict) or root_cert.get("obligation_status") != "PASS":
        return {"code": "MATHEMATICAL_ROOT_NOT_PASS",
                "status": root_cert.get("obligation_status") if root_cert else "MISSING"}

    # 所有 authorization_gate 必须 PASS
    auth_gates = [item for item in registry
                  if item.get("proof_role") == "authorization_gate"
                  and item.get("activation") == "active"
                  and claim in item.get("authorization_for", [])
                  and item.get("id") in certificates]
    failed_gates = [
        item["id"] for item in auth_gates
        if certificates.get(item["id"], {}).get("obligation_status") != "PASS"
    ]
    if failed_gates:
        return {"code": "AUTHORIZATION_GATE_NOT_PASS",
                "failed_gates": failed_gates}

    return None


def aggregate_for_claim(*, claim: str, obligations: list[dict[str, object]] | None = None,
                        registry: list[dict[str, object]],
                        aggregation_spec: dict[str, object] | None = None,
                        verified_status_evidence: dict[str, object] | None = None,
                        verified_certificates: dict[str, dict[str, object]] | None = None,
                        verified_outer_root: str | None = None) -> str:
    """不接受调用方 gate set 的严格 aggregation API。"""
    try:
        closure = claim_dependency_closure(registry, claim)
    except ValueError:
        return "PROOF_BUNDLE_INVALID"
    legacy_guard = reject_legacy_core_for_current_claim(verified_certificates or {})
    if legacy_guard is not None:
        return "UNRESOLVED"
    # proof_role 不变量检查
    if verified_certificates is not None:
        role_check = _check_proof_role_invariants(registry, verified_certificates)
        if role_check is not None:
            return "PROOF_BUNDLE_INVALID"
    # R07 新接口：root 已由 verifier 计算，aggregator 只验证传入的证书、
    # status evidence 与 root 引用一致；它不再自行猜测另一套 root 算法。
    if verified_certificates is not None:
        if verified_outer_root is None or verified_status_evidence is None:
            return "PROOF_BUNDLE_INVALID"
        if set(verified_certificates) != closure or set(verified_status_evidence) != closure:
            return "PROOF_BUNDLE_INVALID"
        normalized: list[dict[str, object]] = []
        for obligation_id in sorted(closure):
            certificate = verified_certificates[obligation_id]
            evidence = verified_status_evidence[obligation_id]
            failure = certificate.get("failure") if isinstance(certificate.get("failure"), dict) else {}
            from formal_toolchain.core.artifact import verify_obligation_certificate
            if (not isinstance(certificate, dict) or not verify_obligation_certificate(certificate)
                    or evidence.get("outer_bundle_root") != verified_outer_root
                    or evidence.get("certificate_hash") != certificate.get("artifact_hash")
                    or evidence.get("obligation_status") != certificate.get("obligation_status")
                    or evidence.get("verified") is not True):
                return "PROOF_BUNDLE_INVALID"
            normalized.append({"id": obligation_id,
                               "obligation_status": certificate.get("obligation_status"),
                               "failure_route": failure.get("route"),
                               "failure_code": failure.get("code")})
        return aggregate(normalized, closure, registry=registry,
                         aggregation_spec=aggregation_spec,
                         _registry_derived_gate_set=True)

    if obligations is None or verified_status_evidence is None:
        return "PROOF_BUNDLE_INVALID"
    supplied = {str(item.get("id")) for item in obligations}
    if len(supplied) != len(obligations) or supplied != closure:
        return "PROOF_BUNDLE_INVALID"
    if verified_status_evidence is None or set(verified_status_evidence) != closure:
        return "PROOF_BUNDLE_INVALID"
    roots = set()
    certificate_records = {}
    for obligation_id in closure:
        evidence = verified_status_evidence[obligation_id]
        if not isinstance(evidence, dict) or evidence.get("obligation_id") != obligation_id:
            return "PROOF_BUNDLE_INVALID"
        certificate_hash = evidence.get("certificate_hash")
        outer_root = evidence.get("outer_bundle_root")
        if (not isinstance(certificate_hash, str) or len(certificate_hash) != 64 or
                not isinstance(outer_root, str) or len(outer_root) != 64 or
                evidence.get("verified") is not True):
            return "PROOF_BUNDLE_INVALID"
        certificate = evidence.get("certificate")
        if not isinstance(certificate, dict):
            return "PROOF_BUNDLE_INVALID"
        certificate_status = certificate.get("obligation_status")
        evidence_status = evidence.get("obligation_status", certificate_status)
        if evidence_status != certificate_status or certificate_status not in {"PASS", "FAIL", "UNRESOLVED"}:
            return "PROOF_BUNDLE_INVALID"
        if sha256_object(certificate) != certificate_hash:
            return "PROOF_BUNDLE_INVALID"
        certificate_records[obligation_id] = certificate_hash
        roots.add(outer_root)
    if len(roots) != 1:
        return "PROOF_BUNDLE_INVALID"
    expected_root = sha256_object({key: certificate_records[key] for key in sorted(certificate_records)})
    if roots != {expected_root}:
        return "PROOF_BUNDLE_INVALID"
    normalized = []
    for item in obligations:
        evidence = verified_status_evidence[str(item["id"])]
        normalized.append({**item, "obligation_status": evidence["obligation_status"]})
    return aggregate(normalized, closure, registry=registry, aggregation_spec=aggregation_spec,
                     _registry_derived_gate_set=True)


def aggregate(obligations: list[dict[str, object]], claim_gates: set[str],
              *, registry: list[dict[str, object]] | None = None,
              aggregation_spec: dict[str, object] | None = None,
              _registry_derived_gate_set: bool = False) -> str:
    """根据 obligation id/status 聚合最终 claim，缺失 gate 必须失败闭合。"""
    if registry is None and any("obligation_status" in item for item in obligations):
        return "PROOF_BUNDLE_INVALID"
    if len({str(item.get("id")) for item in obligations}) != len(obligations):
        return "PROOF_BUNDLE_INVALID"
    if aggregation_spec is None:
        path = Path(__file__).parents[1] / "specs/claim_aggregation.json"
        aggregation_spec = json.loads(path.read_text(encoding="utf-8"))
    priority = tuple(str(item) for item in aggregation_spec.get("priority", PRIORITY))
    allowed = set(str(item) for item in aggregation_spec.get("obligation_statuses", []))
    route_by_id = {str(item["id"]): str(item.get("failure_route", "UNRESOLVED")) for item in (registry or [])}
    if registry is not None and not _registry_derived_gate_set:
        expected_gates = {str(item["id"]) for item in registry
                          if item.get("activation") == "active" and item.get("gates_claims")}
        if set(claim_gates) != expected_gates:
            return "PROOF_BUNDLE_INVALID"
    by_id = {str(item.get("id")): item for item in obligations}
    statuses = []
    for obligation_id in claim_gates:
        item = by_id.get(obligation_id)
        if item is None:
            statuses.append("PROOF_BUNDLE_INVALID")
            continue
        raw_status = str(item.get("obligation_status", item.get("status", "UNRESOLVED")))
        # 兼容 Phase B 早期测试对象：新 certificate 必须使用 obligation_status，
        # 旧的直接 result status 只作为显式失败路由，不参与 PASS 授权。
        if "obligation_status" not in item and raw_status in PRIORITY and raw_status != "DEPLOYED_TREE_PROVED":
            statuses.append(raw_status)
            continue
        status = raw_status
        if status not in allowed:
            return "PROOF_BUNDLE_INVALID"
        if status == "FAIL":
            statuses.append(str(item.get("failure_route") or route_by_id.get(obligation_id, "PROOF_BUNDLE_INVALID")))
        elif status == "UNRESOLVED":
            statuses.append("UNRESOLVED")
        elif status == "NOT_APPLICABLE":
            return "PROOF_BUNDLE_INVALID"
    if not statuses:
        return "DEPLOYED_TREE_PROVED"
    for status in priority:
        if status in statuses:
            return status
    return "PROOF_BUNDLE_INVALID"
