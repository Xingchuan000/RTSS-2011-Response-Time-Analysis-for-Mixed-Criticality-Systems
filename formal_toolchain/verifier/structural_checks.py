"""proof bundle 的结构性 checker。

结构节点同样必须消费真实 bundle 内容。它们不能由 verifier 直接创建 PASS
证书，否则缺失 artifact、前驱 hash 或 root preimage 时仍可能被放行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True)
class StructuralCheckResult:
    status: str
    route: str | None
    code: str | None
    witness: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "route": self.route,
                "code": self.code, "witness": dict(self.witness)}


def _pass(witness: Mapping[str, Any]) -> StructuralCheckResult:
    return StructuralCheckResult("PASS", None, None, dict(witness))


def _fail(code: str, witness: Mapping[str, Any],
          *, route: str = "PROOF_BUNDLE_INVALID") -> StructuralCheckResult:
    return StructuralCheckResult("FAIL", route, code, dict(witness))


def verify_artifact_manifest(*, registry: list[Mapping[str, Any]],
                             certificates: Mapping[str, Mapping[str, Any]],
                             manifest: Mapping[str, Any]) -> StructuralCheckResult:
    """验证 registry 中每个 active artifact 都存在且 hash 与证书一致。"""
    try:
        expected = {str(key): str(value.get("artifact_hash"))
                    for key, value in certificates.items()}
        actual = manifest.get("artifacts")
        if not isinstance(actual, Mapping):
            return _fail("ARTIFACT_MANIFEST_MISSING")
        normalized = {str(key): str(value) for key, value in actual.items()}
        if normalized != expected:
            return _fail("ARTIFACT_MANIFEST_HASH_MISMATCH",
                         {"expected": expected, "actual": normalized})
        return _pass({"certificate_count": len(expected)})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def verify_component_contexts(*, contexts: Mapping[str, Any],
                             expected_contexts: Mapping[str, Mapping[str, Any]]) -> StructuralCheckResult:
    """逐层重算 context hash，拒绝把 hash 自身放进 preimage 的旧格式。"""
    try:
        for name, expected in expected_contexts.items():
            actual = contexts.get(name)
            if not isinstance(actual, Mapping) or actual.get("hash") != expected.get("hash"):
                return _fail("COMPONENT_CONTEXT_MISMATCH", {"context": name})
            body = {key: value for key, value in actual.items() if key != "hash"}
            if sha256_object(body) != actual.get("hash"):
                return _fail("COMPONENT_CONTEXT_HASH_MISMATCH", {"context": name})
        return _pass({"context_count": len(expected_contexts)})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def verify_predecessor_hashes(*, registry: list[Mapping[str, Any]],
                              certificates: Mapping[str, Mapping[str, Any]]) -> StructuralCheckResult:
    """验证每个证书的 direct predecessor 集合和每一个 hash。

    使用 exact predecessor set：不在 certificates 中的前驱也必须声明。
    """
    try:
        by_id = {str(row["id"]): row for row in registry}
        for obligation_id, certificate in certificates.items():
            if obligation_id not in by_id:
                return _fail("UNKNOWN_CERTIFICATE_OBLIGATION", {"obligation_id": obligation_id})
            expected_all = sorted(str(dep) for dep in by_id[obligation_id].get("depends_on", []))
            actual = certificate.get("direct_predecessor_hashes", {})
            actual_ids = sorted(str(key) for key in actual) if isinstance(actual, Mapping) else []
            if actual_ids != expected_all:
                missing = sorted(set(expected_all) - set(actual_ids))
                extra = sorted(set(actual_ids) - set(expected_all))
                return _fail("PREDECESSOR_SET_MISMATCH",
                             {"obligation_id": obligation_id, "expected": expected_all,
                              "actual": actual_ids, "missing": missing, "extra": extra})
            for dependency in expected_all:
                if dependency not in certificates:
                    return _fail("PREDECESSOR_CERTIFICATE_MISSING",
                                 {"obligation_id": obligation_id, "dependency": dependency})
                if actual.get(dependency) != certificates[dependency].get("artifact_hash"):
                    return _fail("PREDECESSOR_HASH_MISMATCH",
                                 {"obligation_id": obligation_id, "dependency": dependency})
        return _pass({"certificate_count": len(certificates)})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def verify_status_evidence(*, status_evidence: Mapping[str, Any],
                           certificates: Mapping[str, Mapping[str, Any]],
                           outer_root: str) -> StructuralCheckResult:
    """验证 status evidence 没有声明不存在的证书或另一个 root。"""
    try:
        if set(status_evidence) != set(certificates):
            return _fail("STATUS_EVIDENCE_SET_MISMATCH")
        for obligation_id, evidence in status_evidence.items():
            certificate = certificates[obligation_id]
            if not isinstance(evidence, Mapping) or evidence.get("obligation_id") != obligation_id:
                return _fail("STATUS_EVIDENCE_ID_MISMATCH", {"obligation_id": obligation_id})
            if evidence.get("certificate_hash") != certificate.get("artifact_hash"):
                return _fail("STATUS_EVIDENCE_CERTIFICATE_HASH_MISMATCH", {"obligation_id": obligation_id})
            if evidence.get("obligation_status") != certificate.get("obligation_status"):
                return _fail("STATUS_EVIDENCE_STATUS_MISMATCH", {"obligation_id": obligation_id})
            if evidence.get("outer_bundle_root") != outer_root or evidence.get("verified") is not True:
                return _fail("STATUS_EVIDENCE_ROOT_MISMATCH", {"obligation_id": obligation_id})
        return _pass({"certificate_count": len(certificates), "outer_bundle_root": outer_root})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def compute_outer_root(*, preimage: Mapping[str, Any]) -> StructuralCheckResult:
    """现场计算 outer root；报告和 summary 不属于 preimage。"""
    try:
        root = sha256_object(dict(preimage))
        return _pass({"preimage": dict(preimage), "outer_bundle_root": root})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def verify_independent_bundle(*, certificates: Mapping[str, Mapping[str, Any]],
                              registry: list[Mapping[str, Any]]) -> StructuralCheckResult:
    """验证所有证书 envelope 的 hash 和 obligation ID。"""
    try:
        known = {str(row["id"]) for row in registry}
        for obligation_id, certificate in certificates.items():
            if obligation_id not in known or not verify_obligation_certificate(certificate):
                return _fail("INDEPENDENT_CERTIFICATE_INVALID", {"obligation_id": obligation_id})
        return _pass({"certificate_count": len(certificates)})
    except Exception as exc:
        return _fail("CHECKER_INTERNAL_ERROR", {"failure": str(exc)})


def verify_claim_aggregation_result(*, result: Mapping[str, Any],
                                    aggregated_status: str,
                                    outer_root: str) -> StructuralCheckResult:
    """验证派生 summary 只记录 aggregator 输出，不参与授权。"""

    if result.get("result_status") != aggregated_status:
        return _fail("CLAIM_AGGREGATION_RESULT_MISMATCH")
    if result.get("outer_bundle_root") != outer_root:
        return _fail("CLAIM_AGGREGATION_ROOT_MISMATCH")
    return _pass({"result_status": aggregated_status, "outer_bundle_root": outer_root})
