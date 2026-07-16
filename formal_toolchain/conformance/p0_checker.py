"""Phase F aggregate checker：只聚合已有细粒度证书。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def aggregate_p0_certificates(certificates: Mapping[str, Mapping[str, Any]], *, registry_entries: list[dict[str, Any]] | None = None,
                              claim: str = "DEPLOYED_HI_SAFETY",
                              verified_status_evidence: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """所有 active P0 证书 PASS 才聚合为 PASS；不替单项证书猜测结论。"""
    if registry_entries is None:
        from formal_toolchain.core.registry import load_registry
        registry_entries = load_registry(Path(__file__).parents[1] / "specs/obligation_registry.json")
    from formal_toolchain.core.registry import active_obligations_for_claim
    phase_ids = {"SCHEDULER_MODEL", "STRICT_PRIORITY_ORDER", "TIME_DOMAIN", "NO_OVERFLOW",
                 "OVERHEAD_PROFILE", "BOOT_INITIALIZATION", "MODE_SEMANTICS_CONFORMANCE",
                 "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT", "REMOVAL_COMPLETENESS",
                 "HI_NONTRUNCATION", "DEADLINE_OBSERVATION", "EFFECTIVE_EVENT_ORDER",
                 "SEQUENCE_ALLOCATION", "PHASE_DAG", "BATCH_CLOSURE", "DEADLINE_BOUNDARY_ORDER",
                 "CONTROLLER_INVISIBILITY", "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS",
                 "WINDOW_MODE_NORMALIZATION", "BUDGET_DOMAIN"}
    phase_ids |= {"OBSERVATION_EXTRACTION", "FEATURE_QUANTIZATION", "ACTION_TRANSITION", "MASK_FALLBACK",
                  "EXECUTABLE_POLICY_SEMANTICS", "CANDIDATE_ENVELOPE", "COMMON_TRANSITION_PRESERVATION",
                  "DEPLOYED_POLICY_PRESERVATION"}
    active = active_obligations_for_claim(registry_entries, claim=claim, phase_ids=phase_ids)
    if not active:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": "ACTIVE_REGISTRY_CLOSURE_EMPTY"}}
    unknown = sorted(set(certificates) - set(active))
    missing = sorted(set(active) - set(certificates))
    if unknown or missing:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "ACTIVE_OBLIGATION_SET_MISMATCH", "missing": missing, "unknown": unknown}}
    if verified_status_evidence is None or set(verified_status_evidence) != set(active):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "VERIFIER_ACCEPTED_EVIDENCE_MISSING"}}
    for obligation_id in active:
        evidence = verified_status_evidence[obligation_id]
        if evidence.get("status") != "PASS" or evidence.get("certificate_hash") != certificates[obligation_id].get("certificate_hash"):
            return {"status": "UNRESOLVED", "route": "PROOF_BUNDLE_INVALID",
                    "failure": {"code": "VERIFIER_STATUS_EVIDENCE_INVALID", "obligation_id": obligation_id}}
    statuses = [certificates[name].get("status", certificates[name].get("obligation_status")) for name in active]
    invalid = sorted(name for name in active if statuses[active.index(name)] not in {"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"})
    if invalid:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "failure": {"code": "UNKNOWN_OBLIGATION_STATUS", "obligations": invalid}}
    unresolved = sorted(name for name in active if certificates[name].get("status", certificates[name].get("obligation_status")) == "UNRESOLVED")
    failed = sorted(name for name in active if certificates[name].get("status", certificates[name].get("obligation_status")) == "FAIL")
    if failed:
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED", "failed": failed, "unresolved": unresolved}
    if unresolved:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "unresolved": unresolved}
    return {"status": "PASS", "schema_version": "p0_model_conformance_v1", "obligations": active}
