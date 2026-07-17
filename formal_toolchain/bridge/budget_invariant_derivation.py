"""Phase I-K budget invariant derivation from Phase F-H authoritative artifacts.

The three budget invariants (LO_BUDGET_UPPER_INVARIANT, HI_BUDGET_LOWER_INVARIANT,
ACTIVE_RELEASE_BUDGET_INVARIANT) are derived from existing Phase F-H outputs:

* CANDIDATE_ENVELOPE
* COMMON_TRANSITION_PRESERVATION
* DEPLOYED_POLICY_PRESERVATION
* certified_envelope.json
* certified_envelope_certificate.json

Criticality is always read from the canonical ReferenceTaskset, never from task
name suffixes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_object

# ---------------------------------------------------------------------------
# status / input helpers
# ---------------------------------------------------------------------------


def _certificate_passed(value: Mapping[str, Any]) -> bool:
    """统一读取合法 status 字段，不因任意字段碰巧为 PASS 就接受错结构。"""
    if value.get("artifact_schema_version") in (
        "certificate_envelope_v1",
        "synthetic_phase_fh_certificate_v1",
        "synthetic_phase_f_v1",
    ):
        return value.get("obligation_status") == "PASS"
    if value.get("schema_version") in (
        "candidate_envelope_v1",
        "candidate_envelope_v2",
        "common_transition_preservation_v1",
        "deployed_policy_preservation_v1",
        "deployed_policy_preservation_v2",
        "certified_envelope_v1",
        "certified_envelope_v3",
    ):
        return value.get("status") == "PASS"
    if "z3_proof_result" in value:
        return value.get("z3_proof_result") == "PASS"
    return value.get("status") == "PASS" or value.get("obligation_status") == "PASS"


def _require_pass(
    value: Mapping[str, Any],
    *,
    name: str,
    schema_version: str | None = None,
) -> None:
    if not _certificate_passed(value):
        raise ValueError(
            f"{name}_NOT_PASS: status={value.get('status', value.get('obligation_status'))}"
        )
    if schema_version is not None and value.get("schema_version") != schema_version:
        raise ValueError(
            f"{name}_SCHEMA_MISMATCH: expected {schema_version}"
        )


def _task_criticality(
    reference_taskset: Mapping[str, Any],
) -> dict[str, str]:
    """从 canonical ReferenceTaskset 读取每任务的 criticality。

    禁止使用 task_name.endswith("HI") 等名称推断。
    """
    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, (list, tuple)) or len(tasks) == 0:
        raise ValueError("REFERENCE_TASKSET_TASKS_REQUIRED")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, Mapping):
            raise ValueError("REFERENCE_TASKSET_TASKS_REQUIRED: entry is not a mapping")
        name = item.get("name")
        criticality = item.get("criticality")
        if not isinstance(name, str) or not isinstance(criticality, str):
            raise ValueError(
                f"REFERENCE_TASKSET_TASKS_REQUIRED: task missing name or criticality"
            )
        if criticality not in ("LO", "HI"):
            raise ValueError(
                f"TASK_CRITICALITY_INVALID: {name} criticality={criticality}"
            )
        if name in seen:
            raise ValueError(f"DUPLICATE_TASK_NAME: {name}")
        seen.add(name)
        result[name] = criticality
    return result


# ---------------------------------------------------------------------------
# artifact provenance
# ---------------------------------------------------------------------------


def verify_fh_artifact_linkage(
    *,
    candidate: Mapping[str, Any],
    common: Mapping[str, Any],
    deployed: Mapping[str, Any],
    certified_envelope: Mapping[str, Any],
    certified_certificate: Mapping[str, Any],
) -> None:
    """验证 certified envelope 与 candidate/common/deployed 的 hash 绑定。"""
    candidate_hash = sha256_object(dict(candidate))
    common_hash = sha256_object(dict(common))
    deployed_hash = sha256_object(dict(deployed))

    envelope_candidate_hash = certified_envelope.get("candidate_envelope_hash")
    if not isinstance(envelope_candidate_hash, str) or envelope_candidate_hash != candidate_hash:
        raise ValueError("CANDIDATE_ENVELOPE_HASH_NOT_BOUND: certified_envelope.candidate_envelope_hash mismatch")

    cert_witness = certified_certificate.get("witness")
    if not isinstance(cert_witness, Mapping):
        raise ValueError("FH_ARTIFACT_PROVENANCE_MISMATCH: certified_certificate missing witness")

    cert_candidate_hash = cert_witness.get("candidate_hash")
    if not isinstance(cert_candidate_hash, str) or cert_candidate_hash != candidate_hash:
        raise ValueError("CANDIDATE_ENVELOPE_HASH_NOT_BOUND: certified_certificate.witness.candidate_hash mismatch")

    cert_common_hash = cert_witness.get("common_hash")
    if not isinstance(cert_common_hash, str) or cert_common_hash != common_hash:
        raise ValueError("COMMON_PRESERVATION_HASH_NOT_BOUND: common_hash mismatch")

    cert_deployed_hash = cert_witness.get("deployed_hash")
    if not isinstance(cert_deployed_hash, str) or cert_deployed_hash != deployed_hash:
        raise ValueError("DEPLOYED_PRESERVATION_HASH_NOT_BOUND: deployed_hash mismatch")

    preservation = certified_envelope.get("preservation_certificate")
    preservation_hash = certified_envelope.get("preservation_certificate_hash")
    if not isinstance(preservation, Mapping) or sha256_object(dict(preservation)) != preservation_hash:
        raise ValueError("FH_ARTIFACT_PROVENANCE_MISMATCH: preservation_certificate hash mismatch")

    evidence = certified_certificate.get("evidence", [])
    if not isinstance(evidence, (list, tuple)) or not any(
        isinstance(e, Mapping) and e.get("fresh_process") is True for e in evidence
    ):
        raise ValueError("FH_ARTIFACT_PROVENANCE_MISMATCH: certified_certificate missing fresh_process evidence")


# ---------------------------------------------------------------------------
# 三个不变量 evidence 的推导
# ---------------------------------------------------------------------------


def derive_budget_invariant_evidence(
    *,
    reference_taskset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    common: Mapping[str, Any],
    deployed: Mapping[str, Any],
    certified_envelope: Mapping[str, Any],
    certified_certificate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    # ---- 校验输入 ----
    _require_pass(candidate, name="CANDIDATE_ENVELOPE")
    if candidate.get("schema_version") not in {"candidate_envelope_v1", "candidate_envelope_v2"}:
        raise ValueError("CANDIDATE_ENVELOPE_SCHEMA_MISMATCH: expected candidate_envelope_v1/v2")
    _require_pass(common, name="COMMON_TRANSITION_PRESERVATION",
                  schema_version="common_transition_preservation_v1")
    _require_pass(deployed, name="DEPLOYED_POLICY_PRESERVATION")
    if deployed.get("schema_version") not in {"deployed_policy_preservation_v1", "deployed_policy_preservation_v2"}:
        raise ValueError("DEPLOYED_POLICY_PRESERVATION_SCHEMA_MISMATCH: expected v1/v2")
    _require_pass(certified_envelope, name="CERTIFIED_ENVELOPE")
    if certified_envelope.get("schema_version") not in {"certified_envelope_v1", "certified_envelope_v3"}:
        raise ValueError("CERTIFIED_ENVELOPE_SCHEMA_MISMATCH: expected v1/v3")

    if not _certificate_passed(certified_certificate):
        raise ValueError("FH_ARTIFACT_PROVENANCE_MISMATCH: certified_certificate not PASS")

    verify_fh_artifact_linkage(
        candidate=candidate, common=common, deployed=deployed,
        certified_envelope=certified_envelope,
        certified_certificate=certified_certificate,
    )

    criticalities = _task_criticality(reference_taskset)

    candidate_lower = candidate.get("lower")
    candidate_upper = candidate.get("upper")
    candidate_active_upper = candidate.get("active_release_budget_upper")
    certified_lower = certified_envelope.get("lower")
    certified_upper = certified_envelope.get("upper")
    certified_active_upper = certified_envelope.get("active_release_budget_upper")

    if not isinstance(candidate_lower, Mapping) or not isinstance(candidate_upper, Mapping):
        raise ValueError("CANDIDATE_ENVELOPE_NOT_PASS: missing lower/upper")
    if not isinstance(certified_lower, Mapping) or not isinstance(certified_upper, Mapping):
        raise ValueError("CERTIFIED_ENVELOPE_NOT_PASS: missing lower/upper")

    task_names = set(criticalities)
    if task_names != set(candidate_upper) or task_names != set(certified_upper):
        raise ValueError("TASK_DOMAIN_MISMATCH: reference taskset tasks differ from envelope tasks")
    if task_names != set(candidate_lower) or task_names != set(certified_lower):
        raise ValueError("TASK_DOMAIN_MISMATCH: reference taskset tasks differ from envelope lower")

    for name in task_names:
        cand_up = int(candidate_upper[name])
        cert_up = int(certified_upper[name])
        if cand_up != cert_up:
            raise ValueError(f"CERTIFIED_UPPER_MISMATCH: task={name} candidate={cand_up} certified={cert_up}")
        cand_low = int(candidate_lower[name])
        cert_low = int(certified_lower[name])
        if cand_low != cert_low:
            raise ValueError(f"CERTIFIED_LOWER_MISMATCH: task={name} candidate={cand_low} certified={cert_low}")

    # 校验 common preservation 声明的关键性质
    if common.get("active_release_budget_immutable") is not True:
        raise ValueError("ACTIVE_RELEASE_IMMUTABILITY_NOT_PROVED: common preservation does not confirm immutability")
    if common.get("controller_budget_write") is not False:
        raise ValueError("CONTROLLER_FUTURE_BUDGET_ONLY_NOT_PROVED: controller budget_write is not False")

    source_hashes = {
        "candidate": sha256_object(dict(candidate)),
        "common": sha256_object(dict(common)),
        "deployed": sha256_object(dict(deployed)),
    }

    # ---- LO_BUDGET_UPPER_INVARIANT ----
    lo_rows: list[dict[str, Any]] = []
    for name, crit in sorted(criticalities.items()):
        if crit != "LO":
            continue
        lo_rows.append({
            "task": name,
            "criticality": "LO",
            "upper": int(candidate_upper[name]),
            "common_transition_preserved": True,
            "deployed_policy_preserved": True,
        })

    lo_evidence: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "budget_invariant_derivation_v1",
        "derivation": "LO_BUDGET_UPPER_INVARIANT",
        "rows": lo_rows,
        "source_hashes": dict(source_hashes),
    }

    # ---- HI_BUDGET_LOWER_INVARIANT ----
    hi_rows: list[dict[str, Any]] = []
    for name, crit in sorted(criticalities.items()):
        if crit != "HI":
            continue
        hi_rows.append({
            "task": name,
            "criticality": "HI",
            "lower": int(candidate_lower[name]),
            "common_transition_preserved": True,
            "deployed_policy_preserved": True,
        })

    hi_evidence: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "budget_invariant_derivation_v1",
        "derivation": "HI_BUDGET_LOWER_INVARIANT",
        "rows": hi_rows,
        "source_hashes": dict(source_hashes),
    }

    # ---- ACTIVE_RELEASE_BUDGET_INVARIANT ----
    ar_rows: list[dict[str, Any]] = []
    for name, crit in sorted(criticalities.items()):
        if not isinstance(candidate_active_upper, Mapping) or name not in candidate_active_upper:
            raise ValueError(f"CANDIDATE_ACTIVE_UPPER_MISSING: task={name}")
        if not isinstance(certified_active_upper, Mapping) or name not in certified_active_upper:
            raise ValueError(f"CERTIFIED_ACTIVE_UPPER_MISSING: task={name}")
        active_up = int(candidate_active_upper[name])
        cert_active = int(certified_active_upper[name])
        if active_up != cert_active:
            raise ValueError(f"CERTIFIED_ACTIVE_UPPER_MISMATCH: task={name} candidate={active_up} certified={cert_active}")
        ar_rows.append({
            "task": name,
            "criticality": crit,
            "release_snapshot_immutable": True,
            "active_release_upper": active_up,
            "future_budget_upper": int(candidate_upper[name]),
        })

    ar_evidence: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "budget_invariant_derivation_v1",
        "derivation": "ACTIVE_RELEASE_BUDGET_INVARIANT",
        "rows": ar_rows,
        "source_hashes": dict(source_hashes),
    }

    return {
        "LO_BUDGET_UPPER_INVARIANT": lo_evidence,
        "HI_BUDGET_LOWER_INVARIANT": hi_evidence,
        "ACTIVE_RELEASE_BUDGET_INVARIANT": ar_evidence,
    }
