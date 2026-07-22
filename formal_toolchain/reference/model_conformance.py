from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object

CHECK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "PERIODIC_SPORADIC_TASK_MODEL": ("REFERENCE_TASKSET",),
    "FULLY_PREEMPTIVE_WORK_CONSERVING_STRICT_FPPS": ("SCHEDULER_MODEL",),
    "NO_BLOCKING_SUSPENSION_NONPREEMPTIVE": ("SCHEDULER_MODEL",),
    "RELEASE_FIXED_DEMAND_DOMINATION": ("BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",),
    "ARRIVAL_SWITCH_RECOVERY_SEMANTICS": ("MODE_SEMANTICS_CONFORMANCE",),
    "VERSION_AND_DEGRADED_SELECTION": ("RELEASE_FIXED_REMOVAL_MAPPING",),
    "STANDARD_EMPTY_LO_INITIALIZATION_AND_RECURRING_HISTORY": ("BOOT_INITIALIZATION",),
    "COMPLETE_SWITCH_DOMAINS_AND_ZERO_BOUNDARY": (
        "CASE1_INTEGER_DOMAIN",
        "CASE2_INTEGER_DOMAIN",
        "ZERO_RELATIVE_START",
    ),
    "DISCRETE_TICK_EMBEDDING": ("DISCRETE_TICK_EMBEDDING",),
    "REFERENCE_PREFIX_EXTENSION": ("REFERENCE_PREFIX_EXTENSION",),
}

REQUIRED_CHECK_IDS = tuple(CHECK_REQUIREMENTS)


def _summary_pass(summary: Mapping[str, Any], obligation_id: str) -> bool:
    return (
        summary.get("obligation_id") == obligation_id
        and summary.get("verified") is True
        and summary.get("obligation_status") == "PASS"
    )


def build_reference_model_conformance_certificate(
    *,
    reference_taskset: Mapping[str, Any],
    context_hash: str,
    predecessor_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    predecessors: Mapping[str, Mapping[str, Any]] | None = None,
    imported_theorem: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("reference taskset missing")
    summaries = dict(predecessor_summaries or predecessors or {})

    checks: dict[str, dict[str, Any]] = {
        "PERIODIC_SPORADIC_TASK_MODEL": {
            "pass": all(
                int(t["period"]) > 0
                and 0 <= int(t.get("offset", 0)) < int(t["period"])
                for t in tasks
            ),
            "source": "REFERENCE_TASKSET",
        },
        "FULLY_PREEMPTIVE_WORK_CONSERVING_STRICT_FPPS": {
            "pass": all(
                0 < int(t["deadline"]) <= int(t["period"])
                for t in tasks
            ),
            "source": "REFERENCE_TASKSET",
        },
    }

    for check_id, obligation_ids in CHECK_REQUIREMENTS.items():
        check_pass = True
        artifact_hashes: dict[str, str] = {}
        for obligation_id in obligation_ids:
            summary = summaries.get(obligation_id, {})
            check_pass = check_pass and _summary_pass(summary, obligation_id)
            if isinstance(summary.get("artifact_hash"), str):
                artifact_hashes[obligation_id] = str(summary["artifact_hash"])
        checks[check_id] = {
            "pass": check_pass,
            "source": list(obligation_ids),
            "artifact_hashes": artifact_hashes,
        }

    missing = sorted(set(REQUIRED_CHECK_IDS) - set(checks))
    status = (
        "PASS"
        if not missing and all(item["pass"] for item in checks.values())
        else "UNRESOLVED"
    )
    direct_hashes = {
        obligation_id: cert["artifact_hash"]
        for obligation_id, cert in predecessors.items()
        if isinstance(cert, Mapping) and isinstance(cert.get("artifact_hash"), str)
    }
    return obligation_certificate(
        obligation_id="REFERENCE_MODEL_CONFORMANCE",
        status=status,
        context_hash=context_hash,
        inputs={
            "reference_taskset_fingerprint": reference_taskset.get("fingerprint"),
            "imported_theorem_id": imported_theorem.get("theorem_id"),
            "imported_theorem_statement_hash": imported_theorem.get("statement_hash"),
        },
        witness={
            "check_ids": list(REQUIRED_CHECK_IDS),
            "checks": checks,
            "missing": missing,
            "conformance_hash": sha256_object(checks),
        },
        direct_predecessor_hashes=direct_hashes,
        checker_id=__name__,
        checker_version="reference-conformance-v1",
        failure=None if status == "PASS" else {
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "REFERENCE_MODEL_CONFORMANCE_INCOMPLETE",
        },
    )
