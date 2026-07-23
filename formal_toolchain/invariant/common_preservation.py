"""Phase H03：非策略转移的预算不变量合同。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.hashing import sha256_proof_object


def check_common_transition_preservation(candidate: Mapping[str, Any], *, transitions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if candidate.get("status") != "PASS":
        raise ValueError("common preservation 必须消费已通过 candidate envelope")
    if transitions is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "COMMON_TRANSITION_EVIDENCE_MISSING"}}
    required = {"boot", "release_snapshot", "completion", "cancellation", "recovery", "active_release_snapshot"}
    if not required <= set(transitions):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "COMMON_TRANSITION_EVIDENCE_INCOMPLETE",
                             "fields": sorted(required - set(transitions))}}
    evidence_fields = {"budget_before", "budget_after", "active_release_before", "active_release_after"}
    lower = candidate.get("lower", {}); upper = candidate.get("upper", {})
    for name in required:
        row = transitions[name]
        if not isinstance(row, Mapping) or not evidence_fields <= set(row):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "COMMON_TRANSITION_WITNESS_INCOMPLETE", "transition": name}}
        after = row["budget_after"]
        if not isinstance(after, Mapping) or set(after) != set(lower) or any(task not in lower or value < lower[task] or value > upper[task]
                                                for task, value in after.items()):
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "COMMON_TRANSITION_EXITED_ENVELOPE", "transition": name}}
        before_active = row["active_release_before"]; after_active = row["active_release_after"]
        before_budget = row["budget_before"]
        if name != "boot" and (not isinstance(before_budget, Mapping) or set(before_budget) != set(lower)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "COMMON_TRANSITION_BEFORE_DOMAIN_INCOMPLETE", "transition": name}}
        if isinstance(before_active, Mapping) and isinstance(after_active, Mapping):
            for job, value in before_active.items():
                if job in after_active and after_active[job] != value:
                    return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                            "failure": {"code": "ACTIVE_RELEASE_SNAPSHOT_MUTATED", "transition": name, "job": job}}
        semantics = row.get("transition_semantics")
        if name == "boot" and (before_active or set(after) != set(lower)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "failure": {"code": "BOOT_INITIAL_BUDGET_WITNESS_INVALID"}}
        if name == "release_snapshot":
            released = row.get("released_job")
            if semantics != "release" or not released or released not in after_active or released in before_active:
                return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                        "failure": {"code": "RELEASE_SNAPSHOT_SEMANTICS_INVALID"}}
            task_name = row.get("released_task")
            if not isinstance(task_name, str) or after_active[released] != before_budget.get(task_name):
                return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                        "failure": {"code": "RELEASE_BUDGET_SNAPSHOT_MISMATCH"}}
        if name in {"completion", "cancellation"}:
            removed = row.get("removed_job")
            if semantics != "remove" or removed not in before_active or removed in after_active or \
               set(after_active) != set(before_active) - {removed}:
                return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                        "failure": {"code": "REMOVAL_SNAPSHOT_SEMANTICS_INVALID", "transition": name}}
        if name in {"recovery", "active_release_snapshot"} and semantics == "preserve" and before_active != after_active:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "RECOVERY_SNAPSHOT_MUTATED", "transition": name}}
    # release snapshot、completion/cancellation/recovery 的共同合同不产生预算写入；
    # 若 caller 提供了显式 transition witness，则只接受声明为 immutable 的记录。
    for name, transition in (transitions or {}).items():
        if transition.get("budget_write") not in (None, False):
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "NON_POLICY_BUDGET_WRITE", "transition": name}}
    return {
        "status": "PASS",
        "schema_version": "common_transition_preservation_v1",
        "active_release_budget_immutable": True,
        "controller_budget_write": False,
        "invariant_checked": True,
        "candidate_envelope_hash": sha256_proof_object(dict(candidate)),
        "safety_polytope_hash": candidate.get("safety_polytope_hash"),
    }
