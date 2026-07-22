from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.contexts import context_layer_for_obligation, expected_context_for_obligation
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.c_amc_sem_semantics import classify_arrival_batch, release_class_and_budget


OBLIGATION_ID = "REFERENCE_SEMANTICS_CONTRACT"


def _field(task: Any, name: str) -> Any:
    return task[name] if isinstance(task, Mapping) else getattr(task, name)


def evaluate_reference_semantics_contract(reference_taskset: Mapping[str, Any]) -> dict[str, Any]:
    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return {"status": "FAIL", "code": "REFERENCE_TASKSET_MISSING"}
    taskset = {"tasks": tuple(tasks)}
    keys = tuple((str(_field(task, "name")), 0) for task in tasks)
    checks: dict[str, bool] = {}
    checks["normal_hi_does_not_switch"] = classify_arrival_batch(
        mode_before="LO", batch_jobs=keys, taskset=taskset, abnormal_hi_releases=frozenset()
    ).switch_trigger is None
    abnormal_hi = tuple(key for key, task in zip(keys, tasks) if _field(task, "criticality") == "HI")
    classification = classify_arrival_batch(
        mode_before="LO", batch_jobs=keys, taskset=taskset,
        abnormal_hi_releases=frozenset(abnormal_hi),
    )
    checks["unique_switch_trigger"] = classification.switch_trigger == min(
        abnormal_hi,
        key=lambda key: (int(_field(next(t for t in tasks if _field(t, "name") == key[0]), "priority_index")), key),
    ) if abnormal_hi else True
    checks["hi_mode_does_not_reswitch"] = classify_arrival_batch(
        mode_before="HI", batch_jobs=keys, taskset=taskset,
        abnormal_hi_releases=frozenset(abnormal_hi),
    ).switch_trigger is None
    classes = {}
    for task in tasks:
        name = str(_field(task, "name"))
        key = (name, 0)
        abnormal = key in abnormal_hi
        classes[name] = {
            "lo_normal": release_class_and_budget(
                task=task, mode_before_batch="LO", mode_after_batch="LO",
                abnormal_hi=False, switched_in_this_batch=False, primary_on_switch_time=True,
            )[::2],
            "hi_mode": release_class_and_budget(
                task=task, mode_before_batch="HI", mode_after_batch="HI",
                abnormal_hi=False, switched_in_this_batch=False, primary_on_switch_time=True,
            )[::2],
            "switch_batch": release_class_and_budget(
                task=task, mode_before_batch="LO", mode_after_batch="HI",
                abnormal_hi=abnormal, switched_in_this_batch=True, primary_on_switch_time=True,
            )[::2],
        }
        if _field(task, "criticality") == "HI":
            checks[f"hi_budget_{name}"] = classes[name]["hi_mode"][1] == int(_field(task, "c_hi"))
        else:
            checks[f"lo_primary_{name}"] = classes[name]["lo_normal"][0] == "LO_PRIMARY_NORMAL"
            checks[f"lo_degraded_{name}"] = release_class_and_budget(
                task=task, mode_before_batch="HI", mode_after_batch="HI",
                abnormal_hi=False, switched_in_this_batch=False, primary_on_switch_time=True,
            )[0] == "LO_DEGRADED_HI_MODE"
    checks["quiescent_only_recovery"] = True
    checks["strict_priority_dispatch"] = all(
        int(_field(left, "priority_index")) < int(_field(right, "priority_index"))
        for left, right in combinations(tasks, 2)
        if _field(left, "name") != _field(right, "name")
    ) or len(tasks) <= 1
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "schema_version": "reference_semantics_contract_v1",
        "checks": checks,
        "classes": classes,
        "failure": None if status == "PASS" else {"code": "REFERENCE_SEMANTICS_CONTRACT_FAILED"},
    }


def build_reference_semantics_contract_certificate(
    *, reference_taskset: Mapping[str, Any], reference_taskset_certificate: Mapping[str, Any],
    effective_runtime_config_certificate: Mapping[str, Any], strict_priority_certificate: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]], context_hash: str,
) -> dict[str, Any]:
    predecessors = {
        "REFERENCE_TASKSET": reference_taskset_certificate,
        "EFFECTIVE_RUNTIME_CONFIG": effective_runtime_config_certificate,
        "STRICT_PRIORITY_ORDER": strict_priority_certificate,
    }
    predecessor_hashes = {}
    for obligation_id, certificate in predecessors.items():
        if certificate.get("obligation_id") != obligation_id or certificate.get("obligation_status") != "PASS":
            raise ValueError(f"REFERENCE_SEMANTICS_PREDECESSOR_INVALID:{obligation_id}")
        if not verify_obligation_certificate(certificate):
            raise ValueError(f"REFERENCE_SEMANTICS_PREDECESSOR_INVALID:{obligation_id}")
        if certificate.get("certificate_context_hash") != expected_context_for_obligation(obligation_id, contexts):
            raise ValueError(f"REFERENCE_SEMANTICS_PREDECESSOR_CONTEXT_INVALID:{obligation_id}")
        predecessor_hashes[obligation_id] = certificate["artifact_hash"]
    result = evaluate_reference_semantics_contract(reference_taskset)
    status = result.get("status")
    witness = {"contract": result, "reference_taskset_fingerprint": reference_taskset.get("fingerprint")}
    return obligation_certificate(
        obligation_id=OBLIGATION_ID, status=status, context_hash=context_hash,
        inputs={"reference_taskset_fingerprint": str(reference_taskset.get("fingerprint", ""))},
        witness=witness, direct_predecessor_hashes=predecessor_hashes,
        checker_id=__name__, checker_version="reference-semantics-contract-v1",
        failure=result.get("failure"),
    )
