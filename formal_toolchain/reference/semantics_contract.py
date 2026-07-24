from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.contexts import context_layer_for_obligation, expected_context_for_obligation
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.c_amc_sem_semantics import (
    ReferenceReleaseDecision,
    classify_arrival_batch,
    decide_reference_release,
)


OBLIGATION_ID = "REFERENCE_SEMANTICS_CONTRACT"


def _effective_config_values(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    witness = certificate.get(
        "witness",
        {},
    )

    if not isinstance(witness, Mapping):
        raise ValueError(
            "REFERENCE_EFFECTIVE_CONFIG_WITNESS_MISSING"
        )

    payload: Mapping[str, Any] = witness

    if isinstance(
        witness.get("config"),
        Mapping,
    ):
        payload = witness["config"]

    fields = payload.get("fields")

    if not isinstance(fields, Mapping):
        raise ValueError(
            "REFERENCE_EFFECTIVE_CONFIG_FIELDS_MISSING"
        )

    return {
        str(name): (
            record.get("value")
            if isinstance(record, Mapping)
            else record
        )
        for name, record in fields.items()
    }


def _field(task: Any, name: str) -> Any:
    return task[name] if isinstance(task, Mapping) else getattr(task, name)


def recovery_is_legal(
    *,
    mode: str,
    active_job_count: int,
    running_job_present: bool,
    pending_release_count: int,
) -> bool:
    return bool(
        mode == "HI"
        and active_job_count == 0
        and not running_job_present
        and pending_release_count == 0
    )


def _decision_record(decision: ReferenceReleaseDecision) -> dict[str, Any]:
    return {
        "release_class": decision.release_class,
        "effective_release_mode": decision.effective_release_mode,
        "release_budget": decision.release_budget,
    }


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
    classes: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        name = str(_field(task, "name"))
        criticality = str(_field(task, "criticality"))
        c_lo = int(_field(task, "c_lo"))
        c_hi = int(_field(task, "c_hi"))
        decisions: dict[str, ReferenceReleaseDecision] = {}
        expected: dict[str, dict[str, Any]] = {}

        if criticality == "HI":
            decisions["HI_NORMAL_LO"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="LO",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=False, primary_on_switch_time=True,
            )
            expected["HI_NORMAL_LO"] = {
                "release_class": "HI_NORMAL",
                "effective_release_mode": "LO",
                "release_budget": c_lo,
            }
            decisions["HI_ABNORMAL_SWITCH_TRIGGER"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="HI",
                abnormal_hi=True, is_switch_trigger=True,
                switched_in_this_batch=True, primary_on_switch_time=True,
            )
            expected["HI_ABNORMAL_SWITCH_TRIGGER"] = {
                "release_class": "HI_ABNORMAL_SWITCH_TRIGGER",
                "effective_release_mode": "LO",
                "release_budget": c_hi,
            }
            decisions["HI_NORMAL_SAME_SWITCH_BATCH"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="HI",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=True, primary_on_switch_time=True,
            )
            expected["HI_NORMAL_SAME_SWITCH_BATCH"] = {
                "release_class": "HI_NORMAL",
                "effective_release_mode": "LO",
                "release_budget": c_lo,
            }
            decisions["HI_NORMAL_HI_MODE"] = decide_reference_release(
                task=task, mode_before_batch="HI", mode_after_batch="HI",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=False, primary_on_switch_time=True,
            )
            expected["HI_NORMAL_HI_MODE"] = {
                "release_class": "HI_NORMAL",
                "effective_release_mode": "HI",
                "release_budget": c_lo,
            }
            decisions["HI_ABNORMAL_NON_TRIGGER_SAME_BATCH"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="HI",
                abnormal_hi=True, is_switch_trigger=False,
                switched_in_this_batch=True, primary_on_switch_time=True,
            )
            expected["HI_ABNORMAL_NON_TRIGGER_SAME_BATCH"] = {
                "release_class": "HI_ABNORMAL",
                "effective_release_mode": "LO",
                "release_budget": c_hi,
            }
            decisions["HI_ABNORMAL_EXISTING_HI_MODE"] = decide_reference_release(
                task=task, mode_before_batch="HI", mode_after_batch="HI",
                abnormal_hi=True, is_switch_trigger=False,
                switched_in_this_batch=False, primary_on_switch_time=True,
            )
            expected["HI_ABNORMAL_EXISTING_HI_MODE"] = {
                "release_class": "HI_ABNORMAL",
                "effective_release_mode": "HI",
                "release_budget": c_hi,
            }
        elif criticality == "LO":
            decisions["LO_PRIMARY_NORMAL"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="LO",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=False, primary_on_switch_time=True,
            )
            expected["LO_PRIMARY_NORMAL"] = {
                "release_class": "LO_PRIMARY_NORMAL",
                "effective_release_mode": "LO",
                "release_budget": c_lo,
            }
            decisions["LO_PRIMARY_SAME_BATCH_SWITCH_TIME"] = decide_reference_release(
                task=task, mode_before_batch="LO", mode_after_batch="HI",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=True, primary_on_switch_time=True,
            )
            expected["LO_PRIMARY_SAME_BATCH_SWITCH_TIME"] = {
                "release_class": "LO_PRIMARY_SAME_BATCH_SWITCH_TIME",
                "effective_release_mode": "LO",
                "release_budget": c_lo,
            }
            decisions["LO_DEGRADED_HI_MODE"] = decide_reference_release(
                task=task, mode_before_batch="HI", mode_after_batch="HI",
                abnormal_hi=False, is_switch_trigger=False,
                switched_in_this_batch=False, primary_on_switch_time=True,
            )
            expected["LO_DEGRADED_HI_MODE"] = {
                "release_class": "LO_DEGRADED_HI_MODE",
                "effective_release_mode": "HI",
                "release_budget": c_hi,
            }
        else:
            checks[f"criticality_valid_{name}"] = False

        classes[name] = {
            case_id: _decision_record(decision)
            for case_id, decision in decisions.items()
        }
        for case_id, expected_record in expected.items():
            checks[f"release_decision_{name}_{case_id}"] = (
                classes[name][case_id] == expected_record
            )

    checks["quiescent_only_recovery"] = (
        recovery_is_legal(
            mode="HI",
            active_job_count=0,
            running_job_present=False,
            pending_release_count=0,
        )
        and not recovery_is_legal(
            mode="LO",
            active_job_count=0,
            running_job_present=False,
            pending_release_count=0,
        )
        and not recovery_is_legal(
            mode="HI",
            active_job_count=1,
            running_job_present=False,
            pending_release_count=0,
        )
        and not recovery_is_legal(
            mode="HI",
            active_job_count=0,
            running_job_present=True,
            pending_release_count=0,
        )
        and not recovery_is_legal(
            mode="HI",
            active_job_count=0,
            running_job_present=False,
            pending_release_count=1,
        )
    )
    priorities = [
        int(_field(task, "priority_index"))
        for task in tasks
    ]
    checks["strict_priority_order_well_formed"] = (
        len(priorities) == len(set(priorities))
        and sorted(priorities) == list(range(len(priorities)))
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "schema_version": "reference_semantics_contract_v2",
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

    effective = _effective_config_values(
        effective_runtime_config_certificate
    )

    required_config = {
        "semantics": "C_AMC_SEM",
        "c_amc_sem_primary_on_switch_time":
            True,
        "drop_lo_jobs_on_hi_switch":
            False,
    }

    config_mismatches = {
        name: {
            "expected": expected,
            "actual": effective.get(name),
        }
        for name, expected
        in required_config.items()
        if effective.get(name) != expected
    }

    if config_mismatches:
        result = {
            "status": "FAIL",
            "schema_version":
                "reference_semantics_contract_v2",
            "checks": {
                "effective_runtime_config":
                    False,
            },
            "effective_runtime_config": {
                "required": required_config,
                "observed": {
                    name: effective.get(name)
                    for name
                    in required_config
                },
                "mismatches":
                    config_mismatches,
            },
            "failure": {
                "code":
                    "REFERENCE_EFFECTIVE_CONFIG_MISMATCH",
            },
        }
    else:
        result = (
            evaluate_reference_semantics_contract(
                reference_taskset
            )
        )

        result = {
            **result,
            "effective_runtime_config": {
                "required": required_config,
                "observed": {
                    name: effective.get(name)
                    for name
                    in required_config
                },
                "matches": True,
            },
        }

    status = result.get("status")
    witness = {"contract": result, "reference_taskset_fingerprint": reference_taskset.get("fingerprint"),
               "effective_runtime_config": result.get("effective_runtime_config", {})}
    return obligation_certificate(
        obligation_id=OBLIGATION_ID, status=status, context_hash=context_hash,
        inputs={"reference_taskset_fingerprint": str(reference_taskset.get("fingerprint", ""))},
        witness=witness, direct_predecessor_hashes=predecessor_hashes,
        checker_id=__name__, checker_version="reference-semantics-contract-v3",
        failure=result.get("failure"),
    )
