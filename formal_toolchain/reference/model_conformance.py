from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.contexts import context_layer_for_obligation, expected_context_for_obligation
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "specs" / "reference_model_conformance_contract.json"

EXPECTED_CONFORMANCE_CONDITIONS: dict[str, dict[str, Any]] = {
    "FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE": {"source_kind": "HYBRID", "predecessor_obligation_ids": ["REFERENCE_TASKSET"]},
    "CONSTRAINED_DEADLINES": {"source_kind": "HYBRID", "predecessor_obligation_ids": ["REFERENCE_TASKSET"]},
    "SINGLE_PROCESSOR_PREEMPTIVE_FPPS": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["SCHEDULER_MODEL", "STRICT_PRIORITY_ORDER", "REFERENCE_SEMANTICS_CONTRACT"]},
    "NO_BLOCKING_SUSPENSION_NONPREEMPTIVE": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["SCHEDULER_MODEL"]},
    "FIXED_POSITIVE_INTEGER_EXECUTION": {"source_kind": "HYBRID", "predecessor_obligation_ids": ["REFERENCE_TASKSET"]},
    "CRITICALITY_MONOTONE_EXECUTION": {"source_kind": "HYBRID", "predecessor_obligation_ids": ["REFERENCE_TASKSET"]},
    "RELEASE_FIXED_DEMAND_DOMINATION": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["DEMAND_ORACLE_BATCH_CONTRACT", "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION", "RELEASE_FIXED_REMOVAL_MAPPING"]},
    "ARRIVAL_CLASSIFICATION_UNIQUE_SWITCH": {"source_kind": "HYBRID", "predecessor_obligation_ids": ["MODE_SEMANTICS_CONFORMANCE", "EFFECTIVE_EVENT_ORDER", "REFERENCE_SEMANTICS_CONTRACT"]},
    "QUIESCENT_LO_RECOVERY": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["MODE_SEMANTICS_CONFORMANCE", "EFFECTIVE_EVENT_ORDER", "REFERENCE_SEMANTICS_CONTRACT"]},
    "RELEASE_VERSION_SELECTION": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["MODE_SEMANTICS_CONFORMANCE", "RELEASE_FIXED_REMOVAL_MAPPING", "REFERENCE_SEMANTICS_CONTRACT"]},
    "STANDARD_EMPTY_LO_INITIALIZATION": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["INITIAL_QUIESCENCE", "BOOT_INITIALIZATION"]},
    "COMPLETE_INTEGER_SWITCH_DOMAINS": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["CASE1_INTEGER_DOMAIN", "CASE2_INTEGER_DOMAIN", "ZERO_RELATIVE_START"]},
    "DISCRETE_TICK_EMBEDDING": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["DISCRETE_TICK_EMBEDDING"]},
    "REFERENCE_PREFIX_EXTENSIBILITY": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER", "REFERENCE_PREFIX_EXTENSION"]},
    "REFERENCE_TRANSITION_SYSTEM_IDENTITY": {"source_kind": "VERIFIED_PREDECESSORS", "predecessor_obligation_ids": ["REFERENCE_TRANSITION_SYSTEM_IDENTITY"]},
}


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != "reference_model_conformance_contract_v1":
        raise ValueError("REFERENCE_CONFORMANCE_CONTRACT_VERSION_INVALID")
    conditions = contract.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(EXPECTED_CONFORMANCE_CONDITIONS):
        raise ValueError(f"REFERENCE_CONFORMANCE_CONTRACT_MUST_HAVE_{len(EXPECTED_CONFORMANCE_CONDITIONS)}_CONDITIONS")
    actual = {}
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("REFERENCE_CONFORMANCE_CONDITION_INVALID")
        condition_id = condition.get("condition_id")
        if condition_id in actual:
            raise ValueError("REFERENCE_CONFORMANCE_CONDITION_IDS_INVALID")
        if condition_id not in EXPECTED_CONFORMANCE_CONDITIONS:
            raise ValueError(f"REFERENCE_CONFORMANCE_UNKNOWN_CONDITION:{condition_id}")
        predecessor_ids = condition.get("predecessor_obligation_ids")
        if not isinstance(predecessor_ids, list) or len(predecessor_ids) != len(set(predecessor_ids)):
            raise ValueError(f"REFERENCE_CONFORMANCE_PREDECESSORS_INVALID:{condition_id}")
        expected = EXPECTED_CONFORMANCE_CONDITIONS[condition_id]
        if condition.get("source_kind") != expected["source_kind"]:
            raise ValueError(f"REFERENCE_CONFORMANCE_SOURCE_KIND_MISMATCH:{condition_id}")
        if predecessor_ids != expected["predecessor_obligation_ids"]:
            raise ValueError(f"REFERENCE_CONFORMANCE_PREDECESSOR_MAPPING_MISMATCH:{condition_id}")
        actual[condition_id] = condition
    if set(actual) != set(EXPECTED_CONFORMANCE_CONDITIONS):
        raise ValueError("REFERENCE_CONFORMANCE_CONDITION_SET_MISMATCH")
    registry_ids = {row["id"] for row in load_registry(CONTRACT_PATH.parent / "obligation_registry.json")}
    all_predecessors = {oid for row in conditions for oid in row["predecessor_obligation_ids"]}
    if not all_predecessors <= registry_ids:
        raise ValueError("REFERENCE_CONFORMANCE_UNKNOWN_PREDECESSOR")
    expected_hash = contract.get("contract_hash")
    actual_hash = sha256_object({key: value for key, value in contract.items() if key != "contract_hash"})
    if expected_hash != actual_hash:
        raise ValueError("REFERENCE_CONFORMANCE_CONTRACT_HASH_MISMATCH")
    return dict(contract)


def load_reference_model_conformance_contract() -> dict[str, Any]:
    import json

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("REFERENCE_CONFORMANCE_CONTRACT_INVALID")
    return _validate_contract(contract)


def _validated_predecessor(
    certificate: Mapping[str, Any], obligation_id: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str, str]:
    if certificate.get("obligation_id") != obligation_id:
        return False, "", ""
    if certificate.get("obligation_status") != "PASS":
        return False, "", ""
    if not verify_obligation_certificate(certificate):
        return False, "", ""
    expected_context_hash = expected_context_for_obligation(obligation_id, contexts)
    if certificate.get("certificate_context_hash") != expected_context_hash:
        return False, "", ""
    layer = context_layer_for_obligation(obligation_id)
    return True, layer, expected_context_hash


def _taskset_direct_checks(reference_taskset: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("REFERENCE_TASKSET_MISSING")
    def integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
    return {
        "FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE": {
            "pass": all(integer(t.get("period")) and t["period"] > 0 for t in tasks)
            and reference_taskset.get("periodic_language_is_sporadic_sub_language") is True,
            "fields": ["period", "periodic_language_is_sporadic_sub_language"],
        },
        "CONSTRAINED_DEADLINES": {
            "pass": all(integer(t.get("deadline")) and integer(t.get("period"))
                        and 0 < t["deadline"] <= t["period"] for t in tasks),
            "fields": ["deadline", "period"],
        },
        "FIXED_POSITIVE_INTEGER_EXECUTION": {
            "pass": all(integer(t.get("c_lo")) and integer(t.get("c_hi"))
                        and t["c_lo"] > 0 and t["c_hi"] > 0 for t in tasks),
            "fields": ["c_lo", "c_hi"],
        },
        "CRITICALITY_MONOTONE_EXECUTION": {
            "pass": all(
                (t.get("criticality") == "LO" and 0 < t["c_hi"] <= t["c_lo"])
                or (t.get("criticality") == "HI" and 0 < t["c_lo"] <= t["c_hi"])
                for t in tasks if isinstance(t, Mapping)
            ),
            "fields": ["criticality", "c_lo", "c_hi"],
        },
    }


def build_reference_model_conformance_certificate(
    *,
    reference_taskset: Mapping[str, Any],
    context_hash: str,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    conformance_contract: Mapping[str, Any],
    imported_theorem: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(verified_predecessors, Mapping):
        raise ValueError("REFERENCE_CONFORMANCE_PREDECESSORS_REQUIRED")
    contract =     _validate_contract(conformance_contract)
    contract_hash = contract["contract_hash"]
    conditions = contract["conditions"]
    condition_ids = [row["condition_id"] for row in conditions]
    theorem_premises = imported_theorem.get("premise_obligation_ids")
    if not isinstance(theorem_premises, list) or not theorem_premises:
        raise ValueError("THEOREM_MACHINE_PREMISES_REQUIRED")

    required_predecessors = {
        predecessor_id
        for condition in conditions
        for predecessor_id in condition.get("predecessor_obligation_ids", [])
    }
    if set(verified_predecessors) != required_predecessors:
        raise ValueError("REFERENCE_CONFORMANCE_EXACT_PREDECESSOR_SET_MISMATCH")
    taskset_fingerprint = reference_taskset.get("fingerprint")
    if not isinstance(taskset_fingerprint, str) or len(taskset_fingerprint) != 64:
        raise ValueError("REFERENCE_TASKSET_FINGERPRINT_REQUIRED")
    reference_certificate = verified_predecessors["REFERENCE_TASKSET"]
    certificate_taskset = reference_certificate.get("witness", {}).get("reference_taskset", {})
    if certificate_taskset.get("fingerprint") != taskset_fingerprint:
        raise ValueError("REFERENCE_TASKSET_FINGERPRINT_MISMATCH")

    direct_checks = _taskset_direct_checks(reference_taskset)
    identity_witness = (
        verified_predecessors[
            "REFERENCE_TRANSITION_SYSTEM_IDENTITY"
        ].get("witness", {})
    )
    direct_checks["REFERENCE_TRANSITION_SYSTEM_IDENTITY"] = {
        "pass": (
            identity_witness.get("transition_system_id")
            == "FIXED_EXECUTABLE_REFERENCE_P0_V3"
            and identity_witness.get("contract", {}).get("status") == "PASS"
            and identity_witness.get("identity_scope", {}).get("event_frontier")
            == "EFFECTIVE_EVENT_FRONTIER_RELATION"
        ),
        "fields": [
            "transition_system_id",
            "contract",
            "identity_scope",
            "source_bindings",
        ],
    }
    rows = []
    all_passed = True
    for condition in conditions:
        condition_id = condition["condition_id"]
        predecessor_ids = list(condition.get("predecessor_obligation_ids", []))
        predecessor_hashes = {}
        predecessor_pass = True
        predecessor_context_layers = {}
        predecessor_context_hashes = {}
        for predecessor_id in predecessor_ids:
            certificate = verified_predecessors[predecessor_id]
            valid, layer, predecessor_context_hash = _validated_predecessor(certificate, predecessor_id, contexts)
            if not valid:
                predecessor_pass = False
            predecessor_hashes[predecessor_id] = certificate.get("artifact_hash")
            predecessor_context_layers[predecessor_id] = layer
            predecessor_context_hashes[predecessor_id] = predecessor_context_hash
        direct = direct_checks.get(condition_id)
        if direct is not None:
            direct = {**direct, "passed": direct["pass"], "checked_fields": direct["fields"], "recomputed_values": {}}
        passed = predecessor_pass and (direct is None or direct["pass"] is True)
        all_passed = all_passed and passed
        rows.append({
            "condition_id": condition_id,
            "passed": passed,
            "source_kind": condition["source_kind"],
            "predecessor_ids": predecessor_ids,
            "predecessor_hashes": predecessor_hashes,
            "predecessor_context_layers": predecessor_context_layers,
            "predecessor_context_hashes": predecessor_context_hashes,
            "direct_check": direct,
        })

    status = "PASS" if all_passed else "FAIL"
    direct_hashes = {oid: verified_predecessors[oid]["artifact_hash"] for oid in sorted(required_predecessors)}
    witness = {
        "contract_version": contract["schema_version"],
        "contract_hash": contract_hash,
        "imported_theorem_id": imported_theorem.get("theorem_id"),
        "imported_theorem_statement_hash": imported_theorem.get("statement_hash"),
        "condition_ids": condition_ids,
        "condition_results": rows,
        "all_conditions_covered": len(rows) == len(condition_ids) and all(row["condition_id"] in condition_ids for row in rows),
        "exact_predecessor_set": sorted(required_predecessors),
        "reference_taskset_fingerprint": taskset_fingerprint,
        "reference_context_hash": context_hash,
    }
    witness["conformance_hash"] = sha256_object(witness)
    return obligation_certificate(
        obligation_id="REFERENCE_MODEL_CONFORMANCE",
        status=status,
        context_hash=context_hash,
        inputs={
            "reference_taskset_fingerprint": taskset_fingerprint,
            "imported_theorem_id": imported_theorem.get("theorem_id"),
            "imported_theorem_statement_hash": imported_theorem.get("statement_hash"),
            "conformance_contract_hash": contract_hash,
        },
        witness=witness,
        direct_predecessor_hashes=direct_hashes,
        checker_id=__name__,
        checker_version="reference-conformance-v3",
        failure=None if status == "PASS" else {
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "REFERENCE_MODEL_CONFORMANCE_CONDITION_FAILED",
        },
    )
