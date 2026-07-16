"""Phase K closed-prefix 的内部前置证书 builders。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.reference.arithmetic import ceil_div_nonnegative
from .event_projection import project_event
from .state_relation import P0ConcreteState, P0ReferenceState, P0Event, relation_holds, p0_state_relation_schema_hash
from .model_bounds import P0ModelBounds


def _theory(theorem_id: str) -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "theory" / "hashes.json"
    return json.loads(path.read_text(encoding="utf-8"))["statements"][theorem_id]


def build_base_relation_certificate(*, context_hash: str,
                                    concrete: P0ConcreteState | None = None,
                                    reference: P0ReferenceState | None = None,
                                    bounds: P0ModelBounds | None = None) -> dict[str, Any]:
    if concrete is None or reference is None:
        return {"status": "UNRESOLVED", "failure": "PRECLOSED_BASE_STATE_REQUIRED"}
    if bounds is None:
        return {"status": "UNRESOLVED", "failure": "MODEL_BOUNDS_REQUIRED"}
    status = "PASS" if relation_holds(concrete, reference) else "FAIL"
    failure = None if status == "PASS" else {"code": "INITIAL_RELATION_FAILED"}
    return obligation_certificate(
        obligation_id="BASE_RELATION", status=status, context_hash=context_hash,
        inputs={"state_relation_schema": p0_state_relation_schema_hash(bounds)},
        witness={"concrete": asdict(concrete), "reference": asdict(reference)},
        checker_id=__name__, checker_version="phase-k-v1", failure=failure)


def build_same_timestamp_closure_certificate(*, context_hash: str, case_proofs: list[Mapping[str, Any]],
                                             event_order_certificate: Mapping[str, Any] | None = None,
                                             coverage_certificate: Mapping[str, Any] | None = None,
                                             decomposition_certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    zero_time = [p.get("inputs", {}).get("case_id") for p in case_proofs
                 if p.get("inputs", {}).get("case_id") and p.get("inputs", {}).get("case_id") != "ONE_SERVICE_TICK"
                 and p.get("inputs", {}).get("case_id") != "JUMP_TO_NEXT_EVENT"]
    theorem = _theory("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")
    valid_cases = all(verify_obligation_certificate(item)
                      and item.get("obligation_status") == "PASS"
                      and item.get("certificate_context_hash") == context_hash
                      for item in case_proofs)
    status = "PASS" if (zero_time and event_order_certificate is not None
                        and coverage_certificate and coverage_certificate.get("status") == "PASS"
                        and decomposition_certificate and decomposition_certificate.get("status") == "PASS"
                        and valid_cases) else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "SAME_TIMESTAMP_INPUT_MISSING"}
    predecessors = {}
    if event_order_certificate is not None:
        predecessors["event_order"] = event_order_certificate["artifact_hash"]
    return obligation_certificate(
        obligation_id="SAME_TIMESTAMP_CLOSURE", status=status, context_hash=context_hash,
        inputs={"theorem": theorem, "zero_time_case_count": len(zero_time),
                "coverage_hash": None if coverage_certificate is None else coverage_certificate.get("artifact_hash"),
                "decomposition_hash": None if decomposition_certificate is None else decomposition_certificate.get("artifact_hash")},
        witness={"zero_time_case_ids": sorted(set(zero_time)), "case_artifact_hashes": [item.get("artifact_hash") for item in case_proofs],
                 "theorem": theorem},
        direct_predecessor_hashes={**predecessors,
                                  **({"coverage": coverage_certificate["artifact_hash"]} if coverage_certificate else {}),
                                  **({"decomposition": decomposition_certificate["artifact_hash"]} if decomposition_certificate else {}),
                                  **{f"case:{index}": item["artifact_hash"] for index, item in enumerate(case_proofs)}},
        checker_id=__name__, checker_version="phase-k-v2",
        failure=failure)


def build_positive_time_service_certificate(*, context_hash: str, case_proofs: list[Mapping[str, Any]],
                                            coverage_certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = {"ONE_SERVICE_TICK", "JUMP_TO_NEXT_EVENT"}
    present = {p.get("inputs", {}).get("case_id") for p in case_proofs}
    theorem = _theory("DISCRETE_TICK_FPPS_EMBEDDING")
    status = "PASS" if required <= present and coverage_certificate and coverage_certificate.get("status") == "PASS" else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "POSITIVE_TIME_CASE_MISSING"}
    predecessors = {p["inputs"]["case_id"]: p["artifact_hash"] for p in case_proofs
                    if p.get("inputs", {}).get("case_id") in required}
    return obligation_certificate(
        obligation_id="POSITIVE_TIME_SERVICE", status=status, context_hash=context_hash,
        inputs={"theorem": theorem, "required_cases": sorted(required)},
        witness={"case_hashes": predecessors, "theorem": theorem},
        direct_predecessor_hashes=predecessors, checker_id=__name__, checker_version="phase-k-v1",
        failure=failure)


def build_controller_postclosure_certificate(*, context_hash: str, case_proofs: list[Mapping[str, Any]],
                                              controller_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = {"CONTROLLER_NO_ACTION", "CONTROLLER_SELECTED_ACTION"}
    present = {p.get("inputs", {}).get("case_id") for p in case_proofs}
    theorem = _theory("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")
    status = "PASS" if required <= present and controller_binding and controller_binding.get("status") == "PASS" else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "CONTROLLER_CASE_MISSING"}
    predecessors = {p["inputs"]["case_id"]: p["artifact_hash"] for p in case_proofs
                    if p.get("inputs", {}).get("case_id") in required}
    return obligation_certificate(
        obligation_id="CONTROLLER_POSTCLOSURE", status=status, context_hash=context_hash,
        inputs={"theorem": theorem, "controller_binding_hash": None if controller_binding is None else controller_binding.get("binding_hash")}, witness={"case_hashes": predecessors, "theorem": theorem},
        direct_predecessor_hashes=predecessors, checker_id=__name__, checker_version="phase-k-v1",
        failure=failure)


def build_event_projection_certificate(*, context_hash: str) -> dict[str, Any]:
    table = {
        "CONTROLLER": None, "TREE": None, "MASK": None, "OBSERVATION": None,
        "BUDGET_UPDATE_LABEL": None, "PRIMARY_LO_CANCELLATION": "JOB_COMPLETION",
        "NORMAL_COMPLETION": "JOB_COMPLETION", "DEGRADED_COMPLETION": "JOB_COMPLETION",
        "HI_COMPLETION": "HI_COMPLETION", "HI_DEADLINE_MISS": "HI_DEADLINE_MISS",
    }
    actual = {}
    for kind, projected in table.items():
        value = project_event(P0Event(0, kind))
        actual[kind] = None if value is None else value.kind
    status = "PASS" if actual == table else "FAIL"
    failure = None if status == "PASS" else {"code": "EVENT_PROJECTION_TABLE_MISMATCH"}
    module_hash = sha256_file(Path(__file__).resolve().parent / "event_projection.py")
    return obligation_certificate(
        obligation_id="EVENT_PROJECTION", status=status, context_hash=context_hash,
        inputs={"semantic_source_hash": module_hash}, witness={"expected": table, "actual": actual},
        checker_id=__name__, checker_version="phase-k-v1", failure=failure)


def build_event_order_certificate(*, context_hash: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    status = "PASS" if binding.get("status") == "PASS" and binding.get("semantics", {}).get("verified") else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "EVENT_ORDER_BINDING_UNRESOLVED"}
    return obligation_certificate(
        obligation_id="EFFECTIVE_EVENT_ORDER", status=status, context_hash=context_hash,
        inputs={"binding_hash": sha256_object(binding)}, witness={"binding": binding},
        checker_id=__name__, checker_version="phase-k-v1", failure=failure)


def build_deadline_observation_certificate(*, context_hash: str, removal_binding: Mapping[str, Any]) -> dict[str, Any]:
    contract = removal_binding.get("p0_contract", {})
    status = "PASS" if removal_binding.get("status") == "PASS" and contract.get("deadline_observe_only") else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "DEADLINE_OBSERVATION_BINDING_UNRESOLVED"}
    return obligation_certificate(
        obligation_id="DEADLINE_OBSERVATION", status=status, context_hash=context_hash,
        inputs={"binding_hash": sha256_object(removal_binding)}, witness={"contract": contract},
        checker_id=__name__, checker_version="phase-k-v1", failure=failure)


def build_hi_nontruncation_certificate(*, context_hash: str, removal_binding: Mapping[str, Any]) -> dict[str, Any]:
    contract = removal_binding.get("p0_contract", {})
    status = "PASS" if removal_binding.get("status") == "PASS" and contract.get("hi_nontruncation") else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "HI_NONTRUNCATION_BINDING_UNRESOLVED"}
    return obligation_certificate(
        obligation_id="HI_NONTRUNCATION", status=status, context_hash=context_hash,
        inputs={"binding_hash": sha256_object(removal_binding)}, witness={"contract": contract},
        checker_id=__name__, checker_version="phase-k-v1", failure=failure)


def build_bridge_prerequisite_certificates(*, context_hash: str, case_proofs: list[Mapping[str, Any]],
                                           event_order_certificate: Mapping[str, Any] | None = None,
                                           concrete_base: P0ConcreteState | None = None,
                                           reference_base: P0ReferenceState | None = None,
                                           bounds: P0ModelBounds | None = None,
                                           coverage_certificate: Mapping[str, Any] | None = None,
                                           decomposition_certificate: Mapping[str, Any] | None = None,
                                           controller_binding: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    base = build_base_relation_certificate(context_hash=context_hash, concrete=concrete_base,
                                           reference=reference_base, bounds=bounds)
    event = build_event_projection_certificate(context_hash=context_hash)
    same = build_same_timestamp_closure_certificate(context_hash=context_hash, case_proofs=case_proofs,
                                                    event_order_certificate=event_order_certificate or event,
                                                    coverage_certificate=coverage_certificate,
                                                    decomposition_certificate=decomposition_certificate)
    positive = build_positive_time_service_certificate(context_hash=context_hash, case_proofs=case_proofs,
                                                       coverage_certificate=coverage_certificate)
    controller = build_controller_postclosure_certificate(context_hash=context_hash, case_proofs=case_proofs,
                                                          controller_binding=controller_binding)
    return {"base_relation": base, "same_timestamp": same, "positive_time": positive,
            "controller_postclosure": controller, "event_projection": event}
