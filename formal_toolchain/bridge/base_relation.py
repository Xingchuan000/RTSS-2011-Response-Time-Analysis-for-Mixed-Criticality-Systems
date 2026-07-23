"""PreClosed(0) 的参数化 boot/batch base certificate。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from .state_relation import P0ConcreteState, P0ReferenceState, relation_holds


REQUIRED_RELEASE_CASES = ("PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE")


def empty_boot_states(*, reference_taskset: Mapping[str, Any]) -> tuple[P0ConcreteState, P0ReferenceState]:
    """构造唯一允许的空 boot；time-0 demand 由后续证据闭合。"""
    budgets = tuple(sorted((str(task["name"]), int(task["c_lo"]))
                           for task in reference_taskset["tasks"]))
    kwargs = {"time": 0, "mode": "LO", "global_future_budgets": budgets,
              "next_controller_boundary": None, "next_timing_boundary": 0}
    return P0ConcreteState(**kwargs), P0ReferenceState(**kwargs)


def build_preclosed0_base_certificate(*, context_hash: str,
                                      reference_taskset: Mapping[str, Any],
                                      demand_oracle_certificate: Mapping[str, Any],
                                      boot_path_certificate: Mapping[str, Any],
                                      arrival_batch_decomposition_certificate: Mapping[str, Any],
                                      handler_decomposition_certificate: Mapping[str, Any] | None = None,
                                      transition_case_certificates: Sequence[Mapping[str, Any]],
                                      event_order_certificate: Mapping[str, Any],
                                      concrete_boot: P0ConcreteState | None = None,
                                      reference_boot: P0ReferenceState | None = None) -> dict[str, Any]:
    """验证空 boot 关系，并以 demand/arrival/release 证据闭合 time-0 batch。"""
    if concrete_boot is None or reference_boot is None or not relation_holds(concrete_boot, reference_boot):
        raise ValueError("EMPTY_BOOT_RELATION_FAILED")
    required = (demand_oracle_certificate, boot_path_certificate,
                arrival_batch_decomposition_certificate, event_order_certificate)
    if demand_oracle_certificate.get("obligation_status") != "PASS":
        raise ValueError("DEMAND_ORACLE_BATCH_CONTRACT_REQUIRED")
    boot_result = boot_path_certificate.get("z3_proof_result", boot_path_certificate.get("witness", {}).get("z3_proof_result"))
    if boot_result != "PASS":
        raise ValueError("BOOT_PATH_PROOF_REQUIRED")
    if arrival_batch_decomposition_certificate.get("obligation_status", arrival_batch_decomposition_certificate.get("status")) != "PASS":
        raise ValueError("ARRIVAL_BATCH_DECOMPOSITION_REQUIRED")
    if (handler_decomposition_certificate is None
            or handler_decomposition_certificate.get("status") != "PASS"):
        raise ValueError("HANDLER_COMPOSITION_REQUIRED")
    arrival_handler = handler_decomposition_certificate.get("handlers", {}).get("arrival_batch", {})
    arrival_results = arrival_handler.get("alternative_results", {})
    if (
        arrival_handler.get("fold_status") != "PASS"
        or arrival_handler.get("fold_theorem")
        != "FINITE_SEQUENCE_INDUCTION_OVER_FRESH_RELEASE_MAP_EXTENSIONS"
        or not isinstance(arrival_results, Mapping)
        or set(arrival_results) != {"ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0"}
        or any(result.get("proof_status") != "PASS" for result in arrival_results.values())
    ):
        raise ValueError("ARRIVAL_MICROSTEP_COMPOSITION_REQUIRED")
    preclosed_composition = handler_decomposition_certificate.get("preclosed0_composition", {})
    if preclosed_composition.get("status") != "PASS":
        raise ValueError("PRECLOSED0_MICROSTEP_COMPOSITION_REQUIRED")
    if event_order_certificate.get("obligation_status") != "PASS":
        raise ValueError("EVENT_ORDER_REQUIRED")
    by_case = {str(item.get("case_id", item.get("inputs", {}).get("case_id"))): item
               for item in transition_case_certificates}
    for case_id in REQUIRED_RELEASE_CASES:
        item = by_case.get(case_id, {})
        proof_status = item.get("z3_proof_result",
                               item.get("witness", {}).get("z3_proof_result",
                               item.get("obligation_status")))
        if proof_status != "PASS":
            raise ValueError(f"RELEASE_CASE_REQUIRED:{case_id}")
    direct = {"demand_oracle": demand_oracle_certificate["artifact_hash"],
              "boot_path": boot_path_certificate["artifact_hash"],
              "arrival_batch": arrival_batch_decomposition_certificate["artifact_hash"],
              "handler_decomposition": handler_decomposition_certificate["artifact_hash"],
              "event_order": event_order_certificate["artifact_hash"]}
    direct.update({f"case:{case_id}": by_case[case_id].get("artifact_hash", sha256_object(by_case[case_id]))
                   for case_id in REQUIRED_RELEASE_CASES})
    return obligation_certificate(
        obligation_id="PRECLOSED0_BASE_RELATION", status="PASS", context_hash=context_hash,
        inputs={"reference_taskset_hash": sha256_object(reference_taskset),
                "task_count": len(reference_taskset["tasks"])},
        direct_predecessor_hashes=direct,
        witness={"base": "EMPTY_BOOT_RELATION",
                 "closure": "PARAMETRIC_TIME0_BATCH_COMPOSITION",
                 "composition": preclosed_composition,
                 "demand_language": "DEMAND_ORACLE_BATCH_CONTRACT"},
        checker_id="formal_toolchain.bridge.base_relation.build_preclosed0_base_certificate",
        checker_version="phase-k-v2")
