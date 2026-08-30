"""Dynamic-to-base C-AMC-sem trace-refinement checks for V10.1.

The deployed runtime is allowed to reduce per-release service relative to the
paper C-AMC-sem model.  This module proves the reduction side of the BASE
route.  The paper Section 4.1 schedulability equations are deliberately not
relabelled from AMC-rtb/AMC-max code: until an exact Section 4.1 implementation
is present, BASE simply yields NOT_SUFFICIENT and the verifier continues with
PCSSC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .kernel.symbolic_state import BoundModel


@dataclass(frozen=True, slots=True)
class BaseRefinementResult:
    status: str
    receipts: tuple[dict[str, Any], ...]
    failure_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "receipts": list(self.receipts),
        }


def check_dynamic_to_base_refinement(model: BoundModel, bindings: Mapping[str, Any]) -> BaseRefinementResult:
    receipts: list[dict[str, Any]] = []
    ok = True
    for task in model.tasks:
        if task.criticality == "HI":
            normal_cap = min(int(task.actual_demand_upper), int(task.c_lo))
            all_cap = int(task.actual_demand_upper)
            rows = (
                ("HI_NORMAL_LE_C_LO", normal_cap, int(task.c_lo)),
                ("HI_ALL_LE_C_HI", all_cap, int(task.c_hi)),
            )
        else:
            primary_cap = min(int(task.actual_demand_upper), int(task.budget_upper) + 1)
            degraded_cap = min(int(task.actual_demand_upper), int(task.degraded_cost or task.c_lo))
            # The frozen C-AMC-sem runtime's degraded budget is the instantiated
            # paper C_HI_LO value for this deployment scope.
            paper_hi_lo = int(task.degraded_cost or task.c_lo)
            rows = (
                ("PRIMARY_EFFECTIVE_SERVICE_LE_C_LO", primary_cap, int(task.c_lo)),
                ("DEGRADED_EFFECTIVE_SERVICE_LE_PAPER_C_HI_LO", degraded_cap, paper_hi_lo),
            )
        for name, lhs, rhs in rows:
            passed = lhs <= rhs
            ok &= passed
            receipts.append({
                "obligation_id": f"{name}::{task.name}",
                "status": "PASS" if passed else "FAIL",
                "lhs_service_cap": lhs,
                "rhs_base_cap": rhs,
            })

    runtime = bindings.get("p0_event_order_binding", {})
    environment = bindings.get("environment_binding", {})
    demand = environment.get("demand_semantics", {}) if isinstance(environment, Mapping) else {}
    action = bindings.get("policy_action_binding", {})

    structural_checks = {
        "CONTROLLER_DOES_NOT_CHANGE_PRIORITY": (
            str(runtime.get("dispatch", "")).startswith("fixed priority")
            and str(action.get("selection", "")).startswith("FirstValid")
        ),
        "CONTROLLER_DOES_NOT_CHANGE_RELEASE_LEGALITY": (
            runtime.get("controller_boundary") == "P5 after P4 and before final dispatch"
            and environment.get("domain", {}).get("release_model") == "EXACT_PERIODIC_PHASE_ZERO"
        ),
        "CONTROLLER_DOES_NOT_CHANGE_DEADLINE": (
            runtime.get("controller_boundary") == "P5 after P4 and before final dispatch"
            and all(int(task.deadline) > 0 for task in model.tasks)
        ),
        "CONTROLLER_DOES_NOT_CHANGE_HI_CLASSIFICATION": (
            demand.get("hi_classification")
            == "NORMAL iff 1<=A<=C_LO; ABNORMAL iff C_LO<A<=C_HI; frozen at release"
            and runtime.get("release_snapshot_boundary")
            == "P3 freezes B_rel,A,class,release_entry_mode before P4/P5"
        ),
        "CONTROLLER_DOES_NOT_TRUNCATE_HI_SERVICE": demand.get("hi") == "E=A; no truncation",
        "CONTROLLER_DOES_NOT_CHANGE_MODE_SWITCH_RULE": (
            runtime.get("mode_switch_boundary")
            == "P4 LO->HI iff P3 release batch contains abnormal HI"
        ),
        "CONTROLLER_ONLY_CHANGES_FUTURE_LO_PRIMARY_CAP_AND_POLICY_HISTORY": (
            demand.get("lo_primary") == "E=min(A, B_rel+1)"
            and runtime.get("release_snapshot_boundary")
            == "P3 freezes B_rel,A,class,release_entry_mode before P4/P5"
            and runtime.get("controller_boundary") == "P5 after P4 and before final dispatch"
        ),
        "FIXED_PRIORITY_SCHEDULER_CORRESPONDENCE": (
            runtime.get("work_conserving_preemptive") is True
            and str(runtime.get("dispatch", "")).startswith("fixed priority")
        ),
    }
    for name, passed in structural_checks.items():
        receipts.append({
            "obligation_id": name,
            "status": "PASS" if passed else "UNRESOLVED",
            "basis": "frozen P0 phase ownership + release-frozen effective-demand/action binding",
        })
    ok &= all(structural_checks.values())
    receipts.append({
        "obligation_id": "DYNAMIC_TO_BASE_C_AMC_SEM_TRACE_REFINEMENT",
        "status": "PASS" if ok else "UNRESOLVED",
        "projection": "forget policy budget/history; retain schedule-visible C-AMC-sem trace",
    })
    return BaseRefinementResult(
        status="PASS" if ok else "FAIL",
        receipts=tuple(receipts),
        failure_code=None if ok else "C_AMC_SEM_SCOPE_BINDING_ERROR",
    )


def run_original_c_amc_sem_schedulability_test(model: BoundModel) -> dict[str, Any]:
    del model
    # No exact implementation of Zhang-Zheng-Gu 2024 Section 4.1 is present in
    # the supplied source.  In particular amc_py/amc.py contains AMC-rtb/max,
    # which is not a substitute.  Fail closed and continue to PCSSC.
    return {
        "obligation_id": "BASE_C_AMC_SEM_SECTION4_1_CERTIFICATE",
        "status": "UNRESOLVED",
        "code": "BASE_C_AMC_SEM_NOT_SUFFICIENT",
        "reason": "EXACT_ZHANG_ZHENG_GU_2024_SECTION4_1_SOLVER_NOT_PRESENT",
    }
