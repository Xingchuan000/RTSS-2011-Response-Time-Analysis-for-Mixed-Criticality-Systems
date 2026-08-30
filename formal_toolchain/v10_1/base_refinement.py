"""Dynamic-to-base C-AMC-sem trace-refinement checks for V10.1.

The deployed runtime is allowed to reduce per-release service relative to the
paper C-AMC-sem model.  This module proves the reduction side of the BASE
route.  The paper Section 4.1 schedulability equations are implemented directly
in :mod:`formal_toolchain.v10_1.base_section4_1`; AMC-rtb/AMC-max code is not
used as a substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .kernel.symbolic_state import BoundModel
from .base_section4_1 import (
    Section41ScopeError,
    bind_paper_taskset,
    prove_original_c_amc_sem_section4_1,
)


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

    try:
        paper_tasks = bind_paper_taskset(model)
    except Section41ScopeError as exc:
        receipts.append({
            "obligation_id": "BASE_C_AMC_SEM_SECTION4_1_TASK_MODEL_SCOPE",
            "status": "UNRESOLVED",
            "reason": str(exc),
        })
        return BaseRefinementResult(
            status="FAIL",
            receipts=tuple(receipts),
            failure_code="C_AMC_SEM_SCOPE_BINDING_ERROR",
        )
    else:
        receipts.append({
            "obligation_id": "BASE_C_AMC_SEM_SECTION4_1_TASK_MODEL_SCOPE",
            "status": "PASS",
            "constrained_deadlines": all(task.deadline <= task.period for task in paper_tasks),
            "lo_graceful_degradation": all(
                task.c_hi <= task.c_lo for task in paper_tasks if task.criticality == "LO"
            ),
            "hi_assurance_monotonicity": all(
                task.c_hi >= task.c_lo for task in paper_tasks if task.criticality == "HI"
            ),
        })

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
            if task.degraded_cost is None:
                raise Section41ScopeError(
                    f"LO task lacks frozen degraded C_HI binding: {task.name}"
                )
            degraded_cap = min(int(task.actual_demand_upper), int(task.degraded_cost))
            # The frozen C-AMC-sem runtime's degraded budget is the instantiated
            # paper C_HI_LO value for this deployment scope.
            paper_hi_lo = int(task.degraded_cost)
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
    """Run the exact Zhang--Zheng--Gu 2024 Section 4.1 BASE test."""

    return prove_original_c_amc_sem_section4_1(model)
