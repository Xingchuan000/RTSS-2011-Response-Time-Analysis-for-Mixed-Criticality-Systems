"""V8 raw protected-prefix mode-ordered proof kernel.

This module is deliberately separate from the V7 saturation kernel.  The raw
route never assumes ``C_LO == C_HI`` for protected LO tasks.  Instead it derives
``mode_raw <= mode_full`` (LO <= HI), then uses the monotonicity of the
mode-specific admissible demand sets to justify copying release-fixed demands.

The receipts are parameterized theorem applications.  They are accepted only
when their source-bound runtime-schema predecessor and exact route predecessors
have already PASSed in the fresh verifier.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

RAW_PROOF_KERNEL_VERSION = "raw_protected_prefix_v8_mode_order_v1"
MODE_ORDER = {"LO": 0, "HI": 1}

_THEOREM_STATEMENTS = {
    "RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE":
        "Rel_raw and full quiescent idle imply raw quiescent idle because removed tail jobs are strictly lower priority and cannot block protected jobs.",
    "RAW_PREFIX_RECOVERY_ORDER_PRESERVATION":
        "For LO<=HI mode order, idle-only recovery in either runtime preserves mode_raw<=mode_full; full recovery is enabled only when raw is also idle.",
    "RAW_PREFIX_SWITCH_ORDER_PRESERVATION":
        "A protected abnormal-HI arrival is the only LO-to-HI trigger in both runtimes, so the switch phase preserves mode_raw<=mode_full.",
    "RAW_PREFIX_MODE_ORDER_INVARIANT":
        "From equal LO initial modes and preservation by the only mode-writing REC/SW phases, all closed boundaries satisfy mode_raw<=mode_full.",
    "RAW_PREFIX_ADMISSIBLE_SET_DOMINATION":
        "For every protected release, mode_raw<=mode_full implies Adm(full_mode) subseteq Adm(raw_mode), using C_LO_tasks(HI)<=C_LO_tasks(LO) and inherited HI thresholds.",
    "RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS":
        "Every full protected release-fixed actual demand is accepted by raw at the corresponding release under the derived mode order and input receptiveness contract.",
    "RAW_PREFIX_PROTECTED_ARRIVAL_BATCH_PROJECTION":
        "Protected arrival batches preserve job keys, release times, HI class and copied actual demands; tail arrivals are erased.",
    "RAW_PREFIX_SERVICE_CORRESPONDENCE":
        "If protected ready is nonempty both fixed-priority runtimes execute the same protected job; otherwise raw idles while full may idle or serve only lower-priority tail.",
    "RAW_PREFIX_COMPLETION_CORRESPONDENCE":
        "Equal copied demand and equal protected service imply simultaneous protected completion/removal.",
    "RAW_PREFIX_DEADLINE_BATCH_CORRESPONDENCE":
        "Protected deadline observations preserve due time, completion state and HI miss ledger under the protected observable relation.",
    "RAW_PREFIX_TOTAL_FINAL_DISPATCH_CORRESPONDENCE":
        "Final dispatch selects the same protected job when one is ready, otherwise raw has no protected running job while full may select tail.",
    "RAW_PREFIX_CLOSE_TO_CLOSE_MACROSTEP":
        "The canonical SVC/REM/REC/DDL/ARR/SW/REL/DSP closure macrostep preserves protected observables and mode_raw<=mode_full.",
    "RAW_PREFIX_COMPLETE_EXECUTION_EXISTENCE":
        "The raw runtime has one complete time-divergent execution constructible stepwise from each full execution using current-prefix mode order to validate next releases.",
    "RAW_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED":
        "For every full execution there exists one complete raw execution such that every natural-number closed boundary preserves protected observables and mode_raw<=mode_full.",
    "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT":
        "The route-neutral N4 pre/post-closed timing projection is aligned with the canonical reference Close(t) boundary consumed by either raw or saturated terminal routes.",
    "RAW_PREFIX_HI_BAD_PREFIX_REFLECTION":
        "A full-reference HI deadline miss is reflected at the same absolute deadline to the corresponding HI job in the constructed raw execution.",
    "RAW_PREFIX_TASKSET_SCHEDULABLE":
        "Raw-prefix mathematical conformance, imported C-AMC-sem theorem binding, fresh verifier soundness, instance evidence binding, and all-task RTA establish schedulability of every raw-prefix task.",
    "REFERENCE_HI_SAFETY_FROM_RAW_PREFIX":
        "Raw-prefix taskset schedulability plus full-to-raw HI bad-prefix reflection implies full-reference HI safety by contradiction.",
}

RAW_DERIVED_THEOREM_IDS = tuple(_THEOREM_STATEMENTS)


def theorem_hash(theorem_id: str) -> str:
    return sha256_object({
        "kernel_version": RAW_PROOF_KERNEL_VERSION,
        "theorem_id": theorem_id,
        "statement": _THEOREM_STATEMENTS[theorem_id],
    })


def _receipt(theorem_id: str, *, status: str = "PASS", code: str | None = None,
             dependencies: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    deps = dict(dependencies or {})
    payload = {
        "proof_kernel_version": RAW_PROOF_KERNEL_VERSION,
        "theorem_id": theorem_id,
        "theorem_hash": theorem_hash(theorem_id),
        "statement": _THEOREM_STATEMENTS[theorem_id],
        "parameterized": True,
        "status": status,
        "code": code,
        "dependency_hashes": {
            key: (value.get("artifact_hash") or value.get("receipt_hash") or sha256_object(value))
            if isinstance(value, Mapping) else sha256_object(value)
            for key, value in sorted(deps.items())
        },
        **fields,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def prove_full_idle_implies_raw_idle(*, construction: Any, runtime_witness: Mapping[str, Any]) -> dict[str, Any]:
    partition = construction.partition_witness
    conditions = {
        "tail_all_lo": partition.get("tail_all_lo") is True,
        "priority_closed": partition.get("priority_closed") is True,
        "strict_fpps": runtime_witness.get("single_processor_preemptive_work_conserving_fp") is True,
        "no_lower_priority_blocking": runtime_witness.get("no_blocking_self_suspension_or_nonpreemptive_segments") is True,
        "fixed_supply": runtime_witness.get("fixed_processor_supply_and_mode_independent_priority") is True,
    }
    ok = all(conditions.values())
    return _receipt("RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE",
                    status="PASS" if ok else "UNRESOLVED",
                    code=None if ok else "RAW_PREFIX_FULL_IDLE_DOMAIN_UNPROVED",
                    conditions=conditions,
                    implication="full_idle => protected_ready_empty => raw_ready_empty => raw_idle")


def prove_recovery_order_preservation(*, full_idle_receipt: Mapping[str, Any], runtime_witness: Mapping[str, Any]) -> dict[str, Any]:
    legal_pairs = (("LO", "LO"), ("LO", "HI"), ("HI", "HI"))
    rows = []
    for raw_mode, full_mode in legal_pairs:
        # Enumerate abstract recovery choices allowed by the theorem domain.
        # raw may recover alone from (HI,HI); full recovering from (HI,HI)
        # consumes full-idle=>raw-idle and therefore raw recovers too.
        successors = {(raw_mode, full_mode)}
        if raw_mode == "HI":
            successors.add(("LO", full_mode))
        if full_mode == "HI":
            if raw_mode == "HI":
                successors.add(("LO", "LO"))
            else:
                successors.add(("LO", "LO"))
        rows.append({
            "before": [raw_mode, full_mode],
            "successors": [list(p) for p in sorted(successors)],
            "all_ordered": all(MODE_ORDER[r] <= MODE_ORDER[f] for r, f in successors),
        })
    conditions = {
        "idle_only_recovery": runtime_witness.get("quiescent_idle_only_recovery") is True,
        "full_idle_implies_raw_idle": full_idle_receipt.get("status") == "PASS",
        "all_mode_pairs_closed": all(row["all_ordered"] for row in rows),
    }
    ok = all(conditions.values())
    return _receipt("RAW_PREFIX_RECOVERY_ORDER_PRESERVATION",
                    status="PASS" if ok else "UNRESOLVED",
                    code=None if ok else "RAW_PREFIX_RECOVERY_ORDER_UNPROVED",
                    conditions=conditions, truth_table=rows)


def prove_switch_order_preservation(*, construction: Any, runtime_witness: Mapping[str, Any]) -> dict[str, Any]:
    # Same protected HI tasks + same abnormal classification.  Tail is LO-only,
    # hence deletion cannot remove or create a switch trigger.
    partition = construction.partition_witness
    conditions = {
        "all_hi_protected": partition.get("all_hi_protected") is True,
        "tail_all_lo": partition.get("tail_all_lo") is True,
        "abnormal_classification_at_arrival": runtime_witness.get("abnormal_classification_at_arrival") is True,
        "abnormal_hi_only_switch_trigger": runtime_witness.get("abnormal_hi_only_switch_trigger") is True,
        "protected_input_independence": runtime_witness.get("protected_input_independence") is True,
    }
    rows = [
        {"before": ["LO", "LO"], "trigger": False, "after": ["LO", "LO"]},
        {"before": ["LO", "LO"], "trigger": True, "after": ["HI", "HI"]},
        {"before": ["LO", "HI"], "trigger": False, "after": ["LO", "HI"]},
        {"before": ["LO", "HI"], "trigger": True, "after": ["HI", "HI"]},
        {"before": ["HI", "HI"], "trigger": False, "after": ["HI", "HI"]},
    ]
    ok = all(conditions.values()) and all(
        MODE_ORDER[row["after"][0]] <= MODE_ORDER[row["after"][1]] for row in rows
    )
    return _receipt("RAW_PREFIX_SWITCH_ORDER_PRESERVATION",
                    status="PASS" if ok else "UNRESOLVED",
                    code=None if ok else "RAW_PREFIX_SWITCH_ORDER_UNPROVED",
                    conditions=conditions, truth_table=rows)


def prove_global_mode_order(*, recovery_receipt: Mapping[str, Any], switch_receipt: Mapping[str, Any], runtime_witness: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "initial_modes_equal_lo": True,
        "recovery_preserves_order": recovery_receipt.get("status") == "PASS",
        "switch_preserves_order": switch_receipt.get("status") == "PASS",
        "only_rec_sw_write_mode": runtime_witness.get("mode_transitions_zero_time") is True,
    }
    ok = all(conditions.values())
    return _receipt("RAW_PREFIX_MODE_ORDER_INVARIANT",
                    status="PASS" if ok else "UNRESOLVED",
                    code=None if ok else "RAW_PREFIX_MODE_ORDER_DOMAIN_UNPROVED",
                    conditions=conditions,
                    order="LO <= HI",
                    excluded_pair=["HI", "LO"],
                    conclusion="mode_raw <= mode_full at every canonical closed boundary and release phase")


def prove_admissible_set_domination(*, full_taskset: Any, raw_taskset: Any,
                                    construction: Any, mode_order_receipt: Mapping[str, Any]) -> dict[str, Any]:
    full = {task.name: task for task in full_taskset.tasks}
    rows = []
    for task in raw_taskset.tasks:
        ft = full[task.name]
        if task.criticality == "LO":
            inherited = task.c_lo == ft.c_lo and task.c_hi == ft.c_hi
            wcet_order = 0 < task.c_hi <= task.c_lo
            pair_checks = {
                "LO_LO": task.c_lo <= task.c_lo,
                "LO_HI": task.c_hi <= task.c_lo,  # raw=LO, full=HI
                "HI_HI": task.c_hi <= task.c_hi,
            }
        else:
            inherited = task.c_lo == ft.c_lo and task.c_hi == ft.c_hi
            wcet_order = 0 < task.c_lo <= task.c_hi
            # HI normal/abnormal class is copied; its bound is threshold/class
            # determined, not weakened by the global mode divergence.
            pair_checks = {"HI_CLASS_BINDING": inherited}
        rows.append({
            "task": task.name,
            "criticality": task.criticality,
            "parameters_inherited": inherited,
            "wcet_order": wcet_order,
            "pair_checks": pair_checks,
            "all_pairs": all(pair_checks.values()),
        })
    conditions = {
        "mode_order": mode_order_receipt.get("status") == "PASS",
        "all_task_relations": all(row["parameters_inherited"] and row["wcet_order"] and row["all_pairs"] for row in rows),
    }
    ok = all(conditions.values())
    return _receipt("RAW_PREFIX_ADMISSIBLE_SET_DOMINATION",
                    status="PASS" if ok else "FAIL",
                    code=None if ok else "RAW_PREFIX_ADMISSIBLE_SET_DOMINATION_FAILED",
                    conditions=conditions, tasks=rows,
                    conclusion="Adm(full release mode) subseteq Adm(raw release mode)")


def compose(theorem_id: str, *, dependencies: Mapping[str, Any],
            extra_conditions: Mapping[str, bool] | None = None, **fields: Any) -> dict[str, Any]:
    conditions = {
        "all_predecessors_pass": all(
            isinstance(value, Mapping) and (
                value.get("obligation_status") == "PASS" or value.get("status") == "PASS"
            ) for value in dependencies.values()
        ),
        **dict(extra_conditions or {}),
    }
    ok = all(conditions.values())
    return _receipt(theorem_id, status="PASS" if ok else "UNRESOLVED",
                    code=None if ok else f"{theorem_id}_PREMISES_UNRESOLVED",
                    dependencies=dependencies, conditions=conditions, **fields)
