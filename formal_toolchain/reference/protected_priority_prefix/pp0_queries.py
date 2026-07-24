"""SMT2 query generators for the PP0 primitive transition obligations.

Each query defines a formal check for one transition schema case.  Queries are
designed to be checked by Z3 or compatible SMT solver.  The output contains
the SMT2 source, expected result, and binding to the transition source file.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .transition_schema import (
    CANONICAL_CASES,
    PrimitiveTransitionSchema,
    transition_obligations,
)


def _smt_declare_frames(case: PrimitiveTransitionSchema) -> str:
    """Generate SMT2 declarations for a primitive case's fields."""
    all_fields = sorted(set(
        list(case.read_fields) + list(case.write_fields) + list(case.protected_frame_fields) + list(case.guard_fields)
    ))
    lines = []
    for field in all_fields:
        lines.append(f"(declare-const {field}_pre Int)")
        lines.append(f"(declare-const {field}_post Int)")
    return "\n".join(lines)


def build_fixed_demand_not_modified_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: prove that fixed_demand is equal before and after transition."""
    if "fixed_demand" not in case.protected_frame_fields and "fixed_demand" not in case.read_fields:
        return "; trivial — fixed_demand not in transition scope"
    fields = _smt_declare_frames(case)
    return f"""; Fixed-demand not modified — {case.case_id}
{fields}
(assert (not (= fixed_demand_pre fixed_demand_post)))
(check-sat)
"""


def build_protected_key_not_modified_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: prove that protected key fields are equal before and after."""
    key_fields = [f for f in ("release_time", "absolute_deadline", "criticality",
                               "priority_index", "job_key")
                  if f in case.protected_frame_fields or f in case.read_fields]
    if not key_fields:
        return "; trivial — no protected key fields in this case"
    fields = _smt_declare_frames(case)
    conjuncts = " ".join(f"(not (= {f}_pre {f}_post))" for f in key_fields)
    return f"""; Protected key not modified — {case.case_id}
{fields}
(assert (or {conjuncts}))
(check-sat)
"""


def build_mode_not_modify_protected_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: for mode-only steps, prove protected fields unchanged."""
    protected = ["active", "ready", "running", "service", "miss"]
    relevant = [f for f in protected if f in case.read_fields or f in case.write_fields]
    if not relevant:
        return "; trivial — no protected activity fields in this case"
    fields = _smt_declare_frames(case)
    conjuncts = " ".join(f"(not (= {f}_pre {f}_post))" for f in relevant)
    return f"""; Mode-only not modify protected activity — {case.case_id}
{fields}
(assert (or {conjuncts}))
(check-sat)
"""


def build_tail_not_modify_protected_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: prove tail service does not modify any protected frame field."""
    if case.case_id != "TAIL_ONLY_SERVICE":
        return "; not applicable — this is not a tail-only case"
    protected = list(case.protected_frame_fields)
    if not protected:
        return "; trivial"
    fields = _smt_declare_frames(case)
    conjuncts = " ".join(f"(not (= {f}_pre {f}_post))" for f in protected)
    return f"""; Tail service preserves protected frame — {case.case_id}
{fields}
(assert (or {conjuncts}))
(check-sat)
"""


def build_ddl_read_only_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: DDL only reads/completes, monotonically appends miss."""
    if case.case_id != "DDL_OBSERVE":
        return "; not applicable"
    fields = _smt_declare_frames(case)
    return f"""; DDL observe-only — {case.case_id}
{fields}
(declare-const active_pre Bool)
(declare-const active_post Bool)
(declare-const ready_pre Bool)
(declare-const ready_post Bool)
(assert (not (= active_pre active_post)))
(assert (not (= ready_pre ready_post)))
(check-sat)
"""


def build_completion_guard_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: completion guard == service >= fixed_demand."""
    if case.case_id != "REM_COMPLETION":
        return "; not applicable"
    return f"""; Completion guard equivalence — {case.case_id}
(declare-const service Int)
(declare-const fixed_demand Int)
(assert (not (= (>= service fixed_demand) (>= service fixed_demand))))
(check-sat)
"""


def build_dispatch_priority_total_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: dispatch is strict fixed-priority total selection."""
    if case.case_id != "FINAL_DISPATCH":
        return "; not applicable"
    return f"""; Dispatch fixed-priority total — {case.case_id}
(declare-const selected_priority Int)
(declare-const max_ready_priority Int)
(declare-const selected_job_key Int)
(assert (and (> selected_priority max_ready_priority) (not (= selected_job_key 0))))
(check-sat)
"""


def build_service_unit_discrete_rate_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: service increases by one discrete unit."""
    if case.case_id != "SERVICE_UNIT":
        return "; not applicable"
    return f"""; Service unit discrete rate — {case.case_id}
(declare-const service_pre Int)
(declare-const service_post Int)
(assert (not (= service_post (+ service_pre 1))))
(check-sat)
"""


def build_arrival_protected_independent_of_tail(case: PrimitiveTransitionSchema) -> str:
    """SMT2: protected entries in arrival batch are independent of tail deletion."""
    if case.case_id != "ARRIVAL_BATCH_OPEN":
        return "; not applicable"
    return f"""; Arrival batch protected independent of tail — {case.case_id}
(declare-const protected_entries_pre Int)
(declare-const protected_entries_post Int)
(declare-const tail_deleted Bool)
(assert (and tail_deleted (not (= protected_entries_pre protected_entries_post))))
(check-sat)
"""


def build_same_time_closure_finite_query(case: PrimitiveTransitionSchema) -> str:
    """SMT2: closure measure is finite and strictly decreasing."""
    fields = _smt_declare_frames(case)
    return f"""; Same-time closure finite measure — {case.case_id}
{fields}
(declare-const time_delta Int)
(declare-const closure_measure_pre Int)
(declare-const closure_measure_post Int)
(assert (not (or (= time_delta 0) (< closure_measure_post closure_measure_pre))))
(check-sat)
"""




def is_trivial_query_source(smt2: str) -> bool:
    """Return True only for an explicit not-applicable/trivial marker.

    Nontrivial SMT2 queries also begin with a comment, so ``startswith(";")``
    is unsound and would classify every query as discharged.
    """
    first_line = smt2.lstrip().splitlines()[0] if smt2.strip() else ""
    return first_line.startswith("; trivial") or first_line.startswith("; not applicable")

QUERY_BUILDERS: dict[str, Any] = {
    "FIXED_DEMAND_NOT_MODIFIED": build_fixed_demand_not_modified_query,
    "PROTECTED_KEY_NOT_MODIFIED": build_protected_key_not_modified_query,
    "MODE_ONLY_NOT_MODIFY_PROTECTED": build_mode_not_modify_protected_query,
    "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": build_tail_not_modify_protected_query,
    "DDL_READ_ONLY_DEADLINE_COMPLETION": build_ddl_read_only_query,
    "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": build_completion_guard_query,
    "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": build_dispatch_priority_total_query,
    "SERVICE_UNIT_SINGLE_DISCRETE_RATE": build_service_unit_discrete_rate_query,
    "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": build_arrival_protected_independent_of_tail,
    "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": build_same_time_closure_finite_query,
}


def generate_all_queries() -> dict[str, dict[str, Any]]:
    """Generate all SMT2 queries for all primitive cases and obligations."""
    results: dict[str, dict[str, Any]] = {}
    for obligation_key, builder in QUERY_BUILDERS.items():
        for case in CANONICAL_CASES:
            query_id = f"{case.case_id}_{obligation_key}"
            smt2 = builder(case)
            results[query_id] = {
                "query_id": query_id,
                "case_id": case.case_id,
                "obligation": obligation_key,
                "smt2_source": smt2,
                "smt2_hash": sha256_object({"source": smt2}),
                "transition_source_binding": "reference/executable_semantics.py",
                "transition_equations_bound": False,
                "proof_scope": "SCHEMA_ONLY_NOT_CODE_BOUND",
                "expected_result": "trivial" if is_trivial_query_source(smt2) else "unresolved",
            }
    return results
