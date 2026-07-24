"""Phase J02-J05: all-task C-AMC-sem reference RTA production."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from typing import Any, Sequence

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.obligation_ids import (
    ALL_TASK_REFERENCE_RTA_ARITHMETIC, PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC,
)

from .arithmetic import ceil_div_nonnegative, floor_div_nonnegative
from .task_mapping import ReferenceTask, ReferenceTaskset


ALL_TASK_RTA_SCHEMA_VERSION = "all_task_rta_v3"


def _task_row_payload(task: ReferenceTask, row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status", "UNRESOLVED")
    return {
        "status": status,
        "task": {
            "name": task.name,
            "priority_index": int(task.priority_index),
            "period": int(task.period),
            "deadline": int(task.deadline),
            "offset": int(task.offset),
            "criticality": task.criticality,
            "c_lo": int(task.c_lo),
            "c_hi": int(task.c_hi),
            "code_c_lo": int(task.code_c_lo),
            "code_c_hi": int(task.code_c_hi),
            "degraded_cost": task.degraded_cost,
        },
        "lo": row.get("lo", {}),
        "worst_case_start": row.get("start", {}),
        "start": row.get("start", {}),
        "case1": row.get("case1", []),
        "case2": row.get("case2", []),
        "zero_relative_start_boundary": row.get("zero_relative_start_boundary", {"applicable": False}),
        "r_lo": int(row.get("r_lo", row.get("lo", {}).get("r_lo", 0))),
        "r_hi": int(row.get("r_hi", 0)),
        "r_star": int(row.get("r_star", max(int(row.get("r_lo", row.get("lo", {}).get("r_lo", 0))), int(row.get("r_hi", 0))))),
        "lo_deadline_holds": bool(row.get("lo_deadline_holds", False)),
        "hi_deadline_holds": bool(row.get("hi_deadline_holds", False)),
    }


def _lo_interference(task: ReferenceTask, higher: Sequence[ReferenceTask], r: int) -> int:
    return sum(ceil_div_nonnegative(r, j.period) * j.c_lo for j in higher)


def lo_postfixed(task: ReferenceTask, higher: Sequence[ReferenceTask], *, max_iter: int = 100000) -> dict[str, Any]:
    """Compute the LO recurrence witness."""

    r = task.c_lo
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        counts = {j.name: ceil_div_nonnegative(r, j.period) for j in higher}
        f = task.c_lo + sum(counts[j.name] * j.c_lo for j in higher)
        trace.append(
            {
                "iteration": iteration,
                "r": r,
                "release_counts": counts,
                "interference": f - task.c_lo,
                "f": f,
            }
        )
        if f <= r:
            return {
                "status": "PASS" if r <= task.deadline else "FAIL",
                "r_lo": r,
                "trace": trace,
                "post_fixed": f <= r,
            }
        r = f
        if r > task.deadline:
            return {"status": "FAIL", "r_lo": r, "trace": trace, "post_fixed": False}
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def _higher_priority_lo_utilization(higher: Sequence[ReferenceTask]) -> Fraction:
    return sum((Fraction(int(task.c_lo), int(task.period)) for task in higher), Fraction(0, 1))


def worst_case_start(task: ReferenceTask, higher: Sequence[ReferenceTask], *, max_iter: int = 100000) -> dict[str, Any]:
    """Compute W_i(LO), failing immediately when no finite post-fixed point can exist."""

    utilization = _higher_priority_lo_utilization(higher)
    if utilization >= 1:
        return {
            "status": "FAIL",
            "trace": [],
            "failure": "HIGHER_PRIORITY_LO_UTILIZATION_NOT_BELOW_ONE",
            "utilization_numerator": int(utilization.numerator),
            "utilization_denominator": int(utilization.denominator),
        }

    w = 0
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        counts = {j.name: floor_div_nonnegative(w, j.period) + 1 for j in higher}
        nxt = sum(counts[j.name] * j.c_lo for j in higher)
        trace.append({"iteration": iteration, "w": w, "release_counts": counts, "next": nxt})
        if nxt <= w:
            return {"status": "PASS", "w_lo": w, "trace": trace}
        w = nxt
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def _case1_candidate(
    task: ReferenceTask,
    higher_lo: Sequence[ReferenceTask],
    higher_hi: Sequence[ReferenceTask],
    *,
    start: int,
    max_iter: int = 100000,
) -> dict[str, Any]:
    r = task.c_lo
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        il_terms = {
            j.name: (
                ceil_div_nonnegative(r, j.period) * j.c_hi
                + (start // j.period + 1) * (j.c_lo - j.c_hi)
            )
            for j in higher_lo
        }
        ih_terms = {
            j.name: (
                ceil_div_nonnegative(r, j.period) * j.c_lo
                + (0 if r <= start else ceil_div_nonnegative(r - start, j.period))
                * (j.c_hi - j.c_lo)
            )
            for j in higher_hi
        }
        raw_f = task.c_lo + sum(il_terms.values()) + sum(ih_terms.values())
        trace.append(
            {
                "iteration": iteration,
                "r": r,
                "start": start,
                "il_terms": il_terms,
                "ih_terms": ih_terms,
                "raw_f": raw_f,
                "f": raw_f,
            }
        )
        if raw_f <= r:
            return {
                "case": "CASE1",
                "start": start,
                "response_for_deadline": r,
                "absolute_response": r,
                "trace": trace,
                "status": "PASS" if r <= task.deadline else "FAIL",
            }
        r = raw_f
        if r > task.deadline:
            return {
                "case": "CASE1",
                "start": start,
                "response_for_deadline": r,
                "absolute_response": r,
                "trace": trace,
                "status": "FAIL",
            }
    return {"case": "CASE1", "start": start, "trace": trace, "status": "UNRESOLVED", "failure": "ITERATION_LIMIT"}


def _case2_candidate(
    task: ReferenceTask,
    higher_lo: Sequence[ReferenceTask],
    higher_hi: Sequence[ReferenceTask],
    *,
    start: int,
    max_iter: int = 100000,
) -> dict[str, Any]:
    absolute_r = start + task.c_hi
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        il_terms = {
            j.name: (
                ceil_div_nonnegative(absolute_r, j.period) * j.c_hi
                + (start // j.period + 1) * (j.c_lo - j.c_hi)
            )
            for j in higher_lo
        }
        ih_terms = {
            j.name: (
                ceil_div_nonnegative(absolute_r, j.period) * j.c_lo
                + max(0, ceil_div_nonnegative(absolute_r - start, j.period))
                * (j.c_hi - j.c_lo)
            )
            for j in higher_hi
        }
        raw_f = task.c_hi + sum(il_terms.values()) + sum(ih_terms.values())
        guarded_f = max(start + task.c_hi, raw_f)
        trace.append(
            {
                "iteration": iteration,
                "absolute_r": absolute_r,
                "start": start,
                "il_terms": il_terms,
                "ih_terms": ih_terms,
                "raw_f": raw_f,
                "guarded_f": guarded_f,
                "f": guarded_f,
                "physical_guard_holds": absolute_r >= start + task.c_hi,
                "post_fixed_holds": guarded_f <= absolute_r,
            }
        )

        if guarded_f <= absolute_r:
            relative = absolute_r - start
            return {
                "case": "CASE2",
                "start": start,
                "absolute_response": absolute_r,
                "relative_response": relative,
                "response_for_deadline": relative,
                "physical_guard_holds": absolute_r >= start + task.c_hi,
                "post_fixed_holds": True,
                "trace": trace,
                "status": "PASS" if relative <= task.deadline else "FAIL",
            }

        absolute_r = guarded_f
        if absolute_r - start > task.deadline:
            return {
                "case": "CASE2",
                "start": start,
                "absolute_response": absolute_r,
                "relative_response": absolute_r - start,
                "response_for_deadline": absolute_r - start,
                "physical_guard_holds": True,
                "post_fixed_holds": False,
                "trace": trace,
                "status": "FAIL",
            }

    return {"case": "CASE2", "start": start, "trace": trace, "status": "UNRESOLVED", "failure": "ITERATION_LIMIT"}


def analyze_reference_task(task: ReferenceTask, higher: Sequence[ReferenceTask]) -> dict[str, Any]:
    """Generate a full task certificate with LO, Case1 and Case2 witnesses."""

    lo = lo_postfixed(task, higher)
    if lo["status"] != "PASS":
        return {
            "status": lo["status"],
            "task": asdict(task),
            "lo": lo,
            "start": {
                "status": "NOT_APPLICABLE",
                "reason": "LO_RTA_NOT_PASS",
                "trace": [],
            },
            "case1": [],
            "case2": [],
            "zero_relative_start_boundary": {"applicable": False},
            "r_lo": int(lo.get("r_lo", 0)),
            "r_hi": 0,
            "r_star": int(lo.get("r_lo", 0)),
            "lo_deadline_holds": False,
            "hi_deadline_holds": False,
        }

    start_result = worst_case_start(task, higher)
    if start_result["status"] != "PASS":
        return {
            "status": start_result["status"],
            "task": asdict(task),
            "lo": lo,
            "start": start_result,
            "case1": [],
            "case2": [],
            "zero_relative_start_boundary": {"applicable": False},
            "r_lo": int(lo["r_lo"]),
            "r_hi": 0,
            "r_star": int(lo["r_lo"]),
            "lo_deadline_holds": int(lo["r_lo"]) <= task.deadline,
            "hi_deadline_holds": False,
        }

    higher_lo = [j for j in higher if j.criticality == "LO"]
    higher_hi = [j for j in higher if j.criticality == "HI"]
    case1 = [_case1_candidate(task, higher_lo, higher_hi, start=s) for s in range(lo["r_lo"])]
    if start_result["w_lo"] == 0:
        case2: list[dict[str, Any]] = []
        zero_boundary = {
            "applicable": True,
            "w_lo": 0,
            "bound": task.c_hi,
            "deadline_holds": task.c_hi <= task.deadline,
        }
    else:
        case2 = [_case2_candidate(task, higher_lo, higher_hi, start=s) for s in range(start_result["w_lo"])]
        zero_boundary = {"applicable": False}

    responses = [item["response_for_deadline"] for item in case1 + case2 if "response_for_deadline" in item]
    if zero_boundary["applicable"]:
        responses.append(int(zero_boundary["bound"]))

    r_hi = max(responses) if responses else 0
    r_star = max(int(lo["r_lo"]), r_hi)
    lo_deadline_holds = int(lo["r_lo"]) <= task.deadline
    hi_deadline_holds = r_hi <= task.deadline
    status = "PASS" if (
        lo["status"] == "PASS"
        and start_result["status"] == "PASS"
        and all(item["status"] == "PASS" for item in case1 + case2)
        and lo_deadline_holds
        and hi_deadline_holds
    ) else "FAIL"

    return {
        "status": status,
        "task": asdict(task),
        "lo": lo,
        "start": start_result,
        "case1": case1,
        "case2": case2,
        "zero_relative_start_boundary": zero_boundary,
        "r_lo": int(lo["r_lo"]),
        "r_hi": r_hi,
        "r_star": r_star,
        "lo_deadline_holds": lo_deadline_holds,
        "hi_deadline_holds": hi_deadline_holds,
    }




def _lo_only_task_analysis(task: ReferenceTask, lo: dict[str, Any]) -> dict[str, Any]:
    """Return a complete fail-closed row without evaluating switch-time domains.

    The all-task theorem is conjunctive.  Once any task's LO recurrence is not
    PASS, Case1/Case2 witnesses cannot authorize schedulability and need not be
    materialized for any task.
    """

    own_status = str(lo.get("status", "UNRESOLVED"))
    return {
        "status": own_status if own_status != "PASS" else "UNRESOLVED",
        "task": asdict(task),
        "lo": lo,
        "start": {
            "status": "NOT_APPLICABLE",
            "reason": (
                "LO_RTA_NOT_PASS"
                if own_status != "PASS"
                else "GLOBAL_LO_RTA_NOT_PASS"
            ),
            "trace": [],
        },
        "case1": [],
        "case2": [],
        "zero_relative_start_boundary": {"applicable": False},
        "r_lo": int(lo.get("r_lo", 0)),
        "r_hi": 0,
        "r_star": int(lo.get("r_lo", 0)),
        "lo_deadline_holds": bool(
            own_status == "PASS"
            and int(lo.get("r_lo", 0)) <= int(task.deadline)
        ),
        "hi_deadline_holds": False,
        "analysis_stage": "LO_ONLY",
    }


def _candidate_domains_complete(task_rows: Sequence[dict[str, Any]]) -> bool:
    """Check exact finite integer domains used by the imported formulas.

    Case 1 must enumerate every integer ``0 <= s < R_i(LO)``.  Case 2
    must enumerate every integer ``0 <= s < W_i(LO)``; when ``W_i(LO)=0``
    the Case-2 list is empty and the separately disclosed zero-start boundary
    record must be present.
    """
    for row in task_rows:
        if row.get("status") != "PASS":
            return False
        r_lo = int(row.get("r_lo", 0))
        case1_starts = [int(item.get("start", -1)) for item in row.get("case1", [])]
        if case1_starts != list(range(r_lo)):
            return False
        start_info = row.get("start", row.get("worst_case_start", {}))
        if start_info.get("status") != "PASS":
            return False
        w_lo = int(start_info.get("w_lo", -1))
        case2_starts = [int(item.get("start", -1)) for item in row.get("case2", [])]
        zero_boundary = row.get("zero_relative_start_boundary", {})
        if w_lo == 0:
            if case2_starts or zero_boundary.get("applicable") is not True:
                return False
        else:
            if case2_starts != list(range(w_lo)) or zero_boundary.get("applicable") is not False:
                return False
    return True

def compute_all_task_rta(taskset: ReferenceTaskset) -> dict[str, Any]:
    """Pure all-task arithmetic; no route or obligation identity is attached."""
    lo_results = [
        lo_postfixed(task, taskset.tasks[:index])
        for index, task in enumerate(taskset.tasks)
    ]

    if any(result.get("status") != "PASS" for result in lo_results):
        task_rows = [
            _task_row_payload(task, _lo_only_task_analysis(task, lo_results[index]))
            for index, task in enumerate(taskset.tasks)
        ]
    else:
        task_rows = [
            _task_row_payload(task, analyze_reference_task(task, taskset.tasks[:index]))
            for index, task in enumerate(taskset.tasks)
        ]

    if len(task_rows) != len(taskset.tasks):
        status, route = "UNRESOLVED", "UNRESOLVED"
    elif any(
        row["lo"].get("status") == "UNRESOLVED"
        or row["worst_case_start"].get("status") == "UNRESOLVED"
        or any(item.get("status") == "UNRESOLVED" for item in row["case1"] + row["case2"])
        for row in task_rows
    ):
        status, route = "UNRESOLVED", "UNRESOLVED"
    elif all(
        row["lo"].get("status") == "PASS"
        and row["worst_case_start"].get("status") == "PASS"
        and all(item.get("status") == "PASS" for item in row["case1"] + row["case2"])
        and row["lo_deadline_holds"]
        and row["hi_deadline_holds"]
        for row in task_rows
    ):
        status, route = "PASS", "PASS"
    else:
        status, route = "FAIL", "REFERENCE_CERTIFICATE_FAILED"

    taskset_dict = taskset.to_dict()
    witness = {
        "schema_version": ALL_TASK_RTA_SCHEMA_VERSION,
        "reference_taskset_fingerprint": taskset_dict["fingerprint"],
        "task_order": [task.name for task in taskset.tasks],
        "task_count_expected": len(taskset.tasks),
        "task_count_analyzed": len(task_rows),
        "all_tasks_covered": len(task_rows) == len(taskset.tasks),
        "all_lo_deadlines_hold": all(row["lo_deadline_holds"] for row in task_rows),
        "all_hi_deadlines_hold": all(row["hi_deadline_holds"] for row in task_rows),
        "all_deadlines_met": (
            all(row["lo_deadline_holds"] for row in task_rows)
            and all(row["hi_deadline_holds"] for row in task_rows)
        ),
        "complete_integer_candidate_domains": _candidate_domains_complete(task_rows),
        "tasks": task_rows,
    }
    return {
        "schema_version": ALL_TASK_RTA_SCHEMA_VERSION,
        "status": status,
        "route": route,
        "taskset": taskset_dict,
        "task_count_expected": len(taskset.tasks),
        "task_count_analyzed": len(task_rows),
        "task_order": witness["task_order"],
        "all_tasks_covered": witness["all_tasks_covered"],
        "all_lo_deadlines_hold": witness["all_lo_deadlines_hold"],
        "all_hi_deadlines_hold": witness["all_hi_deadlines_hold"],
        "all_deadlines_met": witness["all_deadlines_met"],
        "complete_integer_candidate_domains": witness["complete_integer_candidate_domains"],
        "tasks": task_rows,
        "witness": witness,
    }
def build_all_task_rta_certificate(taskset: ReferenceTaskset, *, obligation_id: str,
                                   route_id: str, certificate_context_hash: str | None = None,
                                   checker_version: str = ALL_TASK_RTA_SCHEMA_VERSION) -> dict[str, Any]:
    result = compute_all_task_rta(taskset)
    status = result["status"]
    result["route_id"] = route_id
    result.update(obligation_certificate(
        obligation_id=obligation_id, status=status,
        context_hash=certificate_context_hash or taskset.source_context_hash or "",
        inputs={"taskset_fingerprint": result["taskset"]["fingerprint"],
                "priority_order": list(taskset.priority_order),
                "task_count_expected": len(taskset.tasks), "route_id": route_id},
        witness=result["witness"], checker_id=__name__, checker_version=checker_version,
        failure=None if status == "PASS" else {
            "route": result.get("route", "UNRESOLVED"),
            "code": "ALL_TASK_RTA_INCOMPLETE" if status == "UNRESOLVED" else "ALL_TASK_SUFFICIENT_TEST_FAILED",
        },
    ))
    return result


def all_task_reference_rta(taskset: ReferenceTaskset, *, certificate_context_hash: str | None = None) -> dict[str, Any]:
    return build_all_task_rta_certificate(
        taskset, obligation_id=ALL_TASK_REFERENCE_RTA_ARITHMETIC,
        route_id="strict_full", certificate_context_hash=certificate_context_hash,
        checker_version=ALL_TASK_RTA_SCHEMA_VERSION)


def all_task_protected_prefix_rta(taskset: ReferenceTaskset, *, certificate_context_hash: str | None = None) -> dict[str, Any]:
    return build_all_task_rta_certificate(
        taskset, obligation_id=PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC,
        route_id="protected_prefix", certificate_context_hash=certificate_context_hash,
        checker_version=ALL_TASK_RTA_SCHEMA_VERSION)


def protected_hi_rta(taskset: ReferenceTaskset) -> dict[str, Any]:
    raise RuntimeError(
        "protected_hi_rta is legacy-only and cannot authorize DEPLOYED_HI_SAFETY. "
        "Use all_task_reference_rta() instead."
    )
