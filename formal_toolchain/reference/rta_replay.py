"""Phase J06：不复用 production recurrence 的独立 exhaustive replay。"""

from __future__ import annotations

from typing import Any

from .task_mapping import ReferenceTaskset
from formal_toolchain.core.artifact import verify_obligation_certificate


def ceil_div_nonnegative(value: int, divisor: int) -> int:
    return 0 if value <= 0 else (value + divisor - 1) // divisor


def floor_div_nonnegative(value: int, divisor: int) -> int:
    return 0 if value < 0 else value // divisor


def _fixed_point(task_cost: int, higher: tuple[Any, ...], *, hi_mode: bool = False) -> tuple[int, list[dict[str, Any]], str]:
    """独立计算一个 LO post-fixed witness，不复用 production 的迭代器。"""
    current = task_cost
    trace: list[dict[str, Any]] = []
    for iteration in range(100000):
        counts = {task.name: ceil_div_nonnegative(current, task.period) for task in higher}
        next_value = task_cost + sum(
            counts[task.name] * (task.c_hi if hi_mode else task.c_lo) for task in higher
        )
        trace.append({"iteration": iteration, "r": current, "release_counts": counts,
                      "f": next_value})
        if next_value <= current:
            return current, trace, "PASS"
        current = next_value
    return current, trace, "UNRESOLVED"


def _start_time(higher: tuple[Any, ...]) -> tuple[int, list[dict[str, Any]], str]:
    """独立计算 W_i(LO)，并保留每次 release-count 计算。"""
    current = 0
    trace: list[dict[str, Any]] = []
    for iteration in range(100000):
        counts = {task.name: floor_div_nonnegative(current, task.period) + 1 for task in higher}
        next_value = sum(counts[task.name] * task.c_lo for task in higher)
        trace.append({"iteration": iteration, "w": current,
                      "release_counts": counts, "next": next_value})
        if next_value <= current:
            return current, trace, "PASS"
        current = next_value
    return current, trace, "UNRESOLVED"


def _case1_candidate(task: Any, higher_lo: tuple[Any, ...], higher_hi: tuple[Any, ...], *, start: int) -> dict[str, Any]:
    """独立计算计划原式 Case 1，并直接比较 R。"""
    current = task.c_lo
    trace: list[dict[str, Any]] = []
    for iteration in range(100000):
        lo_terms = {
            higher.name: (
                ceil_div_nonnegative(current, higher.period) * higher.c_hi
                + (start // higher.period + 1) * (higher.c_lo - higher.c_hi)
            )
            for higher in higher_lo
        }
        hi_terms = {higher.name: (ceil_div_nonnegative(current, higher.period) * higher.c_lo
                    + (0 if current <= start else ceil_div_nonnegative(current - start, higher.period))
                    * (higher.c_hi - higher.c_lo)) for higher in higher_hi}
        next_value = task.c_lo + sum(lo_terms.values()) + sum(hi_terms.values())
        trace.append({"iteration": iteration, "r": current, "start": start,
                      "il_terms": lo_terms, "ih_terms": hi_terms, "f": next_value})
        if next_value <= current:
            absolute = current
            return {"case": "CASE1", "start": start,
                    "response_for_deadline": absolute, "absolute_response": absolute,
                    "trace": trace, "status": "PASS" if current <= task.deadline else "FAIL"}
        current = next_value
        if current > task.deadline:
            return {"case": "CASE1", "start": start,
                    "response_for_deadline": current, "absolute_response": current,
                    "trace": trace, "status": "FAIL"}
    return {"case": "CASE1", "start": start, "trace": trace, "status": "UNRESOLVED"}


def _case2_candidate(task: Any, higher_lo: tuple[Any, ...], higher_hi: tuple[Any, ...], *, start: int) -> dict[str, Any]:
    """独立计算 Case 2：初值 s+C_HI 并比较相对响应。"""
    current = start + task.c_hi
    trace: list[dict[str, Any]] = []
    for iteration in range(100000):
        lo_terms = {higher.name: (ceil_div_nonnegative(current, higher.period) * higher.c_hi +
                    (start // higher.period + 1) * (higher.c_lo - higher.c_hi)) for higher in higher_lo}
        hi_terms = {higher.name: (ceil_div_nonnegative(current, higher.period) * higher.c_lo +
                    ceil_div_nonnegative(current - start, higher.period) * (higher.c_hi - higher.c_lo))
                    for higher in higher_hi}
        next_value = task.c_hi + sum(lo_terms.values()) + sum(hi_terms.values())
        trace.append({"iteration": iteration, "r": current, "start": start,
                      "il_terms": lo_terms, "ih_terms": hi_terms, "f": next_value})
        if next_value <= current:
            relative = current - start
            return {"case": "CASE2", "start": start, "response_for_deadline": relative,
                    "absolute_response": current, "relative_response": relative,
                    "trace": trace, "status": "PASS" if relative <= task.deadline else "FAIL"}
        current = next_value
        if current - start > task.deadline:
            return {"case": "CASE2", "start": start, "response_for_deadline": current - start,
                    "absolute_response": current, "relative_response": current - start,
                    "trace": trace, "status": "FAIL"}
    return {"case": "CASE2", "start": start, "trace": trace, "status": "UNRESOLVED"}


def _same_witness(actual: Any, expected: Any) -> bool:
    """递归比较 production 与 replay 的证据字段，禁止只比较最终数值。"""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _same_witness(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_witness(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def replay_rta(taskset: ReferenceTaskset, production: dict[str, Any] | None = None) -> dict[str, Any]:
    """独立重算并逐行检查 production 的 LO、W、Case 1、Case 2 witness。"""
    if production is None:
        return {"schema_version": "rta_replay_v1", "status": "UNRESOLVED",
                "failure": "PRODUCTION_TRACE_REQUIRED"}
    checks = []
    production_rows = {row.get("task", {}).get("name"): row for row in production.get("tasks", [])}
    if production.get("schema_version") != "protected_hi_rta_v1":
        return {"schema_version": "rta_replay_v1", "status": "UNRESOLVED",
                "failure": "PRODUCTION_SCHEMA_MISMATCH"}
    if not verify_obligation_certificate(production):
        return {"schema_version": "rta_replay_v1", "status": "FAIL",
                "failure": "PRODUCTION_CERTIFICATE_HASH_MISMATCH"}
    if (production.get("obligation_id") != "PROTECTED_HI_RTA_ARITHMETIC"
            or production.get("obligation_status") != production.get("status")):
        return {"schema_version": "rta_replay_v1", "status": "FAIL",
                "failure": "PRODUCTION_CERTIFICATE_CONTEXT_MISMATCH"}
    if production.get("taskset") != taskset.to_dict():
        return {"schema_version": "rta_replay_v1", "status": "FAIL",
                "failure": "PRODUCTION_TASKSET_CONTEXT_MISMATCH"}
    for index, task in enumerate(taskset.tasks):
        if task.criticality != "HI":
            continue
        higher = tuple(taskset.tasks[:index])
        row = production_rows.get(task.name)
        if row is None:
            return {"schema_version": "rta_replay_v1", "status": "FAIL",
                    "failure": "PRODUCTION_TASK_MISSING", "task": task.name}
        r_lo, lo_trace, lo_status = _fixed_point(task.c_lo, higher)
        w_lo, start_trace, start_status = _start_time(higher)
        if lo_status != "PASS" or start_status != "PASS":
            return {"schema_version": "rta_replay_v1", "status": lo_status if lo_status != "PASS" else start_status,
                    "failure": "INDEPENDENT_WITNESS_UNRESOLVED", "task": task.name}
        lo_expected = {"status": "PASS" if r_lo <= task.deadline else "FAIL",
                       "r_lo": r_lo,
                       "trace": [dict(item, interference=item["f"] - task.c_lo)
                                 for item in lo_trace],
                       "post_fixed": True}
        start_expected = {"status": "PASS", "w_lo": w_lo, "trace": start_trace}
        lo_ok = _same_witness(row.get("lo"), lo_expected)
        start_ok = _same_witness(row.get("start"), start_expected)
        higher_lo = tuple(item for item in higher if item.criticality == "LO")
        higher_hi = tuple(item for item in higher if item.criticality == "HI")
        case1 = [_case1_candidate(task, higher_lo, higher_hi, start=s)
                 for s in range(r_lo)]
        if w_lo == 0:
            case2 = [{"case": "CASE2", "tag": "ZERO_RELATIVE_START", "start": 0,
                      "response_for_deadline": task.c_hi, "absolute_response": task.c_hi,
                      "relative_response": task.c_hi,
                      "status": "PASS" if task.c_hi <= task.deadline else "FAIL"}]
        else:
            case2 = [_case2_candidate(task, higher_lo, higher_hi, start=s)
                     for s in range(w_lo)]
        case1_ok = _same_witness(row.get("case1"), case1)
        case2_ok = _same_witness(row.get("case2"), case2)
        checks.append({"task": task.name, "lo_trace_match": lo_ok,
                       "start_trace_match": start_ok, "case1_trace_match": case1_ok,
                       "case2_trace_match": case2_ok,
                       "status": "PASS" if lo_ok and start_ok and case1_ok and case2_ok else "FAIL"})
    status = "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL"
    if production.get("status") != status:
        status = "FAIL"
    return {"schema_version": "rta_replay_v1", "status": status, "checks": checks,
            "production_status_match": production.get("status") == status}
