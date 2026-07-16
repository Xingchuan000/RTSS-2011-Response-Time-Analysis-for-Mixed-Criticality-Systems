"""Phase J02-J05：Protected-HI C-AMC-sem 的整数 production checker。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from .arithmetic import ceil_div_nonnegative, floor_div_nonnegative
from .task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.core.artifact import obligation_certificate


def _lo_interference(task: ReferenceTask, higher: Sequence[ReferenceTask], r: int) -> int:
    return sum(ceil_div_nonnegative(r, j.period) * j.c_lo for j in higher)


def lo_postfixed(task: ReferenceTask, higher: Sequence[ReferenceTask], *, max_iter: int = 100000) -> dict[str, Any]:
    """计算 LO recurrence 的 inflationary least post-fixed witness。"""
    r = task.c_lo
    trace = []
    for iteration in range(max_iter):
        counts = {j.name: ceil_div_nonnegative(r, j.period) for j in higher}
        f = task.c_lo + sum(counts[j.name] * j.c_lo for j in higher)
        trace.append({"iteration": iteration, "r": r, "release_counts": counts,
                      "interference": f - task.c_lo, "f": f})
        if f <= r:
            return {"status": "PASS" if r <= task.deadline else "FAIL",
                    "r_lo": r, "trace": trace, "post_fixed": f <= r}
        r = f
        if r > task.deadline:
            return {"status": "FAIL", "r_lo": r, "trace": trace, "post_fixed": False}
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def worst_case_start(task: ReferenceTask, higher: Sequence[ReferenceTask], *, max_iter: int = 100000) -> dict[str, Any]:
    """计算 W_i(LO)，空 hp 集必须得到 0。"""
    w = 0
    trace = []
    for iteration in range(max_iter):
        counts = {j.name: floor_div_nonnegative(w, j.period) + 1 for j in higher}
        nxt = sum(counts[j.name] * j.c_lo for j in higher)
        trace.append({"iteration": iteration, "w": w, "release_counts": counts, "next": nxt})
        if nxt <= w:
            return {"status": "PASS", "w_lo": w, "trace": trace}
        w = nxt
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def _case1_candidate(task: ReferenceTask, higher_lo: Sequence[ReferenceTask],
                     higher_hi: Sequence[ReferenceTask], *, start: int) -> dict[str, Any]:
    """Case 1：按计划原式计算 R，deadline 比较直接使用 R。"""
    r = task.c_lo
    trace = []
    for iteration in range(100000):
        il_terms = {j.name: (ceil_div_nonnegative(r, j.period) * j.c_hi +
                    (start // j.period + 1) * (j.c_lo - j.c_hi)) for j in higher_lo}
        ih_terms = {j.name: (ceil_div_nonnegative(r, j.period) * j.c_lo +
                    (0 if r <= start else ceil_div_nonnegative(r - start, j.period))
                    * (j.c_hi - j.c_lo)) for j in higher_hi}
        nxt = task.c_lo + sum(il_terms.values()) + sum(ih_terms.values())
        trace.append({"iteration": iteration, "r": r, "start": start,
                      "il_terms": il_terms, "ih_terms": ih_terms, "f": nxt})
        if nxt <= r:
            return {"case": "CASE1", "start": start,
                    "response_for_deadline": r, "absolute_response": r,
                    "trace": trace, "status": "PASS" if r <= task.deadline else "FAIL"}
        r = nxt
        if r > task.deadline:
            return {"case": "CASE1", "start": start,
                    "response_for_deadline": r, "absolute_response": r,
                    "trace": trace, "status": "FAIL"}
    return {"case": "CASE1", "start": start, "trace": trace, "status": "UNRESOLVED"}


def _case2_candidate(task: ReferenceTask, higher_lo: Sequence[ReferenceTask],
                     higher_hi: Sequence[ReferenceTask], *, start: int) -> dict[str, Any]:
    """Case 2：初值至少为 s+C_i(HI)，deadline 比较使用 R-s。"""
    r = start + task.c_hi
    trace = []
    for iteration in range(100000):
        il_terms = {j.name: (ceil_div_nonnegative(r, j.period) * j.c_hi +
                    (start // j.period + 1) * (j.c_lo - j.c_hi)) for j in higher_lo}
        ih_terms = {j.name: (ceil_div_nonnegative(r, j.period) * j.c_lo +
                    max(0, ceil_div_nonnegative(r - start, j.period)) * (j.c_hi - j.c_lo)) for j in higher_hi}
        nxt = task.c_hi + sum(il_terms.values()) + sum(ih_terms.values())
        trace.append({"iteration": iteration, "r": r, "start": start,
                      "il_terms": il_terms, "ih_terms": ih_terms, "f": nxt})
        if nxt <= r:
            relative = r - start
            return {"case": "CASE2", "start": start,
                    "response_for_deadline": relative, "absolute_response": r,
                    "relative_response": relative, "trace": trace,
                    "status": "PASS" if relative <= task.deadline else "FAIL"}
        r = nxt
        if r - start > task.deadline:
            return {"case": "CASE2", "start": start,
                    "response_for_deadline": r - start, "absolute_response": r,
                    "relative_response": r - start, "trace": trace, "status": "FAIL"}
    return {"case": "CASE2", "start": start, "trace": trace, "status": "UNRESOLVED"}


def analyze_hi_task(task: ReferenceTask, higher: Sequence[ReferenceTask]) -> dict[str, Any]:
    """生成一个 HI task 的完整 Case 1/Case 2 witness。"""
    if task.criticality != "HI":
        raise ValueError("Protected-HI RTA 只分析 HI task")
    higher_lo = [j for j in higher if j.criticality == "LO"]
    higher_hi = [j for j in higher if j.criticality == "HI"]
    lo = lo_postfixed(task, higher)
    if lo["status"] != "PASS":
        return {"status": lo["status"], "task": asdict(task), "lo": lo}
    start = worst_case_start(task, higher)
    if start["status"] != "PASS":
        return {"status": start["status"], "task": asdict(task), "lo": lo, "start": start}
    case1 = [_case1_candidate(task, higher_lo, higher_hi, start=s)
             for s in range(lo["r_lo"])]
    if start["w_lo"] == 0:
        case2 = [{"case": "CASE2", "tag": "ZERO_RELATIVE_START", "start": 0,
                  "response_for_deadline": task.c_hi, "relative_response": task.c_hi,
                  "absolute_response": task.c_hi,
                  "status": "PASS" if task.c_hi <= task.deadline else "FAIL"}]
    else:
        case2 = [_case2_candidate(task, higher_lo, higher_hi, start=s)
                 for s in range(start["w_lo"])]
    all_w = case1 + case2
    status = "PASS" if all(row["status"] == "PASS" for row in all_w) else "FAIL"
    response_values = [row.get("response_for_deadline", 0) for row in all_w]
    return {"status": status, "task": asdict(task), "lo": lo, "start": start,
            "case1": case1, "case2": case2,
            "r_hi": max(response_values), "r_star": max([lo["r_lo"]] + response_values)}


def protected_hi_rta(taskset: ReferenceTaskset) -> dict[str, Any]:
    """对全部 HI task 生成逐 task certificate；失败只归为参考证书失败。"""
    results = []
    for index, task in enumerate(taskset.tasks):
        if task.criticality == "HI":
            results.append(analyze_hi_task(task, taskset.tasks[:index]))
    if any(row["status"] == "UNRESOLVED" for row in results):
        status = "UNRESOLVED"
        route = "UNRESOLVED"
    else:
        status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
        route = "PASS" if status == "PASS" else "REFERENCE_CERTIFICATE_FAILED"
    result = {"schema_version": "protected_hi_rta_v1", "status": status,
              "route": route, "taskset": taskset.to_dict(), "tasks": results}
    obligation_status = status if status in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED"
    failure = None if obligation_status == "PASS" else {
        "route": route if route != "PASS" else "UNRESOLVED",
        "code": "RTA_WITNESS_INCOMPLETE" if obligation_status == "UNRESOLVED" else "RTA_DEADLINE_FAILED",
        "message": "Protected-HI RTA 未形成完整 deadline 内证书",
        "machine_details": {"task_count": len(results)},
    }
    result.update(obligation_certificate(
        obligation_id="PROTECTED_HI_RTA_ARITHMETIC", status=obligation_status,
        context_hash=taskset.source_context_hash or "",
        inputs={"taskset_fingerprint": taskset.to_dict()["fingerprint"]},
        witness={"tasks": results}, checker_id="formal_toolchain.reference.rta_production",
        checker_version="phase-j-v1", failure=failure,
    ))
    return result
