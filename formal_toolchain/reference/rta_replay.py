"""Independent all-task replay for the reference RTA certificate."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from formal_toolchain.core.artifact import verify_obligation_certificate

from .task_mapping import ReferenceTask, ReferenceTaskset


def ceil_div_nonnegative(value: int, divisor: int) -> int:
    return 0 if value <= 0 else (value + divisor - 1) // divisor


def floor_div_nonnegative(value: int, divisor: int) -> int:
    return 0 if value < 0 else value // divisor


def _replay_lo_postfixed(task: ReferenceTask, higher: tuple[ReferenceTask, ...], *, max_iter: int = 100000) -> dict[str, Any]:
    current = task.c_lo
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        counts = {j.name: ceil_div_nonnegative(current, j.period) for j in higher}
        nxt = task.c_lo + sum(counts[j.name] * j.c_lo for j in higher)
        trace.append({"iteration": iteration, "r": current, "release_counts": counts, "interference": nxt - task.c_lo, "f": nxt})
        if nxt <= current:
            return {"status": "PASS" if current <= task.deadline else "FAIL", "r_lo": current, "trace": trace, "post_fixed": True}
        current = nxt
        if current > task.deadline:
            return {"status": "FAIL", "r_lo": current, "trace": trace, "post_fixed": False}
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def _replay_start_time(higher: tuple[ReferenceTask, ...], *, max_iter: int = 100000) -> dict[str, Any]:
    current = 0
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        counts = {j.name: floor_div_nonnegative(current, j.period) + 1 for j in higher}
        nxt = sum(counts[j.name] * j.c_lo for j in higher)
        trace.append({"iteration": iteration, "w": current, "release_counts": counts, "next": nxt})
        if nxt <= current:
            return {"status": "PASS", "w_lo": current, "trace": trace}
        current = nxt
    return {"status": "UNRESOLVED", "trace": trace, "failure": "ITERATION_LIMIT"}


def _replay_case1_candidate(task: ReferenceTask, higher_lo: tuple[ReferenceTask, ...], higher_hi: tuple[ReferenceTask, ...], *, start: int, max_iter: int = 100000) -> dict[str, Any]:
    current = task.c_lo
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iter):
        il_terms = {
            j.name: (
                ceil_div_nonnegative(current, j.period) * j.c_hi
                + (start // j.period + 1) * (j.c_lo - j.c_hi)
            )
            for j in higher_lo
        }
        ih_terms = {
            j.name: (
                ceil_div_nonnegative(current, j.period) * j.c_lo
                + (0 if current <= start else ceil_div_nonnegative(current - start, j.period))
                * (j.c_hi - j.c_lo)
            )
            for j in higher_hi
        }
        raw_f = task.c_lo + sum(il_terms.values()) + sum(ih_terms.values())
        trace.append({"iteration": iteration, "r": current, "start": start, "il_terms": il_terms, "ih_terms": ih_terms, "raw_f": raw_f, "f": raw_f})
        if raw_f <= current:
            return {
                "case": "CASE1",
                "start": start,
                "response_for_deadline": current,
                "absolute_response": current,
                "trace": trace,
                "status": "PASS" if current <= task.deadline else "FAIL",
            }
        current = raw_f
        if current > task.deadline:
            return {
                "case": "CASE1",
                "start": start,
                "response_for_deadline": current,
                "absolute_response": current,
                "trace": trace,
                "status": "FAIL",
            }
    return {"case": "CASE1", "start": start, "trace": trace, "status": "UNRESOLVED", "failure": "ITERATION_LIMIT"}


def _replay_case2_candidate(task: ReferenceTask, higher_lo: tuple[ReferenceTask, ...], higher_hi: tuple[ReferenceTask, ...], *, start: int, max_iter: int = 100000) -> dict[str, Any]:
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


def _compare_task_witness(candidate: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    comparisons: list[dict[str, Any]] = []
    task = candidate.get("task", {})
    replay_task = replay.get("task", {})
    for field in ("name", "period", "deadline", "c_lo", "c_hi", "criticality", "priority_index", "code_c_lo", "code_c_hi", "degraded_cost", "offset"):
        comparisons.append({
            "field": f"task.{field}",
            "expected": task.get(field),
            "actual": replay_task.get(field),
            "match": task.get(field) == replay_task.get(field),
        })

    for field in ("lo", "start", "case1", "case2", "zero_relative_start_boundary", "r_lo", "r_hi", "r_star", "lo_deadline_holds", "hi_deadline_holds"):
        comparisons.append({
            "field": field,
            "expected": _normalize(candidate.get(field)),
            "actual": _normalize(replay.get(field)),
            "match": _normalize(candidate.get(field)) == _normalize(replay.get(field)),
        })
    if "case1" in candidate:
        comparisons.append({
            "field": "case1_domain",
            "expected": [item.get("start") for item in candidate.get("case1", [])],
            "actual": [item.get("start") for item in replay.get("case1", [])],
            "match": [item.get("start") for item in candidate.get("case1", [])]
            == [item.get("start") for item in replay.get("case1", [])],
        })
    if "case2" in candidate:
        comparisons.append({
            "field": "case2_domain",
            "expected": [item.get("start") for item in candidate.get("case2", [])],
            "actual": [item.get("start") for item in replay.get("case2", [])],
            "match": [item.get("start") for item in candidate.get("case2", [])]
            == [item.get("start") for item in replay.get("case2", [])],
        })
    if candidate.get("zero_relative_start_boundary", {}).get("applicable"):
        comparisons.append({
            "field": "zero_relative_start_boundary",
            "expected": candidate.get("zero_relative_start_boundary"),
            "actual": replay.get("zero_relative_start_boundary"),
            "match": candidate.get("zero_relative_start_boundary") == replay.get("zero_relative_start_boundary"),
        })
    return {
        "status": "PASS" if all(item["match"] for item in comparisons) else "FAIL",
        "comparisons": comparisons,
    }


def _replay_task_independently(task: ReferenceTask, higher: tuple[ReferenceTask, ...]) -> dict[str, Any]:
    lo = _replay_lo_postfixed(task, higher)
    start = _replay_start_time(higher)
    if lo.get("status") != "PASS" or start.get("status") != "PASS":
        return {
            "status": lo.get("status") if lo.get("status") != "PASS" else start.get("status"),
            "task": asdict(task),
            "lo": lo,
            "start": start,
            "case1": [],
            "case2": [],
            "zero_relative_start_boundary": {"applicable": False},
            "r_lo": int(lo.get("r_lo", 0)),
            "r_hi": 0,
            "r_star": int(lo.get("r_lo", 0)),
            "lo_deadline_holds": bool(lo.get("status") == "PASS" and int(lo.get("r_lo", 0)) <= task.deadline),
            "hi_deadline_holds": False,
        }

    higher_lo = tuple(item for item in higher if item.criticality == "LO")
    higher_hi = tuple(item for item in higher if item.criticality == "HI")
    case1 = [_replay_case1_candidate(task, higher_lo, higher_hi, start=s) for s in range(lo["r_lo"])]
    if start["w_lo"] == 0:
        case2: list[dict[str, Any]] = []
        zero_boundary: dict[str, Any] = {
            "applicable": True,
            "w_lo": 0,
            "bound": task.c_hi,
            "deadline_holds": task.c_hi <= task.deadline,
        }
    else:
        case2 = [_replay_case2_candidate(task, higher_lo, higher_hi, start=s) for s in range(start["w_lo"])]
        zero_boundary = {"applicable": False}

    responses = [item["response_for_deadline"] for item in case1 + case2 if "response_for_deadline" in item]
    if zero_boundary["applicable"]:
        responses.append(int(zero_boundary["bound"]))
    r_hi = max(responses) if responses else 0
    return {
        "status": "PASS" if (
            lo["status"] == "PASS"
            and start["status"] == "PASS"
            and all(item["status"] == "PASS" for item in case1 + case2)
            and int(lo["r_lo"]) <= task.deadline
            and r_hi <= task.deadline
        ) else "FAIL",
        "task": asdict(task),
        "lo": lo,
        "start": start,
        "case1": case1,
        "case2": case2,
        "zero_relative_start_boundary": zero_boundary,
        "r_lo": int(lo["r_lo"]),
        "r_hi": r_hi,
        "r_star": max(int(lo["r_lo"]), r_hi),
        "lo_deadline_holds": int(lo["r_lo"]) <= task.deadline,
        "hi_deadline_holds": r_hi <= task.deadline,
    }


def replay_all_task_rta_independently(taskset: ReferenceTaskset) -> dict[str, Any]:
    replay_rows = [
        _replay_task_independently(task, taskset.tasks[:index])
        for index, task in enumerate(taskset.tasks)
    ]
    witness = {
        "schema_version": "all_task_rta_v3",
        "reference_taskset_fingerprint": taskset.to_dict().get("fingerprint"),
        "task_order": [task.name for task in taskset.tasks],
        "task_count_expected": len(taskset.tasks),
        "task_count_analyzed": len(replay_rows),
        "all_tasks_covered": len(replay_rows) == len(taskset.tasks),
        "all_lo_deadlines_hold": all(row["lo_deadline_holds"] for row in replay_rows),
        "all_hi_deadlines_hold": all(row["hi_deadline_holds"] for row in replay_rows),
        "all_deadlines_met": (
            all(row["lo_deadline_holds"] for row in replay_rows)
            and all(row["hi_deadline_holds"] for row in replay_rows)
        ),
        "tasks": replay_rows,
    }
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in replay_rows) else "FAIL",
        "schema_version": "all_task_rta_v3",
        "task_order": witness["task_order"],
        "task_count_expected": witness["task_count_expected"],
        "task_count_analyzed": witness["task_count_analyzed"],
        "all_tasks_covered": witness["all_tasks_covered"],
        "all_lo_deadlines_hold": witness["all_lo_deadlines_hold"],
        "all_hi_deadlines_hold": witness["all_hi_deadlines_hold"],
        "all_deadlines_met": witness["all_deadlines_met"],
        "tasks": replay_rows,
        "witness": witness,
    }


def replay_all_task_rta(taskset: ReferenceTaskset, production: Mapping[str, Any]) -> dict[str, Any]:
    if production.get("schema_version") not in {"all_task_reference_rta_v2", "all_task_rta_v3"}:
        return {"status": "FAIL", "code": "RTA_SCHEMA_MISMATCH"}
    if production.get("task_count_analyzed") != len(taskset.tasks):
        return {"status": "FAIL", "code": "RTA_TASK_COVERAGE_INCOMPLETE"}
    expected_task_names = [task.name for task in taskset.tasks]
    candidate_witness = production.get("witness", production)
    candidate_rows = candidate_witness.get("tasks")
    if not isinstance(candidate_rows, list):
        return {"status": "FAIL", "code": "RTA_TASK_ROWS_MISSING"}
    actual_task_names = [row.get("task", {}).get("name") for row in candidate_rows]
    if actual_task_names != expected_task_names:
        return {"status": "FAIL", "code": "ALL_TASK_RTA_TASK_ORDER_MISMATCH"}
    if len(set(actual_task_names)) != len(actual_task_names):
        return {"status": "FAIL", "code": "ALL_TASK_RTA_DUPLICATE_TASK"}
    expected_fingerprint = taskset.to_dict().get("fingerprint")
    actual_fingerprint = candidate_witness.get("reference_taskset_fingerprint")
    if actual_fingerprint is None:
        candidate_inputs = production.get("inputs", {})
        actual_fingerprint = (
            candidate_inputs.get("reference_taskset_fingerprint")
            if candidate_inputs.get("reference_taskset_fingerprint") is not None
            else candidate_inputs.get("taskset_fingerprint")
        )
    if actual_fingerprint != expected_fingerprint:
        return {"status": "FAIL", "code": "ALL_TASK_RTA_TASKSET_FINGERPRINT_MISMATCH"}
    if production.get("task_count") not in {None, len(taskset.tasks)} and production.get("task_count_expected") != len(taskset.tasks):
        return {"status": "FAIL", "code": "ALL_TASK_RTA_TASK_COUNT_MISMATCH"}
    if candidate_witness.get("all_tasks_covered") is not True and production.get("task_count_expected") != len(taskset.tasks):
        return {"status": "FAIL", "code": "ALL_TASK_RTA_COVERAGE_NOT_ESTABLISHED"}

    replay = replay_all_task_rta_independently(taskset)
    replay_rows = replay["tasks"]
    if len(candidate_rows) != len(replay_rows):
        return {"status": "FAIL", "code": "RTA_TASK_COVERAGE_INCOMPLETE"}

    comparisons = [
        _compare_task_witness(candidate, replay_row)
        for candidate, replay_row in zip(candidate_rows, replay_rows, strict=True)
    ]
    ok = all(item["status"] == "PASS" for item in comparisons)
    expected_all_met = candidate_witness.get("all_deadlines_met")
    if expected_all_met is None:
        expected_all_met = bool(candidate_witness.get("all_lo_deadlines_hold")) and bool(candidate_witness.get("all_hi_deadlines_hold"))
    if expected_all_met != replay["all_deadlines_met"]:
        return {"status": "FAIL", "code": "ALL_TASK_RTA_SUMMARY_MISMATCH"}
    return {
        "status": "PASS" if ok else "FAIL",
        "code": None if ok else "ALL_TASK_RTA_REPLAY_MISMATCH",
        "replay_rows": replay_rows,
        "comparisons": comparisons,
        "witness": replay["witness"],
    }


def replay_rta(taskset: ReferenceTaskset, production: Mapping[str, Any]) -> dict[str, Any]:
    return replay_all_task_rta(taskset, production)
