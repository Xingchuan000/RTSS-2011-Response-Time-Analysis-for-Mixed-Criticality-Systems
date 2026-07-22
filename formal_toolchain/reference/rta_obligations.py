"""Protected-HI RTA 的细粒度义务拆分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def _tasks(rta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = rta.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("RTA_TASKS_MISSING")
    return rows


def semantic_result(semantic_evidence: Mapping[str, Any], obligation_id: str) -> Mapping[str, Any]:
    value = semantic_evidence.get(obligation_id, {})
    return value if isinstance(value, Mapping) else {}


def build_lo_mode_rta_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    rows = _tasks(rta)
    witnesses = [{"task": row["task"]["name"], "lo": row.get("lo")} for row in rows]
    ok = bool(rows) and all(isinstance(row.get("lo"), Mapping) and row["lo"].get("status") == "PASS" and row["lo"].get("post_fixed") is True for row in rows)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "LO_MODE_RTA_NOT_PASS"}, "schema_version": "lo_mode_rta_obligation_v1", "witnesses": witnesses}


def build_worst_case_start_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    rows = _tasks(rta)
    starts = [{"task": row["task"]["name"], "start": row.get("start")} for row in rows]
    ok = bool(rows) and all(isinstance(row.get("start"), Mapping) and row["start"].get("status") == "PASS" and isinstance(row["start"].get("w_lo"), int) and row["start"]["w_lo"] >= 0 for row in rows)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "WORST_CASE_START_NOT_PASS"}, "schema_version": "worst_case_start_obligation_v1", "witnesses": starts}


def build_case1_domain_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for row in _tasks(rta):
        r_lo = int(row["lo"]["r_lo"])
        starts = [int(item["start"]) for item in row.get("case1", [])]
        expected = list(range(r_lo))
        checks.append({"task": row["task"]["name"], "r_lo": r_lo, "expected_starts": expected, "actual_starts": starts, "match": starts == expected})
    ok = bool(checks) and all(item["match"] for item in checks)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "CASE1_INTEGER_DOMAIN_MISMATCH"}, "checks": checks}


def build_case2_domain_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for row in _tasks(rta):
        start = row.get("start")
        if not isinstance(start, Mapping):
            checks.append({"task": row["task"]["name"], "w_lo": None, "expected": None,
                           "actual": None, "match": False})
            continue
        w_lo = int(start["w_lo"])
        case2 = row.get("case2", [])
        if w_lo == 0:
            zero_boundary = row.get("zero_relative_start_boundary", {})
            ok = (
                case2 == []
                and isinstance(zero_boundary, Mapping)
                and zero_boundary.get("applicable") is True
                and zero_boundary.get("bound") == int(row["task"]["c_hi"])
            )
            expected = {"case2": [], "zero_boundary": True}
            actual = {
                "case2": case2,
                "zero_boundary": zero_boundary,
            }
        else:
            expected = list(range(w_lo))
            actual = [int(item["start"]) for item in case2]
            ok = actual == expected
        checks.append({"task": row["task"]["name"], "w_lo": w_lo, "expected": expected, "actual": actual, "match": ok})
    passed = bool(checks) and all(item["match"] for item in checks)
    return {"status": "PASS" if passed else "FAIL", "route": None if passed else "REFERENCE_CERTIFICATE_FAILED", "failure": None if passed else {"code": "CASE2_INTEGER_DOMAIN_MISMATCH"}, "checks": checks}


def build_zero_relative_start_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for row in _tasks(rta):
        start = row.get("start")
        if not isinstance(start, Mapping):
            checks.append({"task": row["task"]["name"], "applicable": False, "status": "NOT_APPLICABLE"})
            continue
        w_lo = int(start["w_lo"])
        if w_lo != 0:
            checks.append({"task": row["task"]["name"], "applicable": False, "status": "NOT_APPLICABLE"})
            continue
        zero_boundary = row.get("zero_relative_start_boundary", {})
        task = row["task"]
        ok = (
            isinstance(zero_boundary, Mapping)
            and zero_boundary.get("applicable") is True
            and zero_boundary.get("bound") == int(task["c_hi"])
            and zero_boundary.get("deadline_holds") == (int(task["c_hi"]) <= int(task["deadline"]))
            and row.get("case2", []) == []
        )
        checks.append({"task": task["name"], "applicable": True, "status": "PASS" if ok else "FAIL", "zero_boundary": zero_boundary})
    applicable = [item for item in checks if item["applicable"]]
    passed = all(item["status"] == "PASS" for item in applicable)
    return {"status": "PASS" if passed else "FAIL", "route": None if passed else "REFERENCE_CERTIFICATE_FAILED", "failure": None if passed else {"code": "ZERO_RELATIVE_START_LEMMA_INSTANCE_FAILED"}, "checks": checks}


def _task_map(rta: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    taskset = _tasks(rta)
    return {str(row["name"]): row for row in (rta.get("taskset", {}).get("tasks", []) if isinstance(rta.get("taskset"), Mapping) else [])} or {row["task"]["name"]: row["task"] for row in taskset}


def build_release_count_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    from formal_toolchain.reference.arithmetic import ceil_div_nonnegative, floor_div_nonnegative
    checks = []
    # `release_counts` 里出现的是“整张 reference taskset”中的干扰任务名，
    # 不只是顶层 RTA 列表里的 HI task。这里必须优先读取 rta 自带的
    # `taskset.tasks`，这样 LO/HI 干扰任务的 period 才能被准确取到。
    # 顶层 `_tasks(rta)` 只保存每个 HI task 的 RTA 结果，不能拿来当完整
    # taskset 使用。
    tasksets = _task_map(rta)
    for row in _tasks(rta):
        lo = row.get("lo")
        start = row.get("start")
        if not isinstance(lo, Mapping) or not isinstance(start, Mapping):
            checks.append({
                "kind": "ROW_SHAPE",
                "task": row.get("task", {}).get("name"),
                "match": False,
            })
            continue
        for item in lo["trace"]:
            r = int(item["r"])
            for name, count in item.get("release_counts", {}).items():
                period = int(tasksets[name]["period"])
                expected = ceil_div_nonnegative(r, period)
                checks.append({"kind": "LO_CEIL", "task": row["task"]["name"], "interferer": name, "expected": expected, "actual": int(count), "match": expected == int(count)})
        for item in start["trace"]:
            w = int(item["w"])
            for name, count in item.get("release_counts", {}).items():
                period = int(tasksets[name]["period"])
                expected = floor_div_nonnegative(w, period) + 1
                checks.append({"kind": "START_FLOOR_PLUS_ONE", "task": row["task"]["name"], "interferer": name, "expected": expected, "actual": int(count), "match": expected == int(count)})
    ok = all(item["match"] for item in checks)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "RELEASE_COUNT_WITNESS_MISMATCH"}, "check_count": len(checks), "checks": checks}


def validate_case_trace(
    *,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from formal_toolchain.reference.arithmetic import ceil_div_nonnegative

    trace = candidate.get("trace")
    if not isinstance(trace, list) or not trace:
        return [{
            "task": row["task"]["name"],
            "case": candidate.get("case"),
            "kind": "trace_nonempty",
            "match": False,
        }]

    task = row["task"]
    start = int(candidate["start"])
    checks: list[dict[str, Any]] = []

    for step in trace:
        absolute_r = int(step.get("absolute_r", step.get("r")))
        actual_il = step.get("il_terms")
        actual_ih = step.get("ih_terms")
        if not isinstance(actual_il, Mapping) or not isinstance(actual_ih, Mapping):
            checks.append({
                "task": task["name"],
                "case": candidate.get("case"),
                "kind": "interference_maps_present",
                "match": False,
            })
            continue

        expected_il: dict[str, int] = {}
        expected_ih: dict[str, int] = {}
        for name, j in task_map.items():
            if int(j["priority_index"]) >= int(task["priority_index"]):
                continue
            if j["criticality"] == "LO":
                expected_il[name] = (
                    ceil_div_nonnegative(absolute_r, int(j["period"]))
                    * int(j["c_hi"])
                    + (start // int(j["period"]) + 1)
                    * (int(j["c_lo"]) - int(j["c_hi"]))
                )
            else:
                carry = (
                    0 if absolute_r <= start
                    else ceil_div_nonnegative(
                        absolute_r - start, int(j["period"])
                    )
                )
                expected_ih[name] = (
                    ceil_div_nonnegative(absolute_r, int(j["period"]))
                    * int(j["c_lo"])
                    + carry * (int(j["c_hi"]) - int(j["c_lo"]))
                )

        own = int(task["c_lo"] if candidate["case"] == "CASE1" else task["c_hi"])
        expected_raw_f = own + sum(expected_il.values()) + sum(expected_ih.values())
        expected_guarded_f = (
            max(start + int(task["c_hi"]), expected_raw_f)
            if candidate["case"] == "CASE2"
            else expected_raw_f
        )

        checks.extend([
            {
                "task": task["name"],
                "case": candidate["case"],
                "kind": "il_terms",
                "expected": expected_il,
                "actual": dict(actual_il),
                "match": dict(actual_il) == expected_il,
            },
            {
                "task": task["name"],
                "case": candidate["case"],
                "kind": "ih_terms",
                "expected": expected_ih,
                "actual": dict(actual_ih),
                "match": dict(actual_ih) == expected_ih,
            },
            {
                "task": task["name"],
                "case": candidate["case"],
                "kind": "raw_f",
                "expected": expected_raw_f,
                "actual": int(step.get("raw_f", step.get("f", -1))),
                "match": int(step.get("raw_f", step.get("f", -1))) == expected_raw_f,
            },
            {
                "task": task["name"],
                "case": candidate["case"],
                "kind": "guarded_f",
                "expected": expected_guarded_f,
                "actual": int(step.get("guarded_f", step.get("f", -1))),
                "match": int(step.get("guarded_f", step.get("f", -1))) == expected_guarded_f,
            },
        ])
        if candidate["case"] == "CASE2":
            guard = absolute_r >= start + int(task["c_hi"])
            checks.append({
                "task": task["name"],
                "case": "CASE2",
                "kind": "physical_guard",
                "expected": True,
                "actual": guard,
                "match": guard,
            })

    return checks


def build_demand_domination_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    task_rows = _tasks(rta)
    expected_count = int(
        rta.get(
            "task_count_expected",
            rta.get("inputs", {}).get("task_count_expected", -1),
        )
    )
    if len(task_rows) != expected_count:
        return {
            "status": "FAIL",
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "failure": {"code": "RTA_TASK_COVERAGE_INCOMPLETE"},
            "checks": [],
        }

    checks: list[dict[str, Any]] = []
    task_map = _task_map(rta)
    for row in task_rows:
        for candidate in row.get("case1", []) + row.get("case2", []):
            checks.extend(
                validate_case_trace(
                    row=row,
                    candidate=candidate,
                    task_map=task_map,
                )
            )

    ok = bool(checks) and all(item.get("match") is True for item in checks)
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "REFERENCE_CERTIFICATE_FAILED",
        "failure": None if ok else {"code": "DEMAND_DOMINATION_TRACE_MISMATCH"},
        "checks": checks,
    }


def build_inherited_hi_domination_evidence(*, rta: Mapping[str, Any], mode_result: Mapping[str, Any]) -> dict[str, Any]:
    data = json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"]
    theorem = data["INHERITED_HI_VIRTUAL_SWITCH_DOMINATION"]
    taskset = rta.get("taskset", {})
    hi_tasks = [task for task in taskset.get("tasks", []) if task.get("criticality") == "HI"] if isinstance(taskset, Mapping) else []
    ok = all((
        mode_result.get("status") == "PASS",
        isinstance(theorem.get("statement_hash"), str),
        len(theorem.get("statement_hash")) == 64,
        bool(hi_tasks),
        all(int(task["c_hi"]) >= int(task["c_lo"]) for task in hi_tasks),
    ))
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "INHERITED_HI_DOMINATION_INSTANCE_FAILED"}, "theorem": theorem, "hi_tasks": hi_tasks, "virtual_switch_origin": 0}


def build_discrete_tick_embedding_evidence(*, time_domain: Mapping[str, Any], scheduler: Mapping[str, Any], overhead: Mapping[str, Any]) -> dict[str, Any]:
    data = json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"]
    theorem = data["DISCRETE_TICK_FPPS_EMBEDDING"]
    ok = all((
        time_domain.get("status") == "PASS",
        scheduler.get("status") == "PASS",
        overhead.get("status") == "PASS",
        overhead.get("overhead") == 0,
        isinstance(theorem.get("statement_hash"), str),
    ))
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "DISCRETE_TICK_EMBEDDING_FAILED"}, "theorem": theorem}


def decompose_rta_obligations(*, rta: Mapping[str, Any], semantic_evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if rta.get("status") not in {"PASS", "FAIL", "UNRESOLVED"}:
        raise ValueError("RTA_STATUS_INVALID")
    return {
        "DISCRETE_TICK_EMBEDDING": build_discrete_tick_embedding_evidence(
            time_domain=semantic_result(semantic_evidence, "TIME_DOMAIN"),
            scheduler=semantic_result(semantic_evidence, "SCHEDULER_MODEL"),
            overhead=semantic_result(semantic_evidence, "OVERHEAD_PROFILE"),
        ),
        "RELEASE_COUNT": build_release_count_evidence(rta),
        "DEMAND_DOMINATION": build_demand_domination_evidence(rta),
        "LO_MODE_RTA": build_lo_mode_rta_evidence(rta),
        "WORST_CASE_START_TIME": build_worst_case_start_evidence(rta),
        "CASE1_INTEGER_DOMAIN": build_case1_domain_evidence(rta),
        "CASE2_INTEGER_DOMAIN": build_case2_domain_evidence(rta),
        "ZERO_RELATIVE_START": build_zero_relative_start_evidence(rta),
        "INHERITED_HI_DOMINATION": build_inherited_hi_domination_evidence(
            rta=rta,
            mode_result=semantic_result(semantic_evidence, "MODE_SEMANTICS_CONFORMANCE"),
        ),
    }
