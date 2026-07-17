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
        w_lo = int(row["start"]["w_lo"])
        case2 = row.get("case2", [])
        if w_lo == 0:
            ok = len(case2) == 1 and case2[0].get("tag") == "ZERO_RELATIVE_START" and case2[0].get("start") == 0
            expected = ["ZERO_RELATIVE_START"]
            actual = [item.get("tag") for item in case2]
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
        w_lo = int(row["start"]["w_lo"])
        if w_lo != 0:
            checks.append({"task": row["task"]["name"], "applicable": False, "status": "NOT_APPLICABLE"})
            continue
        case2 = row.get("case2", [])
        task = row["task"]
        ok = len(case2) == 1 and case2[0].get("tag") == "ZERO_RELATIVE_START" and case2[0].get("start") == 0 and case2[0].get("response_for_deadline") == int(task["c_hi"])
        checks.append({"task": task["name"], "applicable": True, "status": "PASS" if ok else "FAIL", "case2": case2})
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
        for item in row["lo"]["trace"]:
            r = int(item["r"])
            for name, count in item.get("release_counts", {}).items():
                period = int(tasksets[name]["period"])
                expected = ceil_div_nonnegative(r, period)
                checks.append({"kind": "LO_CEIL", "task": row["task"]["name"], "interferer": name, "expected": expected, "actual": int(count), "match": expected == int(count)})
        for item in row["start"]["trace"]:
            w = int(item["w"])
            for name, count in item.get("release_counts", {}).items():
                period = int(tasksets[name]["period"])
                expected = floor_div_nonnegative(w, period) + 1
                checks.append({"kind": "START_FLOOR_PLUS_ONE", "task": row["task"]["name"], "interferer": name, "expected": expected, "actual": int(count), "match": expected == int(count)})
    ok = all(item["match"] for item in checks)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "RELEASE_COUNT_WITNESS_MISMATCH"}, "check_count": len(checks), "checks": checks}


def validate_case_trace(*, row: Mapping[str, Any], candidate: Mapping[str, Any], task_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    from formal_toolchain.reference.arithmetic import ceil_div_nonnegative
    checks: list[dict[str, Any]] = []
    start = int(candidate["start"])
    r = int(candidate["response_for_deadline"] if "response_for_deadline" in candidate else candidate["absolute_response"])
    for name, term in candidate.get("il_terms", {}).items():
        j = task_map[name]
        if candidate.get("case") == "CASE1":
            expected = ceil_div_nonnegative(r, int(j["period"])) * int(j["c_hi"]) + (start // int(j["period"]) + 1) * (int(j["c_lo"]) - int(j["c_hi"]))
        else:
            expected = ceil_div_nonnegative(r, int(j["period"])) * int(j["c_hi"]) + (start // int(j["period"]) + 1) * (int(j["c_lo"]) - int(j["c_hi"]))
        checks.append({"case": candidate.get("case"), "task": row["task"]["name"], "interferer": name, "kind": "il_terms", "expected": expected, "actual": int(term), "match": expected == int(term)})
    for name, term in candidate.get("ih_terms", {}).items():
        j = task_map[name]
        if candidate.get("case") == "CASE1":
            expected = ceil_div_nonnegative(r, int(j["period"])) * int(j["c_lo"]) + (0 if r <= start else ceil_div_nonnegative(r - start, int(j["period"]))) * (int(j["c_hi"]) - int(j["c_lo"]))
        else:
            expected = ceil_div_nonnegative(r, int(j["period"])) * int(j["c_lo"]) + max(0, ceil_div_nonnegative(r - start, int(j["period"]))) * (int(j["c_hi"]) - int(j["c_lo"]))
        checks.append({"case": candidate.get("case"), "task": row["task"]["name"], "interferer": name, "kind": "ih_terms", "expected": expected, "actual": int(term), "match": expected == int(term)})
    checks.append({"case": candidate.get("case"), "task": row["task"]["name"], "kind": "f", "expected": r, "actual": int(candidate.get("f", r)), "match": r == int(candidate.get("f", r))})
    return checks


def build_demand_domination_evidence(rta: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    task_map = _task_map(rta)
    for row in _tasks(rta):
        for candidate in row.get("case1", []) + row.get("case2", []):
            if candidate.get("tag") == "ZERO_RELATIVE_START":
                continue
            checks.extend(validate_case_trace(row=row, candidate=candidate, task_map=task_map))
    ok = all(item["match"] for item in checks)
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "REFERENCE_CERTIFICATE_FAILED", "failure": None if ok else {"code": "DEMAND_DOMINATION_TRACE_MISMATCH"}, "checks": checks}


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
