"""Phase J08：Protected-HI safety corollary 的窄接口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def _expected_theory_refs() -> dict[str, dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "theory" / "hashes.json"
    data = json.loads(path.read_text(encoding="utf-8"))["statements"]
    ids = ("PER_HI_TASK_INDUCTIVE_WCRT", "DISCRETE_TICK_FPPS_EMBEDDING",
           "ZERO_RELATIVE_START_LEMMA", "INHERITED_HI_VIRTUAL_SWITCH_DOMINATION",
           "RECURRING_JOB_CRITICAL_INSTANT_DOMINATION")
    ids = (*ids, "PROTECTED_HI_SAFETY_COROLLARY")
    return {theorem: data[theorem] for theorem in ids}


def protected_hi_safety_corollary(recurring_instances: dict[str, Any]) -> dict[str, Any]:
    """只有全部逐 task instance 通过时才给出 corollary PASS。"""
    instances = recurring_instances.get("instances")
    context_hash = recurring_instances.get("context_hash")
    if (recurring_instances.get("schema_version") != "per_hi_task_inductive_wcrt_v1"
            or recurring_instances.get("theorem") != "PER_HI_TASK_INDUCTIVE_WCRT"
            or not isinstance(instances, list) or not instances
            or not isinstance(context_hash, str) or not context_hash
            or any(instance.get("status") != "PASS" for instance in instances)
            or not verify_obligation_certificate(recurring_instances)
            or recurring_instances.get("obligation_id") != "PER_HI_TASK_INDUCTIVE_WCRT"
            or recurring_instances.get("obligation_status") != "PASS"):
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "theorem": "PROTECTED_HI_SAFETY_COROLLARY"}
    rta = recurring_instances.get("rta_certificate")
    if not isinstance(rta, dict) or not verify_obligation_certificate(rta):
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "theorem": "PROTECTED_HI_SAFETY_COROLLARY",
                "failure": "RTA_PREDECESSOR_INVALID"}
    if recurring_instances.get("rta_artifact_hash") != rta.get("artifact_hash"):
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "theorem": "PROTECTED_HI_SAFETY_COROLLARY",
                "failure": "RTA_PREDECESSOR_HASH_MISMATCH"}
    expected_refs = _expected_theory_refs()
    rta_rows = {
        row.get("task", {}).get("name"): row
        for row in rta.get("tasks", [])
        if row.get("task", {}).get("criticality") == "HI"
    }
    if set(rta_rows) != {item.get("task", {}).get("name") for item in instances}:
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "theorem": "PROTECTED_HI_SAFETY_COROLLARY",
                "failure": "INSTANCE_TASK_SET_MISMATCH"}
    for instance in instances:
        name = instance["task"]["name"]
        row = rta_rows[name]
        expected_hi = max(item["response_for_deadline"] for item in row["case1"] + row["case2"])
        if (instance.get("task") != row.get("task")
                or instance.get("r_lo") != row["lo"].get("r_lo")
                or instance.get("r_hi") != expected_hi
                or instance.get("r_star") != max(row["lo"].get("r_lo"), expected_hi)
                or instance.get("case1_witness_hash") != sha256_object(row["case1"])
                or instance.get("case2_witness_hash") != sha256_object(row["case2"])
                or any(instance.get("theorem_refs", {}).get(key) != expected_refs[key]
                       for key in expected_refs if key != "PROTECTED_HI_SAFETY_COROLLARY")):
            return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                    "theorem": "PROTECTED_HI_SAFETY_COROLLARY",
                    "failure": "INSTANCE_WITNESS_OR_THEORY_MISMATCH", "task": name}
    theorem_hash = expected_refs["PROTECTED_HI_SAFETY_COROLLARY"]["statement_hash"]
    result = {"status": "PASS", "route": "PROTECTED_HI_SAFETY_COROLLARY",
              "theorem": "PROTECTED_HI_SAFETY_COROLLARY", "theorem_hash": theorem_hash,
              "instances": instances}
    result.update(obligation_certificate(
        obligation_id="PROTECTED_HI_SAFETY_COROLLARY", status="PASS",
        context_hash=context_hash, inputs={"instance_count": len(instances)},
        witness={"instances": instances},
        direct_predecessor_hashes={"recurring": str(recurring_instances.get("artifact_hash") or "")},
        checker_id="formal_toolchain.reference.protected_hi", checker_version="phase-j-v1",
    ))
    return result
