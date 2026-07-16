"""Phase J08：逐 HI task 的有限 theorem instance。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from .rta_production import protected_hi_rta
from .task_mapping import ReferenceTaskset


_THEORY_DIR = Path(__file__).resolve().parents[1] / "theory"
_THEOREM_IDS = (
    "PER_HI_TASK_INDUCTIVE_WCRT", "DISCRETE_TICK_FPPS_EMBEDDING",
    "ZERO_RELATIVE_START_LEMMA", "INHERITED_HI_VIRTUAL_SWITCH_DOMINATION",
    "RECURRING_JOB_CRITICAL_INSTANT_DOMINATION",
)


def _theory_hashes() -> dict[str, dict[str, str]]:
    from formal_toolchain.verifier.theory_verifier import verify_theory_library
    verify_theory_library(_THEORY_DIR)
    data = json.loads((_THEORY_DIR / "hashes.json").read_text(encoding="utf-8"))
    result = data.get("statements", {})
    if any(theorem not in result for theorem in _THEOREM_IDS):
        raise ValueError("recurring HI 所需 theorem hash 缺失")
    return {theorem: dict(result[theorem]) for theorem in _THEOREM_IDS}


def build_recurring_hi_instances(taskset: ReferenceTaskset, *,
                                 rta_certificate: dict[str, Any]) -> dict[str, Any]:
    """只消费已验证 RTA certificate，构造带 theorem side conditions 的 instance。"""
    if not verify_obligation_certificate(rta_certificate):
        raise ValueError("RTA certificate hash 无效")
    if (rta_certificate.get("obligation_id") != "PROTECTED_HI_RTA_ARITHMETIC"
            or rta_certificate.get("obligation_status") != "PASS"
            or rta_certificate.get("status") != "PASS"
            or rta_certificate.get("taskset") != taskset.to_dict()):
        raise ValueError("RTA certificate context/status 不满足 recurring theorem 前置条件")
    theory = _theory_hashes()
    instances = []
    for row in rta_certificate.get("tasks", []):
        task = row["task"]
        task_index = taskset.priority_order.index(task["name"])
        hp_lo = [item.name for item in taskset.tasks[:task_index] if item.criticality == "LO"]
        hp_hi = [item.name for item in taskset.tasks[:task_index] if item.criticality == "HI"]
        r_lo = row["lo"]["r_lo"]
        r_hi = max([item["response_for_deadline"] for item in row["case1"] + row["case2"]])
        r_star = max(r_lo, r_hi)
        if not (r_star <= task["deadline"] <= task["period"]):
            raise ValueError(f"task {task['name']} 不满足 R_star<=D<=T")
        case1_hash = sha256_object(row["case1"])
        case2_hash = sha256_object(row["case2"])
        instances.append({
            "task": task, "r_lo": r_lo, "r_hi": r_hi, "r_star": r_star,
            "hp_LO": hp_lo, "hp_HI": hp_hi,
            "r_lo_witness_hash": sha256_object(row["lo"]),
            "w_witness_hash": sha256_object(row["start"]),
            "deadline_side_condition": {"r_star_le_deadline": r_star <= task["deadline"],
                                         "deadline_le_period": task["deadline"] <= task["period"]},
            "case1_witness_hash": case1_hash, "case2_witness_hash": case2_hash,
            "theorem_refs": theory,
            "theorem_side_conditions": {
                "discrete_tick_embedding": theory["DISCRETE_TICK_FPPS_EMBEDDING"]["statement_hash"],
                "zero_relative_start": row["start"]["w_lo"] == 0,
                "inherited_hi_virtual_switch": theory["INHERITED_HI_VIRTUAL_SWITCH_DOMINATION"]["statement_hash"],
                "recurring_job_critical_instant": theory["RECURRING_JOB_CRITICAL_INSTANT_DOMINATION"]["statement_hash"],
            },
            "status": "PASS",
        })
    if not instances:
        raise ValueError("没有 HI task，不能构造 recurring HI theorem instance")
    result = {"schema_version": "per_hi_task_inductive_wcrt_v1",
              "theorem": "PER_HI_TASK_INDUCTIVE_WCRT", "status": "PASS",
              "instances": instances, "context_hash": taskset.source_context_hash,
              "rta_artifact_hash": rta_certificate["artifact_hash"],
              "rta_certificate": rta_certificate}
    result.update(obligation_certificate(
        obligation_id="PER_HI_TASK_INDUCTIVE_WCRT", status="PASS",
        context_hash=taskset.source_context_hash,
        inputs={"rta_artifact_hash": rta_certificate["artifact_hash"]},
        witness={"instances": instances, "theory_refs": theory},
        direct_predecessor_hashes={"rta": rta_certificate["artifact_hash"]},
        checker_id=__name__, checker_version="phase-j-v2",
    ))
    return result
