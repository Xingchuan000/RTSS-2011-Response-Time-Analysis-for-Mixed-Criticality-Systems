"""Phase K 的唯一 P0 semantic case manifest。

该清单描述的是抽象 transition 的固定模板元数据，不包含任何预先填写的
PASS 结果。源码 branch 只能选择清单中的 case，具体证明由 compiler 现场生成。
"""

from __future__ import annotations

from typing import Any

from formal_toolchain.core.hashing import sha256_object

PRIMITIVE_CASE = "PRIMITIVE_CASE"
COMPOSITE_CASE = "COMPOSITE_CASE"
CASE_KINDS = {
    "ARRIVAL_BATCH_NO_SWITCH": COMPOSITE_CASE,
    "ARRIVAL_BATCH_SWITCH_S0": COMPOSITE_CASE,
    "BOOT_TO_PRECLOSED_0": COMPOSITE_CASE,
}


_ROWS = (
    ("BOOT_TO_PRECLOSED_0", "boot", "ZERO_TIME", None, ("time", "mode", "jobs"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("ARRIVAL_BATCH_NO_SWITCH", "arrival", "ZERO_TIME", "JOB_ARRIVAL", ("jobs", "ready", "future_budget"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("ARRIVAL_BATCH_SWITCH_S0", "arrival", "ZERO_TIME", "JOB_ARRIVAL", ("jobs", "ready", "mode", "future_budget"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("PRIMARY_LO_RELEASE", "release_lo", "ZERO_TIME", "JOB_ARRIVAL", ("job_key", "priority", "release", "demand"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("DEGRADED_LO_RELEASE", "release_degraded_lo", "ZERO_TIME", "JOB_ARRIVAL", ("job_key", "priority", "release", "demand"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("HI_RELEASE", "release_hi", "ZERO_TIME", "JOB_ARRIVAL", ("job_key", "priority", "release", "demand", "mode"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("RESCHEDULE_KEEP_SAME", "reschedule_keep_same", "ZERO_TIME", None, ("ready", "running", "priority", "effective_event_frontier"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("RESCHEDULE_TO_IDLE", "reschedule_to_idle", "ZERO_TIME", None, ("ready", "running", "effective_event_frontier"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("PREEMPTION_DISPATCH", "reschedule_dispatch", "ZERO_TIME", None, ("ready", "running", "priority", "effective_event_frontier"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("ONE_SERVICE_TICK", "service", "SERVICE", None, ("time", "service", "remaining", "running"), ("DISCRETE_TICK_FPPS_EMBEDDING",)),
    ("NORMAL_COMPLETION", "completion", "ZERO_TIME", "JOB_COMPLETION", ("jobs", "ready", "running", "service"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("PRIMARY_LO_CANCELLATION", "primary_lo_cancellation", "ZERO_TIME", "JOB_COMPLETION", ("jobs", "ready", "running", "service"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("DEGRADED_COMPLETION", "degraded_completion", "ZERO_TIME", "JOB_COMPLETION", ("jobs", "ready", "running", "service"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("HI_COMPLETION", "hi_completion", "ZERO_TIME", "JOB_COMPLETION", ("jobs", "ready", "running", "service", "hi_complete"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("DEADLINE_OBSERVATION_NO_MISS", "deadline_no_miss", "ZERO_TIME", "DEADLINE_CHECK", ("time", "jobs", "miss"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("DEADLINE_OBSERVATION_FIRST_HI_MISS", "deadline_hi_miss", "ZERO_TIME", "HI_DEADLINE_MISS", ("time", "job_key", "deadline", "service", "miss"), ("FINITE_HI_BAD_PREFIX_REFLECTION",)),
    ("IDLE_RECOVERY", "recovery", "ZERO_TIME", None, ("time", "mode", "jobs", "running"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("CONTROLLER_NO_ACTION", "controller_no_action", "ZERO_TIME", None, ("jobs", "running", "future_budget"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("CONTROLLER_SELECTED_ACTION", "controller_selected_action", "ZERO_TIME", None, ("jobs", "running", "future_budget"), ("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",)),
    ("JUMP_TO_NEXT_EVENT", "jump", "TIME_JUMP", None, ("time", "jobs", "running"), ("DISCRETE_TICK_FPPS_EMBEDDING",)),
)


def p0_case_manifest() -> dict[str, dict[str, Any]]:
    """返回不可变语义清单的拷贝，并附带 deterministic manifest hash。"""
    return {case_id: {"case_id": case_id, "template_id": template_id,
                      "transition_class": transition_class,
                      "projected_event_kind": event_kind,
                      "required_relation_components": list(components),
                      "theorem_dependencies": list(theorems),
                      "case_kind": CASE_KINDS.get(case_id, PRIMITIVE_CASE)}
            for case_id, template_id, transition_class, event_kind, components, theorems in _ROWS}


def p0_case_manifest_hash() -> str:
    return sha256_object(p0_case_manifest())


def require_case(case_id: str) -> dict[str, Any]:
    try:
        return p0_case_manifest()[case_id]
    except KeyError as exc:
        raise ValueError(f"未知 P0 case: {case_id}") from exc


def case_kind(case_id: str) -> str:
    return CASE_KINDS.get(case_id, PRIMITIVE_CASE)
