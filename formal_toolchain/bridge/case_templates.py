"""P0 transition 的独立 reference 模板和 bound-path effect IR。

本模块有意把两条语义链拆开：``compile_case_template`` 只生成 P0
reference machine 的 ``r_*_post`` 方程；runtime 的 concrete 方程由
``compile_bound_path_effect`` 根据已经绑定到真实源码的 effect 列表生成。
因此不能仅修改一个手写模板就同时伪造 concrete 和 reference 的结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from .p0_case_manifest import require_case
from .state_relation import p0_smt_relation_fields


@dataclass(frozen=True, slots=True)
class CompiledTransitionCase:
    case_id: str
    declarations: str
    precondition: str
    concrete_delta: str
    reference_delta: str
    preservation: str
    template_hash: str


_FIELDS = p0_smt_relation_fields()
_DECLARATIONS = "\n".join(
    f"(declare-const {p}_{field}{'_post' if post else ''} Int)"
    for p in ("c", "r") for field in _FIELDS for post in (False, True)
) + "\n" + "\n".join(
    f"(declare-const {name} Int)" for name in (
        "actual_cost", "release_budget", "degraded_cost", "expected_demand",
        "release_job_key", "release_priority", "release_time", "release_deadline",
        "release_category", "next_event_time", "selected_job_key", "elapsed",
    )
)


def state_relation_schema() -> tuple[str, ...]:
    """返回证明实际声明的字段，供 schema hash 绑定，而非绑定常量名称。"""
    return p0_smt_relation_fields()


def _eq(prefix: str, field: str, value: str, post: bool = True) -> str:
    suffix = "_post" if post else ""
    return f"(= {prefix}_{field}{suffix} {value})"


def _all_post(prefix: str, overrides: Mapping[str, str] | None = None) -> str:
    overrides = overrides or {}
    equations = [_eq(prefix, field, overrides.get(field, f"{prefix}_{field}")) for field in _FIELDS]
    return "(and " + " ".join(equations) + ")"


def _relation_precondition() -> str:
    equalities = [f"(= c_{field} r_{field})" for field in _FIELDS]
    return "(and (>= c_time 0) (>= c_active 0) (>= c_ready 0) (>= c_remaining 0) (= c_other_jobs_frame_unchanged 1) (= c_other_task_budgets_frame_unchanged 1) " + " ".join(equalities) + ")"


def _reference_delta(case_id: str) -> str:
    """独立 P0 reference machine 的后继方程，只能出现 r_*_post。"""
    c = {"time": "r_time", "service": "r_service", "remaining": "r_remaining"}
    if case_id == "ONE_SERVICE_TICK":
        c.update(time="(+ r_time elapsed)", service="(+ r_service elapsed)",
                 remaining="(- r_remaining elapsed)",
                 affected_job_service="(+ r_affected_job_service elapsed)")
    elif case_id == "JUMP_TO_NEXT_EVENT":
        c["time"] = "next_event_time"
    elif case_id == "DEADLINE_OBSERVATION_FIRST_HI_MISS":
        c.update(miss="1", affected_job_hi_miss="1")
    elif case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        demand = {
            "PRIMARY_LO_RELEASE": "(ite (<= actual_cost (+ release_budget 1)) actual_cost (+ release_budget 1))",
            "DEGRADED_LO_RELEASE": "(ite (<= actual_cost degraded_cost) actual_cost degraded_cost)",
            "HI_RELEASE": "actual_cost",
        }[case_id]
        c.update(active="(+ r_active 1)", ready="(+ r_ready 1)", service="0",
                 remaining="expected_demand", budget="release_budget", priority="release_priority",
                 release="release_time", deadline="release_deadline", category="release_category",
                 job_key="release_job_key", affected_job_key="release_job_key",
                 affected_job_active="1", affected_job_ready="1",
                 affected_job_running="0", affected_job_priority="release_priority",
                 affected_job_release="release_time", affected_job_deadline="release_deadline",
                 affected_job_category="release_category", affected_job_budget="release_budget",
                 affected_job_demand="expected_demand", affected_job_service="0")
    elif case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "PRIMARY_LO_CANCELLATION"}:
        c.update(active="(- r_active 1)", ready="(- r_ready 1)", running="0",
                 affected_job_active="0", affected_job_ready="0", affected_job_running="0")
    elif case_id == "HI_COMPLETION":
        c.update(active="(- r_active 1)", ready="(- r_ready 1)", running="0", hi_complete="1",
                 affected_job_active="0", affected_job_ready="0", affected_job_running="0",
                 affected_job_hi_complete="1")
    elif case_id == "ARRIVAL_BATCH_SWITCH_S0":
        c["mode"] = "1"
    elif case_id == "IDLE_RECOVERY":
        c["mode"] = "0"
    elif case_id == "PREEMPTION_DISPATCH":
        c.update(running="selected_job_key", affected_job_key="selected_job_key",
                 affected_job_running="1")
    elif case_id == "CONTROLLER_SELECTED_ACTION":
        c.update(future_budget="release_budget", affected_task_budget="release_budget")
    # BOOT and no-op/controller/deadline cases retain the P0 state.
    body = _all_post("r", c)
    if case_id == "PRIMARY_LO_RELEASE":
        return "(and (= expected_demand (ite (<= actual_cost (+ release_budget 1)) actual_cost (+ release_budget 1))) " + body[5:]
    if case_id == "DEGRADED_LO_RELEASE":
        return "(and (= expected_demand (ite (<= actual_cost degraded_cost) actual_cost degraded_cost)) " + body[5:]
    if case_id == "HI_RELEASE":
        return "(and (= expected_demand actual_cost) " + body[5:]
    return body


def compile_bound_path_effect(row: Mapping[str, Any]) -> str:
    """把真实路径的有限 effect IR 编译成 concrete ``c_*_post`` 方程。

    未知 effect 必须显式失败；不能用 unchanged 作为兜底，因为那会把
    尚未建模的 runtime 行为伪装成已经证明。
    """
    effects = tuple(row.get("effects", ()))
    case_id = str(row.get("case_id", ""))
    known = {"mode=HI", "mode=LO", "mode_switch", "same_batch", "arrival_batch",
             "release", "reschedule", "build_job", "release_fixed_budget", "degraded_budget",
             "actual_cost_clamp", "active_add", "ready_add", "highest_priority_select",
             "running_update", "preempt_invalidate", "advance_time", "service_accounting",
             "remaining_update", "executed_to_actual", "active_remove", "running_clear",
             "hi_complete", "cancellation_event", "deadline_observe_only", "hi_miss_flag",
             "recovery_event", "budget_frame", "event_projection", "future_budget_update",
             "next_event_min", "time_jump", "no_service", "boot"}
    unknown = sorted(set(effects) - known)
    if unknown:
        raise ValueError(f"BOUND_PATH_EFFECT_UNRESOLVED:{unknown}")
    o: dict[str, str] = {}
    if "mode=HI" in effects: o["mode"] = "1"
    if "mode=LO" in effects: o["mode"] = "0"
    if "active_add" in effects: o.update(active="(+ c_active 1)", affected_job_active="1")
    if "ready_add" in effects: o.update(ready="(+ c_ready 1)", affected_job_ready="1")
    if "active_remove" in effects:
        o["active"] = "(- c_active 1)"
        o["ready"] = "(- c_ready 1)"
        o.update(affected_job_active="0", affected_job_ready="0")
    if "running_clear" in effects: o.update(running="0", affected_job_running="0")
    if "running_update" in effects: o.update(running="selected_job_key", affected_job_key="selected_job_key", affected_job_running="1")
    if "hi_complete" in effects: o.update(hi_complete="1", affected_job_hi_complete="1")
    if "hi_miss_flag" in effects: o.update(miss="1", affected_job_hi_miss="1")
    if "future_budget_update" in effects: o.update(future_budget="release_budget", affected_task_budget="release_budget")
    if "service_accounting" in effects:
        o.update(time="(+ c_time elapsed)", service="(+ c_service elapsed)",
                 remaining="(- c_remaining elapsed)",
                 affected_job_service="(+ c_affected_job_service elapsed)")
    if "time_jump" in effects:
        o["time"] = "next_event_time"
    if "no_service" in effects and "service_accounting" not in effects:
        o["service"] = "c_service"
    # The batch handler is a sequencing wrapper.  Only a single-release case is
    # allowed to add a job; otherwise a two-arrival batch would count as three.
    if case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        o.update(active="(+ c_active 1)", ready="(+ c_ready 1)", service="0",
                 remaining="expected_demand", budget="release_budget", priority="release_priority",
                 release="release_time", deadline="release_deadline", category="release_category",
                 job_key="release_job_key", affected_job_key="release_job_key",
                 affected_job_active="1", affected_job_ready="1", affected_job_running="0",
                 affected_job_priority="release_priority", affected_job_release="release_time",
                 affected_job_deadline="release_deadline", affected_job_category="release_category",
                 affected_job_budget="release_budget", affected_job_demand="expected_demand",
                 affected_job_service="0")
    body = _all_post("c", o)
    if case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        if case_id == "DEGRADED_LO_RELEASE":
            demand = "(= expected_demand (ite (<= actual_cost degraded_cost) actual_cost degraded_cost))"
        elif case_id == "PRIMARY_LO_RELEASE":
            demand = "(= expected_demand (ite (<= actual_cost (+ release_budget 1)) actual_cost (+ release_budget 1)))"
        else:
            demand = "(= expected_demand actual_cost)"
        return "(and " + demand + " " + body[5:]
    return body


def compile_case_template(case_id: str) -> CompiledTransitionCase:
    metadata = require_case(case_id)
    reference = _reference_delta(case_id)
    preservation = "(and " + " ".join(f"(= c_{f}_post r_{f}_post)" for f in _FIELDS) + ")"
    payload = {"case": metadata, "declarations": _DECLARATIONS, "precondition": _relation_precondition(),
               "reference_delta": reference, "preservation": preservation,
               "schema": state_relation_schema()}
    precondition = _relation_precondition()
    if case_id == "ONE_SERVICE_TICK":
        precondition = "(and " + precondition[5:-1] + " (> elapsed 0) (<= elapsed c_remaining) (> c_affected_job_running 0))"
    if case_id == "JUMP_TO_NEXT_EVENT":
        precondition = "(and " + precondition[5:-1] + " (> next_event_time c_time) (= c_running 0) (= c_ready 0))"
    if case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "HI_COMPLETION", "PRIMARY_LO_CANCELLATION"}:
        precondition = "(and " + precondition[5:-1] + " (> c_active 0))"
    return CompiledTransitionCase(case_id, _DECLARATIONS, precondition, "",
                                  reference, preservation, sha256_object(payload))
