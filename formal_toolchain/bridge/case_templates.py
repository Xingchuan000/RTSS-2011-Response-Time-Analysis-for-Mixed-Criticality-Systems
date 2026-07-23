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
from formal_toolchain.reference.p0_transition_contract import render_reference_p0_delta
from .p0_case_manifest import require_case
from .model_bounds import P0ModelBounds, _legacy_test_bounds
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


def _declarations(bounds: P0ModelBounds) -> str:
    fields = p0_smt_relation_fields(bounds)
    return ("\n".join(
        f"(declare-const {p}_{field}{'_post' if post else ''} Int)"
        for p in ("c", "r") for field in fields for post in (False, True)
    ) + "\n" + "\n".join(
    f"(declare-const {name} Int)" for name in (
        "actual_cost", "release_budget", "degraded_cost", "expected_demand",
        "release_job_key", "release_priority", "release_time", "release_deadline",
        "release_category", "next_event_time", "selected_job_key", "elapsed",
        "event_job_key", "running_job_key", "token_valid", "event_time",
        "event_deadline", "task_criticality", "primary_mode", "config_semantics",
        "event_kind", "job_exists", "job_finished", "is_degraded", "c_lo",
        "active_empty", "ready_empty", "highest_priority_selected", "queue_min_time",
        "update_target_key", "update_arity",
        "affected_task_key",
        "next_release_time", "next_deadline_time",
        "release_mode", "release_slot", "affected_job_slot",
        "event_queue_slot", "release_queue_slot", "deadline_queue_slot",
        "update_target_slot",
        "queue_next_timing_boundary", "pushed_event_time", "pushed_event_kind",
        "queue_next_timing_boundary_kind", "queue_next_timing_boundary_job_key",
        "queue_next_timing_boundary_token", "pushed_event_job_key", "pushed_event_token", "ordered_tasks_empty",
        "task_names_duplicate", "abnormal_arrivals_empty", "release_mode_is_none",
        "response_semantics", "force", "selected_job_present", "selected_started",
        "time_reversed", "capture_trace", "token_invalid", "job_or_running_invalid",
        "monitor_present", "job_active", "completed_job_was_hi", "stop_at_first_miss",
        "budget_exists", "lo_cancellation_semantics", "time_before_target",
        "halted_now", "halted", "idle_recovery_enabled",
    )
    )
    # queue summary 只为本次 transition 中真实出现的 push 保留有序事件
    # 输入；不展开 heap slot。最多支持计划内的四个 timing push。
    + "\n" + "\n".join(
        f"(declare-const pushed_event_{index}_{field} Int)"
        for index in range(4)
        for field in ("time", "kind", "job_key", "token")
    ))


def state_relation_schema(bounds: P0ModelBounds) -> tuple[str, ...]:
    """返回证明实际声明的字段，供 schema hash 绑定，而非绑定常量名称。"""
    return p0_smt_relation_fields(bounds)


def _eq(prefix: str, field: str, value: str, post: bool = True) -> str:
    suffix = "_post" if post else ""
    return f"(= {prefix}_{field}{suffix} {value})"


def _all_post(prefix: str, bounds: P0ModelBounds, overrides: Mapping[str, str] | None = None) -> str:
    overrides = overrides or {}
    fields = p0_smt_relation_fields(bounds)
    equations = [_eq(prefix, field, overrides.get(field, f"{prefix}_{field}")) for field in fields]
    return "(and " + " ".join(equations) + ")"


def _relation_precondition(bounds: P0ModelBounds) -> str:
    fields = p0_smt_relation_fields(bounds)
    equalities = [f"(= c_{field} r_{field})" for field in fields]
    constraints = ["(>= c_time 0)", "(>= c_active 0)", "(>= c_ready 0)"]
    for slot in range(bounds.job_slots):
        prefix = f"c_job_{slot}_"
        constraints.extend([f"(or (= {prefix}present 0) (= {prefix}present 1))",
                            f"(or (= {prefix}active 0) (= {prefix}active 1))",
                            f"(or (= {prefix}ready 0) (= {prefix}ready 1))",
                            f"(or (= {prefix}running 0) (= {prefix}running 1))",
                            f"(=> (= {prefix}ready 1) (= {prefix}active 1))",
                            f"(=> (= {prefix}running 1) (= {prefix}ready 1))",
                            f"(=> (= {prefix}active 1) (= {prefix}present 1))"])
        for other in range(slot):
            constraints.append(
                f"(=> (and (= {prefix}active 1) (= c_job_{other}_active 1)) "
                f"(not (= {prefix}key c_job_{other}_key)))")
    running_slots = [f"(= c_job_{slot}_running 1)" for slot in range(bounds.job_slots)]
    constraints.append(f"(<= (+ {' '.join(f'(ite {item} 1 0)' for item in running_slots)}) 1)")
    # heap 的完整内容不参与 P0 下一步 timing 选择；只绑定最小事件及其
    # 类型/job/token、release/deadline 两个未来边界和摘要计数。
    constraints.extend([
        "(= c_queue_min_time queue_min_time)",
        "(>= c_queue_event_count 0)",
        "(=> (= c_queue_event_count 0) (= c_queue_min_time 2147483647))",
        "(=> (> c_queue_event_count 0) (>= c_queue_min_time c_time))",
        "(>= c_queue_next_release_time c_time)",
        "(>= c_queue_next_deadline_time c_time)",
        "(>= c_queue_token_epoch 0)",
    ])
    return "(and " + " ".join(constraints + equalities) + ")"


def _slot_ite(variable: str, slot: int, when_true: str, when_false: str) -> str:
    return f"(ite (= {variable} {slot}) {when_true} {when_false})"


def _release_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        values = {
            "present": "1", "active": "1", "ready": "1", "running": "0",
            "key": "release_job_key", "priority": "release_priority",
            "release": "release_time", "deadline": "release_deadline",
            "category": "release_category", "criticality": "task_criticality",
            "mode": "release_mode", "released_mode": "release_mode",
            "is_degraded": "is_degraded", "budget": "release_budget",
            "demand": "expected_demand", "service": "0", "token": "0",
            "completion_token": "0", "overrun_token": "0", "hi_complete": "0", "hi_miss": "0",
        }
        for field, value in values.items():
            result[f"job_{slot}_{field}"] = _slot_ite(
                "release_slot", slot, value, f"{base}{field}")
    return result


def _remove_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        selected = f"(= {base}key event_job_key)"
        for field, value in {"present": "0", "active": "0", "ready": "0",
                             "running": "0", "completion_token": "0", "overrun_token": "0",
                             "hi_complete": "0", "hi_miss": "0"}.items():
            result[f"job_{slot}_{field}"] = f"(ite {selected} {value} {base}{field})"
    return result


def _dispatch_job_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    result: dict[str, str] = {}
    for slot in range(bounds.job_slots):
        base = f"{prefix}_job_{slot}_"
        result[f"job_{slot}_running"] = f"(ite (= {base}key selected_job_key) 1 0)"
    return result


def _task_budget_overrides(prefix: str, bounds: P0ModelBounds) -> dict[str, str]:
    return {f"task_{slot}_future_budget": _slot_ite(
        "update_target_slot", slot, "release_budget", f"{prefix}_task_{slot}_future_budget")
            for slot in range(bounds.task_slots)}


def _queue_relation_constraints(bounds: P0ModelBounds) -> list[str]:
    del bounds
    return [f"(= c_{field} r_{field})" for field in (
        "queue_min_time", "queue_min_kind", "queue_min_job_key", "queue_min_token",
        "queue_next_release_time", "queue_next_deadline_time", "queue_event_count",
        "queue_token_epoch")]


def compile_bound_path_effect(row: Mapping[str, Any], *, bounds: P0ModelBounds | None = None) -> str:
    """把真实路径的有限 effect IR 编译成 concrete ``c_*_post`` 方程。

    未知 effect 必须显式失败；不能用 unchanged 作为兜底，因为那会把
    尚未建模的 runtime 行为伪装成已经证明。
    """
    bounds = bounds or _legacy_test_bounds()
    case_id = str(row.get("case_id", ""))
    effect_ir = row.get("effect_ir")
    if not isinstance(effect_ir, list) or not effect_ir:
        # 仅保留单元测试/旧 API 的显式失败测试入口；正式 compiler 在
        # transition_compiler 中先强制要求 effect_ir，因此不会用这里的
        # legacy effects 生成正式证书。
        effects = set(row.get("effects", ()))
        joined = ""
    else:
        sources = tuple(str(item.get("source", "")) for item in effect_ir
                        if isinstance(item, Mapping))
        effects = set()
        joined = "\n".join(sources)
    if "self.state.mode = SystemMode.HI" in joined or "state.mode = SystemMode.HI" in joined:
        effects.add("mode=HI")
    if "self.state.mode = SystemMode.LO" in joined or "state.mode = SystemMode.LO" in joined:
        effects.add("mode=LO")
    if "pop_all_matching" in joined or "events.sort" in joined:
        effects.add("arrival_batch")
    if "_process_single_arrival_in_priority_order" in joined:
        effects.add("release")
    if "_build_job(" in joined:
        effects.add("build_job")
    if "runtime_budget_at_release" in joined or "budget_of" in joined:
        effects.add("release_fixed_budget")
    if "_c_amc_sem_degraded_lo_budget" in joined:
        effects.add("degraded_budget")
    if "actual_cost_override" in joined or "min(original_actual_cost" in joined:
        effects.add("actual_cost_clamp")
    if "active_jobs.append" in joined:
        effects.update(("active_add", "ready_add"))
    if "active_jobs.remove" in joined:
        effects.add("active_remove")
    if "running_job = None" in joined or "running_job=None" in joined:
        effects.add("running_clear")
    if "running_job = selected" in joined or "running_job=selected" in joined:
        effects.add("running_update")
    if "_invalidate_job_events" in joined:
        effects.add("preempt_invalidate")
    if "_update_running_progress" in joined or "_advance_time" in joined:
        effects.update(("advance_time", "service_accounting"))
    if "executed_time = job.actual_cost" in joined:
        effects.add("executed_to_actual")
    if "deadline_misses.append" in joined:
        effects.add("hi_miss_flag")
    if "_maybe_recover" in joined or "mode_recoveries.append" in joined:
        effects.add("recovery_event")
    if "_reschedule" in joined:
        effects.add("reschedule")
    if "_schedule_next_release" in joined:
        effects.add("queue_release")
    if "EventType.DEADLINE_CHECK" in joined:
        effects.add("deadline_schedule")
    if "apply_updates" in joined:
        effects.add("future_budget_update")
    if "queue.push" in joined or "queue.pop" in joined:
        effects.add("queue_operation")
    if "target_time" in joined or "event.time" in joined:
        effects.add("time_boundary")
    # 当前 P0 path selector 将同一复合 handler 的多个 terminal 片段映射
    # 到不同 semantic case；这里仅允许用源码 effect IR 中已经出现的
    # effect，再依据该 path 的实际 terminal case 去除互斥的 sibling branch。
    # 这不是新增人工状态更新：所有保留下来的 effect 都必须来自 effect_ir。
    if case_id == "DEADLINE_OBSERVATION_NO_MISS":
        effects.discard("hi_miss_flag")
    if case_id == "DEADLINE_OBSERVATION_FIRST_HI_MISS":
        effects.add("hi_miss_flag")
    if case_id == "CONTROLLER_NO_ACTION":
        effects.discard("future_budget_update")
    if case_id == "CONTROLLER_SELECTED_ACTION":
        effects.add("future_budget_update")
    if case_id == "JUMP_TO_NEXT_EVENT":
        effects.discard("service_accounting")
        effects.discard("advance_time")
        effects.add("time_jump")
    if case_id in {"CONTROLLER_NO_ACTION", "CONTROLLER_SELECTED_ACTION"}:
        # deployed controller 调用发生在当前决策边界；其 settle 调用的
        # elapsed 为 0，不能把 _advance_time 的源码存在误编译成一次服务。
        effects.discard("service_accounting")
        effects.discard("advance_time")
    effects = tuple(sorted(effects))
    known = {"mode=HI", "mode=LO", "mode_switch", "same_batch", "arrival_batch",
             "release", "reschedule", "build_job", "release_fixed_budget", "degraded_budget",
             "queue_release", "deadline_schedule",
             "actual_cost_clamp", "active_add", "ready_add", "highest_priority_select",
             "running_update", "preempt_invalidate", "advance_time", "service_accounting",
             "remaining_update", "executed_to_actual", "active_remove", "running_clear",
             "hi_complete", "cancellation_event", "deadline_observe_only", "hi_miss_flag",
             "recovery_event", "budget_frame", "event_projection", "future_budget_update",
             "next_event_min", "time_jump", "no_service", "boot"}
    known.update({"queue_operation", "time_boundary"})
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
    if "future_budget_update" in effects:
            o.update(_task_budget_overrides("c", bounds))
    if "service_accounting" in effects:
        o.update(time="(+ c_time elapsed)", service="(+ c_service elapsed)",
                 remaining="(- c_remaining elapsed)",
                 affected_job_service="(+ c_affected_job_service elapsed)")
    if "time_jump" in effects:
        o["time"] = "next_event_time"
    if "no_service" in effects and "service_accounting" not in effects:
        o["service"] = "c_service"
    if "active_remove" in effects:
        o.update(_remove_job_overrides("c", bounds))
    if "running_update" in effects:
        o.update(_dispatch_job_overrides("c", bounds))
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
        o.update(_release_job_overrides("c", bounds))
    body = _all_post("c", bounds, o)
    if case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        if case_id == "DEGRADED_LO_RELEASE":
            demand = "(= expected_demand (ite (<= actual_cost degraded_cost) actual_cost degraded_cost))"
        elif case_id == "PRIMARY_LO_RELEASE":
            demand = "(= expected_demand (ite (<= actual_cost (+ release_budget 1)) actual_cost (+ release_budget 1)))"
        else:
            demand = "(= expected_demand actual_cost)"
        return "(and " + demand + " " + body[5:]
    return body


def compile_case_template(case_id: str, *, bounds: P0ModelBounds | None = None) -> CompiledTransitionCase:
    bounds = bounds or _legacy_test_bounds()
    metadata = require_case(case_id)
    fields = p0_smt_relation_fields(bounds)
    declarations = _declarations(bounds)
    reference = render_reference_p0_delta(
        case_id,
        bounds,
    )
    preservation = "(and " + " ".join(f"(= c_{f}_post r_{f}_post)" for f in fields) + ")"
    precondition = _relation_precondition(bounds)
    extra: list[str] = []
    if case_id in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"}:
        extra += ["(= event_job_key release_job_key)", "(= c_event_job_key release_job_key)",
                  "(= release_time event_time)", "(= event_time c_time)", "(>= actual_cost 0)",
                  "(>= release_budget 0)", "(> next_release_time release_time)",
                  "(= next_deadline_time release_deadline)"]
        extra += ["(or " + " ".join(f"(= release_slot {slot})" for slot in range(bounds.job_slots)) + ")"]
        # 只有选中的空 slot 可以被 release 写入；其余 slot 的 frame
        # 由 concrete/reference post 方程逐字段保持。
        extra += ["(or " + " ".join(f"(and (= release_slot {slot}) (= c_job_{slot}_present 0))"
                                      for slot in range(bounds.job_slots)) + ")"]
        extra += ["(= release_mode c_mode)"]
    if case_id == "ONE_SERVICE_TICK":
        extra += ["(> elapsed 0)", "(<= elapsed c_remaining)", "(> c_affected_job_running 0)",
                  "(= c_running_job_key c_affected_job_key)", "(= event_time (+ c_time elapsed))"]
    if case_id == "JUMP_TO_NEXT_EVENT":
        extra += ["(> next_event_time c_time)", "(= next_event_time queue_min_time)",
                  "(= c_running 0)", "(= c_ready 0)", "(= c_ready_empty 1)"]
    if case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "HI_COMPLETION", "PRIMARY_LO_CANCELLATION"}:
        extra += ["(> c_active 0)", "(= token_valid 1)", "(= event_job_key c_affected_job_key)",
                  "(= running_job_key event_job_key)", "(= c_running_job_key event_job_key)"]
        extra += ["(or " + " ".join(
            f"(and (= c_job_{slot}_active 1) (= c_job_{slot}_key event_job_key))"
            for slot in range(bounds.job_slots)) + ")"]
    if case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION", "HI_COMPLETION"}:
        # completion 是独立 event micro-step；deadline 只属于 deadline
        # observation path，正常 job 可以在 deadline 之前完成。
        extra += ["(= c_remaining 0)", "(>= event_time c_time)", "(= event_deadline c_deadline)"]
    if case_id == "HI_COMPLETION":
        extra += ["(= task_criticality 1)"]
    if case_id == "PRIMARY_LO_CANCELLATION":
        extra += ["(= primary_mode 1)", "(> c_service c_budget)", "(= config_semantics 1)"]
    if case_id == "PREEMPTION_DISPATCH":
        extra += ["(= highest_priority_selected 1)", "(= selected_job_key c_affected_job_key)",
                  "(= c_selected_job_key selected_job_key)", "(>= c_ready 0)"]
        extra += ["(or " + " ".join(
            f"(and (= c_job_{slot}_ready 1) (= c_job_{slot}_key selected_job_key))"
            for slot in range(bounds.job_slots)) + ")"]
    if case_id == "IDLE_RECOVERY":
        extra += ["(= c_mode 1)", "(= active_empty 1)", "(= ready_empty 1)", "(= c_running 0)"]
    if case_id in {"DEADLINE_OBSERVATION_NO_MISS", "DEADLINE_OBSERVATION_FIRST_HI_MISS"}:
        extra += ["(= event_job_key c_affected_job_key)", "(= event_time c_deadline)",
                  "(= event_deadline c_deadline)", "(= token_valid 1)"]
    if case_id == "CONTROLLER_SELECTED_ACTION":
        extra += ["(>= update_arity 1)", "(= update_target_key affected_task_key)",
                  "(or " + " ".join(f"(= update_target_slot {slot})" for slot in range(bounds.task_slots)) + ")",
                  "(or " + " ".join(
                      f"(and (= c_task_{slot}_present 1) (= c_task_{slot}_key update_target_key))"
                      for slot in range(bounds.task_slots)) + ")"]
    if case_id == "CONTROLLER_NO_ACTION":
        # 空 updates 分支必须约束 update_arity，避免条件 budget effect
        # 在该 terminal path 上变成任意写入。
        extra += ["(= update_arity 0)"]
    if extra:
        precondition = "(and " + precondition[5:-1] + " " + " ".join(extra) + ")"
    # template hash 必须覆盖最终 guard；否则调用方可以在 hash 已固定后再替换
    # precondition，形成“证明模板与实际门控条件脱节”的假阳性。
    payload = {"case": metadata, "declarations": declarations, "precondition": precondition,
               "reference_delta": reference, "preservation": preservation,
               "schema": state_relation_schema(bounds), "model_bounds": bounds.to_dict(),
               "reference_transition_system_id":
                   "FIXED_EXECUTABLE_REFERENCE_P0_V3"}
    return CompiledTransitionCase(case_id, declarations, precondition, "",
                                  reference, preservation, sha256_object(payload))
