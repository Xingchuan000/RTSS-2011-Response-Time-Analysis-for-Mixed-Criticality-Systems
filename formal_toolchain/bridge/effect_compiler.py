"""将 CFG 的 EffectIR 编译为有限 P0 concrete delta。

本模块只接受已经由 Python CFG 提取器绑定的 effect；无法识别的状态 effect
必须失败，避免把未知运行时行为错误地当成 identity transition。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model_bounds import P0ModelBounds
from .state_relation import p0_smt_relation_fields


_PHASE_K_STATIC_EFFECT_DEFAULTS: dict[str, bool] = {
    "retroactive_release_budget_mutation": False,
}


def _runtime_config_value(runtime_config: Any | None, name: str, default: Any) -> Any:
    """Read one immutable runtime-config value from object or exported mapping.

    Formal replay normally receives the target runtime config object, while
    some validation paths use the exported ``effective_runtime_config.json``
    shape.  Both are accepted so the static effect binding is deterministic in
    candidate compilation and fresh verification.
    """
    if runtime_config is None:
        return default
    if isinstance(runtime_config, Mapping):
        fields = runtime_config.get("fields")
        if isinstance(fields, Mapping) and name in fields:
            field = fields[name]
            if isinstance(field, Mapping) and "value" in field:
                return field["value"]
            return field
        value = runtime_config.get(name, default)
        if isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value
    return getattr(runtime_config, name, default)


def build_phase_k_static_effect_bindings(runtime_config: Any | None) -> dict[str, bool]:
    """Bind experiment-only helper calls to immutable proof-request config.

    The C3 helper is intentionally called unconditionally in runtime source to
    keep the handler CFG stable.  Under the default/off profile it is a proven
    no-op and must therefore be statically elided by Phase K.  Under the C3
    profile the policy/invariant layer is required to reject the system before
    the closed-prefix bridge is attempted; Phase K fails closed if invoked.
    """
    profile = str(_runtime_config_value(runtime_config, "nonvacuity_profile", "off"))
    return {
        "retroactive_release_budget_mutation":
            profile == "c3_retroactive_release_budget",
    }


@dataclass(frozen=True, slots=True)
class CompiledConcreteEffect:
    equations: tuple[str, ...]
    consumed_effect_hashes: tuple[str, ...]
    non_state_effect_hashes: tuple[str, ...] = ()
    queue_equations: tuple[str, ...] = ()
    modified_components: tuple[str, ...] = ()
    semantic_effect_kinds: tuple[str, ...] = ()
    affected_job_sources: tuple[str, ...] = ()

    def to_smt(self) -> str:
        return "(and " + " ".join(self.equations or ("true",)) + ")"


def is_pure_local_effect(*, kind: str, source: str) -> bool:
    if kind in {"PURE_EXPR", "ASSERT", "RETURN"}:
        return True
    return kind == "ASSIGN" and not any(token in source for token in (
        "self.state", "self.queue", "active_jobs", "running_job", "executed_time",
        "valid_completion_tokens", "valid_overrun_tokens",
        "deadline_misses"))


def _slot_frame(prefix: str, count: int, fields: tuple[str, ...]) -> list[str]:
    return [f"(= {prefix}_{slot}_{field}_post {prefix}_{slot}_{field})"
            for slot in range(count) for field in fields]


def compile_job_insert(*, bounds: P0ModelBounds) -> list[str]:
    equations = ["(>= release_slot 0)", f"(< release_slot {bounds.job_slots})"]
    fields = ("present", "active", "ready", "running", "key", "priority", "release",
              "deadline", "category", "criticality", "mode", "released_mode",
              "is_degraded", "budget", "demand", "service", "completion_token",
              "overrun_token", "hi_complete", "hi_miss")
    values = ("1", "1", "1", "0", "release_job_key", "release_priority", "release_time",
              "release_deadline", "release_category", "task_criticality", "release_mode",
              "release_mode", "is_degraded", "release_budget", "expected_demand", "0",
              "0", "0", "0", "0")
    for slot in range(bounds.job_slots):
        selected = f"(= release_slot {slot})"
        for field, value in zip(fields, values):
            equations.append(f"(= c_job_{slot}_{field}_post (ite {selected} {value} c_job_{slot}_{field}))")
        equations.append(
            f"(=> (= c_job_{slot}_present 1) (not (= c_job_{slot}_key release_job_key)))"
        )
    return equations


def compile_job_remove(*, bounds: P0ModelBounds) -> list[str]:
    equations = ["(or " + " ".join(
        f"(and (= c_job_{slot}_present 1) (= c_job_{slot}_key event_job_key))"
        for slot in range(bounds.job_slots)) + ")"]
    for slot in range(bounds.job_slots):
        selected = f"(= c_job_{slot}_key event_job_key)"
        for field in ("present", "active", "ready", "running", "completion_token",
                      "overrun_token", "hi_complete", "hi_miss"):
            equations.append(f"(= c_job_{slot}_{field}_post (ite {selected} 0 c_job_{slot}_{field}))")
    return equations


def compile_service_update(*, bounds: P0ModelBounds) -> list[str]:
    equations = ["(> elapsed 0)", "(= running_job_key event_job_key)"]
    for slot in range(bounds.job_slots):
        selected = f"(and (= c_job_{slot}_present 1) (= c_job_{slot}_running 1) (= c_job_{slot}_key running_job_key))"
        equations.append(f"(= c_job_{slot}_service_post (ite {selected} (+ c_job_{slot}_service elapsed) c_job_{slot}_service))")
    return equations


def compile_queue_push(*, bounds: P0ModelBounds, index: int = 0,
                       include_count: bool = True) -> list[str]:
    del bounds
    prefix = f"pushed_event_{index}_"
    equations = [
        f"(= c_queue_min_time_post (ite (< {prefix}time c_queue_min_time) {prefix}time c_queue_min_time))",
        f"(= c_queue_min_kind_post (ite (< {prefix}time c_queue_min_time) {prefix}kind c_queue_min_kind))",
        f"(= c_queue_min_job_key_post (ite (< {prefix}time c_queue_min_time) {prefix}job_key c_queue_min_job_key))",
        f"(= c_queue_min_token_post (ite (< {prefix}time c_queue_min_time) {prefix}token c_queue_min_token))",
    ]
    if include_count:
        equations.insert(0, "(= c_queue_event_count_post (+ c_queue_event_count 1))")
    return equations


def compile_queue_pop(*, bounds: P0ModelBounds) -> list[str]:
    del bounds
    return ["(> c_queue_event_count 0)",
            "(= c_queue_event_count_post (- c_queue_event_count 1))",
            "(= c_queue_min_time_post queue_next_timing_boundary)"]


def completion_token_guard(*, bounds: P0ModelBounds) -> str:
    return "(or " + " ".join(
        f"(and (= c_job_{slot}_present 1) (= c_job_{slot}_key event_job_key) (= c_job_{slot}_completion_token event_token))"
        for slot in range(bounds.job_slots)) + ")"


def compile_stale_token_noop(*, bounds: P0ModelBounds) -> str:
    equations = compile_queue_pop(bounds=bounds)
    equations.extend(_slot_frame("c_job", bounds.job_slots, ("present", "key", "active", "ready", "running", "service")))
    equations.extend(_slot_frame("c_task", bounds.task_slots, ("present", "key", "criticality", "future_budget")))
    equations.extend(_slot_frame("c_queue", 0, ()))
    equations.extend(f"(= c_{field}_post c_{field})" for field in (
        "queue_min_kind", "queue_min_job_key", "queue_min_token",
        "queue_next_release_time", "queue_next_deadline_time", "queue_token_epoch"))
    return "(and " + " ".join(equations) + ")"


def compile_effect_ir(effect_ir: list[Mapping[str, Any]], *, bounds: P0ModelBounds,
                      guard_ir: list[Mapping[str, Any]] | None = None,
                      static_effect_bindings: Mapping[str, bool] | None = None
                      ) -> CompiledConcreteEffect:
    equations: list[str] = []
    consumed: list[str] = []
    push_count = 0
    push_kinds: list[str] = []
    pop_count = 0
    token_epoch_delta = 0
    statically_elided: list[str] = []
    modified_components: set[str] = set()
    semantic_effects: set[str] = set()
    affected_job_sources: set[str] = set()
    bindings = dict(_PHASE_K_STATIC_EFFECT_DEFAULTS if static_effect_bindings is None
                    else static_effect_bindings)
    for effect in effect_ir:
        kind, source, ast_hash = str(effect["kind"]), str(effect["source"]), str(effect["ast_hash"])
        if any(token in source for token in ("job", "active_jobs", "jobs_by_key", "running_job")):
            affected_job_sources.add(source)
        if "active_jobs.append" in source:
            semantic_effects.add("JOB_RELEASE"); modified_components.update({"released_ledger", "active_jobs", "ready_order"})
        if "jobs_by_key[" in source and "= job" in source:
            semantic_effects.add("JOB_RELEASE"); modified_components.add("released_ledger")
        if "active_jobs.remove" in source:
            semantic_effects.add("JOB_REMOVE"); modified_components.update({"active_jobs", "ready_order", "running_key"})
        if any(token in source for token in ("completion_time =", "dropped = True", "drop_time =")):
            semantic_effects.add("TERMINAL_MARK"); modified_components.add("terminal_ledger")
        if "deadline_misses.append" in source:
            semantic_effects.add("MISS_APPEND"); modified_components.add("miss_ledger")
        if "executed_time +=" in source or "_update_running_progress" in source:
            semantic_effects.add("SERVICE_UPDATE"); modified_components.update({"active_service", "remaining_to_removal", "time"})
        if "running_job =" in source:
            semantic_effects.add("RUNNING_UPDATE"); modified_components.add("running_key")
        if "state.mode =" in source or "self.state.mode =" in source:
            semantic_effects.add("MODE_UPDATE"); modified_components.add("mode")
        if "apply_updates" in source:
            semantic_effects.add("FUTURE_BUDGET_UPDATE"); modified_components.add("future_budget_ghost")
        if "queue.push" in source:
            semantic_effects.add("QUEUE_PUSH"); modified_components.add("effective_event_frontier")
        if "queue.pop" in source or "pop_all_matching" in source:
            semantic_effects.add("QUEUE_POP"); modified_components.add("effective_event_frontier")
        if "_invalidate_job_events" in source:
            semantic_effects.add("TOKEN_INVALIDATION"); modified_components.add("effective_event_frontier")
        compiled: list[str] | None = None
        assignment_target = source.split("=", 1)[0].strip() if "=" in source else ""
        if "active_jobs.append" in source:
            compiled = compile_job_insert(bounds=bounds)
        elif "active_jobs.remove" in source:
            compiled = compile_job_remove(bounds=bounds)
        elif "executed_time +=" in source or "_update_running_progress" in source:
            compiled = compile_service_update(bounds=bounds)
        elif "queue.push" in source:
            compiled = []
            push_count += 1
            push_kinds.append("1" if "BUDGET_UPDATE" in source else
                              "3" if "DEADLINE_CHECK" in source else "event_kind")
        elif "queue.pop" in source:
            compiled = compile_queue_pop(bounds=bounds)
        elif "_schedule_next_release" in source:
            # 释放后续 arrival 是 timing-relevant queue effect；其具体时间和
            # 事件键由当前有限模型的 pushed_* 符号输入承载。
            compiled = []
            push_count += 1
            push_kinds.append("0")
        elif "state.mode = SystemMode.HI" in source:
            compiled = ["(= c_mode_post 1)"]
        elif "state.mode = SystemMode.LO" in source:
            compiled = ["(= c_mode_post 0)"]
        elif "state.running_job = None" in source:
            compiled = []
        elif "state.running_job = selected" in source:
            compiled = []
        elif "state.current_time = now" in source:
            compiled = ["(= c_time_post event_time)"]
        elif "state.run_started_at =" in source:
            compiled = []
        elif "_invalidate_job_events" in source or "_schedule_running_job_events" in source:
            compiled = []
            token_epoch_delta += 1
        elif "_apply_retroactive_release_budget_mutation" in source:
            enabled = bindings.get("retroactive_release_budget_mutation")
            if not isinstance(enabled, bool):
                raise ValueError(
                    "PHASE_K_STATIC_EFFECT_BINDING_REQUIRED:"
                    "retroactive_release_budget_mutation"
                )
            if enabled:
                # C3 deliberately changes already-released job snapshots.  The
                # ACTIVE_RELEASE_BUDGET_INVARIANT checker must reject this
                # profile before Phase K.  Never silently model it as identity.
                raise ValueError(
                    "NONVACUITY_C3_RETROACTIVE_EFFECT_REQUIRES_"
                    "POLICY_CONTRACT_REJECTION"
                )
            compiled = []
            statically_elided.append(ast_hash)
        elif "budget_state.apply_updates" in source or "runtime_budgets.apply_updates" in source:
            compiled = ["(= c_affected_task_budget_post (ite (> update_arity 0) release_budget c_affected_task_budget))"]
        elif "deadline_misses.append" in source:
            compiled = ["(= c_miss_post 1)"]
        elif kind == "ASSIGN" and assignment_target and not assignment_target.startswith("self."):
            # 读取 state/config 生成局部变量本身不改变 P0；真正改变 state 的
            # assignment 仍由上面的显式分支处理。
            compiled = []
        elif kind == "LOOP_CALL_BOUNDARY":
            compiled = []
        elif kind == "CALL" and any(token in source for token in (
            "jobs_by_key", "all_jobs", "_append_debug_event", "mode_recoveries",
            "job_cancellations", "_record_lo_job_loss", "_maybe_recover",
            "_reschedule", "_advance_time", "monitor.record", "result.",
            "busy_period_start", "response_time_expiry", "job.completion_time",
            "job.executed_time", "job.dropped", "job.drop_time", "run_started_at")):
            compiled = []
        elif kind == "CALL" and not any(token in source for token in (
            "self.state", "self.queue", "active_jobs", "running_job", "executed_time",
            "budget_state", "runtime_budgets", "valid_completion_tokens",
            "valid_overrun_tokens")):
            # 纯局部排序/构造调用必须显式消费，但不产生 P0 状态方程。
            compiled = []
        elif is_pure_local_effect(kind=kind, source=source):
            compiled = []
        if compiled is None:
            raise ValueError(f"UNSUPPORTED_STATE_EFFECT:{source}")
        equations.extend(compiled)
        consumed.append(ast_hash)
    # 以下方程由 EffectIR 中出现的源码操作直接触发，不读取 case_id；它们
    # 是跨 handler 共用的 P0 状态投影。
    joined = "\n".join(str(effect.get("source", "")) for effect in effect_ir)
    if "_schedule_response_time_expiry_for_hi_job" in joined:
        push_count += 1
        push_kinds.append("4")
    if "_schedule_running_job_events" in joined:
        # 一个 running job 会同时获得 completion 和 overrun 两个 timing
        # token/event；二者共享 summary，但必须各占一个 push。
        push_count += 2
        push_kinds.extend(("2", "5"))
    if "queue.pop" in joined or "pop_all_matching" in joined:
        pop_count += 1
    if guard_ir and any(str(item.get("test_source", "")).startswith(
            "event.event_type is EventType.") for item in guard_ir):
        # _process_event 的 event 已由外层 queue.pop 消费；这里把该消费
        # 显式纳入本 micro-step，而不是把它留成源码存在性。
        if not "pop_all_matching" in joined:
            pop_count += 1
    if "active_jobs.append" in joined:
        equations.extend(["(= c_active_post (+ c_active 1))", "(= c_ready_post (+ c_ready 1))",
                          "(= c_affected_job_active_post 1)", "(= c_affected_job_ready_post 1)",
                          "(= c_remaining_post expected_demand)", "(= c_budget_post release_budget)",
                          "(= c_priority_post release_priority)", "(= c_release_post release_time)",
                          "(= c_deadline_post release_deadline)", "(= c_category_post release_category)",
                          "(= c_job_key_post release_job_key)", "(= c_affected_job_key_post release_job_key)",
                          "(= c_affected_job_running_post 0)", "(= c_affected_job_priority_post release_priority)",
                          "(= c_affected_job_release_post release_time)", "(= c_affected_job_deadline_post release_deadline)",
                          "(= c_affected_job_category_post release_category)",
                          "(= c_affected_job_budget_post release_budget)",
                          "(= c_affected_job_demand_post expected_demand)",
                          "(= c_affected_job_service_post 0)", "(= c_service_post 0)"])
        # release demand 按 task criticality 和实际 degraded 标记分类；
        # response-time-expiry 只是可选的后续调度 effect，不能决定 HI
        # release 的 demand 语义。
        if "_c_amc_sem_degraded_lo_budget" in joined or "min(original_actual_cost" in joined:
            equations.append("(= is_degraded 1)")
        else:
            equations.append("(= is_degraded 0)")
        equations.append(
            "(= expected_demand (ite (= task_criticality 1) actual_cost "
            "(ite (= is_degraded 1) "
            "(ite (<= actual_cost degraded_cost) actual_cost degraded_cost) "
            "(ite (<= actual_cost (+ release_budget 1)) actual_cost (+ release_budget 1)))))")
    if "active_jobs.remove" in joined:
        equations.extend(["(= c_active_post (- c_active 1))", "(= c_ready_post (- c_ready 1))",
                          "(= c_affected_job_active_post 0)", "(= c_affected_job_ready_post 0)"])
    if "_update_running_progress" in joined or "executed_time +=" in joined:
        equations.extend(["(= c_time_post (+ c_time elapsed))",
                          "(= c_service_post (+ c_service elapsed))",
                          "(= c_remaining_post (- c_remaining elapsed))",
                          "(= c_affected_job_service_post (+ c_affected_job_service elapsed))"])
    if "running_job = None" in joined or "running_job=None" in joined:
        equations.extend(["(= c_running_post 0)", "(= c_affected_job_running_post 0)"])
    if "running_job = selected" in joined or "running_job=selected" in joined:
        equations.extend(["(= c_running_post selected_job_key)",
                          "(= c_affected_job_key_post selected_job_key)",
                          "(= c_affected_job_running_post 1)"])
        for slot in range(bounds.job_slots):
            selected = f"(= c_job_{slot}_key selected_job_key)"
            equations.append(f"(= c_job_{slot}_running_post (ite {selected} 1 0))")
    if "deadline_misses.append" in joined:
        equations.extend(["(= c_miss_post 1)", "(= c_affected_job_hi_miss_post 1)"])
    if "apply_updates" in joined:
        equations.extend(["(= c_future_budget_post (ite (> update_arity 0) release_budget c_future_budget))",
                          "(= c_affected_task_budget_post (ite (> update_arity 0) release_budget c_affected_task_budget))"])
        for slot in range(bounds.task_slots):
            selected = f"(= update_target_slot {slot})"
            equations.append(
                f"(= c_task_{slot}_future_budget_post (ite (and (> update_arity 0) {selected}) release_budget "
                f"c_task_{slot}_future_budget))")
    if "target_time" in joined and "_advance_time" in joined:
        equations.append("(= c_time_post next_event_time)")
    if push_count or pop_count:
        equations.extend(f"(= pushed_event_{index}_kind {kind})"
                         for index, kind in enumerate(push_kinds))
        count_delta = push_count - pop_count
        if count_delta > 0:
            count_expr = f"(+ c_queue_event_count {count_delta})"
        elif count_delta < 0:
            count_expr = f"(- c_queue_event_count {-count_delta})"
        else:
            count_expr = "c_queue_event_count"
        equations.append(f"(= c_queue_event_count_post {count_expr})")
        base_time = "queue_next_timing_boundary" if pop_count else "c_queue_min_time"
        base_kind = "queue_next_timing_boundary_kind" if pop_count else "c_queue_min_kind"
        base_job = "queue_next_timing_boundary_job_key" if pop_count else "c_queue_min_job_key"
        base_token = "queue_next_timing_boundary_token" if pop_count else "c_queue_min_token"
        for index in range(push_count):
            prefix = f"pushed_event_{index}_"
            previous_time = base_time
            base_time = f"(ite (< {prefix}time {previous_time}) {prefix}time {previous_time})"
            base_kind = f"(ite (< {prefix}time {previous_time}) {prefix}kind {base_kind})"
            base_job = f"(ite (< {prefix}time {previous_time}) {prefix}job_key {base_job})"
            base_token = f"(ite (< {prefix}time {previous_time}) {prefix}token {base_token})"
        equations.extend([f"(= c_queue_min_time_post {base_time})",
                          f"(= c_queue_min_kind_post {base_kind})",
                          f"(= c_queue_min_job_key_post {base_job})",
                          f"(= c_queue_min_token_post {base_token})"])
    if token_epoch_delta:
        equations.append(f"(= c_queue_token_epoch_post (+ c_queue_token_epoch {token_epoch_delta}))")
    # 未被具体 effect 改写的字段必须显式 frame；这样 concrete delta 仍然是
    # 完整的有限 P0 后状态，而不是把未建模字段隐式留给 SMT 任意赋值。
    for field in p0_smt_relation_fields(bounds):
        post_name = f"c_{field}_post"
        if not any(post_name in equation for equation in equations):
            equations.append(f"(= {post_name} c_{field})")
    non_state_candidates = [
        ast_hash for effect, ast_hash in zip(effect_ir, consumed)
        if not any(token in str(effect.get("source", "")) for token in (
            "active_jobs.append", "active_jobs.remove", "executed_time +=",
            "_update_running_progress", "queue.push", "queue.pop",
            "state.mode =", "state.running_job =", "_invalidate_job_events",
            "_schedule_running_job_events", "budget_state.apply_updates",
            "runtime_budgets.apply_updates", "deadline_misses.append",
            "_apply_retroactive_release_budget_mutation"))
    ]
    for ast_hash in statically_elided:
        if ast_hash not in non_state_candidates:
            non_state_candidates.append(ast_hash)
    non_state = tuple(non_state_candidates)
    queue_equations = tuple(equation for equation in equations
                            if "c_queue_" in equation and "_post" in equation)
    return CompiledConcreteEffect(tuple(equations), tuple(consumed), non_state, queue_equations,
                                   tuple(sorted(modified_components)), tuple(sorted(semantic_effects)),
                                   tuple(sorted(affected_job_sources)))
