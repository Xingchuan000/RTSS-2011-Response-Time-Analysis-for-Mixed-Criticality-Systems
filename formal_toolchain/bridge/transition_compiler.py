"""把源码绑定的完整 transition path 编译为可检查的局部证明。"""

from __future__ import annotations

from typing import Any, Mapping

from .case_templates import compile_case_template
from .effect_compiler import (
    build_phase_k_static_effect_bindings,
    compile_effect_ir,
)
from .model_bounds import P0ModelBounds
from .state_relation import p0_state_relation_schema_hash, p0_smt_relation_fields
from .p0_case_manifest import require_case
from .transition_cases import TransitionCaseProof, prove_smt2_case
from formal_toolchain.core.artifact import obligation_certificate


_PHASE_K_STATIC_GUARD_DEFAULTS: dict[str, bool] = {
    "nonvacuity_deadline_cleanup_remove": False,
    "nonvacuity_hi_budget_cap_truncate": False,
    "nonvacuity_recover_without_quiescence": False,
}


def build_phase_k_static_guard_bindings(runtime_config: Any | None) -> dict[str, bool]:
    """从 immutable effective runtime config 导出 Phase K 静态 guard。

    非空洞性 profile 在一次 proof request 内不可变化，因此这些 guard
    不是运行状态变量，应在 GuardIR 编译时折叠为布尔常量。直接调用旧
    API 而未提供 config 时使用安全的 ``off`` 默认值，以保持现有测试和
    工具调用兼容；正式 compile/replay 路径会显式传入 target config。
    """
    if runtime_config is None:
        return dict(_PHASE_K_STATIC_GUARD_DEFAULTS)

    def read(name: str) -> bool:
        if isinstance(runtime_config, Mapping):
            value = runtime_config.get(name, _PHASE_K_STATIC_GUARD_DEFAULTS[name])
        else:
            value = getattr(runtime_config, name, _PHASE_K_STATIC_GUARD_DEFAULTS[name])
        if not isinstance(value, bool):
            raise ValueError(f"PHASE_K_STATIC_GUARD_NOT_BOOL:{name}")
        return value

    return {name: read(name) for name in _PHASE_K_STATIC_GUARD_DEFAULTS}


def _static_bool_formula(bindings: Mapping[str, bool], name: str) -> str:
    if name not in bindings:
        raise ValueError(f"PHASE_K_STATIC_GUARD_BINDING_REQUIRED:{name}")
    value = bindings[name]
    if not isinstance(value, bool):
        raise ValueError(f"PHASE_K_STATIC_GUARD_NOT_BOOL:{name}")
    return "true" if value else "false"


def _guard_formula(source: str, *, static_guard_bindings: Mapping[str, bool] | None = None) -> str:
    """把一个真实源码 predicate 映射到有限 P0 的符号输入。

    这是源码 GuardIR 的唯一编译边界：每一个已提取 predicate 都必须在此
    明确落到符号公式；没有映射时直接失败，不能把 guard 静默跳过。
    """
    bindings = dict(_PHASE_K_STATIC_GUARD_DEFAULTS if static_guard_bindings is None
                    else static_guard_bindings)
    if source == "self.config.nonvacuity_deadline_cleanup_remove":
        return _static_bool_formula(bindings, "nonvacuity_deadline_cleanup_remove")
    if source == ("self.config.nonvacuity_hi_budget_cap_truncate and "
                  "job.task.criticality is Criticality.HI"):
        enabled = _static_bool_formula(bindings, "nonvacuity_hi_budget_cap_truncate")
        return f"(and {enabled} (= task_criticality 1))"
    if source == ("cfg.nonvacuity_recover_without_quiescence and "
                  "state.mode is SystemMode.HI"):
        enabled = _static_bool_formula(bindings, "nonvacuity_recover_without_quiescence")
        return f"(and {enabled} (= c_mode 1))"

    exact = {
        "not ordered_tasks": "(= ordered_tasks_empty 1)",
        "len(task_names) != len(set(task_names))": "(= task_names_duplicate 1)",
        "not _is_c_amc_semantics(self.config.semantics)": "(= config_semantics 0)",
        "not _uses_idle_recovery(cfg.semantics)": "(= idle_recovery_enabled 0)",
        "self.state.mode is not SystemMode.LO": "(not (= c_mode 0))",
        "not abnormal_arrivals": "(= abnormal_arrivals_empty 1)",
        "release_mode is None": "(= release_mode_is_none 1)",
        "task.criticality is Criticality.HI": "(= task_criticality 1)",
        "release_mode is SystemMode.HI": "(= release_mode 1)",
        "release_mode is SystemMode.HI and task.criticality is Criticality.LO": "(and (= release_mode 1) (= task_criticality 0))",
        "_is_c_amc_semantics(self.config.semantics)": "(= config_semantics 1)",
        # HI release 分类只由 task criticality 决定；但这一条源码
        # guard 本身表示“response-based 且 HI”，必须完整保留两个
        # 合取项。C-AMC-sem HI path 对该 guard 取 false 时，由
        # response_semantics=0 满足，不得反向否定 task 的 HI 属性。
        "_is_response_based_semantics(self.config.semantics) and task.criticality is Criticality.HI": "(and (= response_semantics 1) (= task_criticality 1))",
        "selected is state.running_job and (not force)": "(and (= selected_job_key running_job_key) (= force 0))",
        "state.running_job is not None": "(= c_running 1)",
        "selected is None": "(= selected_job_present 1)",
        "selected_key not in state.started_jobs": "(= selected_started 0)",
        "now < old_time": "(= time_reversed 1)",
        "self.config.capture_trace": "(= capture_trace 1)",
        "self.state.valid_completion_tokens.get(key) != event.token": "(= token_invalid 1)",
        "self.state.valid_overrun_tokens.get(key) != event.token": "(= token_invalid 1)",
        "job is None or self.state.running_job is not job": "(= job_or_running_invalid 1)",
        "self.monitor is not None": "(= monitor_present 1)",
        "job in self.state.active_jobs": "(= job_active 1)",
        "self.config.semantics is RuntimeSemantics.AMC_RH and completed_job_was_hi": "(and (= config_semantics 4) (= completed_job_was_hi 1))",
        "event.event_type is EventType.BUDGET_UPDATE": "(= event_kind 1)",
        "event.event_type is EventType.JOB_ARRIVAL": "(= event_kind 0)",
        "event.event_type is EventType.DEADLINE_CHECK": "(= event_kind 3)",
        "event.event_type is EventType.JOB_COMPLETION": "(= event_kind 2)",
        "event.event_type is EventType.RESPONSE_TIME_EXPIRY": "(= event_kind 4)",
        "event.event_type is EventType.BUDGET_OVERRUN": "(= event_kind 5)",
        "job is None": "(= job_exists 0)",
        "not job.finished()": "(= job_finished 0)",
        "job.finished()": "(= job_finished 1)",
        "self.config.stop_at_first_miss": "(= stop_at_first_miss 1)",
        "state.mode is SystemMode.HI and (not state.active_jobs) and (state.running_job is None)": "(and (= c_mode 1) (= active_empty 1) (= c_running 0))",
        "self.state.current_time < target_time and (not halted_now) and (not self.halted)": "(and (= time_before_target 1) (= halted_now 0) (= halted 0))",
    }
    if source in exact:
        return exact[source]
    if source == "_is_response_based_semantics(self.config.semantics) and job.task.criticality is Criticality.HI":
        return "(and (= response_semantics 1) (= task_criticality 1))"
    if source == "budget is None":
        return "(= budget_exists 0)"
    if source == "job.executed_time <= budget":
        return "(<= c_service c_budget)"
    if source.startswith("self.config.semantics in {") and "Criticality.LO" in source:
        return "(and (= lo_cancellation_semantics 1) (= task_criticality 0))"
    raise ValueError(f"UNSUPPORTED_SOURCE_GUARD:{source}")


def compile_source_guards(guard_ir: list[Mapping[str, Any]], *,
                          static_guard_bindings: Mapping[str, bool] | None = None) -> "CompiledSourceGuard":
    """编译完整 GuardIR，并记录每个源码 guard 的 hash 消费情况。"""
    if not isinstance(guard_ir, list):
        raise ValueError("SOURCE_GUARD_IR_REQUIRED")
    formulas: list[str] = []
    consumed: list[str] = []
    for item in guard_ir:
        if not isinstance(item, Mapping):
            raise ValueError("INVALID_SOURCE_GUARD_IR")
        source = str(item.get("test_source", item.get("test", "")))
        formula = _guard_formula(source, static_guard_bindings=static_guard_bindings)
        polarity = bool(item.get("polarity", True))
        formulas.append(formula if polarity else f"(not {formula})")
        consumed.append(str(item["test_ast_hash"]))
    return CompiledSourceGuard("(and " + " ".join(formulas or ["true"]) + ")", tuple(consumed))


class CompiledSourceGuard:
    """GuardIR 的 SMT 结果及其可审计消费记录。"""

    def __init__(self, formula: str, consumed_guard_hashes: tuple[str, ...]):
        self.formula = formula
        self.consumed_guard_hashes = consumed_guard_hashes


def _compile_source_guard(row: Mapping[str, Any]) -> str:
    """兼容旧调用方；正式证明入口使用 ``compile_source_guards``。"""
    guards = row.get("guard_ir")
    if not isinstance(guards, list):
        raise ValueError("SOURCE_GUARD_IR_REQUIRED")
    return compile_source_guards(guards).formula


def compile_and_prove_all_transition_cases(branch_map: Mapping[str, Any], *,
                                           bridge_context_hash: str,
                                           bounds: P0ModelBounds,
                                           runtime_config: Any | None = None) -> dict[str, Any]:
    if branch_map.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "BRANCH_MAP_REQUIRED"}
    if not isinstance(bridge_context_hash, str) or len(bridge_context_hash) != 64:
        return {"status": "UNRESOLVED", "failure": "BRIDGE_CONTEXT_REQUIRED"}
    rows = branch_map.get("paths")
    if not isinstance(rows, list):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_PATHS_REQUIRED"}
    static_guard_bindings = build_phase_k_static_guard_bindings(runtime_config)
    static_effect_bindings = build_phase_k_static_effect_bindings(runtime_config)
    from formal_toolchain.core.hashing import sha256_object
    static_guard_bindings_hash = sha256_object(static_guard_bindings)
    static_effect_bindings_hash = sha256_object(static_effect_bindings)
    proofs: list[TransitionCaseProof] = []
    for row in rows:
        try:
            if not isinstance(row.get("guard_ir"), list) or not row.get("effect_ir"):
                raise ValueError("REAL_GUARD_EFFECT_IR_REQUIRED")
            if row.get("guard") != row.get("guard_ir"):
                raise ValueError("GUARD_FIELD_MUST_BE_SOURCE_AST_IR")
            if row.get("guard_ast_hash") != sha256_object(row["guard_ir"]):
                raise ValueError("GUARD_AST_HASH_MISMATCH")
            if row.get("effect_ir_hash") != sha256_object(row["effect_ir"]):
                raise ValueError("EFFECT_IR_HASH_MISMATCH")
            template = compile_case_template(row["case_id"], bounds=bounds)
            compiled_guard = compile_source_guards(
                row["guard_ir"], static_guard_bindings=static_guard_bindings)
            compiled_effect = compile_effect_ir(
                row["effect_ir"], bounds=bounds, guard_ir=row["guard_ir"],
                static_effect_bindings=static_effect_bindings)
            source_precondition = "(and " + template.precondition[5:-1] + " " + compiled_guard.formula + ")"
            concrete_delta = compiled_effect.to_smt()
            # queue summary 是 concrete 与 reference 共同的 timing projection；
            # 从本次 EffectIR 编译结果复制到 r 域，避免重新按 case_id 猜测。
            queue_reference = []
            for equation in compiled_effect.queue_equations:
                if equation.startswith("(= c_queue_") and equation.endswith(")"):
                    left, right = equation[3:-1].split(" ", 1)
                    if right == left.removesuffix("_post"):
                        continue
                translated = equation.replace("c_", "r_")
                if translated not in queue_reference:
                    queue_reference.append(translated)
            projected_reference_delta = template.reference_delta
            if queue_reference:
                reference_body = template.reference_delta[5:]
                changed_queue_fields = set()
                for equation in queue_reference:
                    if equation.startswith("(= r_queue_"):
                        left = equation[3:].split(" ", 1)[0]
                        changed_queue_fields.add(left.removeprefix("r_").removesuffix("_post"))
                for field in changed_queue_fields:
                    reference_body = reference_body.replace(
                        f"(= r_{field}_post r_{field})", "")
                projected_reference_delta = "(and " + " ".join(queue_reference) + " " + reference_body
            proof = prove_smt2_case(
                case_id=row["case_id"], source_branch_id=row["path_id"],
                declarations=template.declarations, precondition=source_precondition,
                preservation=template.preservation, concrete_delta=concrete_delta,
                projected_reference_delta=projected_reference_delta,
                bound_source_hash=str(branch_map["source_hash"]),
                bounds=bounds,
            )
            required_components = set(require_case(row["case_id"])["required_relation_components"])
            effect_sources = " ".join(item.get("source", "") for item in row["effect_ir"])
            concrete_formula = concrete_delta
            affected_bound = ("job_key" not in required_components or
                              ("affected_job_key" in concrete_formula
                               and any(token in effect_sources for token in
                                       ("job", "task_name", "release_index", "running_job"))))
            # 这些不是调用方传入的 PASS 标志，而是从本次实际编译出的
            # post-state/frame 方程以及源码 effect IR 重新计算的门控事实。
            finite_fields = [f"c_{field}" for field in p0_smt_relation_fields(bounds)]
            frame_bound = (bool(row.get("effect_ir"))
                           and row.get("queue_relation_hash") == sha256_object(row.get("queue_relation", []))
                           and all(field in concrete_formula and f"{field}_post" in concrete_formula
                                  for field in finite_fields))
            if "(= is_degraded 1) (= expected_demand" in concrete_formula:
                demand_semantics = "MIN_ACTUAL_DEGRADED"
            elif row["case_id"] == "PRIMARY_LO_RELEASE":
                demand_semantics = "MIN_ACTUAL_B_PLUS_ONE"
            else:
                demand_semantics = "ACTUAL_COST"
            job_count_delta = sum(
                1 for item in row["effect_ir"]
                if "active_jobs.append" in item.get("source", "")
            )
            idle_precondition_bound = (
                row["case_id"] == "JUMP_TO_NEXT_EVENT"
                and "(= c_ready_empty 1)" in template.precondition
                and "(= next_event_time queue_min_time)" in template.precondition
            )
            proof = TransitionCaseProof(**{**proof.to_dict(),
                "source_branch_id": row["path_id"],
                "precondition_formula": source_precondition,
                "branch_subtree_hash": row["path_effect_hash"],
                "bridge_context_hash": bridge_context_hash,
                "case_template_hash": template.template_hash,
                "concrete_delta_source": "EFFECT_IR",
                "path_id": row["path_id"],
                "path_effect_hash": row["path_effect_hash"],
                "guard_ast_hash": row["guard_ast_hash"],
                "effect_ir_hash": row["effect_ir_hash"],
                "path_ast_hash": row["path_ast_hash"],
                "queue_relation_hash": row["queue_relation_hash"],
                "source_context_hash": str(branch_map["source_hash"]),
                "demand_semantics": demand_semantics,
                "job_count_delta": job_count_delta,
                "idle_precondition_bound": idle_precondition_bound,
                "affected_job_identity_bound": affected_bound,
                "frame_predicates_bound": frame_bound,
                "compiled_guard_hashes": compiled_guard.consumed_guard_hashes,
                "consumed_effect_hashes": compiled_effect.consumed_effect_hashes,
                "non_state_effect_hashes": compiled_effect.non_state_effect_hashes,
            })
            proofs.append(proof)
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "UNRESOLVED", "failure": "CASE_COMPILATION_FAILED", "message": str(exc)}
    from .transition_cases import check_handler_coverage
    coverage = check_handler_coverage([row["path_id"] for row in rows], proofs,
                                      require_all_p0_cases=True)
    status = "PASS" if coverage["status"] == "PASS" else ("UNRESOLVED" if coverage["unresolved_cases"] else "FAIL")
    proof_certificates = [obligation_certificate(
        obligation_id="P0_TRANSITION_CASE", status="PASS" if proof.z3_proof_result == "PASS" else "UNRESOLVED",
        context_hash=bridge_context_hash,
        inputs={"case_id": proof.case_id, "path_id": proof.path_id,
                "source_hash": proof.bound_source_hash, "template_hash": proof.case_template_hash,
                "concrete_delta_source": proof.concrete_delta_source,
                "guard_ast_hash": proof.guard_ast_hash,
                "effect_ir_hash": proof.effect_ir_hash,
                "compiled_guard_hashes": proof.compiled_guard_hashes,
                "consumed_effect_hashes": proof.consumed_effect_hashes,
                "path_ast_hash": proof.path_ast_hash,
                "static_guard_bindings_hash": static_guard_bindings_hash,
                "static_effect_bindings_hash": static_effect_bindings_hash,
                "queue_relation_hash": proof.queue_relation_hash},
        witness=proof.to_dict(), checker_id=__name__, checker_version="phase-k-v2",
        failure=None if proof.z3_proof_result == "PASS" else {"code": "Z3_CASE_UNRESOLVED"})
        for proof in proofs]
    return {"status": status, "proofs": [proof.to_dict() for proof in proofs],
            "proof_certificates": proof_certificates, "coverage": coverage,
            "branch_map_hash": branch_map["path_map_hash"], "source_hash": branch_map["source_hash"],
            "bridge_context_hash": bridge_context_hash,
            "state_relation_schema_hash": p0_state_relation_schema_hash(bounds),
            "model_bounds_hash": bounds.fingerprint,
            "static_guard_bindings": static_guard_bindings,
            "static_guard_bindings_hash": static_guard_bindings_hash,
            "static_effect_bindings": static_effect_bindings,
            "static_effect_bindings_hash": static_effect_bindings_hash}
