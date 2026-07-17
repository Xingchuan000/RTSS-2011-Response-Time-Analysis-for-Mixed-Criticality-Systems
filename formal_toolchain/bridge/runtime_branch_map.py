"""从真实 runtime handler 生成 Phase K 完整 transition-path map。

这里不把任意 ``if`` 节点当作 transition。path selector 是有限的人工维护
绑定，但每条 path 必须绑定实际 handler 的源码 hash、guard hash、effect
hash 和 terminal point；源码变化会使旧 map 失效。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.binding.python_cfg_ir import ExecutablePath, enumerate_function_paths
from .p0_case_manifest import p0_case_manifest_hash
from .transition_cases import REQUIRED_P0_CASE_IDS


def _has_guard(path: ExecutablePath, text: str, polarity: bool | None = None) -> bool:
    return any(text in guard.test_source and (polarity is None or guard.polarity is polarity)
               for guard in path.guards)


def _has_effect(path: ExecutablePath, text: str) -> bool:
    return any(text in effect.source for effect in path.effects)


def _guard_signature(path: ExecutablePath) -> tuple[tuple[str, bool], ...]:
    """返回源码 AST 顺序中的 guard 签名，供 semantic path 精确绑定。"""
    return tuple((guard.test_source, guard.polarity) for guard in path.guards)


def _has_guard_sequence(path: ExecutablePath, sequence: tuple[tuple[str, bool], ...]) -> bool:
    signature = _guard_signature(path)
    return any(signature[index:index + len(sequence)] == sequence
               for index in range(len(signature) - len(sequence) + 1))


def _semantic_path_predicate(case_id: str):
    """为 18 个正式 case 返回基于当前源码 CFG 的唯一 predicate。"""
    def exact(*items: tuple[str, bool]):
        return lambda path: _has_guard_sequence(path, tuple(items))

    release_mode = "release_mode is None"
    task_hi = "task.criticality is Criticality.HI"
    release_hi = "release_mode is SystemMode.HI"
    c_amc = "_is_c_amc_semantics(self.config.semantics)"
    response = "_is_response_based_semantics(self.config.semantics) and task.criticality is Criticality.HI"
    completion = "event.event_type is EventType.JOB_COMPLETION"
    completion_common = ((completion, True),
                         ("self.state.valid_completion_tokens.get(key) != event.token", False),
                         ("job is None or self.state.running_job is not job", False),
                         ("self.monitor is not None", False))
    deadline = "event.event_type is EventType.DEADLINE_CHECK"
    overrun = "event.event_type is EventType.BUDGET_OVERRUN"
    if case_id == "PRIMARY_LO_RELEASE":
        return exact((release_mode, False), (task_hi, False), (release_hi, False),
                     (response, False), (response, False))
    if case_id == "DEGRADED_LO_RELEASE":
        return exact((release_mode, False), (task_hi, False), (release_hi, True),
                     (c_amc, True), (response, False), (response, False))
    if case_id == "HI_RELEASE":
        # C-AMC-sem 不是 response based；两个 response-expiry guard 都必须
        # 走 false，并且必须消费真实 C-AMC-sem guard，确保 raw
        # path 不包含 AMC-RA/AMC-RH 的 expiry scheduling effect。
        return exact((release_mode, False), (task_hi, True), (c_amc, True),
                     (response, False), (response, False))
    if case_id == "PREEMPTION_DISPATCH":
        return exact(("selected is state.running_job and (not force)", False),
                     ("state.running_job is not None", True),
                     ("selected is None", False),
                     ("selected_key not in state.started_jobs", False))
    if case_id == "ONE_SERVICE_TICK":
        return exact(("now < old_time", False), ("self.config.capture_trace", False))
    if case_id in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION"}:
        return exact(*completion_common, ("job in self.state.active_jobs", True),
                     ("self.config.semantics is RuntimeSemantics.AMC_RH and completed_job_was_hi", False))
    if case_id == "HI_COMPLETION":
        return exact(*completion_common, ("job in self.state.active_jobs", True),
                     ("self.config.semantics is RuntimeSemantics.AMC_RH and completed_job_was_hi", True))
    if case_id == "PRIMARY_LO_CANCELLATION":
        return exact((overrun, True),
                     ("self.state.valid_overrun_tokens.get(key) != event.token", False),
                     ("job is None or self.state.running_job is not job", False),
                     ("_is_response_based_semantics(self.config.semantics) and job.task.criticality is Criticality.HI", False),
                     ("budget is None", False), ("job.executed_time <= budget", False),
                     ("self.config.semantics in {RuntimeSemantics.AMC_PLUS, RuntimeSemantics.AMC_RA, RuntimeSemantics.AMC_RH, RuntimeSemantics.C_AMC_SEM} and job.task.criticality is Criticality.LO", True),
                     ("self.monitor is not None", True), ("job in self.state.active_jobs", True))
    if case_id == "DEADLINE_OBSERVATION_NO_MISS":
        return exact((deadline, True), ("job is None", False), ("not job.finished()", False))
    if case_id == "DEADLINE_OBSERVATION_FIRST_HI_MISS":
        return exact((deadline, True), ("job is None", False), ("not job.finished()", True),
                     ("self.config.stop_at_first_miss", False))
    if case_id == "IDLE_RECOVERY":
        return exact(("not _uses_idle_recovery(cfg.semantics)", False),
                     ("state.mode is SystemMode.HI and (not state.active_jobs) and (state.running_job is None)", True))
    if case_id in {"CONTROLLER_NO_ACTION", "CONTROLLER_SELECTED_ACTION"}:
        return exact(("event.event_type is EventType.BUDGET_UPDATE", True))
    if case_id == "JUMP_TO_NEXT_EVENT":
        return lambda path: (_has_guard(path, "self.state.current_time < target_time", True)
                             and _has_effect(path, "self._advance_time(target_time)"))
    if case_id == "ARRIVAL_BATCH_NO_SWITCH":
        return lambda path: not path.guards and _has_effect(path, "self._reschedule(now)")
    if case_id == "ARRIVAL_BATCH_SWITCH_S0":
        return exact(("not _is_c_amc_semantics(self.config.semantics)", False),
                     ("self.state.mode is not SystemMode.LO", False),
                     ("not abnormal_arrivals", False))
    if case_id == "BOOT_TO_PRECLOSED_0":
        return lambda path: (not any(g.polarity for g in path.guards)
                             and _has_effect(path, "engine.queue.push"))
    return lambda path: False


def select_unique_path(source_root: str | Path, spec: "PathSpec") -> ExecutablePath:
    """只返回满足完整 guard/effect predicate 的真实可执行路径。"""
    candidates = [path for path in enumerate_function_paths(source_root, spec.entry_function)
                  if spec.predicate(path)]
    if len(candidates) != 1:
        raise ValueError(f"PATH_NOT_UNIQUE:{spec.semantic_path_id}:candidate_count={len(candidates)}")
    return candidates[0]


class PathSpec:
    def __init__(self, semantic_path_id: str, entry_function: str, case_id: str,
                 predicate: Any) -> None:
        self.semantic_path_id = semantic_path_id
        self.entry_function = entry_function
        self.case_id = case_id
        self.predicate = predicate


# selector 只选择完整 handler path；它不直接把某个 if 节点当 transition。
PATH_SPECS = (
    ("boot/preclosed_0", "EventRuntimeEngine.build", "BOOT_TO_PRECLOSED_0", "initial_state_and_queue", ("boot", "budget_frame")),
    ("arrival_batch/no_switch", "EventRuntimeEngine._process_job_arrival_batch", "ARRIVAL_BATCH_NO_SWITCH", "not switched_by_c_amc_sem_batch", ("arrival_batch", "release", "reschedule")),
    ("arrival_batch/switch_s0", "EventRuntimeEngine._maybe_enter_c_amc_sem_hi_mode_at_arrival", "ARRIVAL_BATCH_SWITCH_S0", "abnormal_arrivals and mode_is_LO", ("mode=HI", "mode_switch", "same_batch")),
    ("release/primary_lo", "EventRuntimeEngine._process_single_arrival_in_priority_order", "PRIMARY_LO_RELEASE", "release_mode=LO and task_is_LO", ("build_job", "release_fixed_budget", "queue_release", "deadline_schedule", "active_add", "ready_add")),
    ("release/degraded_lo", "EventRuntimeEngine._process_single_arrival_in_priority_order", "DEGRADED_LO_RELEASE", "release_mode=HI and task_is_LO and c_amc_sem", ("degraded_budget", "actual_cost_clamp", "queue_release", "deadline_schedule", "active_add", "ready_add")),
    ("release/hi", "EventRuntimeEngine._process_single_arrival_in_priority_order", "HI_RELEASE", "task_is_HI", ("build_job", "release_fixed_budget", "queue_release", "deadline_schedule", "active_add", "ready_add")),
    ("dispatch/preempt", "_reschedule", "PREEMPTION_DISPATCH", "selected_is_not_previous_or_force", ("highest_priority_select", "running_update", "preempt_invalidate")),
    ("service/one_tick", "EventRuntimeEngine._advance_time", "ONE_SERVICE_TICK", "event_before_boundary", ("advance_time", "service_accounting", "remaining_update")),
    ("completion/normal", "EventRuntimeEngine._process_event", "NORMAL_COMPLETION", "event_is_JOB_COMPLETION and job_is_normal", ("executed_to_actual", "active_remove", "running_clear", "recovery_event", "reschedule")),
    ("completion/degraded", "EventRuntimeEngine._process_event", "DEGRADED_COMPLETION", "event_is_JOB_COMPLETION and job_is_degraded", ("executed_to_actual", "active_remove", "running_clear", "recovery_event", "reschedule")),
    ("completion/hi", "EventRuntimeEngine._process_event", "HI_COMPLETION", "event_is_JOB_COMPLETION and job_is_HI", ("executed_to_actual", "active_remove", "running_clear", "hi_complete", "recovery_event", "reschedule")),
    ("cancellation/primary_lo", "EventRuntimeEngine._process_event", "PRIMARY_LO_CANCELLATION", "lo_budget_overrun and c_amc_sem", ("active_remove", "running_clear", "cancellation_event", "recovery_event", "reschedule")),
    ("deadline/no_miss", "EventRuntimeEngine._process_event", "DEADLINE_OBSERVATION_NO_MISS", "job_finished", ()),
    ("deadline/first_hi_miss", "EventRuntimeEngine._process_event", "DEADLINE_OBSERVATION_FIRST_HI_MISS", "not job_finished and task_is_HI", ("hi_miss_flag", "deadline_observe_only")),
    ("recovery/idle", "_maybe_recover_to_lo", "IDLE_RECOVERY", "uses_idle_recovery and HI and quiescent", ("mode=LO", "recovery_event")),
    ("controller/no_action", "EventRuntimeEngine._process_event", "CONTROLLER_NO_ACTION", "event_is_BUDGET_UPDATE and no_updates", ("budget_frame", "event_projection", "reschedule")),
    ("controller/selected_action", "EventRuntimeEngine._process_event", "CONTROLLER_SELECTED_ACTION", "event_is_BUDGET_UPDATE and updates", ("future_budget_update", "event_projection", "reschedule")),
    ("time/jump_to_next_event", "EventRuntimeEngine.run_until", "JUMP_TO_NEXT_EVENT", "ready_empty and next_event_exists", ("next_event_min", "time_jump", "no_service")),
)


def _handler_source(root: Path, qualified: str) -> tuple[str, int, int]:
    module, name = qualified.split(".", 1) if "." in qualified else ("", qualified)
    if qualified.startswith("EventRuntimeEngine."):
        path = root / "amc_py/event_runtime.py"
        class_name, fn_name = qualified.split(".", 1)
    else:
        path = root / "amc_py/event_runtime.py"
        class_name, fn_name = None, qualified
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == fn_name:
            if class_name is None or any(isinstance(parent, ast.ClassDef) and parent.name == class_name for parent in []):
                node = candidate
                break
    if node is None and class_name is not None:
        for cls in [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name]:
            node = next((x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == fn_name), None)
    if node is None:
        raise ValueError(f"runtime handler 不存在: {qualified}")
    return ast.unparse(node), int(node.lineno), int(getattr(node, "end_lineno", node.lineno))


def _path_row(root: Path, spec: tuple[Any, ...]) -> dict[str, Any]:
    path_id, handler, case_id, guard, effects = spec
    source, _handler_start, _handler_end = _handler_source(root, handler)
    evidence = {
        "mode=HI": ("SystemMode.HI",), "mode=LO": ("SystemMode.LO",),
        "mode_switch": ("mode_switches.append",), "same_batch": ("abnormal_arrivals",),
        "arrival_batch": ("pop_all_matching",), "release": ("_process_single_arrival_in_priority_order",),
        "reschedule": ("self._reschedule",), "build_job": ("_build_job",),
        "release_fixed_budget": ("runtime_budget_at_release",),
        "degraded_budget": ("_c_amc_sem_degraded_lo_budget",),
        "actual_cost_clamp": ("actual_cost_override",), "active_add": ("active_jobs.append",),
        "ready_add": ("active_jobs.append",), "highest_priority_select": ("_select_highest_priority_ready_job",),
        "running_update": ("running_job",), "preempt_invalidate": ("_invalidate_job_events",),
        "advance_time": ("_update_running_progress", "current_time"), "service_accounting": ("_update_running_progress",),
        "remaining_update": ("executed_time",), "executed_to_actual": ("executed_time = job.actual_cost",),
        "active_remove": ("active_jobs.remove",), "running_clear": ("running_job = None",),
        "hi_complete": ("completed_job_was_hi",), "cancellation_event": ("job_cancellations.append",),
        "deadline_observe_only": ("deadline_misses",), "hi_miss_flag": ("deadline_misses.append",),
        "recovery_event": ("_maybe_recover_to_lo", "_maybe_recover_rh_to_lo", "mode_recoveries.append"), "budget_frame": ("budget_state",),
        "queue_release": ("_schedule_next_release",), "deadline_schedule": ("EventType.DEADLINE_CHECK",),
        "event_projection": ("_append_debug_event",), "future_budget_update": ("apply_updates",),
        "next_event_min": ("queue",), "time_jump": ("_advance_time", "target_time"), "no_service": ("event.time >= target_time",),
        "boot": ("runtime_budgets",),
    }
    def predicate(path: ExecutablePath) -> bool:
        effect_text = "\n".join(item.source for item in path.effects)
        required_effects = tuple(patterns for effect in effects
                                 for patterns in evidence.get(effect, ())
                                 if effect not in {"release", "reschedule", "recovery_event",
                                                    "budget_frame", "event_projection", "no_service"})
        return (_semantic_path_predicate(case_id)(path)
                and all(pattern in effect_text for pattern in required_effects))

    path = select_unique_path(root, PathSpec(path_id, handler, case_id, predicate))
    guard_ir = [item.to_dict() for item in path.guards]
    effect_ir = [item.to_dict() for item in path.effects]
    # 证据匹配只允许落在已经定位到源码区间的 AST statement 上；不再对
    # 任意整段文本做关键词搜索，避免同名字符串来自另一条分支时误绑定。
    effect_sources = tuple(item.get("source", "") for item in effect_ir)
    for effect in effects:
        patterns = evidence.get(effect)
        if patterns is None or not any(pattern in source for pattern in patterns for source in effect_sources):
            raise ValueError(f"effect {effect} 无真实 AST statement 证据: {path_id}")
    handler_hash = sha256_object({"handler": handler, "source": source})
    queue_relation = [item for item in effect_ir
                      if any(token in item.get("source", "") for token in
                             ("queue", "valid_", "_schedule", "Event(", "token"))]
    path_effect_hash = sha256_object({"handler_hash": handler_hash, "path_ast_hash": path.path_id,
                                      "guard_ir": guard_ir, "effect_ir": effect_ir,
                                      "effects": effects})
    return {"path_id": path_id, "handler": handler, "case_id": case_id,
            "source_file": "amc_py/event_runtime.py",
            # 对外暴露的 guard 不再是 PATH_SPECS 中的人工描述，而是
            # extractor 返回的实际 AST guard 集合；PATH_SPECS 只负责选择
            # 要审计的源码区间。
            "guard": guard_ir, "guard_ir": guard_ir,
            "guard_hash": sha256_object(guard_ir), "guard_ast_hash": sha256_object(guard_ir),
            "effects": list(effects), "path_effect_hash": path_effect_hash,
            "effect_ir": effect_ir, "effect_ir_hash": sha256_object(effect_ir),
            "queue_relation": queue_relation, "queue_relation_hash": sha256_object(queue_relation),
            "path_ast_hash": path.path_id,
            "handler_hash": handler_hash, "terminal": path.terminal}


def build_runtime_branch_map(source_root: str | Path, *, source_hash: str,
                             path_map: Mapping[str, Any] | None = None,
                             case_by_branch: Mapping[str, str] | None = None,
                             schema_ir: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(source_root)
    actual_source_hash = build_source_manifest(root)["semantic_hash"]
    if source_hash != actual_source_hash:
        return {"status": "FAIL", "failure": "SOURCE_HASH_MISMATCH", "expected": actual_source_hash, "provided": source_hash}
    event = bind_event_runtime(root)
    removal = bind_removal_runtime(root)
    if event.get("status") != "PASS" or removal.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "RUNTIME_AST_BINDING_INCOMPLETE"}
    if path_map is None:
        # 兼容旧 API，但不再把 if-node mapping 当成正式路径证明。
        return {"status": "UNRESOLVED", "failure": "COMPLETE_TRANSITION_PATH_MAP_REQUIRED"}
    if path_map.get("schema_version") != "phase_k_transition_path_map_v2_cfg_ir":
        return {"status": "UNRESOLVED", "failure": "TRANSITION_PATH_MAP_SCHEMA_VERSION_REQUIRED"}
    if not isinstance(path_map.get("paths"), Mapping):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_PATH_MAP_SCHEMA_INVALID"}
    expected = {spec[0]: spec for spec in PATH_SPECS}
    if set(path_map["paths"]) != set(expected):
        return {"status": "FAIL", "failure": "TRANSITION_PATH_SET_MISMATCH",
                "missing": sorted(set(expected) - set(path_map["paths"])),
                "unknown": sorted(set(path_map["paths"]) - set(expected))}
    rows = []
    for path_id, spec in expected.items():
        actual = _path_row(root, spec)
        supplied = path_map["paths"][path_id]
        # ``guard`` 只是 ``guard_ir`` 的历史镜像字段；实际语义边界由
        # ``guard_ir`` / hash / effect 这组字段共同约束。重复比较镜像字段
        # 会让不同执行环境的 JSON 规范化细节误判为 stale，因此这里只保留
        # 真正参与证明绑定的字段。
        for field in ("case_id", "handler", "source_file",
                      "guard_ir", "guard_hash", "guard_ast_hash", "effects", "effect_ir",
                      "effect_ir_hash", "path_ast_hash", "queue_relation", "queue_relation_hash",
                      "path_effect_hash", "handler_hash", "terminal"):
            if supplied.get(field) != actual[field]:
                return {"status": "UNRESOLVED", "failure": "TRANSITION_PATH_BINDING_STALE", "path_id": path_id, "field": field}
        rows.append(actual)
    if {row["case_id"] for row in rows} != set(REQUIRED_P0_CASE_IDS):
        return {"status": "UNRESOLVED", "failure": "P0_CASE_PATH_COVERAGE_INCOMPLETE"}
    coverage = build_normal_runtime_path_coverage(root)
    if coverage.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "NORMAL_RUNTIME_PATH_COVERAGE_INCOMPLETE",
                "coverage": coverage}
    return {"status": "PASS", "source_hash": source_hash, "path_count": len(rows),
            "paths": rows, "coverage": coverage, "case_manifest_hash": p0_case_manifest_hash(),
            "path_map_hash": sha256_object({"paths": rows, "coverage": coverage["artifact_hash"]})}


def build_normal_runtime_path_coverage(root: Path) -> dict[str, Any]:
    """检查正常 runtime 中会消费的 no-op、queue 和 controller 路径。

    这些路径不都改变 P0 macro-state，但不能从 coverage 中省略；每个条目
    都绑定到实际源码片段哈希，后续由 closure certificate 消费。
    """
    runtime_path = root / "amc_py/event_runtime.py"
    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"), filename=str(runtime_path))
    wrapper_path = root / "amc_py/rl/runtime_wrapper.py"
    wrapper_tree = ast.parse(wrapper_path.read_text(encoding="utf-8"), filename=str(wrapper_path))

    def function_node(qualified: str) -> ast.FunctionDef | None:
        if "." in qualified:
            class_name, function_name = qualified.split(".", 1)
            for item in runtime_tree.body:
                if isinstance(item, ast.ClassDef) and item.name == class_name:
                    return next((child for child in item.body
                                 if isinstance(child, ast.FunctionDef)
                                 and child.name == function_name), None)
            return None
        return next((item for item in runtime_tree.body
                     if isinstance(item, ast.FunctionDef) and item.name == qualified), None)

    functions = {name: function_node(name) for name in (
        "_reschedule", "_schedule_running_job_events",
        "EventRuntimeEngine._reschedule", "EventRuntimeEngine._process_event",
        "EventRuntimeEngine._process_single_arrival_in_priority_order",
        "EventRuntimeEngine._schedule_running_job_events",
        "EventRuntimeEngine._maybe_recover_to_lo")}

    def has_if_test(function_name: str, predicate) -> bool:
        function = functions.get(function_name)
        return function is not None and any(
            isinstance(node, ast.If) and predicate(node.test) for node in ast.walk(function))

    def has_statement(function_name: str, predicate) -> bool:
        function = functions.get(function_name)
        return function is not None and any(predicate(node) for node in ast.walk(function))

    def source_hash(function_name: str) -> str:
        function = functions.get(function_name)
        return sha256_object({"function": function_name,
                              "ast": ast.dump(function, include_attributes=False) if function else None})

    required = {
        "dispatch_keep_same": ("_reschedule",
                                lambda: has_if_test("_reschedule",
                                                    lambda test: "selected is state.running_job" in ast.unparse(test))),
        "dispatch_to_idle": ("_reschedule",
                              lambda: has_if_test("_reschedule",
                                                  lambda test: "selected is None" in ast.unparse(test))),
        "stale_completion_token": ("EventRuntimeEngine._process_event",
                                    lambda: has_statement("EventRuntimeEngine._process_event",
                                                          lambda node: isinstance(node, ast.Call)
                                                          and ".valid_completion_tokens.get" in ast.unparse(node.func))),
        "stale_overrun_token": ("EventRuntimeEngine._process_event",
                                 lambda: has_statement("EventRuntimeEngine._process_event",
                                                       lambda node: isinstance(node, ast.Call)
                                                       and ".valid_overrun_tokens.get" in ast.unparse(node.func))),
        "missing_or_not_running_job": ("EventRuntimeEngine._process_event",
                                         lambda: has_if_test("EventRuntimeEngine._process_event",
                                                             lambda test: "job is None or self.state.running_job is not job" in ast.unparse(test))),
        "budget_not_yet_overrun": ("EventRuntimeEngine._process_event",
                                    lambda: has_if_test("EventRuntimeEngine._process_event",
                                                        lambda test: "job.executed_time <= budget" in ast.unparse(test))),
        "deadline_job_missing": ("EventRuntimeEngine._process_event",
                                  lambda: has_if_test("EventRuntimeEngine._process_event",
                                                      lambda test: ast.unparse(test) == "job is None")),
        "deadline_finished_observation": ("EventRuntimeEngine._process_event",
                                           lambda: has_if_test("EventRuntimeEngine._process_event",
                                                               lambda test: ast.unparse(test) == "not job.finished()")),
        "lo_deadline_observation": ("EventRuntimeEngine._process_event",
                                     lambda: has_statement("EventRuntimeEngine._process_event",
                                                           lambda node: isinstance(node, ast.Attribute)
                                                           and node.attr == "criticality")),
        "schedule_next_release": ("EventRuntimeEngine._process_single_arrival_in_priority_order",
                                   lambda: has_statement("EventRuntimeEngine._process_single_arrival_in_priority_order",
                                                         lambda node: isinstance(node, ast.Call)
                                                         and ast.unparse(node.func).endswith("_schedule_next_release"))),
        "schedule_deadline": ("EventRuntimeEngine._process_single_arrival_in_priority_order",
                               lambda: has_statement("EventRuntimeEngine._process_single_arrival_in_priority_order",
                                                     lambda node: isinstance(node, ast.Attribute)
                                                     and node.attr == "DEADLINE_CHECK")),
        "token_invalidation": ("EventRuntimeEngine._process_event",
                                lambda: has_statement("EventRuntimeEngine._process_event",
                                                      lambda node: isinstance(node, ast.Call)
                                                      and ast.unparse(node.func).endswith("_invalidate_job_events"))),
        "completion_schedule": ("_schedule_running_job_events",
                                 lambda: has_statement("_schedule_running_job_events",
                                                       lambda node: isinstance(node, ast.Attribute)
                                                       and node.attr == "JOB_COMPLETION")),
        "overrun_schedule": ("_schedule_running_job_events",
                              lambda: has_statement("_schedule_running_job_events",
                                                    lambda node: isinstance(node, ast.Attribute)
                                                    and node.attr == "BUDGET_OVERRUN")),
        "response_expiry_config_branch": ("EventRuntimeEngine._process_single_arrival_in_priority_order",
                                           lambda: has_if_test("EventRuntimeEngine._process_single_arrival_in_priority_order",
                                                               lambda test: "_is_response_based_semantics" in ast.unparse(test))),
    }
    records = []
    missing = []
    for path_id, (function_name, checker) in required.items():
        present = bool(checker())
        if not present:
            missing.append(path_id)
        records.append({"path_id": path_id, "entry_function": function_name,
                        "present": present, "source_hash": source_hash(function_name)})
    controller_calls = [ast.unparse(node) for node in ast.walk(wrapper_tree)
                        if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == "engine.apply_budget_updates"]
    apply_present = bool(controller_calls)
    if not apply_present:
        missing.append("apply_budget_updates")
    records.append({"path_id": "apply_budget_updates",
                    "entry_function": "runtime_wrapper",
                    "present": apply_present,
                    "source_hash": sha256_object({"calls": controller_calls})})
    result = {"status": "PASS" if not missing else "UNRESOLVED",
              "schema_version": "normal_runtime_path_coverage_v1",
              "paths": records, "missing": missing,
              "unreachable": {"response_expiry": {"kind": "EFFECTIVE_CONFIG_UNREACHABLE",
                                                     "condition": "effective_config_non_response_based",
                                                     "source_hash": source_hash("EventRuntimeEngine._process_event")}}}
    result["artifact_hash"] = sha256_object(result)
    return result
