"""实际部署 controller→EventRuntimeEngine.apply_budget_updates 绑定。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from formal_toolchain.binding.python_ast_ir import function_to_ir
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.semantics.frozen_runtime_contract import frozen_event_runtime_path, frozen_runtime_wrapper_path, CONTRACT_VERSION


def _class_method(source: str, class_name: str, method_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return next(
                (item for item in node.body
                 if isinstance(item, ast.FunctionDef) and item.name == method_name),
                None,
            )
    return None


def _top_level_function(source: str, function_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    return next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == function_name),
        None,
    )


def _find_if(function: ast.FunctionDef, predicate) -> ast.If | None:
    matches = [node for node in ast.walk(function)
               if isinstance(node, ast.If) and predicate(ast.unparse(node.test))]
    return matches[0] if len(matches) == 1 else None


def _calls(statements: list[ast.stmt]) -> tuple[str, ...]:
    result = []
    for statement in statements:
        result.extend(
            ast.unparse(node.func)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        )
    return tuple(result)


def _writes(statements: list[ast.stmt]) -> tuple[str, ...]:
    result: list[str] = []
    for statement in statements:
        for node in ast.walk(statement):
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                result.append(ast.unparse(target))
    return tuple(result)


def _statement_sources(statements: list[ast.stmt]) -> tuple[str, ...]:
    return tuple(ast.unparse(statement) for statement in statements)


def _has_assignment(statements: list[ast.stmt], target: str, value: str) -> bool:
    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(ast.unparse(item) == target for item in targets):
            actual = ast.unparse(statement.value) if statement.value is not None else ""
            if actual == value:
                return True
    return False


def _analyze_env_explicit_noop(source: str) -> dict[str, Any]:
    """Bind the production ``AmcBudgetEnv.step`` explicit/fallback noop microstep.

    The proof boundary ends before the later ``run_until`` plant-progress call.
    This is intentionally a narrow structural binding: unrelated mutable-RL code
    may evolve, but any write/call added to the noop branch fails closed.
    """

    function = _class_method(source, "AmcBudgetEnv", "step")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "AMC_ENV_STEP_NOT_FOUND"}}

    action_guard = _find_if(function, lambda text: text == "action_id is not None")
    explicit_guard = _find_if(function, lambda text: text == "action.is_noop")
    if action_guard is None or explicit_guard is None:
        return {"status": "FAIL", "failure": {"code": "EXPLICIT_NOOP_BRANCH_NOT_UNIQUE"}}
    if not (action_guard.lineno <= explicit_guard.lineno <= getattr(action_guard, "end_lineno", action_guard.lineno)):
        return {"status": "FAIL", "failure": {"code": "EXPLICIT_NOOP_BRANCH_OUTSIDE_ACTION_GUARD"}}

    explicit_body = list(explicit_guard.body)
    fallback_body = list(action_guard.orelse)
    explicit_calls = _calls(explicit_body)
    fallback_calls = _calls(fallback_body)
    explicit_writes = _writes(explicit_body)
    fallback_writes = _writes(fallback_body)

    # Fail closed on helper calls/writes, not only on known engine method names:
    # otherwise a future ``self._mutate_*()`` helper could silently bypass the
    # stutter proof.  The current noop branch only needs ``dict`` plus one
    # controller-private statistic write; the implicit fallback only needs int.
    allowed_explicit_calls = {"dict"}
    allowed_fallback_calls = {"int"}
    forbidden_explicit_calls = sorted(set(explicit_calls) - allowed_explicit_calls)
    forbidden_fallback_calls = sorted(set(fallback_calls) - allowed_fallback_calls)
    allowed_explicit_writes = {
        "accepted", "is_explicit_noop_action", "updates", "candidate_budgets",
        "self._selected_explicit_noop_actions",
    }
    allowed_fallback_writes = {"valid_action_count", "masked_action_count"}
    unexpected_explicit_writes = sorted(set(explicit_writes) - allowed_explicit_writes)
    unexpected_fallback_writes = sorted(set(fallback_writes) - allowed_fallback_writes)
    forbidden_explicit_writes = sorted(target for target in unexpected_explicit_writes
                                        if target.startswith("self._engine"))
    forbidden_fallback_writes = sorted(target for target in unexpected_fallback_writes
                                        if target.startswith("self._engine"))

    assignments_ok = all((
        _has_assignment(explicit_body, "accepted", "True"),
        _has_assignment(explicit_body, "is_explicit_noop_action", "True"),
        _has_assignment(explicit_body, "updates", "{}"),
        _has_assignment(explicit_body, "candidate_budgets", "dict(budget_before)"),
    ))

    run_until_calls = [node for node in ast.walk(function)
                       if isinstance(node, ast.Call)
                       and "run_until" in ast.unparse(node.func)]
    plant_progress_separated = (
        len(run_until_calls) == 1
        and run_until_calls[0].lineno > getattr(action_guard, "end_lineno", action_guard.lineno)
    )

    explicit_clean = not forbidden_explicit_calls and not unexpected_explicit_writes
    fallback_clean = not forbidden_fallback_calls and not unexpected_fallback_writes
    ok = assignments_ok and explicit_clean and fallback_clean and plant_progress_separated
    branch_payload = {
        "action_guard": ast.dump(action_guard, include_attributes=False),
        "explicit_noop_guard": ast.dump(explicit_guard, include_attributes=False),
        "run_until": ast.dump(run_until_calls[0], include_attributes=False) if len(run_until_calls) == 1 else None,
    }
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "EXPLICIT_NOOP_RUNTIME_STUTTER_BINDING_FAILED"},
        "source": "amc_py/rl/env.py:AmcBudgetEnv.step",
        "explicit_guard": ast.unparse(explicit_guard.test),
        "implicit_fallback_guard": "action_id is None",
        "assignments_verified": assignments_ok,
        "budget_identity": assignments_ok and not forbidden_explicit_calls,
        "apply_budget_updates_skipped": not any("apply_budget_updates" in call for call in explicit_calls),
        "engine_state_write_free": not forbidden_explicit_writes,
        "runtime_handler_call_free": not forbidden_explicit_calls,
        "controller_private_writes": sorted(target for target in explicit_writes if target.startswith("self.")),
        "unexpected_controller_writes": unexpected_explicit_writes,
        "unexpected_fallback_writes": unexpected_fallback_writes,
        "implicit_fallback_engine_write_free": not forbidden_fallback_writes,
        "implicit_fallback_runtime_handler_call_free": not forbidden_fallback_calls,
        "explicit_and_fallback_same_timing_semantics": explicit_clean and fallback_clean,
        "plant_progress_separated": plant_progress_separated,
        "timing_projection": "STUTTER" if ok else "UNRESOLVED",
        "running_job_unchanged": explicit_clean,
        "effective_event_frontier_unchanged": explicit_clean,
        "released_job_fields_unchanged": explicit_clean,
        "mode_unchanged": explicit_clean,
        "time_unchanged": explicit_clean,
        "controller_time_unchanged": explicit_clean,
        "explicit_branch_source": list(_statement_sources(explicit_body)),
        "fallback_branch_source": list(_statement_sources(fallback_body)),
        "forbidden_explicit_calls": forbidden_explicit_calls,
        "forbidden_fallback_calls": forbidden_fallback_calls,
        "branch_binding_hash": sha256_object(branch_payload),
    }


def _analyze_frozen_wrapper_noop(source: str) -> dict[str, Any]:
    function = _top_level_function(source, "simulate_ordered_taskset_with_agent")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "FROZEN_CONTROLLER_WRAPPER_NOT_FOUND"}}
    noop_guard = _find_if(
        function,
        lambda text: "action is None" in text and "is_noop" in text,
    )
    if noop_guard is None:
        return {"status": "FAIL", "failure": {"code": "FROZEN_NOOP_BRANCH_NOT_UNIQUE"}}
    body = list(noop_guard.body)
    calls = _calls(body)
    allowed_calls = {"dict", "action_log.append"}
    forbidden = sorted(set(calls) - allowed_calls)
    source_text = "\n".join(_statement_sources(body))
    required_tokens = (
        "budget_snapshot = dict(engine.runtime_budgets.budgets)",
        "'updates': {}",
        "'budget_before': budget_snapshot",
        "'budget_after': budget_snapshot",
    )
    required_ok = all(token in source_text for token in required_tokens)
    ok = required_ok and not forbidden
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "FROZEN_NOOP_STUTTER_BINDING_FAILED"},
        "source": "formal_toolchain/semantics/frozen_c_amc_sem_runtime_wrapper.py:simulate_ordered_taskset_with_agent",
        "guard": ast.unparse(noop_guard.test),
        "budget_identity": required_ok,
        "apply_budget_updates_skipped": not any("apply_budget_updates" in call for call in calls),
        "runtime_handler_call_free": not forbidden,
        "explicit_and_fallback_same_branch": True,
        "timing_projection": "STUTTER" if ok else "UNRESOLVED",
        "forbidden_calls": forbidden,
        "branch_binding_hash": sha256_object(ast.dump(noop_guard, include_attributes=False)),
    }


def _analyze_frozen_wrapper_selected_action(source: str) -> dict[str, Any]:
    """Bind the synchronous normal-action branch separately from plant run_until."""
    function = _top_level_function(source, "simulate_ordered_taskset_with_agent")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "FROZEN_CONTROLLER_WRAPPER_NOT_FOUND"}}
    guards = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "action is None or bool(getattr(action, 'is_noop', False))"
    ]
    if len(guards) != 1:
        return {"status": "FAIL", "failure": {"code": "SELECTED_ACTION_BRANCH_NOT_UNIQUE"}}
    guard = guards[0]
    calls = _calls(list(guard.orelse))
    branch_source = _statement_sources(list(guard.orelse))
    apply_calls = [call for call in calls if call == "engine.apply_budget_updates"]
    plant_calls = [call for call in calls if "run_until" in call]
    ok = len(apply_calls) == 1 and not plant_calls
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "SELECTED_ACTION_RUNTIME_BINDING_FAILED"},
        "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "engine.apply_budget_updates",
        "synchronous_budget_update_call": len(apply_calls) == 1,
        "plant_progression_in_branch": bool(plant_calls),
        "plant_progression_separated": not plant_calls,
        "timing_projection": "STUTTER" if ok else "UNRESOLVED",
        "zero_time": True,
        "branch_source": list(branch_source),
        "branch_binding_hash": sha256_object({
            "guard": ast.dump(guard, include_attributes=False),
            "branch_source": branch_source,
        }),
    }


def _analyze_env_selected_action(source: str) -> dict[str, Any]:
    """Bind the normal single-action branch of the deployed environment.

    This is the C1 source boundary.  It establishes only the synchronous
    candidate-evaluation/commit shape; the timing and state-preservation facts
    are added by the later controller-update certificate.
    """

    function = _class_method(source, "AmcBudgetEnv", "step")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "AMC_ENV_STEP_NOT_FOUND"}}

    action_guards = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "action_id is not None"
    ]
    if len(action_guards) != 1:
        return {"status": "FAIL", "failure": {"code": "ACTION_GUARD_NOT_UNIQUE"}}
    action_guard = action_guards[0]

    action_assignments = [
        statement for statement in action_guard.body
        if isinstance(statement, ast.Assign)
        and any(ast.unparse(target) == "action" for target in statement.targets)
        and ast.unparse(statement.value) == "self._actions[action_id]"
    ]
    if len(action_assignments) != 1:
        return {"status": "FAIL", "failure": {"code": "ACTION_LOOKUP_NOT_UNIQUE"}}

    chain = next(
        (statement for statement in action_guard.body
         if isinstance(statement, ast.If) and ast.unparse(statement.test) == "action.is_noop"),
        None,
    )
    if chain is None:
        return {"status": "FAIL", "failure": {"code": "ACTION_BRANCH_CHAIN_MISSING"}}

    branch_tests: list[str] = []
    current = chain
    while isinstance(current, ast.If):
        branch_tests.append(ast.unparse(current.test))
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        normal_body = list(current.orelse)
        break
    else:
        return {"status": "FAIL", "failure": {"code": "NORMAL_ACTION_BRANCH_MISSING"}}

    expected_branch_tests = [
        "action.is_noop",
        "action.is_constraint_guided_pair",
        "action.is_residual_ranked",
    ]
    if branch_tests != expected_branch_tests:
        return {
            "status": "FAIL",
            "failure": {"code": "SINGLE_ACTION_BRANCH_PROFILE_MISMATCH"},
            "branch_tests": branch_tests,
        }

    calls = _calls(normal_body)
    evaluation_calls = [
        node for statement in normal_body for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self.evaluate_budget_candidate"
    ]
    apply_calls = [
        node for statement in normal_body for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._engine.apply_budget_updates"
    ]
    commit_guards = [
        node for statement in normal_body for node in ast.walk(statement)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "accepted"
    ]
    accepted_assignment = _has_assignment(normal_body, "accepted", "evaluation.accepted")
    updates_assignment = _has_assignment(normal_body, "updates", "dict(evaluation.updates)")
    candidate_assignment = _has_assignment(
        normal_body, "candidate_budgets", "dict(evaluation.candidate_budgets)"
    )
    evaluation_before_commit = bool(evaluation_calls and apply_calls)
    if evaluation_before_commit:
        evaluation_before_commit = evaluation_calls[0].lineno < apply_calls[0].lineno

    apply_in_commit_guard = (
        len(commit_guards) == 1
        and len(commit_guards[0].orelse) == 0
        and any(node in apply_calls for node in ast.walk(commit_guards[0]))
    )
    apply_payload_is_updates = (
        len(apply_calls) == 1
        and apply_calls[0].args
        and ast.unparse(apply_calls[0].args[0]) == "updates"
    )

    run_until_calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call) and "run_until" in ast.unparse(node.func)
    ]
    plant_progression_separated = (
        len(run_until_calls) == 1
        and run_until_calls[0].lineno > getattr(action_guard, "end_lineno", action_guard.lineno)
        and not any(
            node in ast.walk(action_guard)
            for node in run_until_calls
        )
    )

    writes = _writes(normal_body)
    state_write_suffixes = {
        "active_jobs", "running_job", "mode", "current_time", "executed_time",
        "runtime_budget_at_release", "service", "executed_service",
    }
    direct_state_writes = sorted(
        target for target in writes
        if target.startswith("self._engine.")
        or target.startswith("job.")
        or target.rsplit(".", 1)[-1] in state_write_suffixes
    )
    runtime_handler_calls = sorted(
        call for call in set(calls)
        if call.startswith("self._engine.")
        and call != "self._engine.apply_budget_updates"
    )

    candidate_evaluation_before_commit = (
        len(evaluation_calls) == 1
        and accepted_assignment
        and updates_assignment
        and candidate_assignment
        and evaluation_before_commit
    )
    rejected_branch_commit_free = apply_in_commit_guard and len(apply_calls) == 1
    ok = all((
        candidate_evaluation_before_commit,
        apply_in_commit_guard,
        apply_payload_is_updates,
        rejected_branch_commit_free,
        not direct_state_writes,
        not runtime_handler_calls,
        plant_progression_separated,
    ))
    branch_payload = {
        "action_guard": ast.dump(action_guard, include_attributes=False),
        "branch_tests": branch_tests,
        "normal_branch": [ast.unparse(statement) for statement in normal_body],
        "run_until": [ast.dump(node, include_attributes=False) for node in run_until_calls],
    }
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "SELECTED_ACTION_RUNTIME_BINDING_FAILED"},
        "source": "amc_py/rl/env.py:AmcBudgetEnv.step",
        "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "self._engine.apply_budget_updates",
        "action_profile": "single25_explicit_noop",
        "candidate_evaluator": "self.evaluate_budget_candidate",
        "candidate_evaluation_before_commit": candidate_evaluation_before_commit,
        "updates_from_evaluation": updates_assignment,
        "candidate_budgets_from_evaluation": candidate_assignment,
        "commit_guard": "accepted",
        "commit_call_count": len(apply_calls),
        "rejected_branch_commit_free": rejected_branch_commit_free,
        "direct_engine_state_write_free": not direct_state_writes,
        "runtime_handler_call_free": not runtime_handler_calls,
        "plant_progression_separated": plant_progression_separated,
        "branch_binding_hash": sha256_object(branch_payload),
        "branch_tests": branch_tests,
        "direct_state_writes": direct_state_writes,
        "runtime_handler_calls": runtime_handler_calls,
    }


def bind_controller_runtime(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    wrapper_path = frozen_runtime_wrapper_path(root)
    engine_path = frozen_event_runtime_path(root)
    env_path = root / "amc_py/rl/env.py"
    wrapper = wrapper_path.read_text(encoding="utf-8")
    engine = engine_path.read_text(encoding="utf-8")
    env_source = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    wrapper_tree = ast.parse(wrapper)
    calls = [ast.unparse(node) for node in ast.walk(wrapper_tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "engine.apply_budget_updates"]
    engine_ir = function_to_ir(engine, "EventRuntimeEngine.apply_budget_updates")
    required = ("_advance_time", "apply_updates", "_reschedule")
    engine_tree = ast.parse(engine)
    engine_function = next((item for cls in engine_tree.body
                            if isinstance(cls, ast.ClassDef) and cls.name == "EventRuntimeEngine"
                            for item in cls.body
                            if isinstance(item, ast.FunctionDef) and item.name == "apply_budget_updates"), None)
    engine_text = ast.unparse(engine_function) if engine_function is not None else ""
    missing = [token for token in required if token not in engine_text]
    frozen_noop = _analyze_frozen_wrapper_noop(wrapper)
    selected_action = (
        _analyze_env_selected_action(env_source)
        if env_source
        else {"status": "FAIL", "failure": {"code": "AMC_ENV_SOURCE_MISSING"}}
    )
    from formal_toolchain.binding.controller_update_binding import bind_controller_budget_update
    from formal_toolchain.bridge.controller_reschedule import (
        build_controller_force_reschedule_certificate,
    )
    update_binding = bind_controller_budget_update(root)
    force_reschedule = build_controller_force_reschedule_certificate(
        source_binding=update_binding,
        scheduler_certificate=update_binding.get("scheduler_certificate", {}),
        effective_frontier_certificate=update_binding.get("effective_frontier_certificate", {}),
        context_hash=sha256_object({"source_root": str(root), "binding": update_binding}),
    )
    force_witness = force_reschedule.get("witness", {})
    atomic = update_binding.get("atomic_budget_commit", {})
    selected_action.update({
        "payload_prevalidated": atomic.get("validate_all_before_commit") is True,
        "no_partial_mutation": atomic.get("partial_mutation_free") is True,
        "zero_time": update_binding.get("zero_time_delta_proved") is True,
        "time_unchanged": update_binding.get("time_unchanged") is True,
        "mode_unchanged": force_witness.get("mode_unchanged") is True,
        "active_jobs_unchanged": force_witness.get("active_keys_unchanged") is True,
        "ready_jobs_unchanged": force_witness.get("ready_keys_unchanged") is True,
        "running_job_unchanged_if_preclosed": force_witness.get("running_key_unchanged_if_preclosed") is True,
        "released_job_fields_unchanged": force_witness.get("release_snapshots_unchanged") is True,
        "released_job_snapshot_unchanged": force_witness.get("released_job_snapshot_unchanged") is True,
        "released_job_service_unchanged": force_witness.get("released_job_service_unchanged") is True,
        "released_job_demand_unchanged": force_witness.get("released_job_demand_unchanged") is True,
        "released_job_classification_unchanged": force_witness.get("released_job_classification_unchanged") is True,
        "completion_miss_unchanged": force_witness.get("completion_miss_unchanged") is True,
        "service_unchanged": force_witness.get("service_unchanged") is True,
        "effective_event_frontier_unchanged_if_preclosed": force_witness.get("effective_event_frontier_unchanged_if_preclosed") is True,
        "plant_progression_separated": selected_action.get("plant_progression_separated") is True,
        "controller_update_binding": update_binding,
        "force_reschedule_certificate": force_reschedule,
    })
    selected_source_required = (
        "payload_prevalidated", "no_partial_mutation", "zero_time", "time_unchanged",
        "mode_unchanged", "active_jobs_unchanged", "ready_jobs_unchanged",
        "released_job_fields_unchanged", "service_unchanged",
        "released_job_snapshot_unchanged", "released_job_service_unchanged",
        "released_job_demand_unchanged", "released_job_classification_unchanged",
        "completion_miss_unchanged", "plant_progression_separated",
        "running_job_unchanged_if_preclosed",
        "effective_event_frontier_unchanged_if_preclosed",
    )
    selected_source_ok = (
        selected_action.get("status") == "PASS"
        and update_binding.get("status") == "PASS"
        and force_reschedule.get("obligation_status") == "PASS"
        and all(selected_action.get(field) is True for field in selected_source_required)
    )
    selected_action["requires_preclosed_boundary"] = True
    selected_action["timing_projection"] = "STUTTER_IF_PRECLOSED" if selected_source_ok else "UNRESOLVED"
    selected_action["status"] = "PASS" if selected_source_ok else "FAIL"
    runtime_noop = (
        _analyze_env_explicit_noop(env_source)
        if env_source
        else {"status": "FAIL", "failure": {"code": "AMC_ENV_SOURCE_MISSING"}}
    )
    noop_ok = frozen_noop.get("status") == "PASS" and runtime_noop.get("status") == "PASS"
    status = "PASS" if calls and engine_ir.get("status") == "PASS" and not missing and noop_ok and selected_action.get("status") == "PASS" else "UNRESOLVED"
    binding_payload = {
        "wrapper_calls": calls,
        "engine_ir": engine_ir,
        "required_effects": required,
        "frozen_noop": frozen_noop,
        "runtime_noop": runtime_noop,
        "selected_action": selected_action,
        "controller_update": update_binding,
        "force_reschedule": force_reschedule,
    }
    return {"status": status, "schema_version": "controller_binding_v3_explicit_noop_stutter",
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "EXPLICIT_NOOP_BRANCH_BLOCKING",
            "wrapper_source_hash": sha256_file(wrapper_path),
            "engine_source_hash": sha256_file(engine_path),
            "runtime_env_source_hash": sha256_file(env_path) if env_path.is_file() else None,
            "wrapper_calls": calls, "engine_ir": engine_ir,
            "required_effects": list(required), "missing": missing,
            "explicit_noop_frozen_binding": frozen_noop,
            "explicit_noop_runtime_binding": runtime_noop,
            "selected_action_runtime_binding": selected_action,
            "binding_hash": sha256_object(binding_payload)}
