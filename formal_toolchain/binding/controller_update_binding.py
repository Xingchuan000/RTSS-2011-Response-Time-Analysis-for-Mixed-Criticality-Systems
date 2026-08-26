"""Source binding for the synchronous controller budget-update operation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.semantics.frozen_runtime_contract import (
    frozen_budget_runtime_path,
    frozen_event_runtime_path,
)


def _method(source: str, class_name: str, method_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = [item for item in node.body
                       if isinstance(item, ast.FunctionDef) and item.name == method_name]
            return methods[0] if len(methods) == 1 else None
    return None


def _function(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    matches = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    return matches[0] if len(matches) == 1 else None


def _calls(node: ast.AST) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)]


def _call_text(node: ast.AST) -> list[str]:
    return [ast.unparse(item.func) for item in _calls(node)]


def _assignment(function: ast.FunctionDef, target: str, value: str) -> ast.Assign | None:
    matches = [node for node in ast.walk(function)
               if isinstance(node, ast.Assign)
               and any(ast.unparse(item) == target for item in node.targets)
               and ast.unparse(node.value) == value]
    return matches[0] if len(matches) == 1 else None


def _exact_call(function: ast.FunctionDef, function_text: str) -> ast.Call | None:
    matches = [node for node in _calls(function) if ast.unparse(node.func) == function_text]
    return matches[0] if len(matches) == 1 else None


def _prove_zero_time_progress() -> dict[str, Any]:
    """Prove the zero-time progress equation used by the controller path.

    The source binder supplies the equal-time call and the non-positive elapsed
    branch.  This receipt proves the resulting arithmetic relation separately,
    so ``service_delta`` is not a compiler-side constant.
    """

    try:
        import z3
    except ImportError:
        return {"status": "UNRESOLVED", "solver": "z3", "failure": {"code": "Z3_UNAVAILABLE"}}

    old_time, now, elapsed = z3.Ints("old_time now elapsed")
    service_before, service_after = z3.Ints("service_before service_after")
    delta = z3.Int("service_delta")
    assumptions = z3.And(
        now == old_time,
        elapsed == now - old_time,
        now >= old_time,
        elapsed <= 0,
        delta == z3.If(elapsed > 0, elapsed, 0),
        service_after == service_before + delta,
    )
    solver = z3.Solver()
    solver.add(assumptions)
    feasible = solver.check() == z3.sat
    delta_proof = z3.Solver()
    delta_proof.add(assumptions, delta != 0)
    unchanged_proof = z3.Solver()
    unchanged_proof.add(assumptions, service_after != service_before)
    delta_unsat = delta_proof.check() == z3.unsat
    unchanged_unsat = unchanged_proof.check() == z3.unsat
    status = "PASS" if feasible and delta_unsat and unchanged_unsat else "FAIL"
    return {
        "status": status,
        "solver": "z3",
        "feasible_zero_time_state": feasible,
        "delta_zero_proved": delta_unsat,
        "service_unchanged_proved": unchanged_unsat,
        "service_delta": 0 if delta_unsat else None,
        "smt2_source": (
            "(assert (= now old_time))\n"
            "(assert (= elapsed (- now old_time)))\n"
            "(assert (>= now old_time))\n"
            "(assert (<= elapsed 0))\n"
            "(assert (= service_delta (ite (> elapsed 0) elapsed 0)))\n"
            "(assert (= service_after (+ service_before service_delta)))\n"
        ),
        "negated_delta_result": "UNSAT" if delta_unsat else "SAT",
        "negated_service_change_result": "UNSAT" if unchanged_unsat else "SAT",
    }


def _atomic_budget_binding(source: str) -> dict[str, Any]:
    function = _method(source, "BudgetState", "apply_updates")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "FROZEN_BUDGET_APPLY_NOT_FOUND"}}

    normalized_assign = next(
        (
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "normalized" for target in node.targets)
            and isinstance(node.value, ast.DictComp)
        ),
        None,
    )
    normalized_key_name: str | None = None
    normalized_value_name: str | None = None
    normalized_source_exact = False
    if normalized_assign is not None:
        comp = normalized_assign.value
        assert isinstance(comp, ast.DictComp)
        if len(comp.generators) == 1:
            generator = comp.generators[0]
            if (
                isinstance(generator.target, ast.Tuple)
                and len(generator.target.elts) == 2
                and all(isinstance(item, ast.Name) for item in generator.target.elts)
                and ast.unparse(generator.iter) == "updates.items()"
            ):
                normalized_key_name = generator.target.elts[0].id
                normalized_value_name = generator.target.elts[1].id
                normalized_source_exact = (
                    ast.unparse(comp.key) == f"str({normalized_key_name})"
                    and ast.unparse(comp.value) == f"int({normalized_value_name})"
                )

    commit = _exact_call(function, "self.budgets.update")
    set_budget_calls = [call for call in _calls(function) if ast.unparse(call.func) == "self.set_budget"]
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    validation_loop = next(
        (
            node
            for node in loops
            if normalized_key_name is not None
            and normalized_value_name is not None
            and ast.unparse(node.target).replace("(", "").replace(")", "")
            == f"{normalized_key_name}, {normalized_value_name}"
            and ast.unparse(node.iter) == "normalized.items()"
        ),
        None,
    )
    normalized_validation = (
        normalized_source_exact
        and validation_loop is not None
        and any(
            isinstance(node, ast.Compare)
            and ast.unparse(node) == f"{normalized_value_name} <= 0"
            for node in ast.walk(validation_loop)
        )
        and any(
            isinstance(node, ast.Compare)
            and ast.unparse(node) == f"{normalized_key_name} not in self.budgets"
            for node in ast.walk(validation_loop)
        )
    )
    commit_after_validation = (
        normalized_source_exact
        and validation_loop is not None
        and commit is not None
        and getattr(commit, "lineno", 0) > getattr(validation_loop, "end_lineno", 0)
        and len(commit.args) == 1
        and ast.unparse(commit.args[0]) == "normalized"
        and not set_budget_calls
    )
    ok = normalized_validation and commit_after_validation
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "ATOMIC_BUDGET_COMMIT_BINDING_FAILED"},
        "normalize_all": normalized_source_exact,
        "validate_all_before_commit": commit_after_validation,
        "commit_call": "self.budgets.update(normalized)" if commit_after_validation else None,
        "partial_mutation_free": commit_after_validation,
        "binding_hash": sha256_object({
            "source": ast.dump(function, include_attributes=False),
            "normalized": normalized_source_exact,
            "validation_loop": ast.dump(validation_loop, include_attributes=False)
            if validation_loop is not None else None,
        }),
    }



def _prove_token_refresh_formulas() -> dict[str, Any]:
    """Prove that a zero-time token refresh preserves logical event times.

    This theorem is intentionally conditional on the release-fixed job fields
    and accumulated service being unchanged.  Those are separate N3 premises;
    this receipt only proves that re-generating completion/overrun events from
    the same logical inputs yields the same logical frontier element.
    """

    try:
        import z3
    except ImportError:
        return {"status": "UNRESOLVED", "solver": "z3", "failure": {"code": "Z3_UNAVAILABLE"}}

    now, actual, service, release_budget = z3.Ints("now actual service release_budget")
    completion_before, completion_after = z3.Ints("completion_before completion_after")
    overrun_before, overrun_after = z3.Ints("overrun_before overrun_after")
    assumptions = z3.And(
        actual >= service,
        release_budget >= 0,
        completion_before == now + (actual - service),
        completion_after == now + (actual - service),
        overrun_before == now + (release_budget + 1 - service),
        overrun_after == now + (release_budget + 1 - service),
    )
    feasible_solver = z3.Solver(); feasible_solver.add(assumptions)
    completion_solver = z3.Solver(); completion_solver.add(assumptions, completion_before != completion_after)
    overrun_solver = z3.Solver(); overrun_solver.add(assumptions, overrun_before != overrun_after)
    feasible = feasible_solver.check() == z3.sat
    completion_equal = completion_solver.check() == z3.unsat
    overrun_equal = overrun_solver.check() == z3.unsat
    status = "PASS" if feasible and completion_equal and overrun_equal else "FAIL"
    return {
        "status": status,
        "solver": "z3",
        "feasible": feasible,
        "completion_time_preserved": completion_equal,
        "overrun_time_preserved": overrun_equal,
        "conditional_inputs": (
            "same_current_time", "same_actual_cost", "same_executed_service",
            "same_runtime_budget_at_release",
        ),
        "negated_completion_result": "UNSAT" if completion_equal else "SAT",
        "negated_overrun_result": "UNSAT" if overrun_equal else "SAT",
    }



def _queue_push_contract(function: ast.FunctionDef, *, ignore_guard_fragment: str | None = None) -> dict[str, Any]:
    """Enumerate logical queue writes in one scheduling helper.

    Production q-AMC code is outside the certified C-AMC-sem projection, so
    queue pushes syntactically nested under the q-AMC-only guard are excluded
    from this projection rather than treated as compatibility behavior.
    """

    rows: list[tuple[str, tuple[str, ...]]] = []

    def walk(node: ast.AST, guards: tuple[str, ...]) -> None:
        next_guards = guards
        if isinstance(node, ast.If):
            next_guards = guards + (ast.unparse(node.test),)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "queue.push":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Call) or ast.unparse(node.args[0].func) != "Event":
                rows.append(("INVALID_QUEUE_PAYLOAD", next_guards))
            else:
                event_type = next((ast.unparse(kw.value) for kw in node.args[0].keywords
                                   if kw.arg == "event_type"), "MISSING_EVENT_TYPE")
                rows.append((event_type, next_guards))
        for child in ast.iter_child_nodes(node):
            walk(child, next_guards)

    walk(function, ())
    projected = [event_type for event_type, guards in rows
                 if not (ignore_guard_fragment and any(ignore_guard_fragment in guard for guard in guards))]
    expected = ["EventType.JOB_COMPLETION", "EventType.BUDGET_OVERRUN", "EventType.BUDGET_OVERRUN"]
    ok = sorted(projected) == sorted(expected) and len(projected) == len(expected)
    return {
        "status": "PASS" if ok else "FAIL",
        "projected_event_types": projected,
        "all_queue_pushes": [{"event_type": event_type, "guards": list(guards)}
                              for event_type, guards in rows],
        "logical_queue_write_set_closed": ok,
    }


def _invalidate_token_contract(event_source: str) -> dict[str, Any]:
    function = _function(event_source, "_invalidate_job_events")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "INVALIDATE_JOB_EVENTS_MISSING"}}
    calls = [ast.unparse(call.func) for call in _calls(function)]
    expected = {
        "_job_key",
        "state.valid_completion_tokens.pop",
        "state.valid_overrun_tokens.pop",
        "state.valid_response_expiry_tokens.pop",
    }
    ok = set(calls) == expected and len(calls) == 4
    return {
        "status": "PASS" if ok else "FAIL",
        "old_running_tokens_invalidated": ok,
        "calls": calls,
    }


def _core_reschedule_contract(event_source: str) -> dict[str, Any]:
    function = _function(event_source, "_reschedule")
    if function is None:
        return {"status": "FAIL", "failure": {"code": "CORE_RESCHEDULE_MISSING"}}
    calls = [ast.unparse(call.func) for call in _calls(function)]
    allowed_calls = {
        "_select_highest_priority_ready_job", "_invalidate_job_events", "_job_key",
        "state.started_jobs.add", "monitor.record_job_start", "_schedule_running_job_events",
    }
    unexpected_calls = sorted(set(calls) - allowed_calls)
    writes: list[str] = []
    for node in ast.walk(function):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        writes.extend(ast.unparse(target) for target in targets)
    state_writes = sorted(target for target in writes if target.startswith("state."))
    allowed_state_writes = {"state.running_job", "state.run_started_at"}
    unexpected_state_writes = sorted(set(state_writes) - allowed_state_writes)
    has_running_assignment = _assignment(function, "state.running_job", "selected") is not None
    has_run_start = _assignment(function, "state.run_started_at", "now") is not None
    ok = not unexpected_calls and not unexpected_state_writes and has_running_assignment and has_run_start
    return {
        "status": "PASS" if ok else "FAIL",
        "unexpected_calls": unexpected_calls,
        "unexpected_state_writes": unexpected_state_writes,
        "active_set_write_free": not any("active_jobs" in target for target in state_writes),
        "mode_write_free": not any(target == "state.mode" for target in state_writes),
        "running_assignment_exact": has_running_assignment,
        "run_start_assignment_exact": has_run_start,
    }


def _production_controller_projection(event_source: str, budget_source: str) -> dict[str, Any]:
    """Bind the deployed C-AMC-sem controller-relevant runtime projection.

    The formal frozen engine remains the semantic contract, but the deployed
    ``AmcBudgetEnv.step`` calls ``amc_py.event_runtime.EventRuntimeEngine``.
    This binding checks the controller-relevant production path directly rather
    than relying on the non-blocking whole-runtime drift audit.
    """

    apply = _method(event_source, "EventRuntimeEngine", "apply_budget_updates")
    advance = _method(event_source, "EventRuntimeEngine", "_advance_time")
    reschedule = _method(event_source, "EventRuntimeEngine", "_reschedule")
    init = _method(event_source, "EventRuntimeEngine", "__post_init__")
    progress = _function(event_source, "_update_running_progress")
    selector = _function(event_source, "_select_highest_priority_ready_job")
    schedule = _function(event_source, "_schedule_running_job_events")
    if any(item is None for item in (apply, advance, reschedule, init, progress, selector, schedule)):
        return {"status": "FAIL", "failure": {"code": "PRODUCTION_CONTROLLER_UPDATE_METHOD_MISSING"}}
    assert apply is not None and advance is not None and reschedule is not None
    assert init is not None and progress is not None and selector is not None and schedule is not None

    advance_call = _exact_call(apply, "self._advance_time")
    budget_apply_call = _exact_call(apply, "self.budget_state.apply_updates")
    reschedule_call = _exact_call(apply, "self._reschedule")
    call_sequence_ok = all((
        advance_call is not None,
        budget_apply_call is not None,
        reschedule_call is not None,
        advance_call is not None and len(advance_call.args) == 1
            and ast.unparse(advance_call.args[0]) == "self.state.current_time",
        budget_apply_call is not None and len(budget_apply_call.args) == 1
            and ast.unparse(budget_apply_call.args[0]) == "update_payload",
        reschedule_call is not None and len(reschedule_call.args) == 1
            and ast.unparse(reschedule_call.args[0]) == "self.state.current_time"
            and len(reschedule_call.keywords) == 1
            and reschedule_call.keywords[0].arg == "force"
            and ast.unparse(reschedule_call.keywords[0].value) == "True",
        advance_call is not None and budget_apply_call is not None and reschedule_call is not None
            and advance_call.lineno < budget_apply_call.lineno < reschedule_call.lineno,
    ))

    advance_ok = all((
        _assignment(advance, "old_time", "self.state.current_time") is not None,
        _assignment(advance, "self.state.current_time", "now") is not None,
        (lambda call: call is not None and [ast.unparse(arg) for arg in call.args] == ["self.state", "now"])(
            _exact_call(advance, "_update_running_progress")
        ),
        any(isinstance(node, ast.Compare) and ast.unparse(node) == "now < old_time"
            for node in ast.walk(advance)),
    ))
    progress_ok = all((
        _assignment(progress, "elapsed", "now - state.run_started_at") is not None,
        any(isinstance(node, ast.If) and ast.unparse(node.test) == "elapsed <= 0"
            and any(isinstance(child, ast.Return) for child in node.body)
            for node in ast.walk(progress)),
    ))

    selected = _exact_call(reschedule, "_select_highest_priority_ready_job")
    scheduler_selection_exact = selected is not None and [ast.unparse(arg) for arg in selected.args] == [
        "self.state.active_jobs", "self.priority_map"
    ]
    force_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and len(node.test.values) == 2
        and ast.unparse(node.test.values[0]) == "previous is selected"
        and ast.unparse(node.test.values[1]).replace("(", "").replace(")", "") == "not force"
        for node in ast.walk(reschedule)
    )
    selector_ok = all((
        _assignment(selector, "best_priority", "-1") is not None,
        any(isinstance(node, ast.Compare) and ast.unparse(node) == "prio < best_priority"
            for node in ast.walk(selector)),
        _assignment(selector, "best", "job") is not None,
        any(isinstance(node, ast.Return) and node.value is not None and ast.unparse(node.value) == "best"
            for node in ast.walk(selector)),
    ))
    priority_map_ok = _assignment(
        init, "self.priority_map", "{task.name: idx for idx, task in enumerate(self.ordered_tasks)}"
    ) is not None

    schedule_text = ast.unparse(schedule)
    queue_contract = _queue_push_contract(
        schedule, ignore_guard_fragment="cfg.semantics is RuntimeSemantics.Q_AMC"
    )
    invalidate_contract = _invalidate_token_contract(event_source)
    core_reschedule_contract = _core_reschedule_contract(event_source)
    schedule_ok = all((
        "time=now + job.remaining()" in schedule_text,
        "budget = job.runtime_budget_at_release" in schedule_text,
        "time=now + remaining_to_overrun" in schedule_text,
        "state.valid_completion_tokens[key]" in schedule_text,
        "state.valid_overrun_tokens[key]" in schedule_text,
        queue_contract.get("status") == "PASS",
        invalidate_contract.get("status") == "PASS",
        core_reschedule_contract.get("status") == "PASS",
    ))
    atomic = _atomic_budget_binding(budget_source)
    ok = all((call_sequence_ok, advance_ok, progress_ok, scheduler_selection_exact,
              force_guard, selector_ok, priority_map_ok, schedule_ok,
              atomic.get("status") == "PASS"))
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "PRODUCTION_CONTROLLER_PROJECTION_BINDING_FAILED"},
        "call_sequence_exact": call_sequence_ok,
        "zero_time_advance_source_exact": advance_ok and progress_ok,
        "scheduler_selection_exact": scheduler_selection_exact and force_guard and selector_ok and priority_map_ok,
        "token_refresh_formula_source_exact": schedule_ok,
        "logical_queue_write_set": queue_contract,
        "token_invalidation_contract": invalidate_contract,
        "core_reschedule_contract": core_reschedule_contract,
        "atomic_budget_commit": atomic,
        "binding_hash": sha256_object({
            "apply": ast.dump(apply, include_attributes=False),
            "advance": ast.dump(advance, include_attributes=False),
            "reschedule": ast.dump(reschedule, include_attributes=False),
            "progress": ast.dump(progress, include_attributes=False),
            "selector": ast.dump(selector, include_attributes=False),
            "schedule": ast.dump(schedule, include_attributes=False),
            "atomic": atomic,
        }),
    }


def bind_controller_budget_update(source_root: str | Path) -> dict[str, Any]:
    """Bind all source facts needed for a zero-time controller update."""

    root = Path(source_root)
    event_path = frozen_event_runtime_path(root)
    budget_path = frozen_budget_runtime_path(root)
    production_event_path = root / "amc_py" / "event_runtime.py"
    production_budget_path = root / "amc_py" / "budget_runtime.py"
    if not all(path.is_file() for path in (event_path, budget_path, production_event_path, production_budget_path)):
        return {"status": "FAIL", "failure": {"code": "CONTROLLER_UPDATE_SOURCE_MISSING"}}
    event_source = event_path.read_text(encoding="utf-8")
    budget_source = budget_path.read_text(encoding="utf-8")
    production_event_source = production_event_path.read_text(encoding="utf-8")
    production_budget_source = production_budget_path.read_text(encoding="utf-8")
    apply = _method(event_source, "EventRuntimeEngine", "apply_budget_updates")
    advance = _method(event_source, "EventRuntimeEngine", "_advance_time")
    reschedule = _method(event_source, "EventRuntimeEngine", "_reschedule")
    init = _method(event_source, "EventRuntimeEngine", "__post_init__")
    progress = _function(event_source, "_update_running_progress")
    selector = _function(event_source, "_select_highest_priority_ready_job")
    schedule = _function(event_source, "_schedule_running_job_events")
    if any(item is None for item in (apply, advance, reschedule, init, progress, selector, schedule)):
        return {"status": "FAIL", "failure": {"code": "CONTROLLER_UPDATE_METHOD_MISSING"}}
    assert apply is not None and advance is not None and reschedule is not None
    assert init is not None and progress is not None and selector is not None and schedule is not None

    advance_call = _exact_call(apply, "self._advance_time")
    budget_apply_call = _exact_call(apply, "self.budget_state.apply_updates")
    reschedule_call = _exact_call(apply, "self._reschedule")
    advance_argument_is_current_time = (
        advance_call is not None
        and len(advance_call.args) == 1
        and ast.unparse(advance_call.args[0]) == "self.state.current_time"
    )
    commit_payload_is_update_payload = (
        budget_apply_call is not None
        and len(budget_apply_call.args) == 1
        and ast.unparse(budget_apply_call.args[0]) == "update_payload"
    )
    force_reschedule_exact = (
        reschedule_call is not None
        and len(reschedule_call.args) == 1
        and ast.unparse(reschedule_call.args[0]) == "self.state.current_time"
        and len(reschedule_call.keywords) == 1
        and reschedule_call.keywords[0].arg == "force"
        and ast.unparse(reschedule_call.keywords[0].value) == "True"
    )
    apply_order = (
        advance_call is not None and budget_apply_call is not None and reschedule_call is not None
        and advance_call.lineno < budget_apply_call.lineno < reschedule_call.lineno
    )

    old_time = _assignment(advance, "old_time", "self.state.current_time")
    progress_call = _exact_call(advance, "_update_running_progress")
    time_commit = _assignment(advance, "self.state.current_time", "now")
    advance_contract = (
        old_time is not None
        and progress_call is not None
        and [ast.unparse(arg) for arg in progress_call.args] == ["self.state", "now"]
        and time_commit is not None
        and any(isinstance(node, ast.Compare) and ast.unparse(node) == "now < old_time"
                for node in ast.walk(advance))
    )
    elapsed = _assignment(progress, "elapsed", "now - state.run_started_at")
    zero_elapsed_return = any(
        isinstance(node, ast.If)
        and ast.unparse(node.test) == "elapsed <= 0"
        and any(isinstance(child, ast.Return) for child in node.body)
        for node in ast.walk(progress)
    )
    progress_contract = elapsed is not None and zero_elapsed_return

    zero_time_proof = _prove_zero_time_progress()
    zero_time_contract = zero_time_proof.get("status") == "PASS"

    selected = _exact_call(reschedule, "_select_highest_priority_ready_job")
    scheduler_selection_exact = (
        selected is not None
        and [ast.unparse(arg) for arg in selected.args]
        == ["self.state.active_jobs", "self.priority_map"]
    )
    force_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and len(node.test.values) == 2
        and ast.unparse(node.test.values[0]) == "previous is selected"
        and ast.unparse(node.test.values[1]).replace("(", "").replace(")", "") == "not force"
        for node in ast.walk(reschedule)
    )
    reschedule_contract = scheduler_selection_exact and force_guard

    selector_contract = all((
        _assignment(selector, "best_priority", "-1") is not None,
        any(isinstance(node, ast.Compare) and ast.unparse(node) == "prio < best_priority"
            for node in ast.walk(selector)),
        _assignment(selector, "best", "job") is not None,
        any(isinstance(node, ast.Return) and ast.unparse(node.value) == "best"
            for node in ast.walk(selector)),
    ))
    priority_map_contract = _assignment(
        init, "self.priority_map", "{task.name: idx for idx, task in enumerate(self.ordered_tasks)}"
    ) is not None
    schedule_text = ast.unparse(schedule)
    event_calls = [
        arg for node in ast.walk(schedule)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "queue.push"
        for arg in node.args
        if isinstance(arg, ast.Call) and ast.unparse(arg.func) == "Event"
    ]
    completion_event = next(
        (node for node in event_calls
         if any(keyword.arg == "event_type"
                and ast.unparse(keyword.value) == "EventType.JOB_COMPLETION"
                for keyword in node.keywords)),
        None,
    )
    completion_event_time = (
        completion_event is not None
        and any(keyword.arg == "time" and ast.unparse(keyword.value) == "now + job.remaining()"
                for keyword in completion_event.keywords)
    )
    logical_removal_source_stable = (
        completion_event_time
        and "state.valid_completion_tokens[key]" in schedule_text
    )
    logical_overrun_source_stable = (
        "budget = job.runtime_budget_at_release" in schedule_text
        and "time=now + remaining_to_overrun" in schedule_text
    )

    controller_path = (apply, advance, reschedule, progress, schedule)
    path_text = "\n".join(ast.unparse(item) for item in controller_path)
    completion_miss_status_source_stable = not any(
        marker in path_text
        for marker in (
            "completion_time =",
            "deadline_misses.append",
            "deadline_misses =",
            "hi_miss",
            "lo_miss",
        )
    )
    released_job_snapshot_source_stable = not any(
        marker in path_text
        for marker in (
            "released_in_mode =",
            "is_degraded =",
            "service_quality_if_completed =",
            "original_actual_cost =",
            "original_runtime_budget_at_release =",
            "actual_cost =",
            "runtime_budget_at_release =",
        )
    )

    frozen_queue_contract = _queue_push_contract(schedule)
    frozen_invalidate_contract = _invalidate_token_contract(event_source)
    frozen_core_reschedule_contract = _core_reschedule_contract(event_source)
    atomic = _atomic_budget_binding(budget_source)
    source_contract = all((
        advance_argument_is_current_time,
        commit_payload_is_update_payload,
        force_reschedule_exact,
        apply_order,
        advance_contract,
        progress_contract,
        zero_time_contract,
        reschedule_contract,
        selector_contract,
        priority_map_contract,
        logical_removal_source_stable,
        logical_overrun_source_stable,
        completion_miss_status_source_stable,
        released_job_snapshot_source_stable,
        frozen_queue_contract.get("status") == "PASS",
        frozen_invalidate_contract.get("status") == "PASS",
        frozen_core_reschedule_contract.get("status") == "PASS",
        atomic.get("status") == "PASS",
    ))
    scheduler_certificate = {
        "status": "PASS" if reschedule_contract else "FAIL",
        "binding_hash": sha256_object({
            "selection": scheduler_selection_exact,
            "force_guard": force_guard,
        }),
        "ready_selects_highest_priority": scheduler_selection_exact,
        "priority_order_unchanged": reschedule_contract,
    }
    token_refresh_proof = _prove_token_refresh_formulas()
    production_projection = _production_controller_projection(
        production_event_source, production_budget_source,
    )
    source_contract = source_contract and production_projection.get("status") == "PASS"
    effective_frontier_certificate = {
        "status": "PASS" if (
            reschedule_contract
            and logical_removal_source_stable
            and logical_overrun_source_stable
            and token_refresh_proof.get("status") == "PASS"
        ) else "FAIL",
        "binding_hash": sha256_object({
            "reschedule": reschedule_contract,
            "logical_removal_source_stable": logical_removal_source_stable,
            "logical_overrun_source_stable": logical_overrun_source_stable,
            "token_refresh_proof": token_refresh_proof,
        }),
        "preserved_if_preclosed": True if (
            reschedule_contract
            and logical_removal_source_stable
            and logical_overrun_source_stable
            and token_refresh_proof.get("status") == "PASS"
        ) else False,
        "requires_release_fixed_snapshot": True,
        "token_refresh_formula_proof": token_refresh_proof,
    }
    frame_source_certificate = {
        "status": "PASS" if source_contract else "FAIL",
        "active_keys_source_stable": reschedule_contract,
        "ready_keys_source_stable": reschedule_contract,
        "running_key_preserved_if_preclosed": reschedule_contract and selector_contract and priority_map_contract,
        "mode_source_stable": advance_contract and reschedule_contract,
        "released_job_snapshot_source_stable": released_job_snapshot_source_stable,
        "released_job_service_source_stable": zero_time_contract and released_job_snapshot_source_stable,
        "completion_miss_source_stable": completion_miss_status_source_stable,
        "conditional_on_preclosed": True,
    }
    return {
        "status": "PASS" if source_contract else "FAIL",
        "failure": None if source_contract else {"code": "CONTROLLER_UPDATE_SOURCE_BINDING_FAILED"},
        "source": "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
        "budget_source": "formal_toolchain/semantics/frozen_c_amc_sem_budget_runtime.py",
        "apply_entrypoint": "EventRuntimeEngine.apply_budget_updates",
        "advance_argument_is_current_time": advance_argument_is_current_time,
        "zero_time_delta_proved": advance_argument_is_current_time and advance_contract and zero_time_contract,
        "zero_time_proof": zero_time_proof,
        "service_delta": zero_time_proof.get("service_delta") if zero_time_contract else None,
        "time_unchanged": advance_argument_is_current_time and advance_contract and zero_time_contract,
        "atomic_budget_commit": atomic,
        "force_reschedule_exact": force_reschedule_exact,
        "scheduler_selection_exact": scheduler_selection_exact,
        "scheduler_selector_contract": selector_contract,
        "priority_map_contract": priority_map_contract,
        "preclosed_frame_source_stable": (
            reschedule_contract and selector_contract and priority_map_contract
            and atomic.get("status") == "PASS"
        ),
        "running_key_preserved_if_preclosed": (
            reschedule_contract and selector_contract and priority_map_contract
        ),
        "released_job_fields_source_stable": advance_argument_is_current_time and progress_contract,
        "released_job_snapshot_source_stable": released_job_snapshot_source_stable,
        "released_job_service_source_stable": zero_time_contract and released_job_snapshot_source_stable,
        "released_job_demand_source_stable": released_job_snapshot_source_stable,
        "released_job_classification_source_stable": released_job_snapshot_source_stable,
        "completion_miss_status_source_stable": completion_miss_status_source_stable,
        "completion_miss_status_unchanged": zero_time_contract and completion_miss_status_source_stable,
        "mode_source_stable": advance_contract and reschedule_contract,
        "event_frontier_refresh_allowed": reschedule_contract,
        "logical_queue_write_set_closed": frozen_queue_contract.get("logical_queue_write_set_closed") is True,
        "old_running_tokens_invalidated": frozen_invalidate_contract.get("old_running_tokens_invalidated") is True,
        "core_reschedule_frame_closed": frozen_core_reschedule_contract.get("status") == "PASS",
        "frozen_queue_contract": frozen_queue_contract,
        "frozen_invalidate_contract": frozen_invalidate_contract,
        "frozen_core_reschedule_contract": frozen_core_reschedule_contract,
        "logical_removal_source_stable": logical_removal_source_stable,
        "logical_overrun_source_stable": logical_overrun_source_stable,
        "scheduler_certificate": scheduler_certificate,
        "effective_frontier_certificate": effective_frontier_certificate,
        "frame_source_certificate": frame_source_certificate,
        "production_projection": production_projection,
        "token_refresh_formula_proof": token_refresh_proof,
        "source_hashes": {
            "frozen_event_runtime": sha256_file(event_path),
            "frozen_budget_runtime": sha256_file(budget_path),
            "production_event_runtime": sha256_file(production_event_path),
            "production_budget_runtime": sha256_file(production_budget_path),
        },
        "binding_hash": sha256_object({
            "apply": ast.dump(apply, include_attributes=False),
            "advance": ast.dump(advance, include_attributes=False),
            "reschedule": ast.dump(reschedule, include_attributes=False),
            "progress": ast.dump(progress, include_attributes=False),
            "atomic": atomic,
        }),
    }
