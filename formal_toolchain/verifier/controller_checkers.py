from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.core.hashing import sha256_object


def _finish(obligation_id: str, *, status: str, witness: Mapping[str, Any], route: str | None = None,
            code: str | None = None) -> dict[str, Any]:
    return {"status": status, "route": route, "code": code, "witness": dict(witness),
            "fresh_input_hashes": {"result_hash": sha256_object(dict(witness))}}


def _binding(raw_inputs: Any) -> dict[str, Any]:
    return bind_controller_runtime(getattr(raw_inputs, "source_root"))


def _status(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return value.get("obligation_status", value.get("status"))


def _predecessor(kwargs: Mapping[str, Any], obligation_id: str) -> Mapping[str, Any]:
    predecessors = kwargs.get("verified_predecessors")
    if not isinstance(predecessors, Mapping):
        return {}
    value = predecessors.get(obligation_id)
    return value if isinstance(value, Mapping) else {}


def _class_method(source: str, class_name: str, method_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            matches = [item for item in node.body
                       if isinstance(item, ast.FunctionDef) and item.name == method_name]
            return matches[0] if len(matches) == 1 else None
    return None


def _run_until_calls(function: ast.FunctionDef) -> list[ast.Call]:
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._engine.run_until"]


def _include_boundary_true(call: ast.Call) -> bool:
    return any(keyword.arg == "include_boundary" and ast.unparse(keyword.value) == "True"
               for keyword in call.keywords)


def _boundary_source_binding(source_root: str | Path) -> dict[str, Any]:
    path = Path(source_root) / "amc_py" / "rl" / "env.py"
    if not path.is_file():
        return {"status": "FAIL", "failure": {"code": "AMC_ENV_SOURCE_MISSING"}}
    source = path.read_text(encoding="utf-8")
    reset = _class_method(source, "AmcBudgetEnv", "reset")
    step = _class_method(source, "AmcBudgetEnv", "step")
    if reset is None or step is None:
        return {"status": "FAIL", "failure": {"code": "AMC_ENV_CONTROLLER_ENTRYPOINT_MISSING"}}

    reset_runs = _run_until_calls(reset)
    step_runs = _run_until_calls(step)
    action_guards = [node for node in ast.walk(step)
                     if isinstance(node, ast.If) and ast.unparse(node.test) == "action_id is not None"]
    reset_returns = [node for node in ast.walk(reset) if isinstance(node, ast.Return)]
    step_returns = [node for node in ast.walk(step) if isinstance(node, ast.Return)]
    reset_boundary = (
        len(reset_runs) == 1 and _include_boundary_true(reset_runs[0])
        and reset_returns and all(node.lineno > reset_runs[0].lineno for node in reset_returns)
    )
    step_boundary = (
        len(step_runs) == 1 and _include_boundary_true(step_runs[0])
        and len(action_guards) == 1
        and action_guards[0].lineno < step_runs[0].lineno
        and getattr(action_guards[0], "end_lineno", action_guards[0].lineno) < step_runs[0].lineno
        and step_returns and all(node.lineno > step_runs[0].lineno for node in step_returns)
    )
    ok = reset_boundary and step_boundary
    return {
        "status": "PASS" if ok else "FAIL",
        "failure": None if ok else {"code": "CONTROLLER_PRECLOSED_BOUNDARY_SOURCE_FAILED"},
        "reset_returns_after_closed_run": reset_boundary,
        "step_action_before_closed_run": step_boundary,
        "run_until_include_boundary": reset_boundary and step_boundary,
        "source_hash": sha256_object(ast.dump(ast.parse(source), include_attributes=False)),
    }


def verify_controller_write_set(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    selected = binding.get("selected_action_runtime_binding", {})
    update = selected.get("controller_update_binding", {}) if isinstance(selected, Mapping) else {}
    frame = update.get("frame_source_certificate", {}) if isinstance(update, Mapping) else {}
    production = update.get("production_projection", {}) if isinstance(update, Mapping) else {}
    ok = all((
        binding.get("status") == "PASS",
        isinstance(selected, Mapping) and selected.get("status") == "PASS",
        selected.get("direct_engine_state_write_free") is True,
        selected.get("runtime_handler_call_free") is True,
        isinstance(frame, Mapping) and frame.get("status") == "PASS",
        isinstance(production, Mapping) and production.get("status") == "PASS",
    ))
    witness = {
        "controller_binding_hash": binding.get("binding_hash"),
        "selected_branch_direct_engine_state_write_free": selected.get("direct_engine_state_write_free"),
        "selected_branch_runtime_handler_call_free": selected.get("runtime_handler_call_free"),
        "transitive_frame_source_certificate": dict(frame) if isinstance(frame, Mapping) else {},
        "production_controller_projection": dict(production) if isinstance(production, Mapping) else {},
        "timing_frame_write_set_closed": ok,
    }
    return _finish("CONTROLLER_WRITE_SET", status="PASS" if ok else "FAIL", witness=witness,
                   route=None if ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if ok else "CONTROLLER_WRITE_SET_FAILED")


def verify_controller_boundary(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    batch = _predecessor(kwargs, "BATCH_CLOSURE")
    source = _boundary_source_binding(raw_inputs.source_root)
    batch_witness = batch.get("witness", {}) if isinstance(batch, Mapping) else {}
    batch_preclosed = (
        _status(batch) == "PASS"
        and isinstance(batch_witness, Mapping)
        and batch_witness.get("batch_reaches_preclosed_state") is True
    )
    ok = source.get("status") == "PASS" and batch_preclosed
    witness = {
        "boundary": "PRECLOSED_TO_CONTROLLER_TO_PRECLOSED",
        "source_binding": source,
        "batch_closure_hash": batch.get("artifact_hash") if isinstance(batch, Mapping) else None,
        "batch_reaches_preclosed_state": batch_preclosed,
        "preclosed_scheduler_consistent": ok,
    }
    return _finish("CONTROLLER_BOUNDARY", status="PASS" if ok else "FAIL", witness=witness,
                   route=None if ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if ok else "CONTROLLER_BOUNDARY_FAILED")


def verify_controller_path_uniqueness(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    selected = binding.get("selected_action_runtime_binding", {})
    noop = binding.get("explicit_noop_runtime_binding", {})
    runtime_config = getattr(raw_inputs.target, "runtime_config", None)
    action_space = runtime_config.get("action_space") if isinstance(runtime_config, Mapping) else getattr(runtime_config, "action_space", None)
    explicit_noop = runtime_config.get("include_explicit_noop") if isinstance(runtime_config, Mapping) else getattr(runtime_config, "include_explicit_noop", None)
    semantics = runtime_config.get("semantics") if isinstance(runtime_config, Mapping) else getattr(runtime_config, "semantics", None)
    semantics_name = getattr(semantics, "name", getattr(semantics, "value", semantics))
    from formal_toolchain.bridge.transition_cases import REQUIRED_PLANT_P0_CASE_IDS, REQUIRED_CONTROLLER_CASE_IDS
    plant_controller_disjoint = not (set(REQUIRED_PLANT_P0_CASE_IDS) & set(REQUIRED_CONTROLLER_CASE_IDS))
    branch_map_path = Path(raw_inputs.source_root) / "formal_toolchain" / "bridge" / "runtime_branch_map.py"
    branch_map_source = branch_map_path.read_text(encoding="utf-8") if branch_map_path.is_file() else ""
    queued_budget_update_excluded = "EventType.BUDGET_UPDATE" not in branch_map_source
    ok = all((
        binding.get("status") == "PASS",
        action_space == "single",
        explicit_noop is True,
        str(semantics_name) == "C_AMC_SEM",
        isinstance(selected, Mapping) and selected.get("status") == "PASS",
        selected.get("source") == "amc_py/rl/env.py:AmcBudgetEnv.step",
        selected.get("source_binding") == "self._engine.apply_budget_updates",
        selected.get("commit_call_count") == 1,
        selected.get("rejected_branch_commit_free") is True,
        isinstance(noop, Mapping) and noop.get("status") == "PASS",
        plant_controller_disjoint,
        queued_budget_update_excluded,
    ))
    witness = {
        "controller_binding_hash": binding.get("binding_hash"),
        "action_profile": "single25_explicit_noop" if action_space == "single" and explicit_noop is True else None,
        "selected_commit_call_count": selected.get("commit_call_count") if isinstance(selected, Mapping) else None,
        "rejected_branch_commit_free": selected.get("rejected_branch_commit_free") if isinstance(selected, Mapping) else None,
        "plant_controller_case_sets_disjoint": plant_controller_disjoint,
        "queued_budget_update_excluded_from_formal_branch_map": queued_budget_update_excluded,
        "runtime_semantics": str(semantics_name),
        "canonical_path_unique": ok,
    }
    return _finish("CONTROLLER_PATH_UNIQUENESS", status="PASS" if ok else "FAIL", witness=witness,
                   route=None if ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if ok else "CONTROLLER_PATH_UNIQUENESS_FAILED")


def verify_update_payload_totality(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    selected = binding.get("selected_action_runtime_binding", {})
    update = selected.get("controller_update_binding", {}) if isinstance(selected, Mapping) else {}
    atomic = update.get("atomic_budget_commit", {}) if isinstance(update, Mapping) else {}
    deployed = _predecessor(kwargs, "DEPLOYED_POLICY_PRESERVATION")
    ok = all((
        binding.get("status") == "PASS",
        _status(deployed) == "PASS",
        isinstance(selected, Mapping) and selected.get("status") == "PASS",
        selected.get("candidate_evaluation_before_commit") is True,
        selected.get("updates_from_evaluation") is True,
        selected.get("candidate_budgets_from_evaluation") is True,
        selected.get("rejected_branch_commit_free") is True,
        isinstance(atomic, Mapping) and atomic.get("status") == "PASS",
        atomic.get("validate_all_before_commit") is True,
        atomic.get("partial_mutation_free") is True,
    ))
    witness = {
        "controller_binding_hash": binding.get("binding_hash"),
        "deployed_policy_hash": deployed.get("artifact_hash") if isinstance(deployed, Mapping) else None,
        "candidate_evaluation_before_commit": selected.get("candidate_evaluation_before_commit") if isinstance(selected, Mapping) else None,
        "updates_from_evaluation": selected.get("updates_from_evaluation") if isinstance(selected, Mapping) else None,
        "validate_all_before_commit": atomic.get("validate_all_before_commit") if isinstance(atomic, Mapping) else None,
        "partial_mutation_free": atomic.get("partial_mutation_free") if isinstance(atomic, Mapping) else None,
        "totality_ok": ok,
    }
    return _finish("UPDATE_PAYLOAD_TOTALITY", status="PASS" if ok else "FAIL", witness=witness,
                   route=None if ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if ok else "UPDATE_PAYLOAD_TOTALITY_FAILED")


def verify_token_refresh_projection(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    selected = binding.get("selected_action_runtime_binding", {})
    update = selected.get("controller_update_binding", {}) if isinstance(selected, Mapping) else {}
    frontier = update.get("effective_frontier_certificate", {}) if isinstance(update, Mapping) else {}
    proof = frontier.get("token_refresh_formula_proof", {}) if isinstance(frontier, Mapping) else {}
    active_release = _predecessor(kwargs, "ACTIVE_RELEASE_BUDGET_INVARIANT")
    ok = all((
        binding.get("status") == "PASS",
        _status(active_release) == "PASS",
        isinstance(update, Mapping) and update.get("status") == "PASS",
        update.get("logical_removal_source_stable") is True,
        update.get("logical_overrun_source_stable") is True,
        update.get("released_job_snapshot_source_stable") is True,
        update.get("logical_queue_write_set_closed") is True,
        update.get("old_running_tokens_invalidated") is True,
        update.get("core_reschedule_frame_closed") is True,
        isinstance(frontier, Mapping) and frontier.get("status") == "PASS",
        frontier.get("preserved_if_preclosed") is True,
        frontier.get("requires_release_fixed_snapshot") is True,
        isinstance(proof, Mapping) and proof.get("status") == "PASS",
        proof.get("completion_time_preserved") is True,
        proof.get("overrun_time_preserved") is True,
    ))
    witness = {
        "controller_binding_hash": binding.get("binding_hash"),
        "active_release_budget_invariant_hash": active_release.get("artifact_hash") if isinstance(active_release, Mapping) else None,
        "token_refresh_formula_proof": dict(proof) if isinstance(proof, Mapping) else {},
        "old_tokens_may_be_stale": True,
        "logical_event_type_time_key_preserved_if_preclosed": ok,
        "effective_frontier_preserved_if_preclosed": ok,
    }
    return _finish("TOKEN_REFRESH_PROJECTION", status="PASS" if ok else "FAIL", witness=witness,
                   route=None if ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if ok else "TOKEN_REFRESH_PROJECTION_FAILED")
