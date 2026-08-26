"""P0 直接上游语义义务的独立 checker 集合。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.conformance.micro_scenarios import SCENARIO_ASSERTION_CONTRACTS
from formal_toolchain.semantics.frozen_runtime_contract import CONTRACT_VERSION


def _fail(code: str, *, route: str = "MODEL_CONFORMANCE_FAILED", **extra: Any) -> dict[str, Any]:
    return {"status": "FAIL", "route": route, "failure": {"code": code}, **extra}


def _unresolved(code: str, *, route: str = "UNRESOLVED", **extra: Any) -> dict[str, Any]:
    return {"status": "UNRESOLVED", "route": route, "failure": {"code": code}, **extra}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def check_initial_quiescence(runtime: Mapping[str, Any]) -> dict[str, Any]:
    initial = _mapping(runtime.get("initial_state"))
    required = {
        "current_time": 0,
        "mode": "LO",
        "running_job": None,
        "active_jobs": [],
        "ready_jobs": [],
        "service_in_progress": False,
        "quiescent": True,
    }
    mismatches = {key: {"expected": expected, "actual": initial.get(key)} for key, expected in required.items() if initial.get(key) != expected}
    if mismatches:
        return _fail("INITIAL_STATE_NOT_QUIESCENT", mismatches=mismatches)
    return {"status": "PASS", "schema_version": "initial_quiescence_v1", "initial_state": dict(initial)}


def check_boot_initialization(runtime: Mapping[str, Any]) -> dict[str, Any]:
    boot = _mapping(runtime.get("boot"))
    mode_after_boot = boot.get("mode_after_boot")
    if (
        boot.get("status") != "PASS"
        or boot.get("boot_time") != 0
        or mode_after_boot not in {"LO", "HI"}
        or boot.get("first_release_batch_defined") is not True
        or boot.get("no_service_before_boot_closure") is not True
        or not isinstance(boot.get("initial_runtime_budget_snapshot"), Mapping)
    ):
        return _fail("BOOT_INITIALIZATION_CONTRACT_FAILED", boot=dict(boot))
    return {"status": "PASS", "schema_version": "boot_initialization_v1", "boot": dict(boot),
            "mode_after_boot": mode_after_boot}


def check_hi_execution_contract(runtime: Mapping[str, Any]) -> dict[str, Any]:
    micro = _mapping(runtime.get("micro_scenarios"))
    scenario = _mapping(micro.get("scenarios", {})).get("hi_nontruncation", {})
    if not isinstance(scenario, Mapping):
        scenario = {}
    removal = _mapping(runtime.get("removal_binding"))
    contract = _mapping(removal.get("p0_contract"))
    assertions = _mapping(scenario.get("assertions"))
    required_assertions = SCENARIO_ASSERTION_CONTRACTS["hi_nontruncation"]
    required = {
        "source_binding_pass": removal.get("status") == "PASS",
        "source_contract_version": removal.get("formal_semantics_contract_version") == CONTRACT_VERSION,
        "source_runtime_decoupled": removal.get("mutable_runtime_binding") == "NON_BLOCKING_AUDIT_ONLY",
        "source_hi_nontruncation": contract.get("hi_nontruncation") is True,
        "scenario_pass": scenario.get("status") == "PASS",
        "scenario_contract_version": scenario.get("formal_semantics_contract") == CONTRACT_VERSION,
        "scenario_runtime_independent": (
            scenario.get("execution_mode") == "PURE_FORMAL_MODEL"
            and scenario.get("mutable_runtime_dependency") == "NONE"
        ),
        **{f"scenario_assertion:{key}": assertions.get(key) is True for key in required_assertions},
    }
    failed = [key for key, value in required.items() if value is not True]
    if failed:
        return _fail("HI_EXECUTION_CONTRACT_FAILED", failed=failed, source_binding=dict(removal), scenario=dict(scenario))
    return {
        "status": "PASS",
        "schema_version": "hi_execution_contract_v2",
        "source_binding_hash": sha256_object(removal),
        "runtime_scenario_hash": sha256_object(scenario),
        "hi_executes_actual_demand": True,
        "hi_not_truncated_at_c_lo": True,
        "hi_not_dropped": True,
        "formal_semantics_contract": CONTRACT_VERSION,
    }


def check_deadline_observation(runtime: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = _mapping(runtime.get("micro_scenarios")).get("scenarios", {})
    scenarios = scenarios if isinstance(scenarios, Mapping) else {}
    completion = _mapping(scenarios.get("completion_at_deadline"))
    observe = _mapping(scenarios.get("deadline_observe_only"))
    removal = _mapping(runtime.get("removal_binding"))
    contract = _mapping(removal.get("p0_contract"))
    completion_assertions = _mapping(completion.get("assertions"))
    observe_assertions = _mapping(observe.get("assertions"))
    ok = all((
        removal.get("status") == "PASS",
        contract.get("deadline_observe_only") is True,
        completion_assertions.get("completion_equals_deadline") is True,
        completion_assertions.get("no_deadline_miss") is True,
        observe_assertions.get("deadline_observed") is True,
        observe_assertions.get("job_not_removed") is True,
    ))
    if not ok:
        return _fail("DEADLINE_OBSERVATION_CONTRACT_FAILED", route="POLICY_CONTRACT_VIOLATION", completion_scenario=dict(completion), observation_scenario=dict(observe), removal_binding=dict(removal))
    return {"status": "PASS", "schema_version": "deadline_observation_v1", "deadline_is_observation_only": True, "completion_precedes_equal_deadline": True}


def check_sequence_allocation(runtime: Mapping[str, Any]) -> dict[str, Any]:
    event = _mapping(runtime.get("event_binding"))
    queue_methods = _mapping(event.get("queue_methods"))
    required_methods = {"push", "pop", "pop_all_matching"}
    queue_ok = required_methods <= set(queue_methods)
    method_ok = all(_mapping(queue_methods.get(name)).get("status") == "PASS" for name in required_methods) if queue_ok else False
    ok = event.get("status") == "PASS" and event.get("fifo_sequence") == "EventQueue._counter" and queue_ok and method_ok
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "MODEL_CONFORMANCE_FAILED",
        "failure": None if ok else {"code": "EVENT_SEQUENCE_ALLOCATION_FAILED"},
        "schema_version": "sequence_allocation_v1",
        "fifo_sequence": event.get("fifo_sequence"),
        "queue_methods": dict(queue_methods),
    }


def check_phase_dag(runtime: Mapping[str, Any]) -> dict[str, Any]:
    edges = _mapping(runtime.get("phase_edges"))
    if not edges:
        return _unresolved("PHASE_DAG_EVIDENCE_MISSING")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("PHASE_DAG_CYCLE")
        if node in visited:
            return
        visiting.add(node)
        for child in edges.get(node, []):
            visit(str(child))
        visiting.remove(node)
        visited.add(node)

    try:
        for node in edges:
            visit(str(node))
    except ValueError:
        return _fail("PHASE_DAG_CYCLE", edges=dict(edges))
    return {"status": "PASS", "schema_version": "phase_dag_v1", "edges": {str(key): list(value) for key, value in edges.items()}, "acyclic": True}


def check_batch_closure(runtime: Mapping[str, Any]) -> dict[str, Any]:
    from formal_toolchain.conformance.boot_controller import check_closure_controller_contract
    result = check_closure_controller_contract(phase_edges=runtime.get("phase_edges"), controller_fields=runtime.get("controller_fields"))
    if result.get("status") != "PASS":
        return result
    result = dict(result)
    result["schema_version"] = "batch_closure_v1"
    result["same_timestamp_batch_finite"] = True
    result["batch_reaches_preclosed_state"] = True
    return result


def check_deadline_boundary_order(runtime: Mapping[str, Any]) -> dict[str, Any]:
    event = _mapping(runtime.get("event_binding"))
    priorities = _mapping(event.get("event_type_priority"))
    normalized: dict[str, int] = {}
    for key, value in priorities.items():
        try:
            normalized[str(key).split(".")[-1]] = int(value)
        except (TypeError, ValueError):
            return _fail("DEADLINE_BOUNDARY_PRIORITY_INVALID", priorities=dict(priorities))
    completion = _mapping(_mapping(runtime.get("micro_scenarios")).get("scenarios", {})).get("completion_at_deadline", {})
    completion = completion if isinstance(completion, Mapping) else {}
    completion_assertions = _mapping(completion.get("assertions"))
    job_completion = normalized.get("JOB_COMPLETION")
    deadline_check = normalized.get("DEADLINE_CHECK")
    job_arrival = normalized.get("JOB_ARRIVAL")
    if None in {job_completion, deadline_check, job_arrival}:
        return _fail("DEADLINE_BOUNDARY_PRIORITY_MISSING", priority=normalized, completion_at_deadline=dict(completion))
    priority_ok = job_completion < deadline_check < job_arrival
    boundary_ok = completion_assertions.get("completion_equals_deadline") is True and completion_assertions.get("no_deadline_miss") is True
    ok = event.get("status") == "PASS" and priority_ok and boundary_ok
    return {"status": "PASS" if ok else "FAIL", "route": None if ok else "MODEL_CONFORMANCE_FAILED", "failure": None if ok else {"code": "DEADLINE_BOUNDARY_ORDER_FAILED"}, "priority": normalized, "completion_at_deadline": dict(completion)}


def check_controller_invisibility(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Compiler-side source candidate for N3.

    The fresh verifier additionally gates this candidate with the five support
    certificates from the obligation registry.  The source binding here is
    therefore deliberately conditional on the certified PreClosed boundary for
    running-key and effective-frontier preservation.
    """
    controller = _mapping(runtime.get("controller_binding"))
    from formal_toolchain.bridge.event_projection import project_event
    from formal_toolchain.bridge.state_relation import P0Event

    projected = project_event(P0Event(0, "CONTROLLER"))
    selected = _mapping(controller.get("selected_action_runtime_binding"))
    selected_unconditional = (
        "payload_prevalidated", "no_partial_mutation", "time_unchanged", "mode_unchanged",
        "active_jobs_unchanged", "ready_jobs_unchanged", "released_job_fields_unchanged",
        "released_job_snapshot_unchanged", "released_job_service_unchanged",
        "released_job_demand_unchanged", "released_job_classification_unchanged",
        "completion_miss_unchanged", "service_unchanged", "plant_progression_separated",
    )
    selected_source_ok = all((
        selected.get("status") == "PASS",
        selected.get("timing_projection") == "STUTTER_IF_PRECLOSED",
        selected.get("source") == "amc_py/rl/env.py:AmcBudgetEnv.step",
        selected.get("source_kind") == "CONTROLLER_SYNCHRONOUS",
        selected.get("source_binding") == "self._engine.apply_budget_updates",
        selected.get("zero_time") is True,
        selected.get("requires_preclosed_boundary") is True,
        selected.get("running_job_unchanged_if_preclosed") is True,
        selected.get("effective_event_frontier_unchanged_if_preclosed") is True,
        all(selected.get(field) is True for field in selected_unconditional),
    ))
    noop = _mapping(controller.get("explicit_noop_runtime_binding"))
    noop_ok = all((
        noop.get("status") == "PASS",
        noop.get("timing_projection") == "STUTTER",
        noop.get("budget_identity") is True,
        noop.get("running_job_unchanged") is True,
        noop.get("effective_event_frontier_unchanged") is True,
        noop.get("released_job_fields_unchanged") is True,
        noop.get("mode_unchanged") is True,
        noop.get("time_unchanged") is True,
        noop.get("explicit_and_fallback_same_timing_semantics") is True,
        noop.get("plant_progress_separated") is True,
    ))
    ok = controller.get("status") == "PASS" and projected is None and selected_source_ok and noop_ok
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "MODEL_CONFORMANCE_FAILED",
        "failure": None if ok else {"code": "CONTROLLER_N3_SOURCE_CANDIDATE_FAILED"},
        "controller_binding": dict(controller),
        "explicit_noop_runtime_binding": dict(noop),
        "selected_action_runtime_binding": dict(selected),
        "selected_action_stutter_conditional_on_preclosed": selected_source_ok,
        "projected_event": None if projected is None else projected.kind,
    }


def check_controller_postclosure(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Compiler-side postclosure candidate.

    Fresh verification additionally requires a verified CONTROLLER_INVISIBILITY
    predecessor.  This function only checks the source-level conditional frame
    and the explicit-noop branch used to construct the candidate artifact.
    """
    controller = _mapping(runtime.get("controller_binding"))
    fields = _mapping(runtime.get("controller_fields"))
    required_effects = {"_advance_time", "apply_updates", "_reschedule"}
    observed = set(controller.get("required_effects", []))
    selected = _mapping(controller.get("selected_action_runtime_binding"))
    selected_source_ok = all((
        selected.get("status") == "PASS",
        selected.get("timing_projection") == "STUTTER_IF_PRECLOSED",
        selected.get("requires_preclosed_boundary") is True,
        selected.get("running_job_unchanged_if_preclosed") is True,
        selected.get("effective_event_frontier_unchanged_if_preclosed") is True,
        selected.get("payload_prevalidated") is True,
        selected.get("no_partial_mutation") is True,
        selected.get("zero_time") is True,
        selected.get("time_unchanged") is True,
        selected.get("mode_unchanged") is True,
        selected.get("active_jobs_unchanged") is True,
        selected.get("ready_jobs_unchanged") is True,
        selected.get("released_job_fields_unchanged") is True,
        selected.get("service_unchanged") is True,
        selected.get("plant_progression_separated") is True,
    ))
    noop_binding = _mapping(controller.get("explicit_noop_runtime_binding"))
    noop_required = (
        "explicit_noop_budget_identity",
        "explicit_noop_macro_stutter",
        "explicit_noop_effective_frontier_stutter",
        "explicit_noop_released_jobs_immutable",
        "explicit_noop_fallback_equivalent",
        "explicit_noop_plant_progress_separated",
    )
    ok = (
        controller.get("status") == "PASS"
        and required_effects <= observed
        and fields.get("active_release_budget_immutable") is True
        and selected_source_ok
        and noop_binding.get("status") == "PASS"
        and noop_binding.get("timing_projection") == "STUTTER"
        and all(fields.get(name) is True for name in noop_required)
    )
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "MODEL_CONFORMANCE_FAILED",
        "failure": None if ok else {"code": "CONTROLLER_POSTCLOSURE_SOURCE_CANDIDATE_FAILED"},
        "controller_binding": dict(controller),
        "controller_fields": dict(fields),
        "selected_action_stutter_conditional_on_preclosed": selected_source_ok,
    }


def check_time_progress(*, source_root: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    from formal_toolchain.binding.time_progress_binding import bind_time_progress_runtime
    binding = bind_time_progress_runtime(source_root)
    ok = all((
        binding.get("status") == "PASS",
        _mapping(binding.get("ready_branch")).get("strict_time_increase") is True,
        _mapping(binding.get("empty_ready_branch")).get("strict_time_increase") is True,
        binding.get("zero_time_closure_finite") is True,
        _mapping(runtime.get("controller_fields")).get("zero_time_stutter_forbidden") is True,
    ))
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "MODEL_CONFORMANCE_FAILED",
        "failure": None if ok else {"code": "TIME_PROGRESS_CONTRACT_FAILED"},
        "binding": dict(binding),
    }


def check_window_mode_normalization(*, mode_result: Mapping[str, Any], event_order_result: Mapping[str, Any], deadline_order_result: Mapping[str, Any], batch_result: Mapping[str, Any]) -> dict[str, Any]:
    predecessors = {"mode": mode_result, "event_order": event_order_result, "deadline_order": deadline_order_result, "batch": batch_result}
    if any(_mapping(item).get("status") != "PASS" for item in predecessors.values()):
        return _unresolved("WINDOW_NORMALIZATION_PREDECESSOR_NOT_PASS")
    return {"status": "PASS", "schema_version": "window_mode_normalization_v1", "entry_point": "POST_SAME_TIMESTAMP_CLOSURE", "entry_mode_stable": True, "mode_values": ["LO", "HI"], "mode_switch_inside_arrival_batch": True, "deadline_boundary_closed": True}


def check_feature_schema_consistency(*, target: Any, inventory: Mapping[str, Any], tree: Any) -> dict[str, Any]:
    target_names = tuple(getattr(target, "feature_names", ()))
    artifact_names = tuple(inventory.get("feature_names", ()))
    indices: list[int] = []
    nodes = getattr(tree, "nodes", None)
    if nodes is None and isinstance(tree, Mapping):
        nodes = tree.get("nodes")
    if nodes is not None:
        for node in nodes:
            index = getattr(node, "feature_index", None)
            if index is not None:
                indices.append(int(index))
    ok = target_names == artifact_names and all(0 <= index < len(target_names) for index in indices)
    return {
        "status": "PASS" if ok else "FAIL",
        "route": None if ok else "PROOF_BUNDLE_INVALID",
        "failure": None if ok else {"code": "FEATURE_SCHEMA_CONSISTENCY_FAILED"},
        "target_feature_names": list(target_names),
        "artifact_feature_names": list(artifact_names),
        "used_feature_indices": sorted(set(indices)),
    }
