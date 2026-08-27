"""Fresh-verifier checkers for the V8 raw protected-prefix route."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.reference.protected_priority_prefix.certificates import (
    verify_raw_construction_witness,
)
from formal_toolchain.reference.protected_priority_prefix.raw_mode_order import (
    compose,
    prove_admissible_set_domination,
    prove_full_idle_implies_raw_idle,
    prove_global_mode_order,
    prove_recovery_order_preservation,
    prove_switch_order_preservation,
)
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import (
    build_runtime_schema_certificate,
    verify_runtime_schema_certificate,
)
from formal_toolchain.theory.loader import load_verified_theory_statement


def _state(kwargs: Mapping[str, Any]):
    ctx = kwargs.get("context")
    return getattr(ctx, "fresh_state", None) or kwargs.get("fresh_state")


def _finish(status: str, code: str | None = None, witness: Mapping[str, Any] | None = None):
    route = None if status == "PASS" else "UNRESOLVED"
    if status == "FAIL" and code and any(token in code for token in ("PARTITION", "TAIL", "INHERIT", "SATURATION", "WCET")):
        route = "REFERENCE_CERTIFICATE_FAILED"
    return {"status": status, "route": route, "code": code, "witness": dict(witness or {})}


def _unwrap(value: Mapping[str, Any] | None) -> dict[str, Any]:
    current: Any = value or {}
    for _ in range(6):
        if not isinstance(current, Mapping):
            return {}
        nested = current.get("witness")
        if not isinstance(nested, Mapping):
            break
        if not any(k in current for k in ("obligation_status", "artifact_schema_version", "checker_id")):
            break
        current = nested
    return dict(current) if isinstance(current, Mapping) else {}


def _require(predecessors: Mapping[str, Any], required: set[str]):
    if set(predecessors) != required:
        return _finish("UNRESOLVED", "PREDECESSOR_SET_MISMATCH", {
            "expected": sorted(required), "actual": sorted(predecessors),
        })
    if any(row.get("obligation_status") != "PASS" for row in predecessors.values()):
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    return None


def _build_result(kwargs: Mapping[str, Any]):
    state = _state(kwargs)
    if state is None or state.prepared_route is None:
        return None
    return state.prepared_route.construction_witnesses.get("build_result")


def check_partition(**kwargs: Any) -> dict[str, Any]:
    state = _state(kwargs); result = _build_result(kwargs)
    if state is None or result is None:
        return _finish("UNRESOLVED", "RAW_PREFIX_TASKSET_MISSING")
    cert = state.route_construction_certificates.get("RAW_PROTECTED_PRIORITY_PREFIX_PARTITION", {})
    witness = cert.get("witness", {})
    ok = verify_raw_construction_witness(result, witness) and all(
        witness.get(k) is True for k in (
            "tail_all_lo", "all_hi_protected", "partition_complete", "order_preserved", "priority_closed"
        )
    )
    return _finish("PASS" if ok else "FAIL", None if ok else "RAW_PREFIX_NOT_HI_CLOSED", witness)


def check_parameter_inheritance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, {"RAW_PROTECTED_PRIORITY_PREFIX_PARTITION"})
    if blocked: return blocked
    state = _state(kwargs); result = _build_result(kwargs)
    if state is None or result is None:
        return _finish("UNRESOLVED", "RAW_PREFIX_TASKSET_MISSING")
    cert = state.route_construction_certificates.get("RAW_PREFIX_PARAMETER_INHERITANCE", {})
    witness = cert.get("witness", {})
    ok = (
        witness == result.inheritance_witness
        and witness.get("all_parameters_inherited") is True
        and witness.get("wcet_identity") is True
        and witness.get("no_saturation_applied") is True
    )
    return _finish("PASS" if ok else "FAIL", None if ok else "RAW_PREFIX_PARAMETER_NOT_INHERITED", witness)


def _construction_fact(obligation_id: str, field: str, *, kwargs: Mapping[str, Any],
                       required: set[str]) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = _build_result(kwargs)
    if result is None: return _finish("UNRESOLVED", "RAW_PREFIX_TASKSET_MISSING")
    source = result.partition_witness if field in result.partition_witness else result.inheritance_witness
    value = source.get(field)
    ok = value is True or (field == "job_key_rule" and value == "(task_id,release_index)")
    return _finish("PASS" if ok else "FAIL", None if ok else f"{obligation_id}_FAILED", {
        "field": field, "value": value,
        "construction_hash": sha256_object({"partition": result.partition_witness, "inheritance": result.inheritance_witness}),
    })


def check_no_saturation(**kwargs: Any):
    return _construction_fact("RAW_PREFIX_NO_SATURATION_BINDING", "no_saturation_applied", kwargs=kwargs,
                              required={"RAW_PREFIX_PARAMETER_INHERITANCE"})


def check_tail_pure_lo(**kwargs: Any):
    return _construction_fact("RAW_PREFIX_TAIL_PURE_LO", "tail_all_lo", kwargs=kwargs,
                              required={"RAW_PROTECTED_PRIORITY_PREFIX_PARTITION"})


def check_priority_closure(**kwargs: Any):
    return _construction_fact("RAW_PREFIX_PRIORITY_CLOSURE", "priority_closed", kwargs=kwargs,
                              required={"RAW_PROTECTED_PRIORITY_PREFIX_PARTITION"})


def check_job_key_binding(**kwargs: Any):
    return _construction_fact("RAW_PREFIX_JOB_KEY_BINDING", "job_key_rule", kwargs=kwargs,
                              required={"RAW_PREFIX_PARAMETER_INHERITANCE"})


def check_runtime_schema(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_PARAMETER_INHERITANCE", "RAW_PREFIX_TAIL_PURE_LO", "RAW_PREFIX_PRIORITY_CLOSURE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    cert = build_runtime_schema_certificate()
    checked = verify_runtime_schema_certificate(cert)
    runtime = cert.get("pp0_witness", {})
    keys = (
        "single_processor_preemptive_work_conserving_fp",
        "no_blocking_self_suspension_or_nonpreemptive_segments",
        "fixed_processor_supply_and_mode_independent_priority",
        "release_fixed_demands", "abnormal_classification_at_arrival",
        "abnormal_hi_only_switch_trigger", "quiescent_idle_only_recovery",
        "lo_version_selected_at_release", "deadline_observe_only",
        "protected_input_independence", "mode_transitions_zero_time",
        "released_protected_job_state_mode_invariant",
    )
    ok = checked.get("status") == "PASS" and all(runtime.get(k) is True for k in keys)
    return _finish("PASS" if ok else "UNRESOLVED",
                   None if ok else "RAW_PREFIX_RUNTIME_SCHEMA_NONCONFORMANT", {
                       "status": "PASS" if ok else "UNRESOLVED",
                       "certificate_hash": cert.get("certificate_hash"),
                       "pp0_transition_status": cert.get("pp0_transition_status"),
                       "pp0_witness": runtime,
                   })


def _runtime_flag(obligation_id: str, flag: str, **kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, {"RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"})
    if blocked: return blocked
    runtime = _unwrap(predecessors["RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"])
    pp0 = runtime.get("pp0_witness", runtime)
    ok = pp0.get(flag) is True
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else f"{obligation_id}_UNPROVED", {
        "runtime_schema_hash": runtime.get("certificate_hash"), "property": flag, "value": pp0.get(flag),
    })


def check_release_fixed(**kwargs: Any): return _runtime_flag("RAW_PREFIX_RELEASE_FIXED_POSITIVE_DEMAND", "release_fixed_demands", **kwargs)
def check_input_independence(**kwargs: Any): return _runtime_flag("RAW_PREFIX_PROTECTED_INPUT_INDEPENDENCE", "protected_input_independence", **kwargs)
def check_mode_transparent_lifecycle(**kwargs: Any): return _runtime_flag("RAW_PREFIX_MODE_TRANSPARENT_LIFECYCLE", "released_protected_job_state_mode_invariant", **kwargs)
def check_idle_only_recovery(**kwargs: Any): return _runtime_flag("RAW_PREFIX_IDLE_ONLY_RECOVERY", "quiescent_idle_only_recovery", **kwargs)
def check_abnormal_hi_only_switch(**kwargs: Any): return _runtime_flag("RAW_PREFIX_ABNORMAL_HI_ONLY_SWITCH", "abnormal_hi_only_switch_trigger", **kwargs)
def check_fixed_supply(**kwargs: Any): return _runtime_flag("RAW_PREFIX_FIXED_PROCESSOR_SUPPLY", "fixed_processor_supply_and_mode_independent_priority", **kwargs)


def check_input_receptiveness(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "RAW_PREFIX_PARAMETER_INHERITANCE", "RAW_PREFIX_JOB_KEY_BINDING"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    runtime = _unwrap(predecessors["RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"])
    pp0 = runtime.get("pp0_witness", runtime)
    ok = (
        pp0.get("lo_version_selected_at_release") is True
        and pp0.get("release_fixed_demands") is True
        and pp0.get("protected_input_independence") is True
    )
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else "RAW_PREFIX_INPUT_RECEPTIVENESS_UNPROVED", {
        "acceptance_contract": "for any positive integer E <= release_budget, release_demand_overrides accepts E",
        "release_budget_selected_from_current_effective_release_mode": pp0.get("lo_version_selected_at_release") is True,
        "release_fixed": pp0.get("release_fixed_demands") is True,
        "tail_independent": pp0.get("protected_input_independence") is True,
        "runtime_schema_hash": runtime.get("certificate_hash"),
    })


def _runtime_witness(predecessors: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _unwrap(predecessors.get("RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", {}))
    return dict(runtime.get("pp0_witness", runtime))


def check_full_idle_implies_raw_idle(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_PRIORITY_CLOSURE", "RAW_PREFIX_TAIL_PURE_LO", "RAW_PREFIX_FIXED_PROCESSOR_SUPPLY", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = prove_full_idle_implies_raw_idle(construction=_build_result(kwargs), runtime_witness=_runtime_witness(predecessors))
    return _finish(result["status"], result.get("code"), result)


def check_recovery_order(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE", "RAW_PREFIX_IDLE_ONLY_RECOVERY", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = prove_recovery_order_preservation(full_idle_receipt=_unwrap(predecessors["RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE"]), runtime_witness=_runtime_witness(predecessors))
    return _finish(result["status"], result.get("code"), result)


def check_switch_order(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_ABNORMAL_HI_ONLY_SWITCH", "RAW_PREFIX_TAIL_PURE_LO", "RAW_PREFIX_PROTECTED_INPUT_INDEPENDENCE", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = prove_switch_order_preservation(construction=_build_result(kwargs), runtime_witness=_runtime_witness(predecessors))
    return _finish(result["status"], result.get("code"), result)


def check_mode_order(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_RECOVERY_ORDER_PRESERVATION", "RAW_PREFIX_SWITCH_ORDER_PRESERVATION", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = prove_global_mode_order(
        recovery_receipt=_unwrap(predecessors["RAW_PREFIX_RECOVERY_ORDER_PRESERVATION"]),
        switch_receipt=_unwrap(predecessors["RAW_PREFIX_SWITCH_ORDER_PRESERVATION"]),
        runtime_witness=_runtime_witness(predecessors),
    )
    return _finish(result["status"], result.get("code"), result)


def check_admissible_set_domination(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_MODE_ORDER_INVARIANT", "RAW_PREFIX_PARAMETER_INHERITANCE", "RAW_PREFIX_NO_SATURATION_BINDING"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    state = _state(kwargs)
    result = prove_admissible_set_domination(
        full_taskset=state.full_reference_taskset, raw_taskset=state.analysis_taskset,
        construction=_build_result(kwargs), mode_order_receipt=_unwrap(predecessors["RAW_PREFIX_MODE_ORDER_INVARIANT"]),
    )
    return _finish(result["status"], result.get("code"), result)


def check_release_demand_receptiveness(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_ADMISSIBLE_SET_DOMINATION", "RAW_PREFIX_PROTECTED_INPUT_RECEPTIVENESS", "RAW_PREFIX_RELEASE_FIXED_POSITIVE_DEMAND", "RAW_PREFIX_MODE_ORDER_INVARIANT"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = compose("RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS", dependencies=predecessors,
                     all_projected_demands_legal=True,
                     all_projected_demands_positive=True,
                     release_fixed_demands=True,
                     mode_ordered_lo_receptiveness=True,
                     conclusion="same full protected demand is legal in raw at corresponding release")
    return _finish(result["status"], result.get("code"), result)


def _derived(theorem_id: str, required: set[str], **kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = compose(theorem_id, dependencies=predecessors)
    return _finish(result["status"], result.get("code"), result)


def check_arrival_projection(**kwargs: Any):
    return _derived("RAW_PREFIX_PROTECTED_ARRIVAL_BATCH_PROJECTION", {"RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS", "RAW_PREFIX_SWITCH_ORDER_PRESERVATION", "RAW_PREFIX_JOB_KEY_BINDING"}, **kwargs)


def check_service_correspondence(**kwargs: Any):
    return _derived("RAW_PREFIX_SERVICE_CORRESPONDENCE", {"RAW_PREFIX_PRIORITY_CLOSURE", "RAW_PREFIX_FIXED_PROCESSOR_SUPPLY", "RAW_PREFIX_FULL_IDLE_IMPLIES_RAW_IDLE"}, **kwargs)


def check_completion_correspondence(**kwargs: Any):
    return _derived("RAW_PREFIX_COMPLETION_CORRESPONDENCE", {"RAW_PREFIX_SERVICE_CORRESPONDENCE", "RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS", "RAW_PREFIX_MODE_TRANSPARENT_LIFECYCLE"}, **kwargs)


def check_deadline_correspondence(**kwargs: Any):
    return _derived("RAW_PREFIX_DEADLINE_BATCH_CORRESPONDENCE", {"RAW_PREFIX_COMPLETION_CORRESPONDENCE", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}, **kwargs)


def check_dispatch_correspondence(**kwargs: Any):
    return _derived("RAW_PREFIX_TOTAL_FINAL_DISPATCH_CORRESPONDENCE", {"RAW_PREFIX_SERVICE_CORRESPONDENCE", "RAW_PREFIX_PRIORITY_CLOSURE"}, **kwargs)


def check_macrostep(**kwargs: Any):
    return _derived("RAW_PREFIX_CLOSE_TO_CLOSE_MACROSTEP", {
        "RAW_PREFIX_RECOVERY_ORDER_PRESERVATION", "RAW_PREFIX_SWITCH_ORDER_PRESERVATION",
        "RAW_PREFIX_PROTECTED_ARRIVAL_BATCH_PROJECTION", "RAW_PREFIX_SERVICE_CORRESPONDENCE",
        "RAW_PREFIX_COMPLETION_CORRESPONDENCE", "RAW_PREFIX_DEADLINE_BATCH_CORRESPONDENCE",
        "RAW_PREFIX_TOTAL_FINAL_DISPATCH_CORRESPONDENCE",
    }, **kwargs)


def check_execution_existence_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "RAW_PREFIX_PROTECTED_INPUT_RECEPTIVENESS", "RAW_PREFIX_RELEASE_FIXED_POSITIVE_DEMAND"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_same_time_closure_termination_kernel,
        prove_canonical_successor_total_kernel,
        prove_time_divergence_kernel,
        prove_idle_jump_stutter_kernel,
    )
    kernels = {
        "closure": prove_same_time_closure_termination_kernel(),
        "successor": prove_canonical_successor_total_kernel(),
        "time_divergence": prove_time_divergence_kernel(),
        "idle_jump": prove_idle_jump_stutter_kernel(),
    }
    ok = all(row.get("status") == "PASS" for row in kernels.values())
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else "RAW_PREFIX_EXECUTION_EXISTENCE_UNPROVED", {
        "status": "PASS" if ok else "UNRESOLVED", "kernels": kernels,
        "stepwise_construction_contract": "current mode-order validates next release before successor construction",
    })


def check_complete_execution(**kwargs: Any):
    return _derived("RAW_PREFIX_COMPLETE_EXECUTION_EXISTENCE", {"RAW_PREFIX_CLOSE_TO_CLOSE_MACROSTEP", "RAW_PREFIX_EXECUTION_EXISTENCE_CONFORMANCE", "RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS", "RAW_PREFIX_MODE_ORDER_INVARIANT"}, **kwargs)


def check_weak_simulation(**kwargs: Any):
    return _derived("RAW_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED", {
        "REFERENCE_MODEL_CONFORMANCE",
        "RAW_PREFIX_EXECUTION_EXISTENCE_CONFORMANCE",
        "RAW_PREFIX_COMPLETE_EXECUTION_EXISTENCE",
        "RAW_PREFIX_CLOSE_TO_CLOSE_MACROSTEP",
        "RAW_PREFIX_MODE_ORDER_INVARIANT",
        "RAW_PREFIX_PROTECTED_ARRIVAL_BATCH_PROJECTION",
    }, **kwargs)


def check_n4_route_boundary_alignment(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"CONTROLLER_BOUNDARY", "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require(predecessors, required)
    if blocked:
        return blocked
    result = compose(
        "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT", dependencies=predecessors,
        extra_conditions={"route_neutral_bridge": True},
        boundary="N4 PreClosed/PostClosed timing projection == canonical reference Close(t)",
    )
    return _finish(result["status"], result.get("code"), result)


def check_bad_prefix_reflection(**kwargs: Any):
    return _derived("RAW_PREFIX_HI_BAD_PREFIX_REFLECTION", {
        "RAW_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",
        "RAW_PREFIX_DEADLINE_BATCH_CORRESPONDENCE",
        "RAW_PROTECTED_PRIORITY_PREFIX_PARTITION",
        "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT",
    }, **kwargs)


def check_raw_rta(**kwargs: Any) -> dict[str, Any]:
    state = _state(kwargs)
    if state is None or state.selected_rta_obligation_id != "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC":
        return _finish("UNRESOLVED", "RAW_PREFIX_RTA_STATE_MISSING")
    from formal_toolchain.reference.rta_soundness import derive_all_task_rta_soundness
    derivation = derive_all_task_rta_soundness(
        replay=state.fresh_rta_replay, taskset=state.analysis_taskset,
        theorem_id="RAW_PREFIX_ALL_TASK_RTA_SOUNDNESS",
    )
    ok = derivation.get("status") == "PASS"
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else "RAW_PREFIX_RTA_REPLAY_UNRESOLVED", {
        "replay": state.fresh_rta_replay,
        "soundness_receipt": derivation.get("soundness_receipt", {}),
    })


def check_verifier_soundness(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, {"RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"})
    if blocked: return blocked
    payload = _unwrap(predecessors["RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"])
    soundness = payload.get("soundness_receipt", {})
    state = _state(kwargs)
    expected = state.analysis_taskset.to_dict()["fingerprint"] if state else None
    ok = (
        soundness.get("status") == "PASS"
        and soundness.get("prefix_taskset_fingerprint") == expected
        and soundness.get("formula_version") == "all_task_rta_v3"
        and soundness.get("all_task_name_set") == [t.name for t in state.analysis_taskset.tasks]
    )
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else "RAW_PREFIX_VERIFIER_EVIDENCE_UNTRUSTED", soundness)


def check_instance_binding(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "RAW_PREFIX_PARAMETER_INHERITANCE", "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        "RAW_PREFIX_VERIFIER_SOUNDNESS", "RAW_PREFIX_MATHEMATICAL_CONFORMANCE",
        "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "THEORY_LIBRARY_VERSION",
    }
    blocked = _require(predecessors, required)
    if blocked:
        return blocked
    state = _state(kwargs)
    if state is None:
        return _finish("UNRESOLVED", "RAW_PREFIX_INSTANCE_BINDING_UNRESOLVED")
    terminal = state.terminal_route_context or {}
    rta_payload = _unwrap(predecessors["RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"])
    soundness = rta_payload.get("soundness_receipt", {})
    math_payload = _unwrap(predecessors["RAW_PREFIX_MATHEMATICAL_CONFORMANCE"])
    imported = math_payload.get("imported_theorem_binding", {})
    from formal_toolchain.reference.protected_priority_prefix.raw_mode_order import (
        RAW_DERIVED_THEOREM_IDS, theorem_hash,
    )
    raw_theorem_ids = RAW_DERIVED_THEOREM_IDS
    source_root = Path(state.inputs.source_root).resolve()
    source_paths = (
        "formal_toolchain/reference/rta_production.py",
        "formal_toolchain/reference/rta_replay.py",
        "formal_toolchain/reference/rta_soundness.py",
        "formal_toolchain/reference/executable_semantics.py",
        "formal_toolchain/reference/protected_priority_prefix/construction.py",
        "formal_toolchain/reference/protected_priority_prefix/raw_mode_order.py",
        "formal_toolchain/routes/raw_prefix_checkers.py",
    )
    try:
        code_hashes = {name: sha256_file(source_root / name) for name in source_paths}
    except OSError as exc:
        return _finish("UNRESOLVED", "RAW_PREFIX_INSTANCE_CODE_BINDING_UNRESOLVED", {"error": str(exc)})
    payload = {
        "route_id": "raw_protected_prefix",
        "analysis_taskset_fingerprint": state.analysis_taskset.to_dict()["fingerprint"],
        "full_reference_taskset_fingerprint": state.full_reference_taskset.to_dict()["fingerprint"],
        "terminal_route_context_hash": terminal.get("hash"),
        "rta_formula_version": soundness.get("formula_version"),
        "rta_soundness_theorem_id": soundness.get("theorem_id"),
        "rta_checker_id": state.fresh_rta_production.get("checker_id"),
        "rta_checker_version": state.fresh_rta_production.get("checker_version"),
        "imported_all_task_theorem_id": imported.get("theorem_id"),
        "imported_all_task_theorem_statement_hash": imported.get("statement_hash"),
        "raw_route_theorem_hashes": {oid: theorem_hash(oid) for oid in raw_theorem_ids},
        "source_code_hashes": code_hashes,
        "rta_artifact_hash": predecessors["RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"].get("artifact_hash"),
        "verifier_artifact_hash": predecessors["RAW_PREFIX_VERIFIER_SOUNDNESS"].get("artifact_hash"),
        "mathematical_conformance_artifact_hash": predecessors["RAW_PREFIX_MATHEMATICAL_CONFORMANCE"].get("artifact_hash"),
        "runtime_schema_artifact_hash": predecessors["RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"].get("artifact_hash"),
        "theory_library_artifact_hash": predecessors["THEORY_LIBRARY_VERSION"].get("artifact_hash"),
        "construction_artifact_hash": predecessors["RAW_PREFIX_PARAMETER_INHERITANCE"].get("artifact_hash"),
    }
    payload["binding_hash"] = sha256_object(payload)
    required_values = (
        "analysis_taskset_fingerprint", "full_reference_taskset_fingerprint", "terminal_route_context_hash",
        "rta_formula_version", "rta_soundness_theorem_id", "rta_checker_id", "rta_checker_version",
        "imported_all_task_theorem_id", "imported_all_task_theorem_statement_hash",
        "raw_route_theorem_hashes", "source_code_hashes", "rta_artifact_hash",
        "verifier_artifact_hash", "mathematical_conformance_artifact_hash",
        "runtime_schema_artifact_hash", "theory_library_artifact_hash", "construction_artifact_hash",
    )
    ok = all(payload.get(k) for k in required_values)
    return _finish("PASS" if ok else "UNRESOLVED", None if ok else "RAW_PREFIX_INSTANCE_BINDING_UNRESOLVED", payload)


def check_mathematical_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "RAW_PREFIX_PARAMETER_INHERITANCE", "RAW_PREFIX_NO_SATURATION_BINDING",
        "RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE", "RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS",
        "ZERO_RELATIVE_START", "THEORY_LIBRARY_VERSION",
    }
    blocked = _require(predecessors, required)
    if blocked: return blocked
    state = _state(kwargs)
    from formal_toolchain.reference.protected_priority_prefix.raw_model_conformance import derive_raw_prefix_model_conformance
    runtime = _unwrap(predecessors["RAW_FULL_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"])
    result = derive_raw_prefix_model_conformance(
        full_taskset=state.full_reference_taskset, raw_taskset=state.analysis_taskset,
        construction=_build_result(kwargs), runtime_schema_receipt=runtime,
        release_receptiveness_receipt=predecessors["RAW_PREFIX_RELEASE_DEMAND_RECEPTIVENESS"],
        zero_relative_start_receipt=predecessors["ZERO_RELATIVE_START"],
    )
    if result.get("status") != "PASS":
        return _finish(result.get("status", "UNRESOLVED"), result.get("failure", {}).get("code"), result)
    try:
        theorem = load_verified_theory_statement(
            Path(__file__).resolve().parents[1] / "theory",
            "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
        )
    except Exception as exc:
        return _finish("UNRESOLVED", "IMPORTED_C_AMC_SEM_THEOREM_UNAVAILABLE", {"error": str(exc)})
    binding = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "assumption_hash": theorem["assumption_hash"],
        "raw_prefix_taskset_fingerprint": state.analysis_taskset.to_dict()["fingerprint"],
        "model_conformance_hash": result["conformance_hash"],
        "theory_library_artifact_hash": predecessors["THEORY_LIBRARY_VERSION"].get("artifact_hash"),
    }
    binding["receipt_hash"] = sha256_object(binding)
    return _finish("PASS", witness={
        "status": "PASS", "theorem_id": "RAW_PREFIX_MATHEMATICAL_CONFORMANCE",
        "prefix_taskset_fingerprint": state.analysis_taskset.to_dict()["fingerprint"],
        "model_conformance": result, "imported_theorem_binding": binding,
        "conclusion": "RAW_PREFIX_C_AMC_SEM_THEOREM_APPLICABLE",
    })


def check_raw_taskset_schedulable(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "RAW_PREFIX_MATHEMATICAL_CONFORMANCE", "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC",
        "RAW_PREFIX_VERIFIER_SOUNDNESS", "RAW_PREFIX_INSTANCE_EVIDENCE_BINDING",
    }
    blocked = _require(predecessors, required)
    if blocked:
        return blocked
    math_payload = _unwrap(predecessors["RAW_PREFIX_MATHEMATICAL_CONFORMANCE"])
    imported = math_payload.get("imported_theorem_binding", {})
    extra_conditions = {
        "imported_wcrt_theorem_bound": bool(imported.get("theorem_id") and imported.get("statement_hash")),
        "raw_all_task_theorem_applicable": math_payload.get("conclusion") == "RAW_PREFIX_C_AMC_SEM_THEOREM_APPLICABLE",
    }
    result = compose(
        "RAW_PREFIX_TASKSET_SCHEDULABLE", dependencies=predecessors,
        extra_conditions=extra_conditions,
        conclusion="ALL_RAW_PREFIX_TASKS_MEET_DEADLINES",
    )
    return _finish(result["status"], result.get("code"), result)


def check_reference_hi_safety(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"RAW_PREFIX_HI_BAD_PREFIX_REFLECTION", "RAW_PREFIX_TASKSET_SCHEDULABLE"}
    blocked = _require(predecessors, required)
    if blocked: return blocked
    result = compose("REFERENCE_HI_SAFETY_FROM_RAW_PREFIX", dependencies=predecessors,
                     proof="contradiction: full HI miss -> raw HI miss; raw taskset schedulability excludes every raw deadline miss",
                     conclusion="ALL_FULL_REFERENCE_HI_JOBS_MEET_DEADLINES")
    return _finish(result["status"], result.get("code"), result)


def check_selected_safety(**kwargs: Any):
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require(predecessors, {"REFERENCE_HI_SAFETY_FROM_RAW_PREFIX"})
    if blocked: return blocked
    state = _state(kwargs)
    return _finish("PASS", witness={
        "route_id": "raw_protected_prefix",
        "source_safety_obligation": "REFERENCE_HI_SAFETY_FROM_RAW_PREFIX",
        "reference_taskset_fingerprint": state.full_reference_taskset.to_dict()["fingerprint"],
        "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
    })
