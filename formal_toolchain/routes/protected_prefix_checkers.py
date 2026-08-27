"""Fresh-verifier checks for the executable prefix scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.reference.protected_priority_prefix.certificates import verify_construction_witness
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import (
    build_runtime_schema_certificate, verify_runtime_schema_certificate,
)
from formal_toolchain.reference.protected_priority_prefix.simulation_domain import check_full_to_prefix_simulation_domain
from formal_toolchain.reference.protected_priority_prefix.macro_step import prove_protected_macro_step_preservation
from formal_toolchain.reference.protected_priority_prefix.simulation_certificate import build_simulation_certificate
from formal_toolchain.reference.protected_priority_prefix.model_conformance import derive_prefix_model_conformance
from formal_toolchain.theory.loader import load_verified_theory_statement


def _phase9_failure_route(code: str | None) -> str:
    """Map the earliest protected-prefix mathematical blocker to Phase 9 routes."""
    value = str(code or "")
    if value.startswith(("PROTECTED_PREFIX_RUNTIME_SCHEMA", "EXECUTABLE_TRANSITION", "PP0_")):
        return "PPP_PP0_UNRESOLVED"
    if value.startswith(("PROTECTED_PREFIX_SAME_TIME", "PROTECTED_PREFIX_CANONICAL", "PROTECTED_PREFIX_TIME", "PROTECTED_PREFIX_IDLE_JUMP")):
        return "PPP_CLOSURE_UNRESOLVED"
    if value.startswith("PROTECTED_PREFIX_COMPLETE_EXECUTION"):
        return "PPP_COMPLETE_EXECUTION_UNRESOLVED"
    if value.startswith("PROTECTED_PREFIX_WEAK_FORWARD"):
        return "PPP_WEAK_SIMULATION_UNRESOLVED"
    if value.startswith("PROTECTED_PREFIX_HI_BAD"):
        return "PPP_BAD_PREFIX_UNRESOLVED"
    if value.startswith(("PROTECTED_PREFIX_RTA", "PP7_B_RTA", "ALL_TASK_RTA")):
        return "PPP_PREFIX_RTA_UNRESOLVED"
    if value.startswith(("REFERENCE_HI_SAFETY", "PP8_", "PP7_A", "IMPORTED_C_AMC", "PROTECTED_PREFIX_THEOREM")):
        return "PPP_SAFETY_COMPOSITION_UNRESOLVED"
    if value.startswith("PPP_"):
        return "PPP_MACRO_STEP_UNRESOLVED"
    return "UNRESOLVED"


def _finish(status: str, code: str | None = None, witness: Mapping[str, Any] | None = None):
    return {"status": status, "route": None if status == "PASS" else _phase9_failure_route(code),
            "code": code, "witness": dict(witness or {})}


def _route_state(kwargs: Mapping[str, Any]):
    ctx = kwargs.get("context")
    return getattr(ctx, "fresh_state", None) or kwargs.get("fresh_state")


def _unwrap_proof_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove verifier/certificate envelopes without eating semantic payloads.

    Some fresh obligations are wrapped twice (route artifact -> recomputed
    obligation certificate -> theorem payload).  Other payloads, notably the
    RTA receipt, legitimately contain their own ``witness`` member.  Therefore
    descend only while the current mapping has envelope markers, rather than
    blindly following every key named ``witness``.
    """
    current: Any = value or {}
    for _ in range(6):
        if not isinstance(current, Mapping):
            return {}
        nested = current.get("witness")
        is_envelope = any(
            marker in current
            for marker in (
                "obligation_status",
                "artifact_schema_version",
                "checker_id",
            )
        )
        if not is_envelope or not isinstance(nested, Mapping):
            break
        current = nested
    return dict(current) if isinstance(current, Mapping) else {}


def _require_pass_predecessors(
    predecessors: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any] | None:
    if set(predecessors) != required:
        return _finish(
            "UNRESOLVED",
            "PREDECESSOR_SET_MISMATCH",
            {"expected": sorted(required), "actual": sorted(predecessors)},
        )
    if any(value.get("obligation_status") != "PASS" for value in predecessors.values()):
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS", {
            "predecessors": sorted(predecessors),
            "required": sorted(required),
        })
    return None


def check_runtime_schema_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"SATURATED_PROTECTED_PREFIX_REFERENCE"},
    )
    if blocked is not None:
        return blocked

    try:
        from formal_toolchain.reference.protected_priority_prefix.transition_ir_validation import validate_all_compiled_ir
        validation = validate_all_compiled_ir()
    except (ImportError, ValueError, TypeError):
        validation = {"status": "UNRESOLVED", "code": "EXECUTABLE_TRANSITION_IR_VALIDATION_UNAVAILABLE"}

    try:
        from formal_toolchain.reference.protected_priority_prefix.pp0_checker import build_pp0_transition_certificate
        pp0_cert = build_pp0_transition_certificate()
    except (ImportError, ValueError, TypeError):
        pp0_cert = {"status": "UNRESOLVED", "code": "PP0_CERTIFICATE_UNAVAILABLE"}

    cert = build_runtime_schema_certificate()
    checked = verify_runtime_schema_certificate(cert)

    if (
        pp0_cert.get("status") == "PASS"
        and validation.get("status") == "PASS"
        and checked.get("status") == "PASS"
        and all(
            cert.get("pp0_witness", {}).get(key) is True
            for key in (
                "single_processor_preemptive_work_conserving_fp",
                "no_blocking_self_suspension_or_nonpreemptive_segments",
                "fixed_processor_supply_and_mode_independent_priority",
                "release_fixed_demands",
                "abnormal_classification_at_arrival",
                "abnormal_hi_only_switch_trigger",
                "quiescent_idle_only_recovery",
                "lo_version_selected_at_release",
                "deadline_observe_only",
                "protected_input_independence",
                "mode_transitions_zero_time",
                "released_protected_job_state_mode_invariant",
            )
        )
    ):
        return _finish("PASS", witness={
            "status": "PASS",
            "pp0_transition_status": pp0_cert.get("status"),
            "compiled_ir_validation": validation,
            "pp0_witness": cert["pp0_witness"],
            "certificate_hash": cert.get("certificate_hash"),
        })

    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_PARAMETRIC_PROOF_MISSING",
        {
            "structural_certificate": cert,
            "structural_certificate_status": checked.get("status"),
            "pp0_certificate_status": pp0_cert.get("status"),
            "compiled_ir_validation": validation,
            "reason": (
                "AST/source-shape checks do not prove the quantified PP0 runtime "
                "schema obligations over all reachable full/prefix states.  "
                "Compiled transition IR from executable semantics must be validated."
            ),
        },
    )


def check_simulation_domain(**kwargs: Any) -> dict[str, Any]:
    """Compose the already verified PP domain predicates.

    This node is a conjunction only; it does not manufacture executions or
    proof receipts.  Once all exact predecessors PASS, the composition helper
    checks quantifier order, oracle identity, demand legality, the single
    execution witness, and the common CloseAt time domain.
    """
    result = check_full_to_prefix_simulation_domain(**kwargs)
    status = result.get("status", "UNRESOLVED")
    return _finish(
        "PASS" if status == "PASS" else ("FAIL" if status == "FAIL" else "UNRESOLVED"),
        result.get("code"), result.get("witness", result),
    )


def check_initial_relation(**kwargs: Any) -> dict[str, Any]:
    """Build the base relation from route state and predecessor artifacts."""
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"PROTECTED_PRIORITY_PREFIX_PARTITION", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"},
    )
    if blocked is not None:
        return blocked
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    from formal_toolchain.reference.executable_semantics import initial_reference_state
    from formal_toolchain.reference.protected_priority_prefix.base_case import (
        prove_standard_initial_relation,
    )
    from formal_toolchain.reference.protected_priority_prefix.observable import (
        observable_schema,
    )
    from formal_toolchain.core.hashing import sha256_object
    full_initial = initial_reference_state(state.full_reference_taskset)
    prefix_initial = initial_reference_state(state.analysis_taskset)
    partition_payload = _unwrap_proof_payload(
        predecessors["PROTECTED_PRIORITY_PREFIX_PARTITION"]
    )
    partition_receipt = {
        "status": "PASS",
        "witness": partition_payload,
        "receipt_hash": predecessors["PROTECTED_PRIORITY_PREFIX_PARTITION"].get(
            "artifact_hash"
        ),
    }
    schema = observable_schema()
    schema_receipt = {
        "status": "PASS",
        "obligation_id": "PROTECTED_OBSERVABLE_SCHEMA",
        "schema": schema,
        "receipt_hash": sha256_object(schema),
    }
    result = prove_standard_initial_relation(
        full_initial,
        prefix_initial,
        partition_receipt,
        schema_receipt,
    )
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_weak_forward_simulation(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {
            "FULL_TO_PREFIX_SIMULATION_DOMAIN",
            "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "PROTECTED_PREFIX_INITIAL_RELATION",
            "PROTECTED_MACRO_STEP_PRESERVATION",
            "PROTECTED_INPUT_STREAM_PROJECTION",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        },
    )
    if blocked is not None:
        return blocked

    state = _route_state(kwargs)
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_weak_forward_simulation_kernel,
    )
    # L8 is an explicit predecessor.  Do not recreate a local outline here.
    macro_step_proof = predecessors["PROTECTED_MACRO_STEP_PRESERVATION"]
    base_case_proof = _unwrap_proof_payload(
        predecessors.get("PROTECTED_PREFIX_INITIAL_RELATION")
    )

    from formal_toolchain.reference.protected_priority_prefix.weak_simulation_kernel import (
        prove_weak_forward_simulation,
    )
    kernel = prove_weak_forward_simulation(
        macro_step_receipt=macro_step_proof,
        execution_existence_receipt=predecessors["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"],
        base_case_receipt=base_case_proof,
        simulation_domain_receipt=predecessors["FULL_TO_PREFIX_SIMULATION_DOMAIN"],
        input_projection_receipt=predecessors["PROTECTED_INPUT_STREAM_PROJECTION"],
        demand_receptiveness_receipt=predecessors["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"],
        proof_kernel_receipt=prove_weak_forward_simulation_kernel(
            macro_step_receipt=macro_step_proof,
            execution_receipt=predecessors["PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS"],
            base_case_receipt=base_case_proof,
            simulation_domain_receipt=predecessors["FULL_TO_PREFIX_SIMULATION_DOMAIN"],
            input_projection_receipt=predecessors["PROTECTED_INPUT_STREAM_PROJECTION"],
            demand_receptiveness_receipt=predecessors["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"],
        ),
    )
    if kernel.get("status") == "PASS":
        return _finish("PASS", witness=kernel)
    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_PARAMETRIC_PROOF_MISSING",
        {
            "required_quantification": (
                "forall full executions, exists one compatible complete prefix "
                "execution, forall closed boundaries"
            ),
            "reason": (
                "Weak forward simulation requires: base case, one internally "
                "constructed complete execution witness, L1-L8 macro-step "
                "induction, and the forall-exists-forall quantifier order."
            ),
            "kernel": kernel,
        },
    )


def check_n4_route_boundary_alignment(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"CONTROLLER_BOUNDARY", "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"}
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked
    runtime = _unwrap_proof_payload(predecessors["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"])
    pp0 = runtime.get("pp0_witness", runtime)
    ok = (
        pp0.get("mode_transitions_zero_time") is True
        and pp0.get("deadline_observe_only") is True
    )
    return _finish(
        "PASS" if ok else "UNRESOLVED",
        None if ok else "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT_UNPROVED",
        {
            "route_id": "protected_prefix",
            "route_neutral_bridge": True,
            "boundary": "N4 PreClosed/PostClosed timing projection == canonical reference Close(t)",
            "runtime_schema_artifact_hash": predecessors["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"].get("artifact_hash"),
            "controller_boundary_artifact_hash": predecessors["CONTROLLER_BOUNDARY"].get("artifact_hash"),
        },
    )


def check_hi_bad_prefix_reflection(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {
            "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",
            "PPP_L5_DEADLINE_BATCH_FOLD",
            "N4_REFERENCE_ROUTE_BOUNDARY_ALIGNMENT",
        },
    )
    if blocked is not None:
        return blocked

    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_hi_bad_prefix_reflection_kernel,
    )
    from formal_toolchain.reference.protected_priority_prefix.observable import observable_schema
    from formal_toolchain.core.hashing import sha256_object
    ddl_batch_proof = _unwrap_proof_payload(
        predecessors["PPP_L5_DEADLINE_BATCH_FOLD"]
    )
    schema = observable_schema()
    observable_schema_receipt = {
        "status": "PASS",
        "obligation_id": "PROTECTED_OBSERVABLE_SCHEMA",
        "schema": schema,
        "receipt_hash": sha256_object(schema),
    }

    from formal_toolchain.reference.protected_priority_prefix.bad_prefix_reflection import (
        derive_hi_bad_prefix_reflection,
    )
    result = derive_hi_bad_prefix_reflection(
        simulation_receipt=(
            _unwrap_proof_payload(
                predecessors["PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"]
            )
        ),
        observable_schema_receipt=observable_schema_receipt,
        deadline_batch_receipt=ddl_batch_proof,
        proof_kernel_receipt=prove_hi_bad_prefix_reflection_kernel(
            simulation_receipt=_unwrap_proof_payload(
                predecessors["PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"]
            ),
            deadline_batch_receipt=ddl_batch_proof,
        ),
        prefix_taskset_fingerprint=(
            _route_state(kwargs).analysis_taskset.to_dict()["fingerprint"]
            if _route_state(kwargs) is not None and _route_state(kwargs).analysis_taskset is not None
            else None
        ),
    )
    if result.get("status") == "PASS":
        return _finish("PASS", witness=result)
    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION_PROOF_MISSING",
        {
            "reason": (
                "Each of the eight reflection fields (job key, criticality, "
                "release time, deadline, demand, service, completion state, "
                "miss ledger) must be derived from the "
                "simulation relation, deadline batch, and observable schema."
            ),
            "result": result,
        },
    )


def check_reference_hi_safety_from_protected_prefix(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    from formal_toolchain.reference.protected_priority_prefix.prefix_safety_lift import (
        prove_reference_hi_safety_from_prefix,
    )
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_pp8_reference_hi_safety_from_prefix_kernel,
    )
    def _bound_receipt(obligation_id: str) -> dict[str, Any]:
        raw = predecessors[obligation_id]
        payload = _unwrap_proof_payload(raw)
        # Keep the fresh verifier's artifact hash alongside the theorem payload;
        # PP8 must record the actual predecessor receipts it consumed.
        payload["artifact_hash"] = raw.get("artifact_hash")
        payload["obligation_id"] = obligation_id
        payload.setdefault("status", raw.get("obligation_status"))
        return payload
    result = prove_reference_hi_safety_from_prefix(
        bad_prefix_reflection_receipt=_bound_receipt("PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"),
        mathematical_conformance_receipt=_bound_receipt("PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"),
        proof_kernel_receipt=prove_pp8_reference_hi_safety_from_prefix_kernel(
            bad_prefix_reflection_receipt=_bound_receipt(
                "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION"
            ),
            mathematical_conformance_receipt=_bound_receipt(
                "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"
            ),
        ),
    )
    if result.get("status") == "PASS":
        return _finish("PASS", witness=result)
    return _finish(
        "UNRESOLVED",
        "REFERENCE_HI_SAFETY_FROM_PREFIX_THEOREM_APPLICATION_MISSING",
        {
            "reason": (
                "By contradiction: full HI miss => PP6 bad-prefix reflection => "
                "prefix HI miss => prefix all-task RTA excludes all misses => "
                "contradiction => full-reference HI safety."
            ),
            "result": result,
        },
    )


def check_full_reference_recurring_input_oracle(**kwargs: Any) -> dict[str, Any]:
    """Establish the symbolic full-execution input-oracle contract.

    The theorem is universally quantified over an arbitrary full reference
    execution.  Supplying one Python object and sampling q=0 is not a proof of
    release-fixed stability or an infinite recurring domain.
    """
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors, {"REFERENCE_MODEL_CONFORMANCE"},
    )
    if blocked is not None:
        return blocked
    state = _route_state(kwargs)
    if state is None or state.full_reference_taskset is None:
        return _finish("UNRESOLVED", "FULL_REFERENCE_TASKSET_MISSING")

    conformance = _unwrap_proof_payload(predecessors["REFERENCE_MODEL_CONFORMANCE"])
    rows = {
        row.get("condition_id"): row
        for row in conformance.get("condition_results", [])
        if isinstance(row, Mapping)
    }
    release_fixed = rows.get("RELEASE_FIXED_DEMAND_DOMINATION", {}).get("passed") is True
    periodic = rows.get("FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE", {}).get("passed") is True
    full_fp = state.full_reference_taskset.to_dict()["fingerprint"]
    from formal_toolchain.reference.protected_priority_prefix.full_execution_input_view import (
        build_symbolic_full_execution_input_theorem,
    )
    receipt = build_symbolic_full_execution_input_theorem(
        full_taskset=state.full_reference_taskset,
        conformance_witness=conformance,
    )
    if receipt.get("status") != "PASS":
        return _finish("UNRESOLVED", "PARAMETRIC_FULL_INPUT_ORACLE_PROOF_MISSING", {
            "release_fixed_reference_conformance": release_fixed,
            "periodic_recurring_reference_conformance": periodic,
            "required_quantification": "forall full reference executions",
            "reason": (
                "A concrete oracle object or a finite q=0 sample cannot discharge "
                "the symbolic input-stream theorem."
            ),
        })
    return _finish("PASS", witness={
        "obligation_id": "FULL_REFERENCE_RECURRING_INPUT_ORACLE",
        "reference_taskset_fingerprint": full_fp,
        "contract_receipt": dict(receipt),
        "contract_receipt_hash": receipt.get("receipt_hash"),
        "authoritative": True,
        "stable_release_fixed_records": True,
        "infinite_recurring_domain": True,
        "demand_not_regenerated_from_wcet": True,
        "quantifier_scope": "forall full reference executions",
    })

def check_protected_input_stream_projection(**kwargs: Any) -> dict[str, Any]:
    """Verify the parametric tail-deletion projection on symbolic input streams."""
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"FULL_REFERENCE_RECURRING_INPUT_ORACLE", "SATURATED_PROTECTED_PREFIX_REFERENCE"},
    )
    if blocked is not None:
        return blocked
    state = _route_state(kwargs)
    construction = (state.prepared_route.construction_witnesses.get("build_result")
                    if state and state.prepared_route else None)
    if construction is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_CONSTRUCTION_MISSING")
    source = _unwrap_proof_payload(predecessors["FULL_REFERENCE_RECURRING_INPUT_ORACLE"])
    prefix_fp = construction.prefix_taskset.to_dict()["fingerprint"]
    from formal_toolchain.reference.protected_priority_prefix.projected_oracle_theorem import (
        build_symbolic_projection_theorem,
    )
    receipt = build_symbolic_projection_theorem(
        full_theorem=source.get("contract_receipt", source),
        protected_task_names=frozenset(construction.protected_task_names),
        full_taskset=state.full_reference_taskset,
        prefix_taskset=construction.prefix_taskset,
        saturation_witness=construction.saturation_witness,
        saturation_certificate_hash=str(
            predecessors["SATURATED_PROTECTED_PREFIX_REFERENCE"].get("artifact_hash", "")
        ),
    )
    if receipt.get("status") != "PASS":
        return _finish("UNRESOLVED", "PARAMETRIC_INPUT_PROJECTION_PROOF_MISSING", {
            "required_quantifier_scope": "forall-full-execution-exists-unique-projected-stream",
            "reason": (
                "A lazy Python wrapper and finitely sampled jobs do not prove the "
                "projection for every release index of an arbitrary full execution."
            ),
        })
    return _finish("PASS", witness={
        "obligation_id": "PROTECTED_INPUT_STREAM_PROJECTION",
        "projection_receipt": dict(receipt),
        "projected_oracle_fingerprint": receipt.get("projected_oracle_fingerprint"),
        "full_oracle_contract_receipt_hash": source.get("contract_receipt_hash"),
        "prefix_taskset_fingerprint": prefix_fp,
        "complete_recurring_stream": True,
        "protected_input_independence": True,
        "quantifier_scope": "forall-full-execution-exists-unique-projected-stream",
    })

def check_protected_input_demand_receptiveness(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"PROTECTED_INPUT_STREAM_PROJECTION", "PROTECTED_PREFIX_LO_SATURATION"},
    )
    if blocked is not None:
        return blocked
    state = _route_state(kwargs)
    if state is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    projection = _unwrap_proof_payload(predecessors["PROTECTED_INPUT_STREAM_PROJECTION"])
    prefix_fp = state.analysis_taskset.to_dict()["fingerprint"]
    from formal_toolchain.reference.protected_priority_prefix.projected_oracle_theorem import (
        build_symbolic_demand_receptiveness_theorem,
    )
    saturation = _unwrap_proof_payload(predecessors["PROTECTED_PREFIX_LO_SATURATION"])
    receipt = build_symbolic_demand_receptiveness_theorem(
        full_taskset=state.full_reference_taskset,
        prefix_taskset=state.analysis_taskset,
        projection_theorem=projection.get("projection_receipt", projection),
        saturation_witness=saturation,
    )
    if receipt.get("status") != "PASS":
        return _finish("UNRESOLVED", "PARAMETRIC_DEMAND_RECEPTIVENESS_PROOF_MISSING", {
            "reason": (
                "Checking q=0 or any finite horizon cannot establish demand "
                "receptiveness for every recurring job."
            ),
            "prefix_taskset_fingerprint": prefix_fp,
        })
    return _finish("PASS", witness={
        "obligation_id": "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        "all_projected_demands_legal": True,
        "all_projected_demands_positive": True,
        "release_fixed_demands": True,
        "mode_independent_lo_receptiveness": True,
        "parameterized": True,
        "prefix_taskset_fingerprint": prefix_fp,
        "projected_oracle_fingerprint": projection.get("projected_oracle_fingerprint"),
        "proof_receipt": dict(receipt),
    })

def check_prefix_reference_prefix_extension(**kwargs: Any) -> dict[str, Any]:
    """Apply the already verified generic reference-prefix extension theorem
    to the transformed prefix taskset.

    This theorem is taskset-parametric; it does not depend on the deleted tail.
    """
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors, {"SATURATED_PROTECTED_PREFIX_REFERENCE"},
    )
    if blocked is not None:
        return blocked
    state = _route_state(kwargs)
    if state is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    prefix = state.analysis_taskset
    tasks = prefix.to_dict().get("tasks", [])
    if not tasks or any(
        not (isinstance(t.get("period"), int) and t["period"] > 0
             and isinstance(t.get("deadline"), int) and 0 < t["deadline"] <= t["period"]
             and isinstance(t.get("offset"), int) and 0 <= t["offset"] < t["period"])
        for t in tasks
    ):
        return _finish("FAIL", "PROTECTED_PREFIX_PERIODIC_DOMAIN_INVALID")
    theory_dir = Path(__file__).resolve().parents[1] / "theory"
    try:
        theorem = load_verified_theory_statement(theory_dir, "REFERENCE_PREFIX_EXTENSION")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION_BACKEND_UNAVAILABLE", {
            "error": str(exc),
        })
    return _finish("PASS", witness={
        "obligation_id": "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        "schema_version": "protected_prefix_reference_extension_v1",
        "generic_theorem_id": theorem["theorem_id"],
        "theorem_statement_hash": theorem["statement_hash"],
        "theorem_assumption_hash": theorem["assumption_hash"],
        "prefix_taskset_fingerprint": prefix.to_dict()["fingerprint"],
        "case_ids": [
            "SAME_TIMESTAMP_CLOSURE",
            "READY_SERVICE_OR_EARLIER_BOUNDARY",
            "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
        ],
        "conclusion": "EVERY_FINITE_PREFIX_EXECUTION_PREFIX_EXTENDS",
    })




def check_prefix_same_time_closure_terminates(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
        prove_same_time_closure_terminates,
    )
    result = prove_same_time_closure_terminates(
        runtime_schema_receipt=predecessors["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"],
        prefix_extension_receipt=predecessors["PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"],
    )
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_prefix_canonical_successor_total(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        "PROTECTED_INPUT_STREAM_PROJECTION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    state = _route_state(kwargs)
    if state is None or state.prepared_route is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")

    from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
        define_next_closed_boundary, prove_canonical_successor_total,
    )
    # The theorem consumes the symbolic projected-oracle contract/fingerprint.
    # It must not require an in-memory oracle object hidden in a predecessor.
    projection = _unwrap_proof_payload(predecessors["PROTECTED_INPUT_STREAM_PROJECTION"])
    successor_def = define_next_closed_boundary(state.analysis_taskset, projection)
    total = prove_canonical_successor_total(
        state.analysis_taskset, successor_def,
        closure_termination_receipt=predecessors["PROTECTED_PREFIX_SAME_TIME_CLOSURE_TERMINATES"],
        prefix_extension_receipt=predecessors["PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"],
        input_projection_receipt=predecessors["PROTECTED_INPUT_STREAM_PROJECTION"],
        demand_receptiveness_receipt=predecessors["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"],
    )
    return _finish(
        "PASS" if total.get("status") == "PASS" else "UNRESOLVED",
        total.get("code"), total,
    )


def check_prefix_time_divergence(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors, {"PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"},
    )
    if blocked is not None:
        return blocked

    from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
        prove_time_divergence,
    )
    result = prove_time_divergence(
        canonical_successor_receipt=predecessors["PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"],
    )
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_prefix_idle_jump_stutter_expansion(**kwargs: Any) -> dict[str, Any]:
    """Discharge the local, parameterized idle-jump frame theorem.

    A finite execution may be supplied for diagnostics, but it can never make
    this obligation PASS.  The theorem is independent of complete execution
    existence, which prevents a witness cycle.
    """
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    from formal_toolchain.reference.protected_priority_prefix.idle_jump_stutter import (
        prove_idle_jump_stutter_expansion,
    )
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_idle_jump_stutter_kernel,
    )
    result = prove_idle_jump_stutter_expansion(
        execution=kwargs.get("prefix_execution") or kwargs.get("finite_execution_diagnostic"),
        observable_projector=kwargs.get("protected_observable_projector"),
        proof_kernel_receipt=prove_idle_jump_stutter_kernel(),
    )
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_prefix_complete_execution_exists(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "PROTECTED_INPUT_STREAM_PROJECTION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    state = _route_state(kwargs)
    if state is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")

    projection = _unwrap_proof_payload(
        predecessors["PROTECTED_INPUT_STREAM_PROJECTION"]
    )
    # A concrete full ledger is optional diagnostic input.  The theorem itself
    # quantifies over an arbitrary full execution and therefore consumes the
    # symbolic projection receipt rather than asking a seed user to supply an
    # infinite ledger manually.
    ledger = (
        kwargs.get("full_execution_release_ledger")
        or kwargs.get("full_execution_ledger")
        or kwargs.get("full_execution_input_view")
    )
    projected_oracle = None
    input_receipt: Mapping[str, Any] = predecessors["PROTECTED_INPUT_STREAM_PROJECTION"]
    if ledger is not None:
        construction = state.prepared_route.construction_witnesses.get("build_result")
        from formal_toolchain.reference.protected_priority_prefix.input_oracle import (
            project_full_execution_ledger,
        )
        from formal_toolchain.reference.protected_priority_prefix.projected_oracle_theorem import (
            bind_projection_to_full_execution_ledger,
        )
        try:
            projected_oracle = project_full_execution_ledger(ledger, construction)
            bound_projection = bind_projection_to_full_execution_ledger(
                symbolic_projection_receipt=projection.get("projection_receipt", projection),
                full_execution_ledger=ledger,
                projected_oracle=projected_oracle,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            return _finish("UNRESOLVED", "FULL_EXECUTION_RELEASE_LEDGER_INVALID", {
                "error": str(exc),
            })
        if bound_projection.get("status") != "PASS":
            return _finish("UNRESOLVED", "PARAMETRIC_INPUT_PROJECTION_BINDING_MISSING", {
                "projection": bound_projection,
            })
        input_receipt = bound_projection

    from formal_toolchain.reference.protected_priority_prefix.execution_builder import (
        prove_complete_execution_exists,
    )
    result = prove_complete_execution_exists(
        canonical_successor_receipt=predecessors["PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL"],
        time_divergence_receipt=predecessors["PROTECTED_PREFIX_TIME_DIVERGENCE"],
        input_projection_receipt=input_receipt,
        demand_receptiveness_receipt=predecessors["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"],
        prefix_taskset=state.analysis_taskset,
        protected_oracle=projected_oracle,
        idle_jump_expansion_receipt=predecessors["PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"],
    )
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_registered_parametric_lemma(**kwargs: Any) -> dict[str, Any]:
    """Build the explicit L1--L8 receipts from fresh executable evidence."""
    predecessors = kwargs.get("verified_predecessors", {})
    if any(value.get("obligation_status") != "PASS" for value in predecessors.values()):
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None or state.full_reference_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    construction = state.prepared_route.construction_witnesses.get("build_result")
    if construction is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")

    from formal_toolchain.core.hashing import sha256_object
    from formal_toolchain.reference.protected_priority_prefix.executable_transition_compiler import compile_function
    from formal_toolchain.reference.protected_priority_prefix.batch_cursor_kernel import (
        build_parameterized_fold_receipt,
    )
    from formal_toolchain.reference.protected_priority_prefix.macro_step import (
        prove_tail_service_exclusion, prove_final_dispatch_correspondence,
        prove_protected_service_correspondence, prove_completion_removal_correspondence,
        prove_deadline_batch_correspondence, prove_arrival_batch_projection,
        prove_mode_tail_phase_join, prove_protected_macro_step_preservation,
    )

    pp0_certificate = _unwrap_proof_payload(
        predecessors.get("PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE")
    )
    pp0_rows = pp0_certificate.get("receipt_results", ())
    pp0: dict[str, dict[str, Any]] = {}
    for row in pp0_rows if isinstance(pp0_rows, list) else ():
        if not isinstance(row, Mapping):
            continue
        normalized = dict(row)
        normalized["status"] = row.get("status")
        pp0[str(row.get("case_id"))] = normalized
        pp0[str(row.get("receipt_id"))] = normalized
    if not pp0:
        from formal_toolchain.reference.protected_priority_prefix.pp0_checker import (
            build_pp0_transition_certificate,
        )
        direct_pp0 = build_pp0_transition_certificate()
        for row in direct_pp0.get("receipt_results", ()):
            if isinstance(row, Mapping):
                normalized = dict(row)
                normalized["status"] = row.get("status")
                pp0[str(row.get("case_id"))] = normalized
                pp0[str(row.get("receipt_id"))] = normalized

    from formal_toolchain.reference.protected_priority_prefix.runtime_semantics_theorems import (
        build_runtime_semantics_theorem_certificate,
    )
    local_runtime = build_runtime_semantics_theorem_certificate()
    local_theorems = dict(local_runtime.get("receipts", {}))

    def transition_receipt(function_name: str, *, case_id: str) -> dict[str, Any]:
        ir = compile_function(function_name)
        receipt = ir.compilation_receipt
        bound = bool(
            ir.is_compiled()
            and receipt is not None
            and receipt.source_ast_hash == ir.source_function_ast_hash
            and receipt.covered_return_path_count == receipt.return_path_count
            and receipt.covered_raise_path_count == receipt.raise_path_count
        )
        payload = {
            "theorem_id": f"PP0_{case_id}_SOURCE_BOUND",
            "status": "PASS" if bound and pp0.get(case_id, {}).get("status") == "PASS" else "UNRESOLVED",
            "case_id": case_id,
            "source_function": function_name,
            "source_ast_hash": ir.source_function_ast_hash,
            "ir_hash": ir.ir_hash(),
            "pp0_receipt_hash": sha256_object(pp0.get(case_id, {})),
            "all_paths_covered": bound,
            "relation_schema": "phase_relation_v4_close_at",
        }
        payload["receipt_hash"] = sha256_object(payload)
        return payload

    idle = _unwrap_proof_payload(
        predecessors.get("PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION")
    )
    scheduler = transition_receipt("_normalize_dispatch", case_id="FINAL_DISPATCH")
    scheduler.update({
        "strict_fixed_priority_dispatch": scheduler["status"] == "PASS",
        "no_mode_dependent_priority": scheduler["status"] == "PASS",
    })
    dispatch = dict(scheduler)
    dispatch["same_protected_ready_set_implies_same_selection"] = scheduler["status"] == "PASS"
    service = transition_receipt("apply_service_tick", case_id="SERVICE_UNIT")
    service.update({
        "same_protected_running_job": service["status"] == "PASS",
        "same_unit_service_increment": service["status"] == "PASS",
        "tail_service_is_protected_stutter": (
            pp0.get("TAIL_ONLY_SERVICE", {}).get("status") == "PASS"
        ),
    })
    removal = transition_receipt("apply_removal", case_id="REM_COMPLETION")
    removal.update({
        "same_removal_guard": removal["status"] == "PASS",
        "same_terminal_record": removal["status"] == "PASS",
        "relation_reestablished_after_removal": removal["status"] == "PASS",
    })
    phase_join = {
        "theorem_id": "PP7_CANONICAL_PHASE_JOIN_SOURCE_BOUND",
        "status": "PASS" if all(
            pp0.get(case, {}).get("status") == "PASS"
            for case in ("RECOVERY", "MODE_SWITCH", "TAIL_ONLY_SERVICE")
        ) and idle.get("status") == "PASS" else "UNRESOLVED",
        "all_symmetric_cases": True,
        "all_asymmetric_cases": True,
        "cases": [
            "FULL_ONLY_RECOVERY", "PREFIX_ONLY_RECOVERY",
            "FULL_ONLY_SWITCH", "PREFIX_ONLY_SWITCH",
            "FULL_TAIL_ONLY_SERVICE", "FULL_TAIL_ONLY_DEADLINE_ENTRY",
            "FULL_TAIL_ONLY_ARRIVAL_ENTRY",
        ],
        "source_receipt_hashes": {
            case: sha256_object(pp0.get(case, {}))
            for case in ("RECOVERY", "MODE_SWITCH", "TAIL_ONLY_SERVICE")
        },
        "relation_schema": "phase_relation_v4_close_at",
    }
    phase_join["receipt_hash"] = sha256_object(phase_join)

    def fold_receipt(phase: str, function_name: str, case_id: str) -> dict[str, Any]:
        ir = compile_function(function_name)
        local_theorem = (
            "DEADLINE_OBSERVE_ONLY" if phase == "DDLCursor"
            else "ABNORMAL_HI_CLASSIFIED_AT_ARRIVAL"
        )
        projection_envelope = _unwrap_proof_payload(
            predecessors.get("PROTECTED_INPUT_STREAM_PROJECTION")
        )
        projection_payload = dict(
            projection_envelope.get("projection_receipt", projection_envelope)
        )
        projection_payload.setdefault("status", "PASS" if predecessors.get(
            "PROTECTED_INPUT_STREAM_PROJECTION", {}
        ).get("obligation_status") == "PASS" else "UNRESOLVED")
        demand_envelope = _unwrap_proof_payload(
            predecessors.get("PROTECTED_INPUT_DEMAND_RECEPTIVENESS")
        )
        demand_payload = dict(
            demand_envelope.get("proof_receipt", demand_envelope)
        )
        demand_payload.setdefault("status", "PASS" if predecessors.get(
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS", {}
        ).get("obligation_status") == "PASS" else "UNRESOLVED")
        return build_parameterized_fold_receipt(
            phase=phase,
            transition_ir=ir,
            pp0_receipt=pp0.get(case_id, {}),
            required_local_theorem_id=local_theorem,
            local_theorem_receipts=local_theorems,
            additional_pp0_receipts={
                key: pp0.get(key, {}) for key in ("MODE_SWITCH", "RELEASE")
            } if phase == "ARRCursor" else {},
            projection_receipt=projection_payload if phase == "ARRCursor" else None,
            demand_receptiveness_receipt=demand_payload if phase == "ARRCursor" else None,
        )

    folds = {
        "DDLCursor": fold_receipt("DDLCursor", "apply_deadline_observation", "DEADLINE_OBSERVATION"),
        "ARRCursor": fold_receipt("ARRCursor", "apply_arrival_batch", "ARRIVAL_BATCH"),
    }
    target = kwargs.get("obligation_id")
    if target == "PPP_L1_TAIL_SERVICE_EXCLUSION":
        result = prove_tail_service_exclusion(
            construction=construction, full_taskset=state.full_reference_taskset,
            scheduler_semantics_receipt=scheduler,
        )
    elif target == "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE":
        result = prove_final_dispatch_correspondence(
            construction=construction,
            tail_service_exclusion_receipt=_unwrap_proof_payload(
                predecessors.get("PPP_L1_TAIL_SERVICE_EXCLUSION")
            ),
            dispatch_semantics_receipt=dispatch,
        )
    elif target == "PPP_L3_SERVICE_CORRESPONDENCE":
        result = prove_protected_service_correspondence(
            construction=construction, pp0_receipts=pp0,
            idle_jump_receipt=idle, service_relation_receipt=service,
        )
    elif target == "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE":
        result = prove_completion_removal_correspondence(
            construction=construction, pp0_receipts=pp0,
            removal_relation_receipt=removal,
        )
    elif target == "PPP_L5_DEADLINE_BATCH_FOLD":
        result = prove_deadline_batch_correspondence(
            construction=construction, fold_kernel_receipt=folds["DDLCursor"],
        )
    elif target == "PPP_L6_ARRIVAL_BATCH_FOLD":
        result = prove_arrival_batch_projection(
            construction=construction, fold_kernel_receipt=folds["ARRCursor"],
        )
    elif target == "PPP_L7_CANONICAL_PHASE_JOIN":
        result = prove_mode_tail_phase_join(
            construction=construction, pp0_receipts=pp0,
            idle_jump_receipt=idle, phase_join_receipt=phase_join,
        )
    elif target == "PROTECTED_MACRO_STEP_PRESERVATION":
        l8_predecessors = {}
        for lemma_id in (
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
            "PPP_L3_SERVICE_CORRESPONDENCE",
            "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
            "PPP_L5_DEADLINE_BATCH_FOLD",
            "PPP_L6_ARRIVAL_BATCH_FOLD",
            "PPP_L7_CANONICAL_PHASE_JOIN",
        ):
            raw = predecessors[lemma_id]
            payload = _unwrap_proof_payload(raw)
            payload["status"] = raw.get("obligation_status")
            payload["artifact_hash"] = raw.get("artifact_hash")
            payload.setdefault("relation_schema", "phase_relation_v4_close_at")
            l8_predecessors[lemma_id] = payload
        result = prove_protected_macro_step_preservation(
            construction=construction, full_taskset=state.full_reference_taskset,
            prefix_taskset=state.analysis_taskset, pp0_receipts=pp0,
            fold_receipts=folds, idle_jump_receipt=idle,
            scheduler_semantics_receipt=scheduler,
            dispatch_semantics_receipt=dispatch,
            phase_join_receipt=phase_join,
            service_relation_receipt=service,
            removal_relation_receipt=removal,
            predecessor_receipts=l8_predecessors,
        )
    else:
        return _finish("UNRESOLVED", "PPP_OBLIGATION_ID_MISSING")
    return _finish(
        "PASS" if result.get("status") == "PASS" else "UNRESOLVED",
        result.get("code"), result,
    )


def check_ppp_l1_tail_service_exclusion(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L1 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L1_TAIL_SERVICE_EXCLUSION"})


def check_ppp_l2_final_dispatch_correspondence(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L2 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE"})


def check_ppp_l3_service_correspondence(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L3 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L3_SERVICE_CORRESPONDENCE"})


def check_ppp_l4_completion_removal_correspondence(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L4 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE"})


def check_ppp_l5_deadline_batch_fold(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L5 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L5_DEADLINE_BATCH_FOLD"})


def check_ppp_l6_arrival_batch_fold(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L6 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L6_ARRIVAL_BATCH_FOLD"})


def check_ppp_l7_canonical_phase_join(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for the explicit L7 DAG node."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PPP_L7_CANONICAL_PHASE_JOIN"})


def check_protected_macro_step_preservation(**kwargs: Any) -> dict[str, Any]:
    """Route adapter for L8; missing any predecessor remains UNRESOLVED."""
    return check_registered_parametric_lemma(**{**kwargs, "obligation_id": "PROTECTED_MACRO_STEP_PRESERVATION"})


def check_partition(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    certificate = state.route_construction_certificates.get(
        "PROTECTED_PRIORITY_PREFIX_PARTITION", {})
    witness = certificate.get("witness", {})
    structural_flags = (
        witness.get("tail_all_lo") is True
        and witness.get("all_hi_protected") is True
        and witness.get("partition_complete") is True
        and witness.get("order_preserved") is True
    )
    ok = (result is not None
          and verify_construction_witness(result, witness)
          and structural_flags)
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_PARTITION_INVALID",
                   witness)


def check_saturation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    certificate = state.route_construction_certificates.get(
        "SATURATED_PROTECTED_PREFIX_REFERENCE", {})
    witness = certificate.get("witness", {})
    expected = result.saturation_witness if result is not None else {}
    equalities = expected.get("lo_saturation_equalities", [])
    equalities_hold = all(
        row.get("C_pp_LO") == row.get("C_pp_HI") == row.get("C_ref_LO")
        for row in equalities
        if isinstance(row, Mapping)
    )
    ok = (witness == expected
          and expected.get("hi_fields_equal") is True
          and expected.get("timing_fields_equal") is True
          and equalities_hold)
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_SATURATION_MISMATCH",
                   witness)


def check_parameter_preservation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    if result is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    full = state.full_reference_taskset
    prefix = state.analysis_taskset
    protected_count = len(result.protected_task_names)
    if len(prefix.tasks) != protected_count or protected_count != result.cutoff_priority_index + 1:
        return _finish("FAIL", "PROTECTED_PREFIX_PARAMETER_PRESERVATION_FAILED")

    fields_equal = True
    for full_task, prefix_task, expected_name in zip(
            full.tasks[:protected_count], prefix.tasks, result.protected_task_names):
        fields_equal = fields_equal and (
            full_task.name == prefix_task.name == expected_name
            and full_task.period == prefix_task.period
            and full_task.deadline == prefix_task.deadline
            and full_task.offset == prefix_task.offset
            and full_task.priority_index == prefix_task.priority_index
            and full_task.criticality == prefix_task.criticality
            and full_task.code_c_lo == prefix_task.code_c_lo
            and full_task.code_c_hi == prefix_task.code_c_hi
            and full_task.degraded_cost == prefix_task.degraded_cost
            and full_task.c_lo == prefix_task.c_lo
            and (full_task.criticality != "HI" or full_task.c_hi == prefix_task.c_hi)
        )
    ok = (fields_equal
          and tuple(t.name for t in full.tasks[:protected_count])
          == tuple(result.protected_task_names)
          and tuple(t.name for t in full.tasks[protected_count:])
          == tuple(result.tail_task_names))
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_PARAMETER_PRESERVATION_FAILED")


def check_lo_saturation(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    result = state.prepared_route.construction_witnesses.get("build_result")
    if result is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    full_by_name = {task.name: task for task in state.full_reference_taskset.tasks}
    ok = all(
        task.c_hi == task.c_lo == full_by_name[task.name].c_lo
        for task in result.prefix_taskset.tasks
        if task.criticality == "LO"
    )
    return _finish(
        "PASS" if ok else "FAIL",
        None if ok else "PROTECTED_PREFIX_SATURATION_MISMATCH",
        result.saturation_witness,
    )


def check_prefix_rta(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.selected_rta_obligation_id != "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC":
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_RTA_STATE_MISSING")
    from formal_toolchain.reference.rta_soundness import derive_all_task_rta_soundness

    replay = state.fresh_rta_replay
    derivation = derive_all_task_rta_soundness(
        replay=replay,
        taskset=state.analysis_taskset,
        theorem_id="PREFIX_ALL_TASK_RTA_SOUNDNESS",
    )
    ok = derivation.get("status") == "PASS"
    return _finish(
        "PASS" if ok else "UNRESOLVED",
        None if ok else "PROTECTED_PREFIX_RTA_REPLAY_UNRESOLVED",
        {
            "replay": replay,
            "soundness_receipt": derivation["soundness_receipt"],
        },
    )


def check_prefix_model_conformance(**kwargs: Any) -> dict[str, Any]:
    """Fresh-verifier check for PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE.

    Derives a prefix-specific model conformance certificate from the
    saturated construction witness, runtime schema, and parameter
    preservation lemmas.  Does NOT reuse full-reference conformance.
    """
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "SATURATED_PROTECTED_PREFIX_REFERENCE",
        "PROTECTED_PREFIX_PARAMETER_PRESERVATION",
        "PROTECTED_PREFIX_LO_SATURATION",
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
        "PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION",
        "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        "ZERO_RELATIVE_START",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    state = _route_state(kwargs)
    if state is None or state.prepared_route is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")

    construction = state.prepared_route.construction_witnesses.get("build_result")
    if construction is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")

    full = state.full_reference_taskset
    prefix = state.analysis_taskset
    runtime_certificate = _unwrap_proof_payload(predecessors["PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"])
    prefix_extension_receipt = predecessors["PROTECTED_PREFIX_REFERENCE_PREFIX_EXTENSION"]
    result = derive_prefix_model_conformance(
        full_taskset=full,
        prefix_taskset=prefix,
        construction=construction,
        runtime_schema_certificate=runtime_certificate,
        prefix_extension_receipt=prefix_extension_receipt,
        demand_receptiveness_receipt=predecessors["PROTECTED_INPUT_DEMAND_RECEPTIVENESS"],
        zero_relative_start_receipt=predecessors["ZERO_RELATIVE_START"],
    )

    if result.get("status") != "PASS":
        status = "FAIL" if result.get("status") == "FAIL" else "UNRESOLVED"
        return _finish(status, result.get("failure", {}).get("code", "PREFIX_MODEL_CONFORMANCE_FAILED"),
                       result.get("witness"))

    return _finish("PASS", witness=result)


def check_mathematical_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {"PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE",
                "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
                "THEORY_LIBRARY_VERSION"}
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked

    state = _route_state(kwargs)
    if state is None or state.analysis_taskset is None:
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_TASKSET_MISSING")
    expected_fp = state.analysis_taskset.to_dict()["fingerprint"]

    conformance_receipt = predecessors["PROTECTED_PREFIX_REFERENCE_MODEL_CONFORMANCE"]
    conformance = _unwrap_proof_payload(conformance_receipt)
    # derive_prefix_model_conformance returns a semantic result object whose
    # own witness is the theorem receipt.  This is not a generic artifact
    # envelope, so unwrap it explicitly at this call site.
    if (
        "prefix_taskset_fingerprint" not in conformance
        and conformance.get("status") == "PASS"
        and isinstance(conformance.get("witness"), Mapping)
    ):
        conformance = dict(conformance["witness"])
    conformance_fp = conformance.get("prefix_taskset_fingerprint") if isinstance(conformance, Mapping) else None
    if conformance_receipt.get("obligation_status") != "PASS":
        return _finish("UNRESOLVED", "PP7_A1_MODEL_CONFORMANCE_UNRESOLVED")

    rta_receipt = predecessors["PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"]
    rta_state = _unwrap_proof_payload(rta_receipt)
    rta_replay = rta_state.get("replay", {}) if isinstance(rta_state, Mapping) else {}
    rta_witness = rta_replay.get("witness", {}) if isinstance(rta_replay, Mapping) else {}
    rta_soundness = rta_state.get("soundness_receipt", {}) if isinstance(rta_state, Mapping) else {}
    if not isinstance(rta_witness, Mapping) or not isinstance(rta_soundness, Mapping):
        return _finish("UNRESOLVED", "PP7_B_RTA_SOUNDNESS_RECEIPT_MISSING")
    rta_fp = rta_witness.get("reference_taskset_fingerprint")
    expected_names = [str(task.name) for task in state.analysis_taskset.tasks]
    if conformance_fp != expected_fp or rta_fp != expected_fp:
        return _finish("FAIL", "PROTECTED_PREFIX_THEOREM_TASKSET_BINDING_MISMATCH", {
            "expected_prefix_fingerprint": expected_fp,
            "conformance_fingerprint": conformance_fp,
            "rta_fingerprint": rta_fp,
        })
    if (rta_receipt.get("obligation_status") != "PASS"
            or rta_soundness.get("status") != "PASS"
            or rta_soundness.get("prefix_taskset_fingerprint") != expected_fp
            or rta_soundness.get("formula_version") != "all_task_rta_v3"
            or rta_soundness.get("all_task_name_set") != expected_names):
        return _finish("UNRESOLVED", "PP7_B_RTA_SOUNDNESS_UNRESOLVED", rta_state)

    try:
        theorem = load_verified_theory_statement(
            Path(__file__).resolve().parents[1] / "theory",
            "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return _finish("UNRESOLVED", "IMPORTED_C_AMC_SEM_THEOREM_UNAVAILABLE", {"error": str(exc)})
    if theorem.get("theorem_id") != "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY":
        return _finish("FAIL", "IMPORTED_THEOREM_ID_MISMATCH")

    from formal_toolchain.core.hashing import sha256_object
    imported_binding = {
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "assumption_hash": theorem["assumption_hash"],
        "prefix_taskset_fingerprint": expected_fp,
        "model_conformance_receipt_hash": conformance_receipt.get("artifact_hash"),
        "rta_receipt_hash": rta_receipt.get("artifact_hash"),
        "theory_library_version_receipt_hash": predecessors["THEORY_LIBRARY_VERSION"].get(
            "artifact_hash"
        ),
    }
    imported_binding["receipt_hash"] = sha256_object(imported_binding)
    from formal_toolchain.reference.protected_priority_prefix.proof_kernel import (
        prove_pp7a_imported_assumptions_kernel,
        prove_pp7b_rta_to_inequalities_kernel,
    )
    pp7a_kernel = prove_pp7a_imported_assumptions_kernel(
        model_conformance_receipt=conformance,
    )
    pp7b_kernel = prove_pp7b_rta_to_inequalities_kernel(
        rta_receipt=rta_soundness,
        imported_theorem_receipt={
            "status": "PASS",
            "statement_hash": theorem["statement_hash"],
            "assumption_hash": theorem["assumption_hash"],
        },
        prefix_taskset_fingerprint=expected_fp,
    )
    if pp7a_kernel.get("status") != "PASS":
        return _finish("UNRESOLVED", "PP7_A1_MODEL_CONFORMANCE_KERNEL_UNRESOLVED", pp7a_kernel)
    if pp7b_kernel.get("status") != "PASS":
        return _finish("UNRESOLVED", "PP7_B_RTA_SOUNDNESS_KERNEL_UNRESOLVED", pp7b_kernel)
    composition = {
        "theorem_id": "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
        "proof_partition": ["PP7-A1", "PP7-A2", "PP7-B"],
        "pp7_a1_model_conformance_hash": conformance_receipt.get("artifact_hash"),
        "pp7_a2_imported_theorem_binding_hash": imported_binding["receipt_hash"],
        "pp7_b_rta_soundness_hash": rta_soundness.get("receipt_hash"),
        "pp7_a1_kernel_hash": pp7a_kernel.get("receipt_hash"),
        "pp7_b_kernel_hash": pp7b_kernel.get("receipt_hash"),
        "prefix_taskset_fingerprint": expected_fp,
        "all_task_name_set": expected_names,
        "conclusion": "PROTECTED_PREFIX_TASKSET_SCHEDULABLE",
    }
    composition["receipt_hash"] = sha256_object(composition)
    return _finish("PASS", witness=composition)


def check_selected_safety(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    predecessors = kwargs.get("verified_predecessors", {})
    expected = ("REFERENCE_HI_SUBSET_SAFETY"
                if state and state.selected_route_id == "strict_full"
                else "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX")
    if set(predecessors) != {expected}:
        return _finish("UNRESOLVED", "PREDECESSOR_SET_MISMATCH",
                       {"expected": [expected], "actual": sorted(predecessors)})
    if predecessors[expected].get("obligation_status") != "PASS":
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    full = state.full_reference_taskset.to_dict()["fingerprint"] if state else None
    return _finish("PASS", witness={"route_id": state.selected_route_id,
                                      "source_safety_obligation": expected,
                                      "reference_taskset_fingerprint": full,
                                      "reference_transition_system_id": "FIXED_EXECUTABLE_REFERENCE_P0_V3",
                                      "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES"})
