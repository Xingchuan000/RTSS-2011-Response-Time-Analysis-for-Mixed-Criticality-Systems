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


def _finish(status: str, code: str | None = None, witness: Mapping[str, Any] | None = None):
    return {"status": status, "route": None if status == "PASS" else "UNRESOLVED",
            "code": code, "witness": dict(witness or {})}


def _route_state(kwargs: Mapping[str, Any]):
    ctx = kwargs.get("context")
    return getattr(ctx, "fresh_state", None) or kwargs.get("fresh_state")


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
        return _finish("UNRESOLVED", "PREDECESSOR_NOT_PASS")
    return None


def check_runtime_schema_conformance(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"SATURATED_PROTECTED_PREFIX_REFERENCE"},
    )
    if blocked is not None:
        return blocked
    certificate = build_runtime_schema_certificate()
    checked = verify_runtime_schema_certificate(certificate)
    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_PARAMETRIC_PROOF_MISSING",
        {
            "structural_certificate": certificate,
            "structural_certificate_status": checked.get("status"),
            "reason": (
                "AST/source-shape checks do not prove the quantified PP0 runtime "
                "schema obligations over all reachable full/prefix states."
            ),
        },
    )


def check_simulation_domain(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    required = {
        "REFERENCE_MODEL_CONFORMANCE",
        "PROTECTED_PRIORITY_PREFIX_PARTITION",
        "PROTECTED_PREFIX_LO_SATURATION",
        "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE",
    }
    blocked = _require_pass_predecessors(predecessors, required)
    if blocked is not None:
        return blocked
    return _finish(
        "UNRESOLVED",
        "FULL_TO_PREFIX_COMPLETE_EXECUTION_DOMAIN_PROOF_MISSING",
        {
            "reason": (
                "The current implementation checks only an initial batch and one "
                "closed timestamp; it does not construct one complete prefix execution "
                "for the complete projected protected input stream."
            )
        },
    )


def check_weak_forward_simulation(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"FULL_TO_PREFIX_SIMULATION_DOMAIN"},
    )
    if blocked is not None:
        return blocked
    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_PARAMETRIC_PROOF_MISSING",
        {
            "required_quantification": (
                "forall full executions, exists one compatible complete prefix "
                "execution, forall closed boundaries"
            ),
            "reason": (
                "Named lemma receipts and non-empty protected-set checks are not a "
                "proof of canonical macro-step preservation."
            ),
        },
    )


def check_hi_bad_prefix_reflection(**kwargs: Any) -> dict[str, Any]:
    predecessors = kwargs.get("verified_predecessors", {})
    blocked = _require_pass_predecessors(
        predecessors,
        {"PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED"},
    )
    if blocked is not None:
        return blocked
    return _finish(
        "UNRESOLVED",
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION_PROOF_MISSING",
        {
            "reason": (
                "Equality of protected job key, deadline, fixed demand, service and "
                "miss-ledger membership at the deadline must be derived from the "
                "simulation relation, not populated as constant True fields."
            )
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
    return _finish(
        "UNRESOLVED",
        "REFERENCE_HI_SAFETY_FROM_PREFIX_THEOREM_APPLICATION_MISSING",
        {
            "reason": (
                "The contradiction theorem may close only after a genuine bad-prefix "
                "reflection theorem and prefix all-task schedulability theorem are verified."
            )
        },
    )

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
    return _finish("PASS" if ok else "FAIL",
                   None if ok else "PROTECTED_PREFIX_SATURATION_MISMATCH")


def check_prefix_rta(**kwargs: Any) -> dict[str, Any]:
    state = _route_state(kwargs)
    if state is None or state.selected_rta_obligation_id != "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC":
        return _finish("UNRESOLVED", "PROTECTED_PREFIX_RTA_STATE_MISSING")
    replay = state.fresh_rta_replay
    witness = replay.get("witness", replay.get("replay", {}).get("witness", {}))
    ok = (replay.get("status") == "PASS"
          and (witness.get("all_tasks_covered") is True
               or replay.get("replay", {}).get("all_tasks_covered") is True)
          and witness.get("all_deadlines_met") is True)
    return _finish("PASS" if ok else "UNRESOLVED",
                   None if ok else "PROTECTED_PREFIX_RTA_REPLAY_UNRESOLVED",
                   replay)


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
    runtime_certificate = build_runtime_schema_certificate()

    result = derive_prefix_model_conformance(
        full_taskset=full,
        prefix_taskset=prefix,
        construction=construction,
        runtime_schema_certificate=runtime_certificate,
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
    conformance = conformance_receipt.get("witness", {})
    if conformance.get("status") == "PASS":
        conformance_fp = conformance.get("prefix_taskset_fingerprint")
    else:
        conformance_fp = conformance.get("prefix_taskset_fingerprint")
        if conformance_fp is None and isinstance(conformance.get("witness"), Mapping):
            conformance_fp = conformance["witness"].get("prefix_taskset_fingerprint")

    rta_receipt = predecessors["PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"]
    rta_state = rta_receipt.get("witness", {})
    rta_witness = rta_state.get("witness", rta_state)
    if not isinstance(rta_witness, Mapping):
        return _finish("FAIL", "PROTECTED_PREFIX_RTA_WITNESS_INVALID")
    rta_fp = rta_witness.get("reference_taskset_fingerprint")
    if rta_fp is None and isinstance(rta_state.get("replay"), Mapping):
        rta_fp = rta_state["replay"].get("witness", {}).get("reference_taskset_fingerprint")

    if conformance_fp != expected_fp or rta_fp != expected_fp:
        return _finish("FAIL", "PROTECTED_PREFIX_THEOREM_TASKSET_BINDING_MISMATCH", {
            "expected_prefix_fingerprint": expected_fp,
            "conformance_fingerprint": conformance_fp,
            "rta_fingerprint": rta_fp,
        })
    if rta_witness.get("all_tasks_covered") is not True or rta_witness.get("all_deadlines_met") is not True:
        return _finish("FAIL", "PROTECTED_PREFIX_ALL_TASK_SCHEDULABILITY_NOT_ESTABLISHED", rta_state)

    try:
        theorem = load_verified_theory_statement(
            Path(__file__).resolve().parents[1] / "theory",
            "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return _finish("UNRESOLVED", "IMPORTED_C_AMC_SEM_THEOREM_UNAVAILABLE", {"error": str(exc)})
    if theorem.get("theorem_id") != "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY":
        return _finish("FAIL", "IMPORTED_THEOREM_ID_MISMATCH")

    return _finish("PASS", witness={
        "theorem_id": theorem["theorem_id"],
        "statement_hash": theorem["statement_hash"],
        "assumption_hash": theorem["assumption_hash"],
        "prefix_taskset_fingerprint": expected_fp,
        "model_conformance_artifact_hash": conformance_receipt.get("artifact_hash"),
        "all_task_rta_artifact_hash": rta_receipt.get("artifact_hash"),
        "all_tasks_covered": True,
        "all_deadlines_met": True,
        "conclusion": "PROTECTED_PREFIX_TASKSET_SCHEDULABLE",
    })


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
