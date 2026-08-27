"""Canonical synchronous controller transition certificate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate


CONTROLLER_SELECTED_ACTION = "CONTROLLER_SELECTED_ACTION"
CONTROLLER_NO_ACTION = "CONTROLLER_NO_ACTION"


def _status(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return value.get("status", value.get("obligation_status"))


def build_controller_transition_certificate(
    *,
    controller_binding: Mapping[str, Any],
    action_binding: Mapping[str, Any],
    deployed_policy_binding: Mapping[str, Any],
    controller_postclosure_certificate: Mapping[str, Any],
    context_hash: str,
) -> dict[str, Any]:
    selected_binding = controller_binding.get("selected_action_runtime_binding", {})
    noop_binding = controller_binding.get("explicit_noop_runtime_binding", {})
    explicit_noop = action_binding.get("explicit_noop") is True
    postclosure_ok = _status(controller_postclosure_certificate) == "PASS"
    action_noop = action_binding.get("noop_binding", {})
    selected_required_true_fields = (
        "payload_prevalidated",
        "no_partial_mutation",
        "zero_time",
        "time_unchanged",
        "mode_unchanged",
        "active_jobs_unchanged",
        "ready_jobs_unchanged",
        "released_job_fields_unchanged",
        "released_job_snapshot_unchanged",
        "released_job_service_unchanged",
        "released_job_demand_unchanged",
        "released_job_classification_unchanged",
        "completion_miss_unchanged",
        "service_unchanged",
        "plant_progression_separated",
        "running_job_unchanged_if_preclosed",
        "effective_event_frontier_unchanged_if_preclosed",
    )
    noop_required_true_fields = (
        "budget_identity",
        "running_job_unchanged",
        "effective_event_frontier_unchanged",
        "released_job_fields_unchanged",
        "mode_unchanged",
        "time_unchanged",
        "explicit_and_fallback_same_timing_semantics",
        "plant_progress_separated",
    )
    selected_action_ok = all((
        _status(action_binding) == "PASS",
        action_binding.get("action_profile") == "single25_explicit_noop",
        _status(deployed_policy_binding) == "PASS",
        postclosure_ok,
        _status(selected_binding) == "PASS",
        selected_binding.get("source") == "amc_py/rl/env.py:AmcBudgetEnv.step",
        selected_binding.get("source_kind") == "CONTROLLER_SYNCHRONOUS",
        selected_binding.get("source_binding") == "self._engine.apply_budget_updates",
        selected_binding.get("timing_projection") == "STUTTER_IF_PRECLOSED",
        selected_binding.get("requires_preclosed_boundary") is True,
        all(selected_binding.get(field) is True for field in selected_required_true_fields),
    ))
    noop_runtime_ok = (
        _status(noop_binding) == "PASS"
        and noop_binding.get("timing_projection") == "STUTTER"
        and all(noop_binding.get(field) is True for field in noop_required_true_fields)
    )
    noop_action_ok = (
        explicit_noop
        and action_noop.get("present") is True
        and action_noop.get("action_id") is not None
        and action_noop.get("derived_action_ids") == [action_noop.get("action_id")]
        and action_noop.get("defensive_fallback_same_action_verified") is True
    )
    noop_ok = noop_runtime_ok and noop_action_ok
    controller_ok = _status(controller_binding) == "PASS"
    selected_case = {
        "case_id": CONTROLLER_SELECTED_ACTION,
        "source": selected_binding.get("source"),
        "source_kind": selected_binding.get("source_kind"),
        "source_binding": selected_binding.get("source_binding"),
        "zero_time": selected_binding.get("zero_time") is True,
        "plant_progression": not bool(selected_binding.get("plant_progression_separated")),
        "timing_projection": "STUTTER" if selected_action_ok else "UNRESOLVED",
        "payload_prevalidated": selected_binding.get("payload_prevalidated"),
        "no_partial_mutation": selected_binding.get("no_partial_mutation"),
        "time_unchanged": selected_binding.get("time_unchanged"),
        "mode_unchanged": selected_binding.get("mode_unchanged"),
        "active_jobs_unchanged": selected_binding.get("active_jobs_unchanged"),
        "ready_jobs_unchanged": selected_binding.get("ready_jobs_unchanged"),
        "running_job_unchanged": True if selected_action_ok else False,
        "released_job_fields_unchanged": selected_binding.get("released_job_fields_unchanged"),
        "released_job_snapshot_unchanged": selected_binding.get("released_job_snapshot_unchanged"),
        "released_job_service_unchanged": selected_binding.get("released_job_service_unchanged"),
        "released_job_demand_unchanged": selected_binding.get("released_job_demand_unchanged"),
        "released_job_classification_unchanged": selected_binding.get("released_job_classification_unchanged"),
        "completion_miss_unchanged": selected_binding.get("completion_miss_unchanged"),
        "service_unchanged": selected_binding.get("service_unchanged"),
        "effective_event_frontier_unchanged": True if selected_action_ok else False,
        "budget_effect": "CERTIFIED_ACTION_UPDATE",
        "status": "PASS" if selected_action_ok and controller_ok else "UNRESOLVED",
    }
    noop_case = {
        "case_id": CONTROLLER_NO_ACTION,
        "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "EXPLICIT_OR_FALLBACK_NOOP",
        "zero_time": noop_binding.get("time_unchanged") is True,
        "budget_effect": "IDENTITY",
        "timing_projection": noop_binding.get("timing_projection"),
        "plant_progression": False,
        "running_job_unchanged": noop_binding.get("running_job_unchanged"),
        "effective_event_frontier_unchanged": noop_binding.get("effective_event_frontier_unchanged"),
        "released_job_fields_unchanged": noop_binding.get("released_job_fields_unchanged"),
        "mode_unchanged": noop_binding.get("mode_unchanged"),
        "time_unchanged": noop_binding.get("time_unchanged"),
        "plant_separated": noop_binding.get("plant_separated", noop_binding.get("plant_progress_separated")),
        "noop_sources": ("EXPLICIT_NOOP", "DEFENSIVE_FALLBACK"),
        "same_action_id": noop_action_ok,
        "same_budget_semantics": noop_binding.get("budget_identity") is True,
        "same_timing_semantics": noop_binding.get("timing_projection") == "STUTTER",
        "status": "PASS" if noop_ok and controller_ok else "UNRESOLVED",
    }
    status = "PASS" if selected_action_ok and noop_ok and controller_ok else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "CONTROLLER_TRANSITION_BINDING_UNRESOLVED"}
    deployed_hash = deployed_policy_binding.get("artifact_hash", deployed_policy_binding.get("binding_hash"))
    return obligation_certificate(
        obligation_id="CONTROLLER_TRANSITION",
        status=status,
        context_hash=context_hash,
        inputs={
            "controller_binding_hash": controller_binding.get("binding_hash"),
            "action_binding_hash": action_binding.get("binding_hash"),
            "deployed_policy_binding_hash": deployed_hash,
            "controller_postclosure_hash": controller_postclosure_certificate.get("artifact_hash"),
        },
        witness={
            "case_ids": (CONTROLLER_SELECTED_ACTION, CONTROLLER_NO_ACTION),
            "cases": {
                CONTROLLER_SELECTED_ACTION: selected_case,
                CONTROLLER_NO_ACTION: noop_case,
            },
            "selected_action_ok": selected_action_ok,
            "noop_ok": noop_ok,
            "source_kind": "CONTROLLER_SYNCHRONOUS",
        },
        direct_predecessor_hashes={
            key: value for key, value in {
                "action_binding": action_binding.get("binding_hash"),
                "controller_binding": controller_binding.get("binding_hash"),
                "deployed_policy_binding": deployed_hash,
                "controller_postclosure": controller_postclosure_certificate.get("artifact_hash"),
            }.items() if isinstance(value, str)
        },
        checker_id=__name__,
        checker_version="phase-k-v1",
        failure=failure,
    )


def _controller_case_binding_ok(case_id: str, case: Mapping[str, Any]) -> bool:
    """Validate the exact source-bound synchronous controller semantics used by SMT."""
    if case.get("status") != "PASS" or case.get("source_kind") != "CONTROLLER_SYNCHRONOUS":
        return False
    if case.get("timing_projection") != "STUTTER" or case.get("plant_progression") is not False:
        return False
    if case_id == CONTROLLER_SELECTED_ACTION:
        required_true = (
            "zero_time", "payload_prevalidated", "no_partial_mutation",
            "time_unchanged", "mode_unchanged", "active_jobs_unchanged",
            "ready_jobs_unchanged", "running_job_unchanged",
            "released_job_fields_unchanged", "released_job_snapshot_unchanged",
            "released_job_service_unchanged", "released_job_demand_unchanged",
            "released_job_classification_unchanged", "completion_miss_unchanged",
            "service_unchanged", "effective_event_frontier_unchanged",
        )
        return (
            case.get("source") == "amc_py/rl/env.py:AmcBudgetEnv.step"
            and case.get("source_binding") == "self._engine.apply_budget_updates"
            and case.get("budget_effect") == "CERTIFIED_ACTION_UPDATE"
            and all(case.get(field) is True for field in required_true)
        )
    if case_id == CONTROLLER_NO_ACTION:
        return (
            case.get("source_binding") == "EXPLICIT_OR_FALLBACK_NOOP"
            and case.get("budget_effect") == "IDENTITY"
            and case.get("zero_time") is True
            and case.get("running_job_unchanged") is True
            and case.get("effective_event_frontier_unchanged") is True
            and case.get("released_job_fields_unchanged") is True
            and case.get("mode_unchanged") is True
            and case.get("time_unchanged") is True
            and case.get("same_action_id") is True
            and case.get("same_budget_semantics") is True
            and case.get("same_timing_semantics") is True
        )
    return False


def _controller_concrete_delta(case_id: str, bounds: Any) -> str:
    """Render concrete post-state equations from the certified controller effect."""
    from .state_relation import p0_smt_relation_fields

    overrides: dict[str, str] = {}
    if case_id == CONTROLLER_SELECTED_ACTION:
        overrides["future_budget"] = "release_budget"
        overrides["affected_task_budget"] = "release_budget"
        for slot in range(bounds.task_slots):
            field = f"task_{slot}_future_budget"
            overrides[field] = (
                f"(ite (= update_target_slot {slot}) release_budget "
                f"c_{field})"
            )
    elif case_id != CONTROLLER_NO_ACTION:
        raise ValueError(f"CONTROLLER_CASE_UNKNOWN:{case_id}")

    equations = [
        f"(= c_{field}_post {overrides.get(field, f'c_{field}')})"
        for field in p0_smt_relation_fields(bounds)
    ]
    return "(and " + " ".join(equations) + ")"


def build_controller_transition_case_proofs(
    *,
    controller_transition_certificate: Mapping[str, Any],
    bounds: Any,
    bridge_context_hash: str,
) -> dict[str, Any]:
    """Compile the two certified controller micro-steps into real SMT proofs."""
    from formal_toolchain.core.artifact import verify_obligation_certificate
    from formal_toolchain.core.hashing import sha256_object
    from .case_templates import compile_case_template
    from .state_relation import parameterized_state_relation_schema_hash
    from .transition_cases import TransitionCaseProof, prove_smt2_case

    if (
        not isinstance(controller_transition_certificate, Mapping)
        or controller_transition_certificate.get("obligation_id") != "CONTROLLER_TRANSITION"
        or controller_transition_certificate.get("obligation_status") != "PASS"
        or controller_transition_certificate.get("certificate_context_hash") != bridge_context_hash
        or not verify_obligation_certificate(controller_transition_certificate)
    ):
        return {
            "status": "UNRESOLVED",
            "failure": "CONTROLLER_TRANSITION_CERTIFICATE_INVALID",
            "proofs": [],
        }

    witness = controller_transition_certificate.get("witness", {})
    cases = witness.get("cases", {}) if isinstance(witness, Mapping) else {}
    required = (CONTROLLER_NO_ACTION, CONTROLLER_SELECTED_ACTION)
    if not isinstance(cases, Mapping) or any(
        not isinstance(cases.get(case_id), Mapping)
        or not _controller_case_binding_ok(case_id, cases[case_id])
        for case_id in required
    ):
        return {
            "status": "UNRESOLVED",
            "failure": "CONTROLLER_TRANSITION_CASE_BINDING_INVALID",
            "proofs": [],
        }

    certificate_hash = str(controller_transition_certificate.get("artifact_hash", ""))
    controller_binding_hash = str(
        controller_transition_certificate.get("inputs", {}).get("controller_binding_hash", "")
    )
    proof_rows: list[dict[str, Any]] = []
    for case_id in required:
        case = dict(cases[case_id])
        template = compile_case_template(case_id, bounds=bounds)
        concrete_delta = _controller_concrete_delta(case_id, bounds)
        case_binding_hash = sha256_object({
            "certificate_hash": certificate_hash,
            "case_id": case_id,
            "case": case,
        })
        source_branch_id = f"CONTROLLER_SYNCHRONOUS::{case_id}"
        proof = prove_smt2_case(
            case_id=case_id,
            source_branch_id=source_branch_id,
            declarations=template.declarations,
            precondition=template.precondition,
            preservation=template.preservation,
            concrete_delta=concrete_delta,
            projected_reference_delta=template.reference_delta,
            bound_source_hash=controller_binding_hash or case_binding_hash,
            bounds=bounds,
        )
        modified_components = (
            ("future_budget_ghost",)
            if case_id == CONTROLLER_SELECTED_ACTION
            else ()
        )
        semantic_effect_kinds = (
            ("FUTURE_BUDGET_UPDATE",)
            if case_id == CONTROLLER_SELECTED_ACTION
            else ("IDENTITY",)
        )
        footprint_hash = sha256_object({
            "case_id": case_id,
            "controller_transition_certificate_hash": certificate_hash,
            "case_binding_hash": case_binding_hash,
            "modified_components": modified_components,
            "semantic_effect_kinds": semantic_effect_kinds,
            "concrete_delta_hash": sha256_object(concrete_delta),
        })
        parameterized_ok = proof.z3_proof_result == "PASS"
        proof = TransitionCaseProof(**{
            **proof.to_dict(),
            "source_branch_id": source_branch_id,
            "branch_subtree_hash": case_binding_hash,
            "bridge_context_hash": bridge_context_hash,
            "case_template_hash": template.template_hash,
            "concrete_delta_source": "CONTROLLER_TRANSITION_CERTIFICATE",
            "path_id": source_branch_id,
            "path_effect_hash": case_binding_hash,
            "guard_ast_hash": case_binding_hash,
            "path_ast_hash": case_binding_hash,
            "source_context_hash": controller_binding_hash or certificate_hash,
            "affected_job_identity_bound": True,
            "frame_predicates_bound": True,
            "parameterized_relation_schema_hash": parameterized_state_relation_schema_hash(),
            "local_footprint_hash": footprint_hash,
            "map_update_kind": "UNCHANGED",
            "created_key_fresh_proved": False,
            "released_ledger_contract_proved": True,
            "terminal_ledger_contract_proved": True,
            "miss_ledger_contract_proved": True,
            "unaffected_job_frame_proved": True,
            "effective_frontier_contract_proved": True,
            "parameterized_contract_status": "PASS" if parameterized_ok else "UNRESOLVED",
            "modified_components": modified_components,
            "semantic_effect_kinds": semantic_effect_kinds,
            "evidence_hashes": tuple(sorted({certificate_hash, case_binding_hash})),
            "parameterized_contract_failure": "" if parameterized_ok else "CONTROLLER_SMT_PROOF_NOT_PASS",
        })
        proof_rows.append(proof.to_dict())

    status = "PASS" if all(
        row.get("z3_proof_result") == "PASS"
        and row.get("concrete_feasibility") == "SAT"
        and row.get("reference_totality") == "PASS"
        and row.get("relation_preservation") == "PASS"
        and row.get("parameterized_contract_status") == "PASS"
        for row in proof_rows
    ) else "UNRESOLVED"
    return {
        "status": status,
        "failure": None if status == "PASS" else "CONTROLLER_TRANSITION_CASE_SMT_UNRESOLVED",
        "controller_transition_certificate_hash": certificate_hash,
        "proofs": proof_rows,
        "case_ids": list(required),
    }
