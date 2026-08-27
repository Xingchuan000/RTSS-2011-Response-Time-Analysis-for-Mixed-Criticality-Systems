from __future__ import annotations

from copy import deepcopy

from formal_toolchain.bridge.controller_transition import build_controller_transition_case_proofs
from formal_toolchain.bridge.model_bounds import P0ModelBounds
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object

_CONTEXT = "0" * 64


def _controller_certificate(*, selected_budget_effect: str = "CERTIFIED_ACTION_UPDATE") -> dict:
    selected = {
        "case_id": "CONTROLLER_SELECTED_ACTION",
        "source": "amc_py/rl/env.py:AmcBudgetEnv.step",
        "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "self._engine.apply_budget_updates",
        "zero_time": True, "plant_progression": False, "timing_projection": "STUTTER",
        "payload_prevalidated": True, "no_partial_mutation": True,
        "time_unchanged": True, "mode_unchanged": True,
        "active_jobs_unchanged": True, "ready_jobs_unchanged": True,
        "running_job_unchanged": True, "released_job_fields_unchanged": True,
        "released_job_snapshot_unchanged": True, "released_job_service_unchanged": True,
        "released_job_demand_unchanged": True, "released_job_classification_unchanged": True,
        "completion_miss_unchanged": True, "service_unchanged": True,
        "effective_event_frontier_unchanged": True,
        "budget_effect": selected_budget_effect, "status": "PASS",
    }
    noop = {
        "case_id": "CONTROLLER_NO_ACTION", "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "EXPLICIT_OR_FALLBACK_NOOP", "zero_time": True,
        "budget_effect": "IDENTITY", "timing_projection": "STUTTER",
        "plant_progression": False, "running_job_unchanged": True,
        "effective_event_frontier_unchanged": True, "released_job_fields_unchanged": True,
        "mode_unchanged": True, "time_unchanged": True,
        "same_action_id": True, "same_budget_semantics": True, "same_timing_semantics": True,
        "status": "PASS",
    }
    return obligation_certificate(
        obligation_id="CONTROLLER_TRANSITION", status="PASS", context_hash=_CONTEXT,
        inputs={"controller_binding_hash": sha256_object({"controller": "source-bound"})},
        witness={"case_ids": ("CONTROLLER_SELECTED_ACTION", "CONTROLLER_NO_ACTION"),
                 "cases": {"CONTROLLER_SELECTED_ACTION": selected, "CONTROLLER_NO_ACTION": noop},
                 "selected_action_ok": True, "noop_ok": True, "source_kind": "CONTROLLER_SYNCHRONOUS"},
        checker_id="test.controller_transition", checker_version="test-v1",
    )


def _bounds() -> P0ModelBounds:
    return P0ModelBounds(task_slots=2, job_slots=2, queue_slots=4, max_preemptions_per_job=2)


def test_controller_certificate_is_compiled_into_two_real_smt_transition_proofs() -> None:
    result = build_controller_transition_case_proofs(
        controller_transition_certificate=_controller_certificate(), bounds=_bounds(), bridge_context_hash=_CONTEXT)
    assert result["status"] == "PASS"
    assert [row["case_id"] for row in result["proofs"]] == ["CONTROLLER_NO_ACTION", "CONTROLLER_SELECTED_ACTION"]
    for row in result["proofs"]:
        assert row["concrete_feasibility"] == "SAT"
        assert row["reference_totality"] == "PASS"
        assert row["relation_preservation"] == "PASS"
        assert row["z3_proof_result"] == "PASS"
        assert row["parameterized_contract_status"] == "PASS"


def test_controller_transition_case_proofs_are_deterministic_for_fresh_replay() -> None:
    certificate = _controller_certificate()
    first = build_controller_transition_case_proofs(controller_transition_certificate=certificate, bounds=_bounds(), bridge_context_hash=_CONTEXT)
    second = build_controller_transition_case_proofs(controller_transition_certificate=deepcopy(certificate), bounds=_bounds(), bridge_context_hash=_CONTEXT)
    assert first == second


def test_controller_smt_bridge_fails_closed_if_certified_effect_is_tampered() -> None:
    result = build_controller_transition_case_proofs(
        controller_transition_certificate=_controller_certificate(selected_budget_effect="IDENTITY"),
        bounds=_bounds(), bridge_context_hash=_CONTEXT)
    assert result["status"] == "UNRESOLVED"
    assert result["failure"] == "CONTROLLER_TRANSITION_CASE_BINDING_INVALID"
