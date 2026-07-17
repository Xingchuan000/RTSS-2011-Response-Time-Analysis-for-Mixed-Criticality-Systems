"""Phase I/J/K 的黄金、边界和 fail-closed 测试。"""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from formal_toolchain.bridge.deadline_removal import (
    build_release_fixed_removal_certificate,
    exact_removal_demand,
)
from formal_toolchain.bridge.event_projection import project_event
from formal_toolchain.bridge.state_relation import (
    P0ConcreteState,
    P0Event,
    P0Job,
    P0ReferenceState,
    relation_holds,
)
from formal_toolchain.bridge.transition_cases import prove_smt2_case
from formal_toolchain.bridge.case_templates import compile_bound_path_effect, compile_case_template
from formal_toolchain.bridge.transition_cases import TransitionCaseProof, check_handler_coverage
from formal_toolchain.bridge.job_mapping import (
    build_parameterized_release_mapping_certificate,
    verify_parameterized_release_mapping_certificate,
)
from formal_toolchain.compiler.compile import compile_request
from formal_toolchain.verifier.recompute import (
    _fresh_bridge_proofs,
    _fresh_reference_taskset,
    candidate_evidence,
    load_verifier_inputs,
)
from formal_toolchain.reference.arithmetic import ceil_div_nonnegative, floor_div_nonnegative, post_count
from formal_toolchain.reference.rta_production import protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta
from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
from formal_toolchain.verifier.reference_mapping_verifier import verify_reference_mapping
from formal_toolchain.reference.task_mapping import (
    ReferenceTask,
    ReferenceTaskset,
    build_reference_taskset,
    degraded_cost,
)
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.verifier.envelope_checker import independently_verify_envelope
from formal_toolchain.workflow.seed_workspace import freeze_seed_workspace


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_p0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TOY_CONTEXT_HASH = "a" * 64


def certified_test_envelope(upper):
    preservation = {"obligation_status": "PASS", "test_input": "explicit"}
    return {"schema_version": "certified_envelope_v1", "status": "PASS",
            "upper": dict(upper), "active_release_budget_upper": dict(upper),
            "preservation_certificate": preservation,
            "preservation_certificate_hash": sha256_object(preservation)}


def test_degraded_cost_uses_python_ties_to_even_and_clamps():
    assert degraded_cost(5, xf=0.5) == 2
    assert degraded_cost(4, xf=0.5) == 2
    assert degraded_cost(9, xf=0.01) == 1
    assert degraded_cost(9, xf=2.0) == 9


def test_reference_mapping_keeps_lo_code_hi_out_of_degraded_cost():
    task = ReferenceTask("lo", 10, 10, 4, 4, "LO", 0, 4, 99, 2)
    assert task.c_hi == 4
    assert task.degraded_cost == 2


def test_reference_mapping_fails_closed_when_budget_provenance_is_missing():
    from amc_py.models import Criticality, Task
    with pytest.raises(ValueError, match="budget provenance"):
        build_reference_taskset(
            (Task("lo", 10, 10, 4, 4, Criticality.LO),),
            {},
            xf=0.5,
            certified_envelope={"upper": {"lo": 8}},
            semantic_context_hash=TOY_CONTEXT_HASH,
            effective_runtime_config_hash=TOY_CONTEXT_HASH,
        )


def test_reference_mapping_uses_certified_b_bar_not_budget_floor():
    from amc_py.models import Criticality, Task
    envelope = certified_test_envelope({"lo": 8})
    reference = build_reference_taskset(
        (Task("lo", 10, 10, 4, 4, Criticality.LO),),
    {"lo": {"b_bar": 8, "budget_floor": 0,
                 "certified_envelope_hash": sha256_object(envelope)}},
        xf=0.5,
        certified_envelope=envelope,
        semantic_context_hash=TOY_CONTEXT_HASH,
        effective_runtime_config_hash=TOY_CONTEXT_HASH,
    )
    assert reference.tasks[0].c_lo == 9
    assert reference.tasks[0].c_hi == 2


def test_reference_mapping_rejects_provenance_that_forges_certified_upper():
    from amc_py.models import Criticality, Task
    envelope = certified_test_envelope({"lo": 8})
    with pytest.raises(ValueError, match="provenance b_bar"):
        build_reference_taskset(
            (Task("lo", 10, 10, 4, 4, Criticality.LO),),
            {"lo": {"b_bar": 7, "certified_envelope_hash": sha256_object(envelope)}},
            xf=0.5, certified_envelope=envelope,
            semantic_context_hash=TOY_CONTEXT_HASH,
            effective_runtime_config_hash=TOY_CONTEXT_HASH,
        )


def test_independent_mapping_verifier_rejects_tampered_reference_cost():
    from amc_py.models import Criticality, Task
    envelope = certified_test_envelope({"lo": 8})
    tasks = (Task("lo", 10, 10, 4, 4, Criticality.LO),)
    metadata = {"lo": {"b_bar": 8, "budget_floor": 0,
                        "certified_envelope_hash": sha256_object(envelope)}}
    reference = build_reference_taskset(
        tasks, metadata, xf=0.5, certified_envelope=envelope,
        semantic_context_hash=TOY_CONTEXT_HASH,
        effective_runtime_config_hash=TOY_CONTEXT_HASH,
    )
    tampered = replace(reference, tasks=(replace(reference.tasks[0], c_lo=7),))
    result = verify_reference_mapping(
        reference=tampered, ordered_tasks=tasks, budget_by_task=metadata,
        certified_envelope=envelope, xf=0.5,
        semantic_context_hash=TOY_CONTEXT_HASH,
        effective_runtime_config_hash=TOY_CONTEXT_HASH,
    )
    assert result["obligation_status"] == "FAIL"


def toy_taskset():
    return ReferenceTaskset((
        ReferenceTask("t1", 10, 10, 2, 1, "LO", 0, 2, 1),
        ReferenceTask("t2", 40, 40, 10, 12, "HI", 1, 10, 12),
        ReferenceTask("t3", 200, 200, 40, 60, "HI", 2, 40, 60),
    ), source_context_hash=TOY_CONTEXT_HASH)


def test_toy_rta_production_and_independent_replay():
    result = protected_hi_rta(toy_taskset())
    assert result["status"] == "PASS"
    by_name = {row["task"]["name"]: row for row in result["tasks"]}
    assert (by_name["t2"]["lo"]["r_lo"], by_name["t2"]["start"]["w_lo"]) == (14, 2)
    assert (by_name["t3"]["lo"]["r_lo"], by_name["t3"]["start"]["w_lo"]) == (76, 14)
    assert max(row["relative_response"] for row in by_name["t3"]["case2"]) == 108
    # 计划规定 Case 2 的有效域是 range(W)，因此 s=W=14 不得出现。
    assert max(row["start"] for row in by_name["t3"]["case2"] if "start" in row) == 13
    assert replay_rta(toy_taskset(), result)["status"] == "PASS"


def test_case1_uses_plan_response_formula_and_toy_golden_values():
    result = protected_hi_rta(toy_taskset())
    by_name = {row["task"]["name"]: row for row in result["tasks"]}
    assert result["status"] == "PASS"
    assert by_name["t2"]["r_hi"] == 15
    assert by_name["t2"]["r_star"] == 15
    assert by_name["t3"]["r_hi"] == 108
    assert by_name["t3"]["r_star"] == 108


def test_replay_rejects_tampered_production_trace():
    production = protected_hi_rta(toy_taskset())
    tampered = deepcopy(production)
    tampered["tasks"][0]["case1"][0]["trace"][0]["f"] += 1
    assert replay_rta(toy_taskset(), tampered)["status"] == "FAIL"


def test_recurring_and_corollary_require_verified_rta_predecessor():
    rta = protected_hi_rta(toy_taskset())
    recurring = build_recurring_hi_instances(toy_taskset(), rta_certificate=rta)
    corollary = protected_hi_safety_corollary(recurring)
    assert recurring["obligation_status"] == "PASS"
    assert corollary["status"] == "PASS"
    tampered = deepcopy(recurring)
    tampered["instances"][0]["r_star"] += 1
    assert protected_hi_safety_corollary(tampered)["status"] == "FAIL"


def test_obligation_certificate_hash_is_checked_independently():
    production = protected_hi_rta(toy_taskset())
    assert verify_obligation_certificate(production)
    production["tasks"][0]["status"] = "FAIL"
    assert not verify_obligation_certificate(production)


def test_exact_arithmetic_rejects_float_bool_and_negative_period():
    assert ceil_div_nonnegative(5, 2) == 3
    assert floor_div_nonnegative(5, 2) == 2
    assert post_count(0, 11, 10) == 2
    with pytest.raises((TypeError, ValueError)):
        ceil_div_nonnegative(True, 2)
    with pytest.raises((TypeError, ValueError)):
        post_count(0, 1, -1)


def test_release_fixed_removal_boundaries():
    assert exact_removal_demand(actual_cost=3, primary_mode="LO", release_budget=3) == 3
    assert exact_removal_demand(actual_cost=4, primary_mode="LO", release_budget=3) == 4
    assert exact_removal_demand(actual_cost=7, primary_mode="LO", release_budget=3) == 4
    assert exact_removal_demand(actual_cost=7, primary_mode="DEGRADED_LO", degraded_cost=2) == 2
    assert exact_removal_demand(actual_cost=7, primary_mode="HI") == 7


def test_event_projection_erases_controller_and_relabels_cancellation():
    assert project_event(P0Event(0, "CONTROLLER")) is None
    assert project_event(P0Event(1, "PRIMARY_LO_CANCELLATION", ("lo", 0))).kind == "JOB_COMPLETION"
    assert project_event(P0Event(2, "BUDGET_UPDATE_LABEL", payload=(("task", "lo"),))) is None


def test_state_relation_does_not_use_raw_concrete_budget_as_remaining():
    concrete = P0ConcreteState(0, "LO")
    reference = P0ReferenceState(0, "LO")
    assert relation_holds(concrete, reference)
    assert not relation_holds(
        P0ConcreteState(0, "LO", global_future_budgets=(("lo", 2),)),
        P0ReferenceState(0, "LO", global_future_budgets=(("lo", 3),)),
    )


def test_state_relation_keeps_release_budget_and_certificate_requires_context():
    job_c = P0Job(("lo", 0), 0, 0, 10, "normal", 3, 2)
    job_r = P0Job(("lo", 0), 0, 0, 10, "normal", 4, 2)
    assert not relation_holds(P0ConcreteState(0, "LO", (job_c,)),
                              P0ReferenceState(0, "LO", (job_r,)))
    with pytest.raises(ValueError, match="不能为空"):
        build_release_fixed_removal_certificate([], source_context_hash=TOY_CONTEXT_HASH)


def test_transition_proof_never_uses_unverified_pass_string():
    proof = prove_smt2_case(
        case_id="BOOT_TO_PRECLOSED_0", source_branch_id="boot",
        declarations="(declare-const x Int)", precondition="(>= x 0)",
        preservation="(>= x 0)", concrete_delta="(= x x)",
        projected_reference_delta="(= x x)", bound_source_hash="a" * 64,
    )
    assert proof.z3_proof_result in {"PASS", "UNRESOLVED"}
    assert proof.z3_proof_result != "FAIL"


def test_transition_checker_rejects_vacuous_reference_successor():
    template = compile_case_template("HI_RELEASE")
    # Reproduce the old bug: concrete used the LO B+1 clamp while reference
    # requires actual_cost.  actual=10,B=2 is a feasible concrete transition
    # with no related reference successor.
    wrong = compile_bound_path_effect({
        "case_id": "PRIMARY_LO_RELEASE",
        "effects": ["build_job", "release_fixed_budget", "active_add", "ready_add"],
    })
    proof = prove_smt2_case(
        case_id="HI_RELEASE", source_branch_id="release/hi",
        declarations=template.declarations,
        precondition=f"(and {template.precondition} (= actual_cost 10) (= release_budget 2))",
        preservation=template.preservation, concrete_delta=wrong,
        projected_reference_delta=template.reference_delta,
        bound_source_hash="a" * 64,
    )
    assert proof.concrete_feasibility == "SAT"
    assert proof.reference_totality == "FAIL"
    assert proof.relation_preservation == "FAIL"
    assert proof.z3_proof_result == "FAIL"


def test_compiler_defers_phase_k_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    imported = freeze_seed_workspace(FIXTURE_ROOT, "best_overall", workspace, code_root=PROJECT_ROOT)

    def _explode(*args, **kwargs):
        raise AssertionError("compiler should not call the Phase K generator")

    monkeypatch.setattr("formal_toolchain.bridge.compile_bridge.compile_phase_k", _explode)
    summary = compile_request(Path(imported["request"]), tmp_path / "candidate")
    assert summary["phase_k_candidate_status"] == "UNRESOLVED"
    assert summary["phase_k_candidate_failure"] == "FRESH_VERIFIER_REQUIRED"


def test_verifier_invokes_fresh_phase_k_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / "workspace"
    imported = freeze_seed_workspace(FIXTURE_ROOT, "best_overall", workspace, code_root=PROJECT_ROOT)
    compile_out = tmp_path / "candidate"
    compile_request(Path(imported["request"]), compile_out)

    inputs = load_verifier_inputs(Path(imported["request"]), source_root=PROJECT_ROOT)
    candidate_common = json.loads((compile_out / "artifacts" / "COMMON_TRANSITION_PRESERVATION.json").read_text(encoding="utf-8"))
    candidate_deployed = json.loads((compile_out / "artifacts" / "DEPLOYED_POLICY_PRESERVATION.json").read_text(encoding="utf-8"))
    envelope_state = independently_verify_envelope(
        candidate_envelope=candidate_evidence(
            json.loads((compile_out / "artifacts" / "CANDIDATE_ENVELOPE.json").read_text(encoding="utf-8"))
        ) or {},
        common_preservation=candidate_evidence(candidate_common) or {},
        deployed_preservation=candidate_evidence(candidate_deployed) or {},
        raw_inputs=inputs,
        invariant_context_hash=str(inputs.contexts["invariant_context"]["hash"]),
    )
    fresh_reference = _fresh_reference_taskset(inputs, envelope_state.certified_envelope)
    candidate_release_mapping = json.loads((compile_out / "artifacts" / "RELEASE_FIXED_REMOVAL_MAPPING.json").read_text(encoding="utf-8"))
    rta = protected_hi_rta(fresh_reference)
    recurring = build_recurring_hi_instances(fresh_reference, rta_certificate=rta)
    corollary = protected_hi_safety_corollary(recurring)
    fresh_certificates = {
        "SCHEDULER_MODEL": json.loads((compile_out / "artifacts" / "SCHEDULER_MODEL.json").read_text(encoding="utf-8")),
        "MODE_SEMANTICS_CONFORMANCE": json.loads((compile_out / "artifacts" / "MODE_SEMANTICS_CONFORMANCE.json").read_text(encoding="utf-8")),
        "DEMAND_ORACLE_BATCH_CONTRACT": json.loads((compile_out / "artifacts" / "DEMAND_ORACLE_BATCH_CONTRACT.json").read_text(encoding="utf-8")),
        "HI_EXECUTION_CONTRACT": json.loads((compile_out / "artifacts" / "HI_EXECUTION_CONTRACT.json").read_text(encoding="utf-8")),
        "REMOVAL_COMPLETENESS": json.loads((compile_out / "artifacts" / "REMOVAL_COMPLETENESS.json").read_text(encoding="utf-8")),
        "HI_NONTRUNCATION": json.loads((compile_out / "artifacts" / "HI_NONTRUNCATION.json").read_text(encoding="utf-8")),
        "DEADLINE_OBSERVATION": json.loads((compile_out / "artifacts" / "DEADLINE_OBSERVATION.json").read_text(encoding="utf-8")),
        "EFFECTIVE_EVENT_ORDER": json.loads((compile_out / "artifacts" / "EFFECTIVE_EVENT_ORDER.json").read_text(encoding="utf-8")),
        "BATCH_CLOSURE": json.loads((compile_out / "artifacts" / "BATCH_CLOSURE.json").read_text(encoding="utf-8")),
        "CONTROLLER_POSTCLOSURE": json.loads((compile_out / "artifacts" / "CONTROLLER_POSTCLOSURE.json").read_text(encoding="utf-8")),
        "TIME_PROGRESS": json.loads((compile_out / "artifacts" / "TIME_PROGRESS.json").read_text(encoding="utf-8")),
        "WINDOW_MODE_NORMALIZATION": json.loads((compile_out / "artifacts" / "WINDOW_MODE_NORMALIZATION.json").read_text(encoding="utf-8")),
        "CERTIFIED_ENVELOPE": {"obligation_id": "CERTIFIED_ENVELOPE", "obligation_status": "PASS"},
        "PROTECTED_HI_SAFETY_COROLLARY": corollary,
        "RELEASE_FIXED_REMOVAL_MAPPING": candidate_release_mapping,
    }

    calls = {"count": 0}

    def _wrapped(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("bridge generator was reached")

    monkeypatch.setattr(
        "formal_toolchain.bridge.runtime_branch_map.build_runtime_branch_map",
        lambda *args, **kwargs: {
            "status": "PASS",
            "source_hash": "a" * 64,
            "path_map_hash": "b" * 64,
            "paths": [],
            "coverage": {"status": "PASS"},
        },
    )
    monkeypatch.setattr("formal_toolchain.bridge.compile_bridge.compile_phase_k", _wrapped)
    with pytest.raises(AssertionError, match="bridge generator was reached"):
        _fresh_bridge_proofs(
            inputs=inputs, fresh_certificates=fresh_certificates, fresh_reference=fresh_reference)
    assert calls["count"] >= 1


def test_release_batch_and_time_templates_bind_runtime_semantics():
    hi = compile_bound_path_effect({"case_id": "HI_RELEASE",
        "effects": ["build_job", "release_fixed_budget", "active_add", "ready_add"]})
    degraded = compile_bound_path_effect({"case_id": "DEGRADED_LO_RELEASE",
        "effects": ["degraded_budget", "actual_cost_clamp", "active_add", "ready_add"]})
    batch = compile_bound_path_effect({"case_id": "ARRIVAL_BATCH_NO_SWITCH",
        "effects": ["arrival_batch", "release", "reschedule"]})
    service = compile_case_template("ONE_SERVICE_TICK")
    jump = compile_case_template("JUMP_TO_NEXT_EVENT")
    assert "(= expected_demand actual_cost)" in hi
    assert "degraded_cost" in degraded
    assert "(+ c_active 1)" not in batch and "(+ c_ready 1)" not in batch
    assert "elapsed" in service.precondition and "(+ r_service elapsed)" in service.reference_delta
    assert "(= c_running 0)" in jump.precondition and "(= c_ready 0)" in jump.precondition


def test_affected_job_identity_mismatch_fails_even_when_counts_match():
    declarations = "\n".join([
        "(declare-const c_affected_job_key_post Int)",
        "(declare-const r_affected_job_key_post Int)",
    ])
    proof = prove_smt2_case(
        case_id="NORMAL_COMPLETION", source_branch_id="completion/identity",
        declarations=declarations, precondition="true",
        concrete_delta="(= c_affected_job_key_post 1)",
        projected_reference_delta="(= r_affected_job_key_post 2)",
        preservation="(= c_affected_job_key_post r_affected_job_key_post)",
        bound_source_hash="a" * 64,
    )
    assert proof.concrete_feasibility == "SAT"
    assert proof.relation_preservation == "FAIL"
    assert proof.z3_proof_result == "FAIL"


def test_parameterized_release_mapping_and_many_to_one_case_mapping():
    certificate = build_parameterized_release_mapping_certificate(source_context_hash=TOY_CONTEXT_HASH)
    assert verify_parameterized_release_mapping_certificate(certificate)
    first = TransitionCaseProof("BOOT_TO_PRECLOSED_0", "true", "true", "true", "true",
                                "PASS", "a" * 64, "branch-a", True)
    second = TransitionCaseProof("BOOT_TO_PRECLOSED_0", "true", "true", "true", "true",
                                 "PASS", "a" * 64, "branch-b", True)
    coverage = check_handler_coverage(("branch-a", "branch-b"), (first, second))
    assert coverage["status"] == "PASS"
