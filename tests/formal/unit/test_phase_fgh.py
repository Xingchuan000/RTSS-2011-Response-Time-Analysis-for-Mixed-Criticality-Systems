"""Phase F/G/H 的最小正向和负向合同测试。"""

from dataclasses import dataclass

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from formal_toolchain.adapters.batch_frozen_scenario import BatchFrozenExecutionScenario
from formal_toolchain.conformance.scheduler import check_scheduler_model
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.invariant.certified_envelope import certify_envelope
from formal_toolchain.policy.mask_fallback import select_first_valid
from formal_toolchain.adapters.synthetic_policy import mask_and_reasons, build_transition_witness
from formal_toolchain.policy.quantization import replay_quantize


@pytest.fixture
def tasks():
    return (Task("L", 10, 10, 2, 2, Criticality.LO), Task("H", 20, 20, 3, 4, Criticality.HI))


def test_phase_f_rejects_bool_as_formal_integer(tasks):
    bad = Task("L", 10, 10, 2, 2, Criticality.LO)
    object.__setattr__(bad, "period", True)
    facts = {name: True for name in ("ready_selects_highest_priority", "tick_boundary_preemption",
                                     "work_conserving", "no_blocking", "no_self_suspension",
                                     "no_non_preemptive_sections", "sporadic_release_contract")}
    facts["evidence"] = {"source": "test"}
    facts["binding"] = facts["evidence"]
    facts["binding_hash"] = __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(facts["binding"])
    with pytest.raises(ValueError):
        check_scheduler_model((bad, tasks[1]), scheduler_facts=facts)


def test_batch_freezes_simultaneous_reads(tasks):
    calls = []
    def demand(task, index):
        calls.append((task.name, index))
        return index + 1
    adapter = BatchFrozenExecutionScenario(tasks, demand)
    assert adapter.demand(tasks[0], 0) == 1
    assert adapter.demand(tasks[1], 0) == 1
    assert calls == [("L", 0), ("H", 0)]


def test_phase_g_quantization_and_fallback():
    config = {"input_min": 0.0, "input_max": 1.0, "scale": 1_000_000, "output_min": 0, "output_max": 1_000_000}
    assert replay_quantize(0.5, config)[0] == 500000
    assert select_first_valid((2, 0, 1), (False, True, False), action_dim=3) == 1
    assert select_first_valid((2, 0, 1), (False, False, False), action_dim=3) is None


def test_phase_h_requires_domain_and_certifies(tasks):
    metadata = {"L": {"initial_runtime_budget": 2, "budget_floor": 1, "budget_cap": 10},
                "H": {"initial_runtime_budget": 3, "budget_floor": 3, "budget_cap": 4}}
    domain = build_budget_domain(tasks, metadata, runtime_config=type("Config", (), {"processor_overhead": 0})())
    domain["context_hash"] = "a" * 64
    actions = build_budget_action_space(tasks, action_space="single", budget_increase_ratio=.02, budget_decrease_ratio=.02)
    candidate = synthesize_candidate_envelope(domain, actions, tasks, context_hash="a" * 64)
    transitions = build_transition_witness(domain, tasks)
    common = check_common_transition_preservation(candidate, transitions=transitions)
    runtime_state = {"budgets": {"L": 2, "H": 3}, "criticality": {"L": "LO", "H": "HI"}, "floor": {}}
    runtime_mask, runtime_reasons = mask_and_reasons(runtime_state, actions, tasks)
    deployed = check_deployed_policy_preservation(candidate, actions, tasks, leaves=(0,),
        selected_cases=tuple({"leaf_id": 0, "rank_position": index, "action_id": action.action_id, "valid": index == 0, "mask_reasons": runtime_reasons,
                              "ranking": tuple(item.action_id for item in actions), "mask": runtime_mask,
                              "runtime_state": runtime_state,
                              "action_definitions": tuple({"target_task": "L", "direction": "increase"} for _ in actions)}
                             for index, action in enumerate(actions)))
    assert candidate["status"] == common["status"] == deployed["status"] == "PASS"
    attestation = {"fresh_process": True,
                   "candidate_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(candidate),
                   "common_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(common),
                   "deployed_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(deployed)}
    with pytest.raises(ValueError):
        certify_envelope(candidate, common, deployed, context_hash="a" * 64,
                         verifier_attestation=attestation)


def test_phase_h_rejects_unresolved_dynamic_action(tasks):
    metadata = {"L": {"initial_runtime_budget": 2, "budget_floor": 1, "budget_cap": 10},
                "H": {"initial_runtime_budget": 3, "budget_floor": 3, "budget_cap": 4}}
    domain = build_budget_domain(tasks, metadata, runtime_config=type("Config", (), {"processor_overhead": 0})())
    domain["context_hash"] = "a" * 64
    actions = build_budget_action_space(tasks, action_space="residual_ranked")
    candidate = synthesize_candidate_envelope(domain, actions, tasks, context_hash="a" * 64)
    assert candidate["status"] == "UNRESOLVED"
    with pytest.raises(ValueError):
        check_deployed_policy_preservation(candidate, actions, tasks)


def test_phase_fh_empty_registry_and_direct_certification_fail_closed():
    from formal_toolchain.conformance.p0_checker import aggregate_p0_certificates
    assert aggregate_p0_certificates({})["status"] != "PASS"
    with pytest.raises(ValueError):
        certify_envelope({"status": "PASS"}, {"status": "PASS"}, {"status": "PASS"}, context_hash="a" * 64)


def test_top1_valid_else_noop_is_safe_selection_semantics(tasks):
    metadata = {
        "L": {"initial_runtime_budget": 2, "budget_floor": 1, "budget_cap": 10},
        "H": {"initial_runtime_budget": 3, "budget_floor": 3, "budget_cap": 4},
    }
    domain = build_budget_domain(
        tasks,
        metadata,
        runtime_config=type("Config", (), {"processor_overhead": 0})(),
    )
    domain["context_hash"] = "a" * 64
    actions = build_budget_action_space(
        tasks,
        action_space="single",
        budget_increase_ratio=.02,
        budget_decrease_ratio=.02,
    )
    candidate = synthesize_candidate_envelope(
        domain, actions, tasks, context_hash="a" * 64
    )
    candidate["schema_version"] = "candidate_envelope_v2"
    candidate["safety_polytope_hash"] = "b" * 64
    from formal_toolchain.policy.actions import build_action_transition_table

    action_transition = build_action_transition_table(
        actions,
        tasks,
        domain["tasks"],
        rounding_mode="ceil_floor",
        min_budget_delta=1,
    )
    ranking = tuple(action.action_id for action in actions)
    mask_fallback = {
        "status": "PASS",
        "leaves": [{"leaf_id": 0, "ranking": ranking}],
    }
    mask_contract = {
        "shared_with_step": True,
        "check_safety": True,
        "selection": "top1_valid_else_noop",
    }
    result = check_deployed_policy_preservation(
        candidate,
        actions,
        tasks,
        mask_fallback_certificate=mask_fallback,
        action_transition_certificate=action_transition,
        mask_contract=mask_contract,
        selection_semantics="top1_valid_else_noop",
    )
    assert result["status"] == "PASS"
    assert result["implicit_noop_checked"] is True
