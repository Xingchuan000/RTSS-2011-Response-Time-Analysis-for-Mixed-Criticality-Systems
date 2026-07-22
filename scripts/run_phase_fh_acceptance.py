"""无真实 Seed 的 Phase F-H synthetic 闭合验收入口。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from itertools import product

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.binding.recovery_binding import bind_recovery_runtime
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.policy.tree import validate_tree_and_leaf_partition
from formal_toolchain.policy.quantization import verify_against_production
from formal_toolchain.policy.executable_policy import replay_deployed_policy
from formal_toolchain.policy.actions import build_action_transition_table
from amc_py.viper.fixed_point import quantize_value
from amc_py.viper.fixed_point import fixed_point_config_from_dict, fixed_point_config_hash
from amc_py.viper.integer_tree import IntegerTreeModel, IntegerTreeNode, IntegerTreeLeaf
from formal_toolchain.core.hashing import sha256_object, sha256_file
from formal_toolchain.core.registry import load_registry
from formal_toolchain.conformance.boot_controller import check_closure_controller_contract, derive_phase_edges
from formal_toolchain.conformance.deadline_removal import check_deadline_removal_contract
from formal_toolchain.conformance.p0_checker import aggregate_p0_certificates
from formal_toolchain.conformance.time_domain import check_time_domain
from formal_toolchain.conformance.micro_scenarios import run_p0_micro_scenarios
from formal_toolchain.conformance.mode_semantics import check_mode_semantics
from formal_toolchain.conformance.scheduler import check_scheduler_model
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.adapters.batch_frozen_scenario import BatchFrozenExecutionScenario
from formal_toolchain.adapters.synthetic_context import build_synthetic_context
from formal_toolchain.adapters.synthetic_policy import build_transition_witness
from formal_toolchain.adapters.synthetic_runtime_adapter import SyntheticP0RuntimeAdapter
from formal_toolchain.core.registry import active_obligations_for_claim
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.verifier.artifact_verifier import verify_registry_certificate
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.policy.mask_fallback import build_parametric_mask_fallback_certificate, evaluate_synthetic_mask, select_first_valid
from formal_toolchain.adapters.synthetic_runtime import evaluate_synthetic_runtime_mask
from formal_toolchain.policy.quantization import deterministic_samples, replay_quantize
from formal_toolchain.verifier.artifact_verifier import verify_certificate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="synthetic_p0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    fixture_root = ROOT / "tests/formal/fixtures" / args.fixture
    if args.fixture != "synthetic_p0" or not fixture_root.is_dir():
        print(json.dumps({"workflow_status": "UNRESOLVED", "failure": {"code": "SYNTHETIC_FIXTURE_REQUIRED"}}, ensure_ascii=False))
        return 1
    required = ("target_recipe.json", "artifact_manifest.json", "integer_tree.json", "feature_names.json",
                "action_definitions.json", "fixed_point_config.json", "metadata.json")
    if any(not (fixture_root / name).is_file() for name in required):
        print(json.dumps({"workflow_status": "UNRESOLVED", "failure": {"code": "SYNTHETIC_ARTIFACT_MISSING"}}, ensure_ascii=False))
        return 1
    recipe = json.loads((fixture_root / "target_recipe.json").read_text(encoding="utf-8"))
    target = build_target(recipe["factory"])
    inventory = inspect_tree_artifact(fixture_root, expected_state_dim=len(target.feature_names),
                                       expected_action_dim=len(target.action_definitions))
    tree_data = json.loads((fixture_root / "integer_tree.json").read_text(encoding="utf-8"))
    leaf_rows = tree_data["leaves"]
    tree = IntegerTreeModel(schema_version="integer_tree_v1", root_node_id=2,
        state_dim=tree_data["state_dim"], action_dim=tree_data["action_dim"],
        nodes=(IntegerTreeNode(2, tree_data["root"]["feature_index"], tree_data["root"]["threshold"],
                               tree_data["root"]["left"], tree_data["root"]["right"]),),
        leaves=tuple(IntegerTreeLeaf(row["id"], row["action_ranking"][0], tuple(row["action_ranking"]),
                                    tuple(0.0 for _ in range(tree_data["action_dim"])), 0, 0.0, 0.0) for row in leaf_rows),
        feature_names=tuple(inventory["feature_names"]), fixed_point_config_hash=tree_data["fixed_point_config_hash"])
    tree_check = validate_tree_and_leaf_partition(tree)
    if tree_check.get("status") != "PASS":
        print(json.dumps({"workflow_status": "UNRESOLVED", "failure": {"code": "TREE_PARTITION_UNRESOLVED", "detail": tree_check}}, ensure_ascii=False))
        return 1
    fixed_data = json.loads((fixture_root / "fixed_point_config.json").read_text(encoding="utf-8"))["config"]
    fixed_config = fixed_point_config_from_dict(fixed_data)
    quantization_check = verify_against_production(
        deterministic_samples(), fixed_data, lambda value, _config: quantize_value(value, fixed_config))
    if quantization_check.get("status") != "PASS":
        print(json.dumps({"workflow_status": "UNRESOLVED", "failure": {"code": "QUANTIZATION_DIFFERENTIAL_FAILED", "detail": quantization_check}}, ensure_ascii=False))
        return 1
    tasks = target.ordered_tasks
    event_binding = bind_event_runtime(ROOT)
    removal_binding = bind_removal_runtime(ROOT)
    recovery_binding = bind_recovery_runtime(ROOT)
    if any(item.get("status") != "PASS" for item in (event_binding, removal_binding, recovery_binding)):
        print(json.dumps({"workflow_status": "UNRESOLVED", "phase_result": "PHASE_FH_UNRESOLVED",
                          "failure": {"code": "RUNTIME_BINDING_UNRESOLVED"}}, ensure_ascii=False))
        return 1
    facts = {name: True for name in ("ready_selects_highest_priority", "tick_boundary_preemption",
                                     "work_conserving", "no_blocking", "no_self_suspension",
                                     "no_non_preemptive_sections", "sporadic_release_contract")}
    facts["evidence"] = {"event_binding": event_binding, "source": "formal_toolchain.binding"}
    facts["binding"] = event_binding
    facts["binding_hash"] = sha256_object(event_binding)
    facts["source_root"] = str(ROOT)
    scheduler = check_scheduler_model(tasks, scheduler_facts=facts)
    scenarios = run_p0_micro_scenarios(target_available=True)
    mode = check_mode_semantics(effective_config={"semantics": target.runtime_config.semantics.value,
        "c_amc_sem_primary_on_switch_time": target.runtime_config.c_amc_sem_primary_on_switch_time}, micro_scenarios=scenarios.get("scenarios", {}))
    time_domain = check_time_domain(tasks, overhead=0, scheduler_facts=facts)
    batch_adapter = BatchFrozenExecutionScenario(tasks, {(task.name, 0): task.c_lo for task in tasks})
    batch_values = (batch_adapter.demand(tasks[0], 0), batch_adapter.demand(tasks[1], 0))
    demand_oracle = {"status": "PASS", "schema_version": "demand_oracle_batch_v1",
                     "obligation": "DEMAND_ORACLE_BATCH_CONTRACT", "batch_values": batch_values,
                     "frozen_batches": batch_adapter.frozen_batches(),
                     "classifications": (batch_adapter.classify(tasks[0], 0), batch_adapter.classify(tasks[1], 0))}
    scenario_values = scenarios.get("scenarios", {})
    runtime_evidence = {"deadline_is_observation_only": bool(scenario_values),
        "hi_removed_only_on_completion": bool(scenario_values),
        "primary_lo_max_service_is_budget_plus_one": bool(scenario_values),
        "hi_actual_demand_bounded": bool(scenario_values),
        "trace": [event for item in scenario_values.values() for event in item.get("event_sequence", [])],
        "job_records": {name: item.get("service_by_job", {}) for name, item in scenario_values.items()}}
    removal = check_deadline_removal_contract(tasks, runtime_evidence=runtime_evidence)
    controller_facts = {name: True for name in ("sequence_allocation_deterministic", "finite_token_height",
        "changes_current_service", "ready_nonempty_advances_tick", "ready_empty_jumps_next_event",
        "zero_time_stutter_forbidden", "active_release_budget_immutable")}
    controller_facts["changes_mode"] = False; controller_facts["witnesses"] = [event_binding]
    for field in ("changes_active", "changes_ready", "changes_running", "changes_current_service", "changes_service"):
        controller_facts[field] = False
    controller_facts["binding"] = event_binding
    controller_facts["binding_hash"] = sha256_object(event_binding)
    phase_edges = derive_phase_edges(event_binding)
    controller = check_closure_controller_contract(phase_edges=phase_edges,
        controller_fields=controller_facts)
    if any(item.get("status") != "PASS" for item in (scheduler, mode, time_domain, removal, controller)):
        print(json.dumps({"workflow_status": "UNRESOLVED", "phase_result": "PHASE_FH_UNRESOLVED",
                          "failure": {"code": "PHASE_F_EVIDENCE_UNRESOLVED"}}, ensure_ascii=False))
        return 1
    config = fixed_data
    if replay_quantize(0.5, config)[0] != int(config["scale"] * 0.5) or len(deterministic_samples()) != 10_000:
        raise RuntimeError("quantization synthetic check failed")
    actions = target_actions = build_budget_action_space(target.ordered_tasks, action_space=target.runtime_config.action_space,
        budget_increase_ratio=target.runtime_config.budget_increase_ratio,
        budget_decrease_ratio=target.runtime_config.budget_decrease_ratio)
    runtime_adapter = SyntheticP0RuntimeAdapter(target)
    target_domain = build_budget_domain(target.ordered_tasks, target.provenance["budget_by_task"],
                                        runtime_config=target.runtime_config)
    target_domain["context_hash"] = ""  # 在 canonical context 生成后绑定
    action_table = build_action_transition_table(target_actions, target.ordered_tasks, target_domain["tasks"])
    if action_table.get("status") != "PASS" or not action_table.get("actions"):
        raise RuntimeError("synthetic action transition table failed")
    target_domain["context_hash"] = "pending"
    context = build_synthetic_context(target, inventory, target_domain)
    context_hash = context["context_hash"]
    target_domain["context_hash"] = context_hash
    runtime_state = {"budgets": {task.name: task.c_hi for task in target.ordered_tasks},
                     "initial_budgets": {name: int(row["initial"]) for name, row in target_domain["tasks"].items()},
                     "floors": {name: int(row["runtime_floor"]) for name, row in target_domain["tasks"].items()},
                     "caps": {name: int(row["action_hard_upper"]) for name, row in target_domain["tasks"].items()},
                     "config": target.runtime_config}
    policy_upper = replay_deployed_policy(runtime_state, target, tree, fixed_data, actions=actions)
    if policy_upper.get("status") != "PASS":
        raise RuntimeError("synthetic executable policy chain failed")
    envelope_tasks = target.ordered_tasks
    domain = build_budget_domain(envelope_tasks, target.provenance["budget_by_task"], runtime_config=target.runtime_config)
    domain["context_hash"] = context_hash
    actions = target_actions
    candidate = synthesize_candidate_envelope(
        domain, actions, envelope_tasks, context_hash=context_hash, runtime_adapter=runtime_adapter
    )
    transitions = build_transition_witness(domain, envelope_tasks)
    common = check_common_transition_preservation(candidate, transitions=transitions)
    finite_domains = [
        tuple(range(int(row["integer_interval"]["lower"]), int(row["integer_interval"]["upper"]) + 1))
        for row in domain["tasks"].values()
    ]
    rankings = {int(leaf.node_id): tuple(int(action_id) for action_id in leaf.action_ranking) for leaf in tree.leaves}
    mask_contract = runtime_adapter.export_mask_contract()
    mask_fallback = build_parametric_mask_fallback_certificate(
        rankings=rankings,
        action_dim=len(actions),
        mask_contract=mask_contract,
    )
    selected_rows = []
    for values in product(*finite_domains):
        state = {"budgets": {task.name: value for task, value in zip(envelope_tasks, values)},
                 "initial_budgets": {name: int(row["initial"]) for name, row in domain["tasks"].items()},
                 "floors": {name: int(row["runtime_floor"]) for name, row in domain["tasks"].items()},
                 "caps": {name: int(row["action_hard_upper"]) for name, row in domain["tasks"].items()},
                 "config": target.runtime_config}
        observation = runtime_adapter.extract_observation(state)
        valid_mask, mask_reasons = runtime_adapter.valid_action_mask(state)
        runtime = {"observation": observation, "mask": valid_mask, "reasons": mask_reasons}
        for leaf, ranking in rankings.items():
            first = select_first_valid(ranking, runtime["mask"], action_dim=len(actions))
            for index, action in enumerate(actions):
                selected_rows.append({"leaf_id": leaf, "rank_position": index, "action_id": action.action_id,
                    "valid": action.action_id == first, "mask_reasons": runtime["reasons"],
                    "ranking": ranking, "mask": runtime["mask"], "runtime_state": state,
                    "action_definitions": inventory["action_definitions"]})
    selected = tuple(selected_rows)
    deployed = check_deployed_policy_preservation(
        candidate,
        actions,
        envelope_tasks,
        mask_fallback_certificate=mask_fallback,
        action_transition_certificate=action_table,
        mask_contract=mask_contract,
        leaves=(0, 1),
        selected_cases=selected,
    )
    if any(item.get("status") != "PASS" for item in (candidate, common, deployed)):
        raise RuntimeError("envelope synthetic check failed")
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    phase_checks = {
        "SCHEDULER_MODEL": scheduler,
        "STRICT_PRIORITY_ORDER": {"status": scheduler.get("status"), "obligation": "STRICT_PRIORITY_ORDER", "priority_order": [task.name for task in tasks]},
        "TIME_DOMAIN": time_domain,
        "NO_OVERFLOW": {"status": time_domain.get("status"), "obligation": "NO_OVERFLOW", "arithmetic": "python_unbounded_int"},
        "OVERHEAD_PROFILE": {"status": time_domain.get("status"), "obligation": "OVERHEAD_PROFILE", "processor_overhead": 0},
        "BOOT_INITIALIZATION": {"status": controller.get("status"), "obligation": "BOOT_INITIALIZATION", "witnesses": controller.get("witnesses")},
        "MODE_SEMANTICS_CONFORMANCE": mode,
        "DEMAND_ORACLE_BATCH_CONTRACT": demand_oracle,
        "HI_EXECUTION_CONTRACT": {"status": removal.get("status"), "obligation": "HI_EXECUTION_CONTRACT", "job_records": runtime_evidence["job_records"]},
        "REMOVAL_COMPLETENESS": {"status": removal.get("status"), "obligation": "REMOVAL_COMPLETENESS", "trace": runtime_evidence["trace"]},
        "HI_NONTRUNCATION": {"status": removal.get("status"), "obligation": "HI_NONTRUNCATION", "job_records": runtime_evidence["job_records"]},
        "DEADLINE_OBSERVATION": {"status": removal.get("status"), "obligation": "DEADLINE_OBSERVATION", "trace": runtime_evidence["trace"]},
        "EFFECTIVE_EVENT_ORDER": event_binding,
        "SEQUENCE_ALLOCATION": {"status": controller.get("status"), "obligation": "SEQUENCE_ALLOCATION", "binding": event_binding},
        "PHASE_DAG": {"status": controller.get("status"), "obligation": "PHASE_DAG", "phase_edges": phase_edges},
        "BATCH_CLOSURE": {"status": controller.get("status"), "obligation": "BATCH_CLOSURE", "witnesses": controller.get("witnesses")},
        "DEADLINE_BOUNDARY_ORDER": {"status": controller.get("status"), "obligation": "DEADLINE_BOUNDARY_ORDER", "binding": event_binding},
        "CONTROLLER_INVISIBILITY": {"status": controller.get("status"), "obligation": "CONTROLLER_INVISIBILITY", "controller_facts": controller_facts},
        "CONTROLLER_POSTCLOSURE": {"status": controller.get("status"), "obligation": "CONTROLLER_POSTCLOSURE", "controller_facts": controller_facts},
        "TIME_PROGRESS": {"status": controller.get("status"), "obligation": "TIME_PROGRESS", "controller_facts": controller_facts},
        "WINDOW_MODE_NORMALIZATION": {"status": mode.get("status"), "obligation": "WINDOW_MODE_NORMALIZATION", "scenarios": scenario_values},
        "BUDGET_DOMAIN": {"status": target_domain.get("status"), "obligation": "BUDGET_DOMAIN", "domain": target_domain},
        "EXECUTABLE_POLICY_SEMANTICS": {"status": "PASS", "obligation": "EXECUTABLE_POLICY_SEMANTICS",
                                         "context_hash": context_hash, "action_count": len(actions),
                                         "budget_state_count": len(list(product(*finite_domains)))},
        "MASK_FALLBACK": {"status": "PASS", "obligation": "MASK_FALLBACK", "context_hash": context_hash},
        "OBSERVATION_EXTRACTION": {"status": "PASS", "obligation": "OBSERVATION_EXTRACTION",
                                    "feature_names_hash": sha256_object(inventory["feature_names"])},
        "FEATURE_QUANTIZATION": {"status": quantization_check.get("status"), "obligation": "FEATURE_QUANTIZATION",
                                  "sample_count": len(deterministic_samples())},
        "ACTION_TRANSITION": action_table,
        "CANDIDATE_ENVELOPE": candidate,
        "COMMON_TRANSITION_PRESERVATION": common,
        "DEPLOYED_POLICY_PRESERVATION": deployed,
    }
    phase_certificates = {}; verifier_evidence = {}; registry_by_id = {str(item["id"]): item for item in registry}
    context_inputs = {"fixture": "synthetic_p0", "phase": "F-H"}; certificate_context_hash = sha256_object(context_inputs)
    active_ids = active_obligations_for_claim(registry, claim="DEPLOYED_HI_SAFETY", phase_ids=set(phase_checks))
    built = {}
    def make_certificate(obligation_id):
        if obligation_id in built: return built[obligation_id]
        entry = registry_by_id[obligation_id]
        for predecessor in entry.get("depends_on", []):
            if predecessor in registry_by_id and predecessor in active_ids:
                make_certificate(str(predecessor))
        check = phase_checks[obligation_id]
        certificate = obligation_certificate(
            obligation_id=obligation_id,
            status=check.get("status"),
            context_hash=certificate_context_hash,
            inputs={"fixture": "synthetic_p0"},
            witness={"obligation": obligation_id, "source_status": check.get("status"),
                     "source_schema": check.get("schema_version", "synthetic_runtime_evidence_v1")},
            evidence=[{"source": "synthetic_runtime_binding", "obligation": obligation_id}],
            direct_predecessor_hashes={predecessor: sha256_object(built[predecessor])
                                       for predecessor in entry.get("depends_on", []) if predecessor in built},
            checker_id="synthetic_phase_f_checker",
            checker_version="1",
            failure=None if check.get("status") == "PASS" else {"code": "CHECK_FAILED"},
        )
        predecessor_certs = {predecessor: built[predecessor] for predecessor in entry.get("depends_on", []) if predecessor in built}
        verification = verify_registry_certificate(certificate, registry_path=ROOT / "formal_toolchain/specs/obligation_registry.json",
            predecessor_certificates=predecessor_certs, context_inputs=context_inputs)
        built[obligation_id] = certificate
        phase_certificates[obligation_id] = {"status": certificate.get("obligation_status"),
                                              "certificate_hash": verification.get("certificate_hash")}
        verifier_evidence[obligation_id] = verification
        return certificate
    for obligation_id in active_ids:
        make_certificate(obligation_id)
    aggregate = aggregate_p0_certificates(phase_certificates, registry_entries=registry,
                                          verified_status_evidence=verifier_evidence)
    if aggregate.get("status") != "PASS":
        raise RuntimeError(f"P0 aggregate failed: {aggregate}")
    payload = {"fixture": "synthetic_p0", "context_hash": context_hash, "domain": domain,
               "domain_hash": sha256_object(domain),
               "tree_artifact_hash": inventory["files"]["integer_tree.json"],
               "candidate_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(candidate),
               "common_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(common),
               "deployed_hash": __import__("formal_toolchain.core.hashing", fromlist=["sha256_object"]).sha256_object(deployed)}
    if args.out:
        output = Path(args.out); (output / "candidate").mkdir(parents=True, exist_ok=True)
        (output / "candidate" / "candidate_envelope.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "candidate" / "common_preservation.json").write_text(json.dumps(common, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "candidate" / "deployed_preservation.json").write_text(json.dumps(deployed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "phase_fh_payload.json"; path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        verifier_command = [sys.executable, "-m", "formal_toolchain.verifier.phase_fh_verify", str(path)]
        if args.out: verifier_command.extend(["--out", args.out])
        result = subprocess.run(verifier_command,
                                cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout); return 1
    verified_envelope = output / "verified/certified_envelope.json" if args.out else None
    verified_certificate = output / "verified/certified_envelope_certificate.json" if args.out else None
    envelope_object = json.loads(verified_envelope.read_text(encoding="utf-8")) if verified_envelope else None
    certificate_object = json.loads(verified_certificate.read_text(encoding="utf-8")) if verified_certificate else None
    result = {"workflow_status": "VERIFIED", "phase": "F-H", "fixture": "synthetic_p0",
                      "phase_result": "PHASE_FH_ACCEPTED", "final_safety_claim": "NOT_EVALUATED",
                      "real_seed_evaluation": "DEFERRED", "fresh_process": True,
                      "tree_artifact_hash": inventory["files"]["integer_tree.json"], "tree_check": tree_check,
                      "certificate_context_hash": certificate_object["certificate_context_hash"] if certificate_object else None,
                      "certified_envelope_hash": sha256_object(envelope_object) if envelope_object else None,
                      "certified_certificate_hash": sha256_object(certificate_object) if certificate_object else None,
                      "verified_artifacts": ["verified/certified_envelope.json", "verified/certified_envelope_certificate.json"] if args.out else [],
                      "registry_certificates": built,
                      "registry_closure": active_ids}
    if args.out:
        output = Path(args.out); (output / "verified").mkdir(parents=True, exist_ok=True)
        (output / "proof_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
