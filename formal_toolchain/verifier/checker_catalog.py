"""verifier-side obligation→checker 的静态映射。

这里禁止从 registry 动态创建 ``status`` passthrough checker。每个 active
obligation 必须在这张显式表中出现；尚未实现算法的入口使用显式
UNRESOLVED checker，不能借用 candidate 或 preflight 的状态。
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping

from amc_py.models import Criticality
from amc_py.rl.actions import build_budget_action_space
from amc_py.viper.fixed_point import fixed_point_config_from_dict, quantize_value
from amc_py.viper.integer_tree import evaluate_integer_tree
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.adapters.runtime_manifest import (
    build_checker_version_manifest,
    build_dependency_manifest,
    build_runtime_environment_manifest,
    check_dependency_policy,
)
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.bridge.budget_invariant_derivation import derive_budget_invariant_evidence
from formal_toolchain.core.contexts import expected_context_for_obligation
from formal_toolchain.core.hashing import sha256_object, sha256_proof_object
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.registry import load_registry
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.conformance.active_release_budget import (
    check_active_release_budget_source_contract,
)
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.certified_envelope import _certify_envelope_from_verifier
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.policy.actions import build_action_transition_table
from formal_toolchain.policy.mask_fallback import (
    build_mask_fallback_certificate,
    build_parametric_mask_fallback_certificate,
    select_first_valid,
    select_by_semantics,
)
from formal_toolchain.policy.quantization import deterministic_samples, replay_quantize, verify_against_production
from formal_toolchain.policy.selected_regions import selected_action_regions, selected_action_regions_v2
from formal_toolchain.policy.tree import validate_tree_and_leaf_partition
from formal_toolchain.policy.tree_io import integer_tree_from_dict
from formal_toolchain.reference.model_conformance import build_reference_model_conformance_certificate
from formal_toolchain.reference.rta_production import all_task_reference_rta, protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_all_task_rta, replay_all_task_rta_independently
from formal_toolchain.reference.task_mapping import build_reference_taskset, validate_reference_mapping
from formal_toolchain.verifier.budget_domination_checker import verify_budget_to_reference_domination
from formal_toolchain.verifier.controller_checkers import (
    verify_controller_boundary, verify_controller_path_uniqueness,
    verify_controller_write_set, verify_token_refresh_projection,
    verify_update_payload_totality,
)
from formal_toolchain.verifier.early_stop_gate_checker import verify_early_stop_configuration_gate
from formal_toolchain.verifier.effective_frontier_checker import verify_effective_event_frontier_relation
from formal_toolchain.verifier.finite_contradiction_checker import verify_finite_bad_prefix_contradiction
from formal_toolchain.verifier.reference_conformance_checker import verify_reference_model_conformance
from formal_toolchain.verifier.reference_theorem_checker import (
    verify_reference_hi_subset_safety,
    verify_reference_taskset_schedulable,
)
from formal_toolchain.verifier.bootstrap_checks import (
    build_interface_coverage_report,
    verify_json_schema_file,
    verify_migration_manifest,
    verify_obligation_registry,
    verify_source_manifest,
)
from formal_toolchain.verifier.theory_verifier import verify_theory_library
from formal_toolchain.verifier.semantic_checkers import (
    verify_batch_closure, verify_boot_initialization, verify_case1_integer_domain,
    verify_case2_integer_domain, verify_closed_prefix_refinement, verify_code_reference_upper_bound_mapping,
    verify_controller_invisibility, verify_controller_postclosure, verify_deadline_boundary_order,
    verify_deadline_observation, verify_demand_domination, verify_demand_oracle_contract,
    verify_discrete_tick_embedding, verify_event_order, verify_feature_schema_consistency,
    verify_feature_totality, verify_hi_bad_closed_prefix_reflection, verify_hi_execution_contract,
    verify_hi_nontruncation, verify_initial_quiescence, verify_inherited_hi_domination,
    verify_lo_mode_rta, verify_mode_semantics, verify_no_overflow, verify_observation_extraction,
    verify_overhead_profile, verify_per_hi_task_inductive_wcrt, verify_phase_dag,
    verify_protected_hi_safety_corollary, verify_release_count, verify_release_fixed_removal_mapping,
    verify_removal_completeness, verify_reference_prefix_extension, verify_scheduler_model,
    verify_sequence_allocation, verify_strict_priority_order, verify_time_domain, verify_time_progress,
    verify_unresolved, verify_window_mode_normalization, verify_worst_case_start_time,
    verify_zero_relative_start,
)

Checker = Callable[..., dict[str, Any]]


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _raw_inputs_result(obligation_id: str, *, code: str) -> dict[str, Any]:
    return {"status": "UNRESOLVED", "route": "UNRESOLVED",
            "code": code, "witness": {"obligation_id": obligation_id}}


def _finish(obligation_id: str, result: Mapping[str, Any], *, expected_context_hash: str | None = None,
            candidate_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status = result.get("status", result.get("obligation_status"))
    if status not in {"PASS", "FAIL", "UNRESOLVED"}:
        return _raw_inputs_result(obligation_id, code="SEMANTIC_CHECK_STATUS_INVALID")
    witness = dict(result.get("witness", result)) if isinstance(result, Mapping) else {}
    # Diagnostic/production-safe canonicalization: fresh verifier witnesses may
    # contain recipe/config floats. Proof objects require canonical decimal strings.
    from formal_toolchain.verifier.replay_inputs import _proof_safe
    witness = _proof_safe(witness)
    if expected_context_hash is not None:
        witness["fresh_context_hash"] = expected_context_hash
    if isinstance(candidate_evidence, Mapping):
        candidate_status = candidate_evidence.get("status", candidate_evidence.get("obligation_status"))
        if candidate_status in {"PASS", "FAIL", "UNRESOLVED"}:
            witness["candidate_status_compared"] = candidate_status
    failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
    return {"status": status, "route": None if status == "PASS" else result.get("route", failure.get("route", "UNRESOLVED")),
            "code": None if status == "PASS" else result.get("code", failure.get("code", "SEMANTIC_CHECK_FAILED")),
            "witness": witness, "fresh_input_hashes": {"result_hash": sha256_object(witness)}}


def _raw_inputs(kwargs: Mapping[str, Any], obligation_id: str):
    value = kwargs.get("raw_inputs")
    if value is None:
        if not kwargs.get("evidence") and not kwargs.get("candidate_evidence"):
            return None, _raw_inputs_result(obligation_id, code="OBLIGATION_EVIDENCE_MISSING")
        return None, _raw_inputs_result(obligation_id, code="VERIFIER_RAW_INPUTS_MISSING")
    return value, None


def _target(raw_inputs: Any) -> Any:
    return _field(raw_inputs, "target")


def _artifact_dir(raw_inputs: Any) -> Path:
    return Path(_field(raw_inputs, "artifact_dir"))


def _inventory(raw_inputs: Any) -> Mapping[str, Any]:
    value = _field(raw_inputs, "inventory")
    if not isinstance(value, Mapping):
        raise ValueError("verifier inputs 缺少 inventory")
    return value


def _contexts(raw_inputs: Any) -> Mapping[str, Mapping[str, Any]] | None:
    value = _field(raw_inputs, "contexts")
    return value if isinstance(value, Mapping) else None


def _expected_context(raw_inputs: Any, obligation_id: str, expected_context_hash: str | None) -> str | None:
    if expected_context_hash is not None:
        return expected_context_hash
    contexts = _contexts(raw_inputs)
    if contexts is None:
        return None
    return expected_context_for_obligation(obligation_id, contexts)


def _tree_model(raw_inputs: Any):
    artifact_dir = _artifact_dir(raw_inputs)
    inventory = _inventory(raw_inputs)
    tree_data = inventory.get("tree")
    if not isinstance(tree_data, Mapping):
        tree_path = artifact_dir / "integer_tree.json"
        tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    return integer_tree_from_dict(tree_data)


def _fixed_point_config(raw_inputs: Any) -> dict[str, Any]:
    inventory = _inventory(raw_inputs)
    fixed_data = inventory.get("fixed_point_config")
    if isinstance(fixed_data, Mapping):
        fixed_data = fixed_data.get("config", fixed_data)
    if not isinstance(fixed_data, Mapping):
        fixed_path = _artifact_dir(raw_inputs) / "fixed_point_config.json"
        fixed_json = json.loads(fixed_path.read_text(encoding="utf-8"))
        fixed_data = fixed_json.get("config", fixed_json)
    if not isinstance(fixed_data, Mapping):
        raise ValueError("fixed_point_config 缺失")
    return dict(fixed_data)


def _actions(raw_inputs: Any):
    target = _target(raw_inputs)
    return build_budget_action_space(
        target.ordered_tasks,
        action_space=str(getattr(target.runtime_config, "action_space")),
        budget_increase_ratio=float(getattr(target.runtime_config, "budget_increase_ratio")),
        budget_decrease_ratio=float(getattr(target.runtime_config, "budget_decrease_ratio")),
    )


def _runtime_adapter(raw_inputs: Any):
    target = _target(raw_inputs)
    adapter = _field(target, "runtime_adapter")
    if adapter is None:
        raise ValueError("FORMAL_RUNTIME_ADAPTER_MISSING")
    return adapter


def _domain(raw_inputs: Any):
    target = _target(raw_inputs)
    return build_budget_domain(
        target.ordered_tasks,
        target.provenance.get("budget_by_task"),
        runtime_config=target.runtime_config,
    )


def _candidate_envelope(raw_inputs: Any, *, context_hash: str | None) -> dict[str, Any]:
    target = _target(raw_inputs)
    actions = _actions(raw_inputs)
    domain = _domain(raw_inputs)
    if context_hash is None:
        raise ValueError("CANDIDATE_CONTEXT_MISSING")
    domain["context_hash"] = context_hash
    adapter = _runtime_adapter(raw_inputs)
    return synthesize_candidate_envelope(
        domain,
        actions,
        target.ordered_tasks,
        context_hash=context_hash,
        runtime_adapter=adapter,
    )


def _fresh_structural_envelope_pipeline(
    raw_inputs: Any,
    *,
    context_hash: str,
) -> dict[str, Any]:
    target = _target(raw_inputs)
    tree = _tree_model(raw_inputs)
    actions = _actions(raw_inputs)
    domain = _domain(raw_inputs)
    domain["context_hash"] = context_hash
    adapter = _runtime_adapter(raw_inputs)

    action_cert = build_action_transition_table(
        actions, target.ordered_tasks, domain["tasks"],
        rounding_mode=str(getattr(target.runtime_config, "budget_rounding_mode", "ceil_floor")),
        min_budget_delta=int(getattr(target.runtime_config, "min_budget_delta", 1)),
    )
    candidate = synthesize_candidate_envelope(
        domain,
        actions,
        target.ordered_tasks,
        context_hash=context_hash,
        runtime_adapter=adapter,
    )
    transitions = _transition_witness(raw_inputs, domain)
    common = check_common_transition_preservation(
        candidate,
        transitions=transitions,
    )

    rankings = {
        int(leaf.node_id): tuple(int(a) for a in leaf.action_ranking)
        for leaf in tree.leaves
    }
    mask_contract = adapter.export_mask_contract()
    selection_semantics = str(mask_contract.get("selection", "ranked_first_valid"))
    mask = build_parametric_mask_fallback_certificate(
        rankings=rankings,
        action_dim=len(actions),
        mask_contract=mask_contract,
    )
    regions = selected_action_regions_v2(_leaf_guards(tree), rankings,
                                         selection_semantics=selection_semantics)
    deployed = check_deployed_policy_preservation(
        candidate,
        actions,
        target.ordered_tasks,
        mask_fallback_certificate=mask,
        action_transition_certificate=action_cert,
        mask_contract=mask_contract,
        forbid_decreasing_hi_budgets=bool(getattr(target.runtime_config, "forbid_decreasing_hi_budgets")),
        selection_semantics=selection_semantics,
        disabled_guards=tuple(mask_contract.get("disabled_guards", ())),
    )
    return {
        "domain": domain,
        "action": action_cert,
        "candidate": candidate,
        "common": common,
        "mask": mask,
        "regions": regions,
        "deployed": deployed,
        "mask_contract": mask_contract,
        "transitions": transitions,
    }


def _transition_witness(raw_inputs: Any, domain: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _target(raw_inputs)
    from formal_toolchain.adapters.synthetic_policy import build_transition_witness
    return build_transition_witness(domain, target.ordered_tasks)


def _leaf_guards(tree) -> dict[int, tuple[dict[str, Any], ...]]:
    leaves = {leaf.node_id for leaf in tree.leaves}
    nodes = {node.node_id: node for node in tree.nodes}
    guards: dict[int, tuple[dict[str, Any], ...]] = {}

    def walk(node_id: int, path: list[dict[str, Any]]) -> None:
        if node_id in leaves:
            guards[node_id] = tuple(path)
            return
        node = nodes[node_id]
        walk(node.left_child, path + [{"feature_index": int(node.feature_index), "operator": "<=", "threshold": int(node.threshold_int)}])
        walk(node.right_child, path + [{"feature_index": int(node.feature_index), "operator": ">", "threshold": int(node.threshold_int)}])

    walk(tree.root_node_id, [])
    return guards


def _policy_samples(raw_inputs: Any, tree, fixed_data: Mapping[str, Any], actions, domain: Mapping[str, Any]):
    target = _target(raw_inputs)
    adapter = _runtime_adapter(raw_inputs)
    names = [str(task.name) for task in target.ordered_tasks]
    policy_states = [
        {name: int(domain["tasks"][name]["initial"]) for name in names},
        {name: int(domain["tasks"][name]["runtime_deploy_cap"]) for name in names},
    ]
    selected_cases: list[dict[str, Any]] = []
    rankings: list[list[int]] = []
    masks: list[list[bool]] = []
    reasons: list[list[str]] = []
    executable: list[dict[str, Any]] = []
    for budgets in policy_states:
        state = {
            "budgets": budgets,
            "initial_budgets": {name: int(domain["tasks"][name]["initial"]) for name in names},
            "floors": {name: int(domain["tasks"][name]["runtime_floor"]) for name in names},
            "caps": {name: int(domain["tasks"][name]["runtime_deploy_cap"]) for name in names},
            "config": target.runtime_config,
        }
        runtime_state = adapter.build_runtime_state_from_budget_vector(budgets)
        runtime_mask, runtime_reasons = adapter.valid_action_mask(runtime_state)
        runtime = {"observation": tuple(adapter.extract_observation(runtime_state)),
                   "mask": tuple(runtime_mask), "reasons": tuple(runtime_reasons)}
        quantized = tuple(replay_quantize(value, fixed_data)[0] for value in runtime["observation"])
        replay = evaluate_integer_tree(tree, quantized)
        selection_semantics = str(adapter.export_mask_contract().get("selection", "ranked_first_valid"))
        selected = select_by_semantics(replay.action_ranking, runtime["mask"], action_dim=len(actions),
                                       selection_semantics=selection_semantics)
        after = adapter.apply_action(runtime_state, selected)
        executable.append({
            "status": "PASS",
            "quantized": quantized,
            "leaf_id": int(replay.leaf_id),
            "ranking": tuple(int(action_id) for action_id in replay.action_ranking),
            "selected_action": selected,
            "mask": tuple(runtime["mask"]),
            "mask_reasons": tuple(runtime["reasons"]),
            "implicit_noop": selected is None,
            "budget_after": after,
        })
        rankings.append([int(action_id) for action_id in replay.action_ranking])
        masks.append(list(runtime["mask"]))
        reasons.append(list(runtime["reasons"]))
        ranking_map = {int(leaf.node_id): tuple(int(action_id) for action_id in leaf.action_ranking) for leaf in tree.leaves}
        for leaf_id, ranking in ranking_map.items():
            first = select_by_semantics(ranking, runtime["mask"], action_dim=len(actions),
                                        selection_semantics=selection_semantics)
            for action in actions:
                selected_cases.append({
                    "leaf_id": int(leaf_id),
                    "rank_position": int(ranking.index(int(action.action_id))),
                    "action_id": int(action.action_id),
                    "valid": first == action.action_id,
                    "mask": list(runtime["mask"]),
                    "mask_reasons": list(runtime["reasons"]),
                    "ranking": list(ranking),
                    "runtime_state": state,
                    "action_definitions": list(_inventory(raw_inputs)["action_definitions"]),
                })
    return {
        "status": "PASS",
        "selected_cases": selected_cases,
        "policy_states": policy_states,
        "rankings": rankings,
        "masks": masks,
        "reasons": reasons,
        "executable": executable,
    }


def _explicit_unresolved(obligation_id: str) -> Checker:
    def checker(*, evidence: Mapping[str, Any] | None = None, raw_inputs: Any = None, **kwargs: Any) -> dict[str, Any]:
        # 兼容旧的单元调用：完全没有 fresh 输入时，先报告 evidence 缺失；
        # 一旦进入 verifier 主链，则明确报告算法尚未实现。
        if raw_inputs is None and not evidence:
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "OBLIGATION_EVIDENCE_MISSING", "witness": {}}
        return verify_unresolved(obligation_id, raw_inputs=raw_inputs, **kwargs)

    checker.__name__ = f"verify_{obligation_id.lower()}"
    return checker


def _candidate_status_checker(obligation_id: str) -> Checker:
    """返回 obligation 对应的 fresh verifier。

    这个分发器保留旧名字，但不再把 bootstrap / reference 义务固定连接到
    unresolved 占位符。所有 active obligation 都必须落到可独立重算的
    verifier 上，或在确实没有实现时显式进入 unresolved。
    """
    bootstrap_schema_files = {
        "REGISTRY_META_SCHEMA": ("registry_meta_schema.json", "certificate_status"),
        "P0_PROFILE_SCHEMA": ("p0_profile_schema.json", "certificate_status"),
        "CONTEXT_SCHEMA": ("certificate_context_schema.json", "certificate_status"),
        "CANONICAL_SERIALIZATION": ("canonical_serialization.json", "certificate_status"),
    }
    if obligation_id in bootstrap_schema_files:
        schema_file, _ = bootstrap_schema_files[obligation_id]
        return lambda **kwargs: _verify_schema_file_obligation(obligation_id, schema_file, **kwargs)
    if obligation_id in {"THEORY_MANIFEST", "THEORY_LIBRARY_VERSION", "ASSURANCE_POLICY"}:
        return lambda **kwargs: _verify_theory_library_obligation(obligation_id, **kwargs)
    if obligation_id == "OBLIGATION_REGISTRY":
        return _verify_obligation_registry_obligation
    if obligation_id == "CLAIM_AGGREGATION":
        return _verify_claim_aggregation_obligation
    if obligation_id == "INTERFACE_COVERAGE":
        return _verify_interface_coverage_obligation
    if obligation_id == "MIGRATION_MANIFEST":
        return _verify_migration_manifest_obligation
    if obligation_id == "PROOF_REQUEST":
        return _verify_proof_request_obligation
    if obligation_id == "SOURCE_TREE_INTEGRITY":
        return _verify_source_tree_integrity_obligation
    if obligation_id == "RUNTIME_ENVIRONMENT":
        return _verify_runtime_environment_obligation
    if obligation_id == "DEPENDENCY_LOCK":
        return _verify_dependency_lock_obligation
    if obligation_id == "CHECKER_VERSION":
        return _verify_checker_version_obligation
    if obligation_id == "IMMUTABLE_INPUT_HASH":
        return _verify_immutable_input_hash_obligation
    if obligation_id == "EFFECTIVE_RUNTIME_CONFIG":
        return _verify_effective_runtime_config_obligation
    if obligation_id == "REFERENCE_TASKSET":
        return _verify_reference_taskset_obligation
    if obligation_id == "REFERENCE_SEMANTICS_CONTRACT":
        return _verify_reference_semantics_contract
    if obligation_id == "ALL_TASK_REFERENCE_RTA_ARITHMETIC":
        return _verify_all_task_reference_rta_obligation
    if obligation_id == "REFERENCE_TRANSITION_SYSTEM_IDENTITY":
        return _verify_reference_transition_system_identity
    if obligation_id == "REFERENCE_MODEL_CONFORMANCE":
        return _verify_reference_model_conformance
    if obligation_id == "PROTECTED_HI_RTA_ARITHMETIC":
        return _verify_protected_hi_rta_obligation
    return _explicit_unresolved(obligation_id)


def _specs_root(raw_inputs: Any) -> Path:
    return Path(raw_inputs.source_root) / "formal_toolchain" / "specs"


def _theory_root(raw_inputs: Any) -> Path:
    return Path(raw_inputs.source_root) / "formal_toolchain" / "theory"


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_schema_file_obligation(obligation_id: str, schema_filename: str, *, raw_inputs=None,
                                   candidate_evidence=None, expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, obligation_id)
    if error:
        return error
    result = verify_json_schema_file(_specs_root(raw) / schema_filename)
    return _finish(obligation_id, result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_theory_library_obligation(obligation_id: str, *, raw_inputs=None,
                                      candidate_evidence=None, expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                               "candidate_evidence": candidate_evidence}, obligation_id)
    if error:
        return error

    context = kwargs.get("context")
    route_id = None
    if context is not None:
        fresh_state = getattr(context, "fresh_state", None)
        if fresh_state is not None:
            route_id = getattr(fresh_state, "selected_route_id", None)

    try:
        if route_id is not None:
            result = verify_theory_library(_theory_root(raw), route_id=route_id)
        else:
            result = verify_theory_library(_theory_root(raw))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                  "failure": {"code": "THEORY_LIBRARY_INVALID", "detail": str(exc)}}
    if result.get("status") != "PASS":
        route = "UNRESOLVED" if result.get("status") == "UNRESOLVED" else "PROOF_BUNDLE_INVALID"
        return _finish(obligation_id, {"status": result.get("status"), "route": route,
                                       "failure": {"code": "THEORY_LIBRARY_INVALID",
                                                   "detail": result}},
                       expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)
    witness = {"library_version": result.get("library_version"),
               "theorem_count": result.get("theorem_count"),
               "theorem_ids": result.get("theorem_ids"),
               "route_id": result.get("route_id")}
    if obligation_id == "ASSURANCE_POLICY":
        policy = _read_json(_theory_root(raw) / "assurance_policy.json")
        witness["assurance_policy"] = policy
    elif obligation_id == "THEORY_MANIFEST":
        witness["theory_manifest"] = _read_json(_theory_root(raw) / "theory_manifest.json")
    else:
        witness["theory_dir"] = str(_theory_root(raw))
    return _finish(obligation_id, {"status": "PASS", "witness": witness},
                   expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_obligation_registry_obligation(*, raw_inputs=None, candidate_evidence=None,
                                           expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "OBLIGATION_REGISTRY")
    if error:
        return error
    registry = load_registry(_specs_root(raw) / "obligation_registry.json")
    result = verify_obligation_registry(registry=registry)
    return _finish("OBLIGATION_REGISTRY", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_claim_aggregation_obligation(*, raw_inputs=None, candidate_evidence=None,
                                         expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "CLAIM_AGGREGATION")
    if error:
        return error
    path = _specs_root(raw) / "claim_aggregation.json"
    try:
        data = _read_json(path)
        if data.get("schema_version") != "claim_aggregation_v1":
            raise ValueError("claim aggregation schema_version mismatch")
        if list(data.get("priority", ())) != [
                "PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED",
                "CONCRETE_TIMING_COUNTEREXAMPLE", "POLICY_CONTRACT_VIOLATION",
                "REFERENCE_COUNTEREXAMPLE", "REFERENCE_CERTIFICATE_FAILED",
                "UNRESOLVED", "DEPLOYED_TREE_PROVED"]:
            raise ValueError("claim aggregation priority mismatch")
        if list(data.get("obligation_statuses", ())) != ["PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"]:
            raise ValueError("claim aggregation obligation statuses mismatch")
        if data.get("fail_status_uses_registry_failure_route") is not True:
            raise ValueError("claim aggregation failure route policy mismatch")
        result = {"status": "PASS", "schema_version": "claim_aggregation_v1",
                  "witness": {"claim_aggregation": data}}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        result = {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                  "failure": {"code": "CLAIM_AGGREGATION_INVALID", "detail": str(exc)}}
    return _finish("CLAIM_AGGREGATION", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_interface_coverage_obligation(*, raw_inputs=None, candidate_evidence=None,
                                          expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "INTERFACE_COVERAGE")
    if error:
        return error
    registry = load_registry(_specs_root(raw) / "obligation_registry.json")
    report = build_interface_coverage_report(
        registry=registry, spec_root=_specs_root(raw),
        checker_catalog=VERIFIER_CHECKERS,
        structural_ids={
            "ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY", "DIRECT_PREDECESSOR_HASHES",
            "STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
            "CLAIM_AGGREGATION_RESULT",
        },
    )
    if report.get("status") != "PASS":
        return _finish("INTERFACE_COVERAGE", {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                                              "failure": {"code": "INTERFACE_COVERAGE_INVALID",
                                                          "detail": report}},
                       expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)
    return _finish("INTERFACE_COVERAGE", {"status": "PASS", "witness": report},
                   expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_migration_manifest_obligation(*, raw_inputs=None, candidate_evidence=None,
                                          expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "MIGRATION_MANIFEST")
    if error:
        return error
    registry = load_registry(_specs_root(raw) / "obligation_registry.json")
    migration = _read_json(_specs_root(raw) / "migration_manifest.json")
    result = verify_migration_manifest(
        migration=migration, registry=registry, current_schema_version="obligation_registry_v5")
    return _finish("MIGRATION_MANIFEST", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_proof_request_obligation(*, raw_inputs=None, candidate_evidence=None,
                                     expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "PROOF_REQUEST")
    if error:
        return error
    request_path = Path(raw.workspace) / "request" / "proof_request.json"
    try:
        request = _read_json(request_path)
        if request != dict(raw.request):
            raise ValueError("proof request 与 verifier inputs 不一致")
        if request.get("profile") != "P0" or request.get("primary_claim") != "DEPLOYED_HI_SAFETY":
            raise ValueError("proof request profile/claim mismatch")
        result = {"status": "PASS", "witness": {"proof_request": request}}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                  "failure": {"code": "PROOF_REQUEST_INVALID", "detail": str(exc)}}
    return _finish("PROOF_REQUEST", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_source_tree_integrity_obligation(*, raw_inputs=None, candidate_evidence=None,
                                             expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "SOURCE_TREE_INTEGRITY")
    if error:
        return error
    result = verify_source_manifest(manifest=raw.source_manifest, source_root=Path(raw.source_root))
    return _finish("SOURCE_TREE_INTEGRITY", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_runtime_environment_obligation(*, raw_inputs=None, candidate_evidence=None,
                                           expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "RUNTIME_ENVIRONMENT")
    if error:
        return error
    manifest = build_runtime_environment_manifest()
    valid = (
        manifest.get("schema_version") == "runtime_environment_manifest_v1"
        and isinstance(manifest.get("float_info", {}).get("max"), str)
        and manifest.get("source_encoding") == "UTF-8"
    )
    result = {"status": "PASS" if valid else "FAIL",
              "route": None if valid else "MODEL_CONFORMANCE_FAILED",
              "failure": None if valid else {"code": "RUNTIME_ENVIRONMENT_INVALID"},
              "witness": {"runtime_environment_manifest": manifest}}
    return _finish("RUNTIME_ENVIRONMENT", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _lock_path(raw_inputs: Any) -> Path | None:
    try:
        from pathlib import Path
        specs = Path(str(raw_inputs.source_root)) / "formal_toolchain" / "specs"
        lock = specs / "proof_dependency_lock.json"
        if lock.is_file():
            import json
            return json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _verify_dependency_lock_obligation(*, raw_inputs=None, candidate_evidence=None,
                                        expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "DEPENDENCY_LOCK")
    if error:
        return error
    manifest = build_dependency_manifest()
    lock = _lock_path(raw) if raw is not None else None
    result = check_dependency_policy(manifest, lock=lock)
    status = "PASS" if result.get("status") == "PASS" else "FAIL"
    failure = None if status == "PASS" else {
        "code": result.get("code", "DEPENDENCY_LOCK_INCOMPLETE"),
        "detail": {k: v for k, v in result.items() if k != "status"},
    }
    return _finish("DEPENDENCY_LOCK", {"status": status, "route": None if status == "PASS" else "PROOF_BUNDLE_INVALID",
                                        "failure": failure,
                                        "witness": {"dependency_manifest": manifest, "lock_check": result}},
                   expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_checker_version_obligation(*, raw_inputs=None, candidate_evidence=None,
                                       expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "CHECKER_VERSION")
    if error:
        return error
    manifest = build_checker_version_manifest(Path(raw.source_root))
    valid = (
        manifest.get("schema_version") == "checker_version_manifest_v1"
        and isinstance(manifest.get("checker_build_hash"), str)
        and manifest.get("schema_versions", {}).get("common_certificate") == "common_certificate_v1"
        and manifest.get("schema_versions", {}).get("p0_profile") == "p0_profile_v1"
        and manifest.get("checker_source_files")
    )
    result = {"status": "PASS" if valid else "FAIL",
              "route": None if valid else "PROOF_BUNDLE_INVALID",
              "failure": None if valid else {"code": "CHECKER_VERSION_INVALID"},
              "witness": {"checker_version_manifest": manifest}}
    return _finish("CHECKER_VERSION", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _immutable_inputs_payload(raw_inputs: Any) -> dict[str, Any]:
    specs_root = _specs_root(raw_inputs)
    spec_files = {
        path.relative_to(specs_root).as_posix(): _read_json(path)
        for path in sorted(specs_root.rglob("*.json"))
    }
    tree = _read_json(Path(raw_inputs.artifact_dir) / "integer_tree.json")
    return {
        "source_manifest": raw_inputs.source_manifest,
        "runtime_manifest": build_runtime_environment_manifest(),
        "dependency_manifest": build_dependency_manifest(),
        "checker_manifest": build_checker_version_manifest(Path(raw_inputs.source_root)),
        "taskset": raw_inputs.preflight.get("taskset"),
        "priority": list(raw_inputs.preflight.get("taskset", {}).get("priority_order", [])),
        "tree": tree,
        "features": list(raw_inputs.inventory.get("feature_names", [])),
        "actions": list(raw_inputs.inventory.get("action_definitions", [])),
        "fixed_point": raw_inputs.inventory.get("fixed_point_config"),
        "effective_config": export_formal_target_config(raw_inputs.target),
        "theory": _read_json(_theory_root(raw_inputs) / "theory_manifest.json"),
        "specs": spec_files,
    }


def _verify_immutable_input_hash_obligation(*, raw_inputs=None, candidate_evidence=None,
                                            expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "IMMUTABLE_INPUT_HASH")
    if error:
        return error
    from formal_toolchain.verifier.immutable_input_verifier import recompute_immutable_input_root
    payload = _immutable_inputs_payload(raw)
    try:
        expected = recompute_immutable_input_root(payload)
    except (TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                  "failure": {"code": "IMMUTABLE_INPUT_RECOMPUTE_FAILED", "detail": str(exc)}}
    else:
        result = {"status": "PASS", "witness": expected}
    return _finish("IMMUTABLE_INPUT_HASH", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_effective_runtime_config_obligation(*, raw_inputs=None, candidate_evidence=None,
                                                expected_context_hash=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "EFFECTIVE_RUNTIME_CONFIG")
    if error:
        return error
    result = export_formal_target_config(raw.target)
    return _finish("EFFECTIVE_RUNTIME_CONFIG", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _reference_taskset_witness(raw_inputs: Any, certified_envelope: Mapping[str, Any],
                               fresh_reference: Any | None) -> dict[str, Any]:
    if fresh_reference is not None:
        return {"reference_taskset": fresh_reference.to_dict() if hasattr(fresh_reference, "to_dict") else fresh_reference}
    envelope_hash = sha256_object(dict(certified_envelope))
    from formal_toolchain.reference.task_mapping import build_reference_taskset
    budget_by_task = {
        str(name): {**dict(row), "b_bar": int(certified_envelope["upper"][name]),
                    "certified_envelope_hash": envelope_hash}
        for name, row in raw_inputs.target.provenance["budget_by_task"].items()
    }
    reference = build_reference_taskset(
        raw_inputs.target.ordered_tasks, budget_by_task,
        xf=raw_inputs.target.runtime_config.c_amc_sem_lo_degradation_ratio,
        certified_envelope=certified_envelope,
        semantic_context_hash=str(raw_inputs.contexts["semantic_context"]["hash"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(raw_inputs.target)),
    )
    return {"reference_taskset": reference.to_dict()}


def _verify_reference_taskset_obligation(*, raw_inputs=None, candidate_evidence=None,
                                         expected_context_hash=None, certified_envelope=None,
                                         fresh_reference=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "REFERENCE_TASKSET")
    if error:
        return error
    try:
        if not isinstance(certified_envelope, Mapping):
            return _raw_inputs_result("REFERENCE_TASKSET", code="CERTIFIED_ENVELOPE_REQUIRED")
        budget_by_task = {
            str(name): {**dict(row), "b_bar": int(certified_envelope["upper"][name]),
                        "certified_envelope_hash": sha256_object(dict(certified_envelope))}
            for name, row in raw.target.provenance["budget_by_task"].items()
        }
        if hasattr(fresh_reference, "to_dict"):
            reference_obj = fresh_reference
        else:
            reference_obj = build_reference_taskset(
                raw.target.ordered_tasks, budget_by_task,
                xf=raw.target.runtime_config.c_amc_sem_lo_degradation_ratio,
                certified_envelope=certified_envelope,
                semantic_context_hash=str(raw.contexts["semantic_context"]["hash"]),
                effective_runtime_config_hash=sha256_object(export_formal_target_config(raw.target)),
            )
        result = validate_reference_mapping(
            reference=reference_obj, ordered_tasks=raw.target.ordered_tasks,
            budget_by_task=budget_by_task,
            certified_envelope=certified_envelope,
            xf=raw.target.runtime_config.c_amc_sem_lo_degradation_ratio,
            semantic_context_hash=str(raw.contexts["semantic_context"]["hash"]),
            effective_runtime_config_hash=sha256_object(export_formal_target_config(raw.target)),
        )
        status = result.get("obligation_status", result.get("status"))
        if status != "PASS":
            result = {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                      "failure": {"code": "REFERENCE_TASKSET_INVALID", "detail": result}}
        else:
            result = {"status": "PASS", "witness": {"reference_taskset": reference_obj.to_dict() if hasattr(reference_obj, "to_dict") else reference_obj}}
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                  "failure": {"code": "REFERENCE_TASKSET_INVALID", "detail": str(exc)}}
    return _finish("REFERENCE_TASKSET", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_reference_semantics_contract(*, raw_inputs=None, candidate_evidence=None,
                                         expected_context_hash=None, fresh_reference=None,
                                         verified_predecessors=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "REFERENCE_SEMANTICS_CONTRACT")
    if error:
        return error
    if fresh_reference is None:
        return _raw_inputs_result("REFERENCE_SEMANTICS_CONTRACT", code="FRESH_REFERENCE_TASKSET_MISSING")
    predecessors = verified_predecessors if isinstance(verified_predecessors, Mapping) else {}
    required = {"REFERENCE_TASKSET", "EFFECTIVE_RUNTIME_CONFIG", "STRICT_PRIORITY_ORDER"}
    if set(predecessors) & required != required:
        return _raw_inputs_result("REFERENCE_SEMANTICS_CONTRACT", code="REFERENCE_SEMANTICS_PREDECESSORS_MISSING")
    try:
        from formal_toolchain.reference.semantics_contract import build_reference_semantics_contract_certificate
        reference_taskset = fresh_reference.to_dict() if hasattr(fresh_reference, "to_dict") else fresh_reference
        result = build_reference_semantics_contract_certificate(
            reference_taskset=reference_taskset,
            reference_taskset_certificate=predecessors["REFERENCE_TASKSET"],
            effective_runtime_config_certificate=predecessors["EFFECTIVE_RUNTIME_CONFIG"],
            strict_priority_certificate=predecessors["STRICT_PRIORITY_ORDER"],
            contexts=raw.contexts,
            context_hash=expected_context_hash or str(raw.contexts["reference_context"]["hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                  "failure": {"code": "REFERENCE_SEMANTICS_CONTRACT_FAILED", "detail": str(exc)}}
    return _finish("REFERENCE_SEMANTICS_CONTRACT", result,
                   expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_protected_hi_rta_obligation(*, raw_inputs=None, candidate_evidence=None,
                                        expected_context_hash=None, certified_envelope=None,
                                        fresh_reference=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "PROTECTED_HI_RTA_ARITHMETIC")
    if error:
        return error
    if fresh_reference is None:
        return _raw_inputs_result("PROTECTED_HI_RTA_ARITHMETIC", code="FRESH_REFERENCE_TASKSET_MISSING")
    try:
        from formal_toolchain.reference.rta_replay import replay_all_task_rta

        production = protected_hi_rta(fresh_reference)
        replay = replay_all_task_rta(fresh_reference, production)
        if not verify_obligation_certificate(production):
            raise ValueError("protected_hi_rta certificate hash invalid")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        result = {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "PROTECTED_HI_RTA_INVALID",
            "failure": {"code": "PROTECTED_HI_RTA_INVALID", "detail": str(exc)},
        }
    else:
        if replay.get("status") != "PASS":
            result = {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "PROTECTED_HI_RTA_REPLAY_MISMATCH",
                "failure": {"code": "PROTECTED_HI_RTA_REPLAY_MISMATCH", "detail": replay},
            }
        elif candidate_evidence:
            candidate_witness = candidate_evidence.get("witness") if isinstance(candidate_evidence, Mapping) else {}
            candidate_inputs = candidate_evidence.get("inputs") if isinstance(candidate_evidence, Mapping) else {}
            if isinstance(candidate_witness, Mapping) and candidate_witness != {"production": production, "replay": replay}:
                result = {
                    "status": "FAIL",
                    "route": "PROOF_BUNDLE_INVALID",
                    "code": "PROTECTED_HI_RTA_REPLAY_MISMATCH",
                    "failure": {"code": "PROTECTED_HI_RTA_REPLAY_MISMATCH", "detail": "candidate witness mismatch"},
                }
            elif isinstance(candidate_inputs, Mapping) and candidate_inputs:
                result = {
                    "status": "FAIL",
                    "route": "PROOF_BUNDLE_INVALID",
                    "code": "PROTECTED_HI_RTA_REPLAY_MISMATCH",
                    "failure": {"code": "PROTECTED_HI_RTA_REPLAY_MISMATCH", "detail": "candidate inputs mismatch"},
                }
            else:
                result = {"status": "PASS", "witness": {"production": production, "replay": replay}}
        else:
            result = {"status": "PASS", "witness": {"production": production, "replay": replay}}
    return _finish("PROTECTED_HI_RTA_ARITHMETIC", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_all_task_reference_rta_obligation(*, raw_inputs=None, candidate_evidence=None,
                                              expected_context_hash=None, fresh_reference=None,
                                              **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "ALL_TASK_REFERENCE_RTA_ARITHMETIC")
    if error:
        return error
    if fresh_reference is None:
        return _raw_inputs_result("ALL_TASK_REFERENCE_RTA_ARITHMETIC", code="FRESH_REFERENCE_TASKSET_MISSING")
    try:
        replay = replay_all_task_rta_independently(fresh_reference)
    except (KeyError, TypeError, ValueError) as exc:
        result = {
            "status": "FAIL",
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "ALL_TASK_REFERENCE_RTA_INVALID",
            "failure": {"code": "ALL_TASK_REFERENCE_RTA_INVALID", "detail": str(exc)},
        }
        return _finish("ALL_TASK_REFERENCE_RTA_ARITHMETIC", result,
                       expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)
    result = dict(replay)
    if isinstance(candidate_evidence, Mapping):
        candidate_witness = candidate_evidence.get("witness", candidate_evidence)
        if candidate_witness and candidate_witness != replay.get("witness", {}):
            result = {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "ALL_TASK_REFERENCE_RTA_REPLAY_MISMATCH",
                "failure": {"code": "ALL_TASK_REFERENCE_RTA_REPLAY_MISMATCH",
                            "detail": "candidate witness mismatch"},
                "witness": replay.get("witness", {}),
            }
    return _finish("ALL_TASK_REFERENCE_RTA_ARITHMETIC", result,
                   expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_reference_transition_system_identity(*, raw_inputs=None, candidate_evidence=None,
                                                  expected_context_hash=None, fresh_reference=None,
                                                  verified_predecessors=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence},
                             "REFERENCE_TRANSITION_SYSTEM_IDENTITY")
    if error:
        return error
    try:
        if fresh_reference is None:
            return _raw_inputs_result("REFERENCE_TRANSITION_SYSTEM_IDENTITY", code="FRESH_REFERENCE_TASKSET_MISSING")
        from formal_toolchain.reference.transition_identity import build_reference_transition_identity_certificate
        from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
        reference_dict = getattr(fresh_reference, "to_dict", lambda: fresh_reference)()
        model_bounds = derive_p0_model_bounds(reference_dict)
        predecessors = verified_predecessors if isinstance(verified_predecessors, Mapping) else {}
        contexts = getattr(raw, "contexts", {})
        result = build_reference_transition_identity_certificate(
            reference_taskset=reference_dict,
            model_bounds=model_bounds,
            verified_predecessors=predecessors,
            contexts=contexts,
            context_hash=expected_context_hash or str(contexts.get("reference_context", {}).get("hash", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        result = {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "failure": {"code": "REFERENCE_TRANSITION_SYSTEM_IDENTITY_FAILED", "detail": str(exc)},
        }
    return _finish("REFERENCE_TRANSITION_SYSTEM_IDENTITY", result,
                   expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_reference_model_conformance(*, raw_inputs=None, candidate_evidence=None,
                                        expected_context_hash=None, fresh_reference=None,
                                        verified_predecessors=None, **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "REFERENCE_MODEL_CONFORMANCE")
    if error:
        return error
    try:
        if fresh_reference is None:
            return _raw_inputs_result("REFERENCE_MODEL_CONFORMANCE", code="FRESH_REFERENCE_TASKSET_MISSING")
        from formal_toolchain.reference.model_conformance import build_reference_model_conformance_certificate
        if not isinstance(verified_predecessors, Mapping):
            verified_predecessors = {}
        candidate = candidate_evidence if isinstance(candidate_evidence, Mapping) else {}
        result = verify_reference_model_conformance(
            candidate_certificate=candidate,
            raw_inputs=raw,
            verified_predecessors=verified_predecessors,
            expected_context_hash=expected_context_hash or str(raw.contexts["reference_context"]["hash"]),
            fresh_reference=fresh_reference,
        )
    except (KeyError, TypeError, ValueError) as exc:
        result = {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "REFERENCE_MODEL_CONFORMANCE_FAILED",
            "failure": {"code": "REFERENCE_MODEL_CONFORMANCE_FAILED", "detail": str(exc)},
        }
    return _finish("REFERENCE_MODEL_CONFORMANCE", result, expected_context_hash=expected_context_hash,
                   candidate_evidence=candidate_evidence)


def _verify_final_claim_composition(*, raw_inputs=None, candidate_evidence=None,
                                    expected_context_hash=None, verified_predecessors=None,
                                    **kwargs):
    raw, error = _raw_inputs({"raw_inputs": raw_inputs, "evidence": kwargs.get("evidence"),
                              "candidate_evidence": candidate_evidence}, "FINAL_CLAIM_COMPOSITION")
    if error:
        return error
    if not isinstance(verified_predecessors, Mapping):
        return _raw_inputs_result("FINAL_CLAIM_COMPOSITION", code="VERIFIED_PREDECESSORS_MISSING")

    from formal_toolchain.theory.loader import load_verified_theory_statement
    from .predecessor_contract import require_verified_predecessor as rvp
    from .predecessor_contract import require_exact_predecessor_set

    theorem = load_verified_theory_statement(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "theory",
        "FINAL_DEPLOYED_HI_SAFETY_COMPOSITION",
    )
    premise_obligation_ids = theorem.get("premise_obligation_ids")
    if not isinstance(premise_obligation_ids, list) or not premise_obligation_ids:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "FINAL_THEOREM_MACHINE_PREMISES_MISSING",
            "failure": {"code": "FINAL_THEOREM_MACHINE_PREMISES_MISSING"},
        }

    from .predecessor_contract import PredecessorContractError
    contexts = getattr(raw, "contexts", {}) if raw is not None else {}
    try:
        require_exact_predecessor_set(predecessors=verified_predecessors, expected_ids=set(premise_obligation_ids))
        for premise_id in premise_obligation_ids:
            rvp(predecessors=verified_predecessors, obligation_id=premise_id, contexts=contexts)
    except PredecessorContractError as exc:
        return {
            "status": "FAIL",
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "FINAL_CLAIM_THEOREM_PREMISE_FAILED",
            "failure": {"code": "FINAL_CLAIM_THEOREM_PREMISE_FAILED", "detail": str(exc)},
        }

    witness_rows = []
    for obligation_id in sorted(premise_obligation_ids):
        certificate = verified_predecessors[obligation_id]
        valid = (
            isinstance(certificate, Mapping)
            and certificate.get("obligation_id") == obligation_id
            and certificate.get("obligation_status") == "PASS"
            and verify_obligation_certificate(certificate)
        )
        witness_rows.append({
            "obligation_id": obligation_id,
            "status": certificate.get("obligation_status") if isinstance(certificate, Mapping) else None,
            "valid": valid,
            "artifact_hash": certificate.get("artifact_hash") if isinstance(certificate, Mapping) else None,
        })

    if not all(row["valid"] for row in witness_rows):
        result = {
            "status": "FAIL",
            "route": "REFERENCE_CERTIFICATE_FAILED",
            "code": "FINAL_CLAIM_PREMISES_INVALID",
            "failure": {"code": "FINAL_CLAIM_PREMISES_INVALID", "witness": witness_rows},
        }
    else:
        result = {
            "status": "PASS",
            "witness": {
                "theorem_id": theorem.get("theorem_id"),
                "theorem_statement_hash": theorem.get("statement_hash"),
                "premise_obligation_ids": sorted(premise_obligation_ids),
                "premises": witness_rows,
                "final_claim": "DEPLOYED_HI_SAFETY",
            },
        }
    return _finish("FINAL_CLAIM_COMPOSITION", result,
                   expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_tree(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("TREE_WELLFORMEDNESS", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        tree = _tree_model(raw_inputs)
        result = validate_tree_and_leaf_partition(tree)
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "TREE_WELLFORMEDNESS_FAILED", "detail": str(exc)}}
    return _finish("TREE_WELLFORMEDNESS", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_leaf_guard_partition(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("LEAF_GUARD_PARTITION", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        tree = _tree_model(raw_inputs)
        result = validate_tree_and_leaf_partition(tree)
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "LEAF_GUARD_PARTITION_FAILED", "detail": str(exc)}}
    return _finish("LEAF_GUARD_PARTITION", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_quantization(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("FEATURE_QUANTIZATION", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        fixed_data = _fixed_point_config(raw_inputs)
        from amc_py.viper.fixed_point import fixed_point_config_from_dict, quantize_value
        config = fixed_point_config_from_dict(fixed_data)
        result = verify_against_production(
            deterministic_samples(),
            fixed_data,
            lambda value, _config: quantize_value(value, config),
        )
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "FEATURE_QUANTIZATION_FAILED", "detail": str(exc)}}
    return _finish("FEATURE_QUANTIZATION", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_action_transition(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("ACTION_TRANSITION", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        target = _target(raw_inputs)
        actions = _actions(raw_inputs)
        domain = _domain(raw_inputs)
        result = build_action_transition_table(
        actions, target.ordered_tasks, domain["tasks"],
        rounding_mode=str(getattr(target.runtime_config, "budget_rounding_mode", "ceil_floor")),
        min_budget_delta=int(getattr(target.runtime_config, "min_budget_delta", 1)),
    )
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "ACTION_TRANSITION_FAILED", "detail": str(exc)}}
    return _finish("ACTION_TRANSITION", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_budget_domain(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("BUDGET_DOMAIN", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        result = _domain(raw_inputs)
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                  "failure": {"code": "BUDGET_DOMAIN_FAILED", "detail": str(exc)}}
    return _finish("BUDGET_DOMAIN", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_mask_fallback(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("MASK_FALLBACK", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        context_hash = _expected_context(raw_inputs, "MASK_FALLBACK", expected_context_hash)
        if context_hash is None:
            return _raw_inputs_result("MASK_FALLBACK", code="CANDIDATE_CONTEXT_MISSING")
        result = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)["mask"]
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "MASK_FALLBACK_FAILED", "detail": str(exc)}}
    return _finish("MASK_FALLBACK", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_selected_action_regions(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("SELECTED_ACTION_REGIONS", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        context_hash = _expected_context(raw_inputs, "SELECTED_ACTION_REGIONS", expected_context_hash)
        if context_hash is None:
            return _raw_inputs_result("SELECTED_ACTION_REGIONS", code="CANDIDATE_CONTEXT_MISSING")
        tree = _tree_model(raw_inputs)
        regions = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)["regions"]
        result = {
            "status": "PASS",
            "schema_version": "selected_action_regions_v2",
            "regions": regions["regions"],
            "leaf_count": len(_leaf_guards(tree)),
            "action_count": len(_actions(raw_inputs)),
            "implicit_noop_region_count": sum(1 for region in regions["regions"] if region.get("implicit_noop_predicate")),
            "universal_over_policy_inputs": True,
            "state_enumeration_used": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "SELECTED_ACTION_REGIONS_FAILED", "detail": str(exc)}}
    return _finish("SELECTED_ACTION_REGIONS", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_executable_policy_semantics(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("EXECUTABLE_POLICY_SEMANTICS", code="OBLIGATION_EVIDENCE_MISSING")
    try:
        context_hash = _expected_context(raw_inputs, "EXECUTABLE_POLICY_SEMANTICS", expected_context_hash)
        if context_hash is None:
            return _raw_inputs_result("EXECUTABLE_POLICY_SEMANTICS", code="CANDIDATE_CONTEXT_MISSING")
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        target = _target(raw_inputs)
        tree = _tree_model(raw_inputs)
        fixed_data = _fixed_point_config(raw_inputs)
        adapter = _runtime_adapter(raw_inputs)
        names = [str(task.name) for task in target.ordered_tasks]
        sample_vectors = [
            tuple(int(pipeline["domain"]["tasks"][name]["initial"]) for name in names),
            tuple(int(pipeline["domain"]["tasks"][name]["action_hard_upper"]) for name in names),
        ]
        policy_states = []
        for vector in sample_vectors:
            state = {
                "budgets": dict(zip(names, vector)),
                "initial_budgets": {name: int(pipeline["domain"]["tasks"][name]["initial"]) for name in names},
                "floors": {name: int(pipeline["domain"]["tasks"][name]["runtime_floor"]) for name in names},
                "caps": {name: int(pipeline["domain"]["tasks"][name]["action_hard_upper"]) for name in names},
                "config": export_formal_target_config(target),
            }
            runtime_state = adapter.build_runtime_state_from_budget_vector(state["budgets"])
            runtime_mask, _ = adapter.valid_action_mask(runtime_state)
            observation = adapter.extract_observation(runtime_state)
            quantized = tuple(replay_quantize(value, fixed_data)[0] for value in observation)
            leaf_id = int(evaluate_integer_tree(tree, quantized).leaf_id)
            policy_states.append({
                "state": state,
                "leaf_id": leaf_id,
                "mask": list(runtime_mask),
            })
        result = {
            "status": "PASS",
            "schema_version": "executable_policy_semantics_v1",
            "states": policy_states,
            "selected_case_count": len(policy_states),
            "budget_domain_hash": sha256_object(pipeline["domain"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "EXECUTABLE_POLICY_SEMANTICS_FAILED", "detail": str(exc)}}
    return _finish("EXECUTABLE_POLICY_SEMANTICS", result, expected_context_hash=expected_context_hash, candidate_evidence=candidate_evidence)


def _verify_candidate_envelope(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("CANDIDATE_ENVELOPE", code="OBLIGATION_EVIDENCE_MISSING")
    context_hash = _expected_context(raw_inputs, "CANDIDATE_ENVELOPE", expected_context_hash)
    if context_hash is None:
        return _raw_inputs_result("CANDIDATE_ENVELOPE", code="CANDIDATE_CONTEXT_MISSING")
    try:
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        result = pipeline["candidate"]
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "CANDIDATE_ENVELOPE_FAILED", "detail": str(exc)}}
    return _finish("CANDIDATE_ENVELOPE", result, expected_context_hash=context_hash, candidate_evidence=candidate_evidence)


def _verify_common_transition_preservation(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("COMMON_TRANSITION_PRESERVATION", code="OBLIGATION_EVIDENCE_MISSING")
    context_hash = _expected_context(raw_inputs, "COMMON_TRANSITION_PRESERVATION", expected_context_hash)
    if context_hash is None:
        return _raw_inputs_result("COMMON_TRANSITION_PRESERVATION", code="CANDIDATE_CONTEXT_MISSING")
    try:
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        result = pipeline["common"]
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "COMMON_TRANSITION_PRESERVATION_FAILED", "detail": str(exc)}}
    return _finish("COMMON_TRANSITION_PRESERVATION", result, expected_context_hash=context_hash, candidate_evidence=candidate_evidence)


def _verify_deployed_policy_preservation(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("DEPLOYED_POLICY_PRESERVATION", code="OBLIGATION_EVIDENCE_MISSING")
    context_hash = _expected_context(raw_inputs, "DEPLOYED_POLICY_PRESERVATION", expected_context_hash)
    if context_hash is None:
        return _raw_inputs_result("DEPLOYED_POLICY_PRESERVATION", code="CANDIDATE_CONTEXT_MISSING")
    try:
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        result = pipeline["deployed"]
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "DEPLOYED_POLICY_PRESERVATION_FAILED", "detail": str(exc)}}
    return _finish("DEPLOYED_POLICY_PRESERVATION", result, expected_context_hash=context_hash, candidate_evidence=candidate_evidence)


def _verify_certified_envelope(*, raw_inputs=None, candidate_evidence=None, expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result("CERTIFIED_ENVELOPE", code="OBLIGATION_EVIDENCE_MISSING")
    context_hash = _expected_context(raw_inputs, "CERTIFIED_ENVELOPE", expected_context_hash)
    if context_hash is None:
        return _raw_inputs_result("CERTIFIED_ENVELOPE", code="CANDIDATE_CONTEXT_MISSING")
    try:
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        candidate = pipeline["candidate"]
        common = pipeline["common"]
        deployed = pipeline["deployed"]
        if deployed.get("status") == "FAIL":
            result = {
                "status": "FAIL",
                "route": str(deployed.get("route", "POLICY_CONTRACT_VIOLATION")),
                "failure": dict(deployed.get("failure", {})),
                "witness": {"deployed_policy_failure": deployed},
            }
        elif candidate.get("status") == "FAIL" or common.get("status") == "FAIL":
            failed = candidate if candidate.get("status") == "FAIL" else common
            result = {
                "status": "FAIL",
                "route": str(failed.get("route", "POLICY_CONTRACT_VIOLATION")),
                "failure": dict(failed.get("failure", {})),
                "witness": {"envelope_precondition_failure": failed},
            }
        elif any(item.get("status") != "PASS" for item in (candidate, common, deployed)):
            result = {"status": "UNRESOLVED", "route": "UNRESOLVED",
                      "failure": {"code": "CERTIFIED_ENVELOPE_PRECONDITION_FAILED",
                                  "common": common, "deployed": deployed}}
        else:
            attestation = {
                "fresh_process": True,
                "candidate_hash": sha256_proof_object(candidate),
                "common_hash": sha256_proof_object(common),
                "deployed_hash": sha256_proof_object(deployed),
            }
            result = _certify_envelope_from_verifier(
                candidate,
                common,
                deployed,
                context_hash=context_hash,
                verifier_attestation=attestation,
            )
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": "CERTIFIED_ENVELOPE_FAILED", "detail": str(exc)}}
    return _finish("CERTIFIED_ENVELOPE", result, expected_context_hash=context_hash, candidate_evidence=candidate_evidence)


def _verify_budget_invariant(obligation_id: str, *, raw_inputs=None, candidate_evidence=None,
                             expected_context_hash=None, **kwargs):
    if raw_inputs is None:
        return _raw_inputs_result(obligation_id, code="OBLIGATION_EVIDENCE_MISSING")
    context_hash = _expected_context(raw_inputs, obligation_id, expected_context_hash)
    if context_hash is None:
        return _raw_inputs_result(obligation_id, code="CANDIDATE_CONTEXT_MISSING")
    if obligation_id == "ACTIVE_RELEASE_BUDGET_INVARIANT":
        source_result = check_active_release_budget_source_contract(
            _field(raw_inputs, "source_root")
        )
        if source_result.get("status") == "FAIL":
            return _finish(
                obligation_id,
                source_result,
                expected_context_hash=context_hash,
                candidate_evidence=candidate_evidence,
            )
    try:
        pipeline = _fresh_structural_envelope_pipeline(raw_inputs, context_hash=context_hash)
        candidate = pipeline["candidate"]
        common = pipeline["common"]
        deployed = pipeline["deployed"]
        if deployed.get("status") == "FAIL":
            result = {
                "status": "FAIL",
                "route": str(deployed.get("route", "POLICY_CONTRACT_VIOLATION")),
                "failure": dict(deployed.get("failure", {})),
                "witness": {"deployed_policy_failure": deployed},
            }
        elif candidate.get("status") == "FAIL" or common.get("status") == "FAIL":
            failed = candidate if candidate.get("status") == "FAIL" else common
            result = {
                "status": "FAIL",
                "route": str(failed.get("route", "POLICY_CONTRACT_VIOLATION")),
                "failure": dict(failed.get("failure", {})),
                "witness": {"envelope_precondition_failure": failed},
            }
        elif any(item.get("status") != "PASS" for item in (candidate, common, deployed)):
            result = {"status": "UNRESOLVED", "route": "UNRESOLVED",
                      "failure": {"code": "BUDGET_INVARIANT_PRECONDITION_FAILED"}}
        else:
            cert = _certify_envelope_from_verifier(
                candidate,
                common,
                deployed,
                context_hash=context_hash,
                verifier_attestation={
                    "fresh_process": True,
                    "candidate_hash": sha256_proof_object(candidate),
                    "common_hash": sha256_proof_object(common),
                    "deployed_hash": sha256_proof_object(deployed),
                },
            )
            target = _target(raw_inputs)
            reference_taskset = {
                "tasks": [
                    {
                        "name": str(task.name),
                        "criticality": str(getattr(task.criticality, "value", task.criticality)),
                    }
                    for task in target.ordered_tasks
                ]
            }
            cert_certificate = {
                "artifact_schema_version": "synthetic_phase_fh_certificate_v1",
                "obligation_id": "CERTIFIED_ENVELOPE",
                "obligation_status": "PASS",
                "certificate_context_hash": context_hash,
                "direct_predecessor_hashes": {},
                "checker_id": "fresh_verifier",
                "checker_version": "1",
                "inputs": {"fixture": "fresh_verifier"},
                "witness": {"candidate_hash": sha256_proof_object(candidate),
                            "common_hash": sha256_proof_object(common),
                            "deployed_hash": sha256_proof_object(deployed)},
                "evidence": [{"fresh_process": True}],
                "failure": None,
            }
            derived = derive_budget_invariant_evidence(
                reference_taskset=reference_taskset,
                candidate=candidate,
                common=common,
                deployed=deployed,
                certified_envelope=cert,
                certified_certificate=cert_certificate,
            )
            result = derived[obligation_id]
    except (KeyError, TypeError, ValueError) as exc:
        result = {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                  "failure": {"code": f"{obligation_id}_FAILED", "detail": str(exc)}}
    return _finish(obligation_id, result, expected_context_hash=context_hash, candidate_evidence=candidate_evidence)


VERIFIER_CHECKERS: dict[str, Checker] = {
    # bootstrap / implementation boundary；这些文件由 bootstrap checker
    # 在 bundle 主流程中独立重算，catalog 仍显式列出它们以闭合 coverage。
    "REGISTRY_META_SCHEMA": _candidate_status_checker("REGISTRY_META_SCHEMA"),
    "P0_PROFILE_SCHEMA": _candidate_status_checker("P0_PROFILE_SCHEMA"),
    "THEORY_MANIFEST": _candidate_status_checker("THEORY_MANIFEST"),
    "THEORY_LIBRARY_VERSION": _candidate_status_checker("THEORY_LIBRARY_VERSION"),
    "ASSURANCE_POLICY": _candidate_status_checker("ASSURANCE_POLICY"),
    "OBLIGATION_REGISTRY": _candidate_status_checker("OBLIGATION_REGISTRY"),
    "CLAIM_AGGREGATION": _candidate_status_checker("CLAIM_AGGREGATION"),
    "CONTEXT_SCHEMA": _candidate_status_checker("CONTEXT_SCHEMA"),
    "CANONICAL_SERIALIZATION": _candidate_status_checker("CANONICAL_SERIALIZATION"),
    "INTERFACE_COVERAGE": _candidate_status_checker("INTERFACE_COVERAGE"),
    "MIGRATION_MANIFEST": _candidate_status_checker("MIGRATION_MANIFEST"),
    "PROOF_REQUEST": _candidate_status_checker("PROOF_REQUEST"),
    "SOURCE_TREE_INTEGRITY": _candidate_status_checker("SOURCE_TREE_INTEGRITY"),
    "RUNTIME_ENVIRONMENT": _candidate_status_checker("RUNTIME_ENVIRONMENT"),
    "DEPENDENCY_LOCK": _candidate_status_checker("DEPENDENCY_LOCK"),
    "CHECKER_VERSION": _candidate_status_checker("CHECKER_VERSION"),
    "IMMUTABLE_INPUT_HASH": _candidate_status_checker("IMMUTABLE_INPUT_HASH"),
    "EFFECTIVE_RUNTIME_CONFIG": _candidate_status_checker("EFFECTIVE_RUNTIME_CONFIG"),
    # semantic
    "SCHEDULER_MODEL": verify_scheduler_model,
    "STRICT_PRIORITY_ORDER": verify_strict_priority_order,
    "TIME_DOMAIN": verify_time_domain,
    "NO_OVERFLOW": verify_no_overflow,
    "OVERHEAD_PROFILE": verify_overhead_profile,
    "MODE_SEMANTICS_CONFORMANCE": verify_mode_semantics,
    "DEMAND_ORACLE_BATCH_CONTRACT": verify_demand_oracle_contract,
    "HI_EXECUTION_CONTRACT": verify_hi_execution_contract,
    "REMOVAL_COMPLETENESS": verify_removal_completeness,
    "HI_NONTRUNCATION": verify_hi_nontruncation,
    "DEADLINE_OBSERVATION": verify_deadline_observation,
    "EFFECTIVE_EVENT_ORDER": verify_event_order,
    "OBSERVATION_EXTRACTION": verify_observation_extraction,
    "FEATURE_TOTALITY": verify_feature_totality,
    # explicit unresolved semantic entries; absence of implementation cannot pass
    "INITIAL_QUIESCENCE": verify_initial_quiescence,
    "BOOT_INITIALIZATION": verify_boot_initialization,
    "SEQUENCE_ALLOCATION": verify_sequence_allocation,
    "PHASE_DAG": verify_phase_dag,
    "BATCH_CLOSURE": verify_batch_closure,
    "DEADLINE_BOUNDARY_ORDER": verify_deadline_boundary_order,
    "CONTROLLER_INVISIBILITY": verify_controller_invisibility,
    "CONTROLLER_POSTCLOSURE": verify_controller_postclosure,
    "TIME_PROGRESS": verify_time_progress,
    "WINDOW_MODE_NORMALIZATION": verify_window_mode_normalization,
    "FEATURE_SCHEMA_CONSISTENCY": verify_feature_schema_consistency,
    # policy/invariant/reference algorithms are handled by their dedicated
    # verifier paths; they remain explicit here rather than gaining a passthrough.
    "TREE_WELLFORMEDNESS": _verify_tree,
    "LEAF_GUARD_PARTITION": _verify_leaf_guard_partition,
    "FEATURE_QUANTIZATION": _verify_quantization,
    "ACTION_TRANSITION": _verify_action_transition,
    "MASK_FALLBACK": _verify_mask_fallback,
    "SELECTED_ACTION_REGIONS": _verify_selected_action_regions,
    "EXECUTABLE_POLICY_SEMANTICS": _verify_executable_policy_semantics,
    "CANDIDATE_ENVELOPE": _verify_candidate_envelope,
    "BUDGET_DOMAIN": _verify_budget_domain,
    "LO_BUDGET_UPPER_INVARIANT": lambda **kwargs: _verify_budget_invariant("LO_BUDGET_UPPER_INVARIANT", **kwargs),
    "HI_BUDGET_LOWER_INVARIANT": lambda **kwargs: _verify_budget_invariant("HI_BUDGET_LOWER_INVARIANT", **kwargs),
    "ACTIVE_RELEASE_BUDGET_INVARIANT": lambda **kwargs: _verify_budget_invariant("ACTIVE_RELEASE_BUDGET_INVARIANT", **kwargs),
    "COMMON_TRANSITION_PRESERVATION": _verify_common_transition_preservation,
    "DEPLOYED_POLICY_PRESERVATION": _verify_deployed_policy_preservation,
    "CERTIFIED_ENVELOPE": _verify_certified_envelope,
    "CODE_REFERENCE_UPPER_BOUND_MAPPING": verify_code_reference_upper_bound_mapping,
    "REFERENCE_TASKSET": _verify_reference_taskset_obligation,
    "DISCRETE_TICK_EMBEDDING": verify_discrete_tick_embedding,
    "RELEASE_COUNT": verify_release_count,
    "DEMAND_DOMINATION": verify_demand_domination,
    "LO_MODE_RTA": verify_lo_mode_rta,
    "WORST_CASE_START_TIME": verify_worst_case_start_time,
    "CASE1_INTEGER_DOMAIN": verify_case1_integer_domain,
    "CASE2_INTEGER_DOMAIN": verify_case2_integer_domain,
    "ZERO_RELATIVE_START": verify_zero_relative_start,
    "INHERITED_HI_DOMINATION": verify_inherited_hi_domination,
    "ALL_TASK_REFERENCE_RTA_ARITHMETIC": _verify_all_task_reference_rta_obligation,
    "REFERENCE_SEMANTICS_CONTRACT": _verify_reference_semantics_contract,
    "REFERENCE_TRANSITION_SYSTEM_IDENTITY": _candidate_status_checker("REFERENCE_TRANSITION_SYSTEM_IDENTITY"),
    "REFERENCE_MODEL_CONFORMANCE": lambda **kwargs: _verify_reference_model_conformance(**kwargs),
    "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION": lambda **kwargs: verify_budget_to_reference_domination(**kwargs),
    "REFERENCE_TASKSET_SCHEDULABLE": lambda **kwargs: verify_reference_taskset_schedulable(**kwargs),
    "REFERENCE_HI_SUBSET_SAFETY": lambda **kwargs: verify_reference_hi_subset_safety(**kwargs),
    "FINITE_BAD_PREFIX_CONTRADICTION": lambda **kwargs: verify_finite_bad_prefix_contradiction(**kwargs),
    "FINAL_CLAIM_COMPOSITION": _verify_final_claim_composition,
    "ARTIFACT_MANIFEST": lambda **kwargs: _raw_inputs_result("ARTIFACT_MANIFEST", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "COMPONENT_CONTEXT_INTEGRITY": lambda **kwargs: _raw_inputs_result("COMPONENT_CONTEXT_INTEGRITY", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "DIRECT_PREDECESSOR_HASHES": lambda **kwargs: _raw_inputs_result("DIRECT_PREDECESSOR_HASHES", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "STATUS_EVIDENCE": lambda **kwargs: _raw_inputs_result("STATUS_EVIDENCE", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "OUTER_BUNDLE_ROOT": lambda **kwargs: _raw_inputs_result("OUTER_BUNDLE_ROOT", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "INDEPENDENT_BUNDLE_VERIFICATION": lambda **kwargs: _raw_inputs_result("INDEPENDENT_BUNDLE_VERIFICATION", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "CLAIM_AGGREGATION_RESULT": lambda **kwargs: _raw_inputs_result("CLAIM_AGGREGATION_RESULT", code="STRUCTURAL_CHECK_ROUTED_ELSEWHERE"),
    "PROTECTED_HI_RTA_ARITHMETIC": _candidate_status_checker("PROTECTED_HI_RTA_ARITHMETIC"),
    "EARLY_STOP_CONFIGURATION_GATE": lambda **kwargs: verify_early_stop_configuration_gate(**kwargs),
    "EFFECTIVE_EVENT_FRONTIER_RELATION": lambda **kwargs: verify_effective_event_frontier_relation(**kwargs),
    "CONTROLLER_WRITE_SET": lambda **kwargs: verify_controller_write_set(**kwargs),
    "CONTROLLER_BOUNDARY": lambda **kwargs: verify_controller_boundary(**kwargs),
    "CONTROLLER_PATH_UNIQUENESS": lambda **kwargs: verify_controller_path_uniqueness(**kwargs),
    "UPDATE_PAYLOAD_TOTALITY": lambda **kwargs: verify_update_payload_totality(**kwargs),
    "TOKEN_REFRESH_PROJECTION": lambda **kwargs: verify_token_refresh_projection(**kwargs),
    "PER_HI_TASK_INDUCTIVE_WCRT": verify_per_hi_task_inductive_wcrt,
    "PROTECTED_HI_SAFETY_COROLLARY": verify_protected_hi_safety_corollary,
    "RELEASE_FIXED_REMOVAL_MAPPING": verify_release_fixed_removal_mapping,
    "CLOSED_PREFIX_REFINEMENT": verify_closed_prefix_refinement,
    "REFERENCE_PREFIX_EXTENSION": lambda **kwargs: verify_reference_prefix_extension(**kwargs),
    "HI_BAD_CLOSED_PREFIX_REFLECTION": lambda **kwargs: verify_hi_bad_closed_prefix_reflection(**kwargs),
}


def checker_for(obligation_id: str, *, route_strategy: Any | None = None) -> Checker | None:
    route_catalog = route_strategy.checker_catalog() if route_strategy is not None else {}
    collisions = set(VERIFIER_CHECKERS).intersection(route_catalog)
    if collisions:
        raise ValueError(f"ROUTE_CHECKER_ID_COLLISION:{sorted(collisions)}")
    checker = VERIFIER_CHECKERS.get(obligation_id)
    if checker is not None:
        return checker
    return route_catalog.get(obligation_id)
