"""Phase L/M 共同消费的原始输入重放和证明证据计算。

这里没有任何 ``PASS`` 常量捷径：每一项结果都由当前 seed artifact、target
factory、预算有限域以及现有独立 checker 的实际返回值决定。compiler 可以
把这些结果写成 candidate，verifier 则在新进程中再次调用本模块并重新生成
证书；两者之间不通过“上一次运行成功”的文本传递信任。
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from amc_py.models import Criticality
from amc_py.rl.actions import build_budget_action_space
from amc_py.viper.fixed_point import fixed_point_config_from_dict, fixed_point_config_hash, quantize_value
from amc_py.viper.integer_tree import evaluate_integer_tree

from formal_toolchain.adapters.amc_taskset import export_taskset
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.adapters.seed_directory import ALLOWED_VARIANTS
from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.adapters.synthetic_policy import build_transition_witness
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.conformance.preflight import preflight_formal_target
from formal_toolchain.conformance.runtime_evidence import build_p0_runtime_evidence
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.core.contexts import (
    build_bootstrap_context, build_bridge_context, build_bundle_context,
    build_composition_context, build_implementation_context,
    build_invariant_context, build_policy_context,
    build_reference_context_layer, build_semantic_context,
)
from formal_toolchain.core.registry import load_registry, registry_fingerprint
from formal_toolchain.compiler.dag_runner import topological_order
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.policy.actions import build_action_transition_table
from formal_toolchain.policy.executable_policy import replay_deployed_policy
from formal_toolchain.policy.mask_fallback import (
    build_mask_fallback_certificate,
    build_parametric_mask_fallback_certificate,
    select_first_valid,
)
from formal_toolchain.policy.quantization import (
    deterministic_samples,
    replay_quantize,
    verify_against_production,
)
from formal_toolchain.policy.selected_regions import selected_action_regions_v2
from formal_toolchain.policy.tree import validate_tree_and_leaf_partition
from formal_toolchain.policy.tree_io import integer_tree_from_dict


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def proof_safe(value: Any) -> Any:
    """把运行时诊断中的 float 转为稳定十进制字符串后再进入证书 hash。"""

    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, Mapping):
        return {str(key): proof_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [proof_safe(item) for item in value]
    return value


def workspace_for_request(request_path: Path) -> Path:
    """请求文件固定位于 ``<workspace>/request``，拒绝从任意路径猜工作区。"""

    request_path = Path(request_path).resolve()
    if request_path.parent.name != "request":
        raise ValueError("proof_request.json 必须位于 workspace/request 目录")
    return request_path.parent.parent


def load_request_inputs(request_path: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    """加载请求、artifact 和 target；不读取 HOUT 或外部 seed 路径。"""

    request_path = Path(request_path)
    request = _read(request_path)
    if request.get("schema_version") != "proof_request_v2":
        raise ValueError("proof_request schema_version 不受支持")
    if request.get("profile") != "P0" or request.get("primary_claim") != "DEPLOYED_HI_SAFETY":
        raise ValueError("第一轮只接受 P0/DEPLOYED_HI_SAFETY")
    if request.get("target_kind") is None:
        raise ValueError("TARGET_KIND_MISSING")
    workspace = workspace_for_request(request_path)
    artifact_dir = workspace / str(request["tree_artifact_dir"])
    if workspace not in artifact_dir.resolve().parents and artifact_dir.resolve() != workspace:
        raise ValueError("tree artifact 越出 proof workspace")
    recipe = request.get("target_recipe")
    if not isinstance(recipe, Mapping) or not isinstance(recipe.get("factory"), str):
        raise ValueError("request 缺少唯一 target_recipe.factory")
    target = build_target(str(recipe["factory"]), dict(recipe.get("kwargs", {})))
    return {
        "request": request,
        "workspace": workspace,
        "artifact_dir": artifact_dir,
        "target": target,
        "source_root": Path(source_root or Path.cwd()).resolve(),
    }


def _canonical_fixture_checks(inputs: Mapping[str, Any], inventory: Mapping[str, Any],
                              preflight: Mapping[str, Any]) -> dict[str, Any]:
    """核对 fixture manifest 中的 canonical target 身份，禁止自动重排。"""

    workspace = Path(inputs["workspace"])
    source_dir = workspace / "request" / "inputs"
    fixture_manifest_path = source_dir / "formal_target_manifest.json"
    if not fixture_manifest_path.is_file():
        fixture_manifest_path = source_dir / "fixture_manifest.json"
    if not fixture_manifest_path.is_file():
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "TARGET_MANIFEST_MISSING"}}
    manifest = _read(fixture_manifest_path)
    if manifest.get("schema_version") == "formal_target_manifest_v1":
        target_id = manifest.get("target_id")
        target_kind = manifest.get("target_kind")
    else:
        target_id = manifest.get("fixture_id")
        target_kind = manifest.get("fixture_kind")
    if target_kind not in {"SYNTHETIC_P0", "REAL_VIPER_SEED"}:
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "TARGET_KIND_UNSUPPORTED"}}
    canonical_task_path = source_dir / "formal_inputs" / "code_taskset_canonical.json"
    priority_path = source_dir / "formal_inputs" / "priority_order.json"
    config_path = source_dir / "formal_inputs" / "effective_runtime_config.json"
    if any(not path.is_file() for path in (canonical_task_path, priority_path, config_path)):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "CANONICAL_TARGET_INPUT_MISSING"}}
    expected_tasks = _read(canonical_task_path)
    expected_priority = _read(priority_path)
    expected_config = _read(config_path)
    target = inputs["target"]
    target_kind = inputs["request"].get("target_kind")
    if target_kind is None:
        raise ValueError("TARGET_KIND_MISSING")
    target_kind = str(target_kind)
    if target_kind == "REAL_VIPER_SEED":
        # 真实 seed 必须提供自身的 runtime adapter；这里拒绝把 synthetic
        # adapter 偷换成 real target 的证明语义。
        if target.runtime_adapter is None or target.provenance.get("adapter_kind") == "SYNTHETIC_P0":
            raise ValueError("REAL_TARGET_SYNTHETIC_ADAPTER_FORBIDDEN")
    actual_tasks = export_taskset(target.ordered_tasks, target.provenance.get("budget_by_task"))
    actual_config = export_formal_target_config(target)
    if expected_tasks != actual_tasks:
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "CANONICAL_TASKSET_MISMATCH"},
                "expected": expected_tasks, "actual": actual_tasks}
    if expected_priority != {"schema_version": "priority_order_v1",
                             "priority_order": actual_tasks["priority_order"]}:
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "PRIORITY_ORDER_MISMATCH"}}
    if expected_config != actual_config:
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "EFFECTIVE_RUNTIME_CONFIG_MISMATCH"}}
    return {"status": "PASS", "target_kind": target_kind,
            "target_id": target_id,
            "target_hash": sha256_object(actual_tasks),
            "priority_order": actual_tasks["priority_order"],
            "effective_config_hash": sha256_object(actual_config)}


def _leaf_guards(tree: Any) -> dict[int, list[dict[str, Any]]]:
    leaves = {leaf.node_id for leaf in tree.leaves}
    nodes = {node.node_id: node for node in tree.nodes}
    guards: dict[int, list[dict[str, Any]]] = {}

    def walk(node_id: int, path: list[dict[str, Any]]) -> None:
        if node_id in leaves:
            guards[node_id] = list(path)
            return
        node = nodes[node_id]
        walk(node.left_child, path + [{"feature_index": int(node.feature_index), "operator": "<=", "threshold": int(node.threshold_int)}])
        walk(node.right_child, path + [{"feature_index": int(node.feature_index), "operator": ">", "threshold": int(node.threshold_int)}])

    walk(tree.root_node_id, [])
    return guards


def _make_selected_cases(target: Any, tree: Any, fixed_data: Mapping[str, Any],
                         actions: tuple[Any, ...], domain: Mapping[str, Any],
                         inventory: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """枚举有限 budget domain，并记录每个 leaf/rank 的实际 mask/fallback。"""

    names = [str(task.name) for task in target.ordered_tasks]
    domains = [tuple(int(value) for value in domain["tasks"][name]["finite_integer_domain"])
               for name in names]
    rankings = {int(leaf.node_id): tuple(int(action_id) for action_id in leaf.action_ranking)
                for leaf in tree.leaves}
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    fallback = False
    noop = False
    for values in product(*domains):
        budgets = dict(zip(names, values))
        adapter = target.runtime_adapter
        if adapter is None:
            raise ValueError("FORMAL_RUNTIME_ADAPTER_MISSING")
        runtime_state = adapter.build_runtime_state_from_budget_vector(budgets)
        observation = adapter.extract_observation(runtime_state)
        runtime_mask, runtime_reasons = adapter.valid_action_mask(runtime_state)
        runtime = {"observation": tuple(observation), "mask": tuple(runtime_mask), "reasons": tuple(runtime_reasons)}
        quantized = tuple(replay_quantize(value, fixed_data)[0]
                          for value in runtime["observation"])
        leaf_id = int(evaluate_integer_tree(tree, quantized).leaf_id)
        # 对每个 leaf 都保留一条 region witness。这样 verifier 可以检查
        # artifact 的所有 ranking，而不是只检查 runtime 当前恰好选到的 leaf。
        for leaf_id_for_region, ranking in rankings.items():
            first = select_first_valid(ranking, runtime["mask"], action_dim=len(actions))
            if first is not None:
                selected_ids.add(first)
                if ranking.index(first) > 0:
                    fallback = True
            else:
                noop = True
            for rank_position, action in enumerate(actions):
                selected.append({
                    "leaf_id": leaf_id_for_region,
                    "rank_position": rank_position,
                    "action_id": int(action.action_id),
                    "valid": first == action.action_id,
                    "mask": list(runtime["mask"]),
                    "mask_reasons": list(runtime["reasons"]),
                    "ranking": list(ranking),
                    "runtime_state": runtime_state,
                    "action_definitions": proof_safe(list(inventory["action_definitions"])),
                    "actual_tree_leaf_id": leaf_id,
                })
    return selected, {"fallback_success": fallback, "implicit_noop": noop,
                      "selected_action_ids": sorted(selected_ids),
                      "state_count": len(list(product(*domains)))}


def calculate_raw_evidence(request_path: Path, *, source_root: Path | None = None,
                           include_reference: bool = False) -> dict[str, Any]:
    """从原始请求计算 L/M 证据；``include_reference`` 只由 verifier 开启。"""

    inputs = load_request_inputs(request_path, source_root=source_root)
    target = inputs["target"]
    artifact_dir = Path(inputs["artifact_dir"])
    inventory = inspect_tree_artifact(
        artifact_dir,
        expected_state_dim=len(target.feature_names),
        expected_action_dim=len(target.action_definitions),
        expected_seed=None,
    )
    preflight = preflight_formal_target(target, artifact_dir)
    fixture_check = _canonical_fixture_checks(inputs, inventory, preflight)
    if preflight.get("obligation_status") != "PASS":
        raise ValueError(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    if fixture_check.get("status") != "PASS":
        raise ValueError(json.dumps(fixture_check, ensure_ascii=False, sort_keys=True))

    fixed_json = _read(artifact_dir / "fixed_point_config.json")
    fixed_data = fixed_json.get("config", fixed_json)
    fixed_config = fixed_point_config_from_dict(fixed_data)
    if inventory["fixed_point_config_hash"] != fixed_point_config_hash(fixed_config):
        raise ValueError("fixed-point semantic hash 重算不一致")
    tree_data = _read(artifact_dir / "integer_tree.json")
    tree = integer_tree_from_dict(tree_data)
    tree_check = validate_tree_and_leaf_partition(tree, state_min=0, state_max=int(fixed_data["output_max"]))
    quantization = verify_against_production(
        deterministic_samples(), fixed_data,
        lambda value, _config: quantize_value(value, fixed_config),
    )
    actions = build_budget_action_space(
        target.ordered_tasks,
        action_space=str(getattr(target.runtime_config, "action_space")),
        budget_increase_ratio=float(getattr(target.runtime_config, "budget_increase_ratio")),
        budget_decrease_ratio=float(getattr(target.runtime_config, "budget_decrease_ratio")),
    )
    if len(actions) != len(target.action_definitions):
        raise ValueError("target action space 与 artifact action dimension 不一致")
    domain = build_budget_domain(
        target.ordered_tasks, target.provenance.get("budget_by_task"),
        runtime_config=target.runtime_config,
    )
    # 分层 context 只吸收本层及显式上游输入，不把任何下游 PASS 摘要放入
    # preimage。candidate 仍使用 semantic context 作为语义输入边界；
    # verifier 会按相同字段现场重算并把 context hash 纳入 outer root。
    source_manifest = build_source_manifest(inputs["source_root"])
    registry_hash = registry_fingerprint(load_registry(Path(__file__).parents[1] / "specs/obligation_registry.json"))
    bootstrap_context = build_bootstrap_context(registry_hash=registry_hash,
                                                profile="P0", claim="DEPLOYED_HI_SAFETY")
    implementation_context = build_implementation_context(
        bootstrap_context_hash=bootstrap_context["hash"],
        source_manifest_hash=source_manifest["semantic_hash"],
        runtime_config_hash=sha256_object(export_formal_target_config(target)))
    semantic_context = build_semantic_context(
        implementation_context_hash=implementation_context["hash"],
        taskset_fingerprint=sha256_object(preflight["taskset"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(target)),
        budget_domain_hash=sha256_object(domain))
    policy_context = build_policy_context(
        semantic_context_hash=semantic_context["hash"],
        tree_inventory_hash=sha256_object(inventory["files"]),
        fixed_point_config_hash=inventory["fixed_point_config_hash"],
        feature_schema_hash=sha256_object(inventory["feature_names"]),
        action_schema_hash=sha256_object(proof_safe(inventory["action_definitions"])))
    invariant_context = build_invariant_context(
        policy_context_hash=policy_context["hash"],
        domain_hash=sha256_object(domain),
        action_transition_input_hash=sha256_object({"actions": proof_safe(inventory["action_definitions"])}))
    reference_context = build_reference_context_layer(
        invariant_context_hash=invariant_context["hash"],
        reference_input_mode="FROZEN_FORMAL_INPUTS")
    bridge_context = build_bridge_context(
        reference_context_hash=reference_context["hash"],
        source_manifest_hash=source_manifest["semantic_hash"])
    composition_context = build_composition_context(
        bridge_context_hash=bridge_context["hash"],
        mathematical_root_id="FINAL_CLAIM_COMPOSITION",
        claim="DEPLOYED_HI_SAFETY")
    bundle_context = build_bundle_context(
        composition_context_hash=composition_context["hash"],
        target_id=fixture_check.get("target_id"), claim="DEPLOYED_HI_SAFETY")
    contexts = {
        "bootstrap_context": bootstrap_context,
        "implementation_context": implementation_context,
        "semantic_context": semantic_context,
        "policy_context": policy_context,
        "invariant_context": invariant_context,
        "reference_context": reference_context,
        "bridge_context": bridge_context,
        "composition_context": composition_context,
        "bundle_context": bundle_context,
    }
    context_body = {"contexts": contexts, "fixture": fixture_check,
                    "tree_files": inventory["files"], "taskset": preflight["taskset"],
                    "effective_config": export_formal_target_config(target), "domain": domain,
                    "source_manifest": source_manifest, "registry_hash": registry_hash}
    context_hash = invariant_context["hash"]
    domain["context_hash"] = context_hash
    # inventory 需要同时被 inspect_target JSON 输出和后续 builder 消费。
    # 这里保留 integer_tree.json 的原始字典形态，避免把运行时模型对象
    # 直接塞进 preflight 输出导致 fresh 进程无法序列化。
    inventory["tree"] = tree_data
    actions_table = build_action_transition_table(
        actions, target.ordered_tasks, domain["tasks"],
        rounding_mode=str(getattr(target.runtime_config, "budget_rounding_mode", "ceil_floor")),
        min_budget_delta=int(getattr(target.runtime_config, "min_budget_delta", 1)),
    )
    transitions = build_transition_witness(domain, target.ordered_tasks)
    adapter = target.runtime_adapter
    if adapter is None:
        raise ValueError("FORMAL_RUNTIME_ADAPTER_MISSING")
    candidate = synthesize_candidate_envelope(
        domain,
        actions,
        target.ordered_tasks,
        context_hash=context_hash,
        runtime_adapter=adapter,
    )
    common = (
        check_common_transition_preservation(candidate, transitions=transitions)
        if candidate.get("status") == "PASS"
        else {"status": "UNRESOLVED", "route": "UNRESOLVED",
              "failure": {"code": "CANDIDATE_ENVELOPE_NOT_PASS"}}
    )
    if str(getattr(target.runtime_config, "nonvacuity_profile", "off")) == "c3_retroactive_release_budget":
        common = {
            "status": "FAIL",
            "route": "POLICY_CONTRACT_VIOLATION",
            "failure": {
                "code": "ACTIVE_RELEASE_BUDGET_RETROACTIVELY_MUTATED",
                "obligation_id": "ACTIVE_RELEASE_BUDGET_INVARIANT",
            },
            "active_release_budget_immutable": False,
            "controller_budget_write": True,
        }
    rankings = {
        int(leaf.node_id): tuple(int(action_id) for action_id in leaf.action_ranking)
        for leaf in tree.leaves
    }
    mask_contract = adapter.export_mask_contract()
    selection_semantics = str(mask_contract.get("selection", "ranked_first_valid"))
    mask_fallback = build_parametric_mask_fallback_certificate(
        rankings=rankings,
        action_dim=len(actions),
        mask_contract=mask_contract,
    )
    regions = selected_action_regions_v2(
        _leaf_guards(tree), rankings, selection_semantics=selection_semantics
    )

    names = [str(task.name) for task in target.ordered_tasks]
    initial_vector = tuple(int(domain["tasks"][name]["initial"]) for name in names)
    upper_vector = tuple(int(domain["tasks"][name]["action_hard_upper"]) for name in names)
    mid_vector = tuple((int(domain["tasks"][name]["initial"]) + int(domain["tasks"][name]["action_hard_upper"])) // 2 for name in names)
    sample_vectors: list[tuple[int, ...]] = []
    for vector in (initial_vector, mid_vector, upper_vector):
        if vector not in sample_vectors:
            sample_vectors.append(vector)
    policy_states = [
        adapter.build_runtime_state_from_budget_vector(dict(zip(names, values)))
        for values in sample_vectors
    ]
    executable = [replay_deployed_policy(state, target, tree, fixed_data, actions=actions)
                  for state in policy_states]
    regression_samples = {
        "policy_states": policy_states,
        "executable": executable,
        "sample_count": len(policy_states),
    }
    deployed = (
        check_deployed_policy_preservation(
            candidate,
            actions,
            target.ordered_tasks,
            mask_fallback_certificate=mask_fallback,
            action_transition_certificate=actions_table,
            mask_contract=mask_contract,
            forbid_decreasing_hi_budgets=bool(
                getattr(target.runtime_config, "forbid_decreasing_hi_budgets")
            ),
            selection_semantics=selection_semantics,
            disabled_guards=tuple(mask_contract.get("disabled_guards", ())),
        )
        if candidate.get("status") == "PASS" and common.get("status") == "PASS" and mask_fallback.get("status") == "PASS"
        else {"status": "UNRESOLVED", "route": "UNRESOLVED",
              "failure": {"code": "DEPLOYED_POLICY_PRESERVATION_UNRESOLVED"}}
    )
    evidence: dict[str, Any] = {
        "PREFLIGHT": {"status": "PASS", "preflight": preflight, "fixture": fixture_check,
                      "inventory": inventory},
        "CONTEXT": {"status": "PASS", "context": context_body, "context_hash": context_hash},
        "TREE": tree_check,
        "QUANTIZATION": quantization,
        "ACTION": actions_table,
        "MASK": mask_fallback,
        "SELECTED_REGIONS": regions,
        "EXECUTABLE": {"status": "PASS" if executable and all(item.get("status") == "PASS" for item in executable) else "UNRESOLVED",
                        "states": executable},
        "DOMAIN": domain,
        "CANDIDATE": candidate,
        "COMMON": common,
        "DEPLOYED": deployed,
        "TRANSITIONS": transitions,
        "regression_samples": regression_samples,
    }
    evidence["P0_RUNTIME_EVIDENCE"] = build_p0_runtime_evidence(
        target=target,
        source_root=inputs["source_root"],
    ).to_dict()
    # 每个 active semantic obligation 必须经过显式 builder。这里不再把
    # preflight 的总体 PASS 复制成同名义务的 PASS；没有 builder 的义务只
    # 能获得明确的 UNRESOLVED evidence。
    from formal_toolchain.compiler.semantic_evidence_builders import SEMANTIC_EVIDENCE_BUILDERS
    from formal_toolchain.core.obligation_ids import (
        ALL_TASK_REFERENCE_RTA_ARITHMETIC,
        BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION,
        FINITE_BAD_PREFIX_CONTRADICTION,
        FINAL_CLAIM_COMPOSITION,
        REFERENCE_HI_SUBSET_SAFETY,
        REFERENCE_MODEL_CONFORMANCE,
        REFERENCE_SEMANTICS_CONTRACT,
        REFERENCE_TASKSET_SCHEDULABLE,
        REFERENCE_TRANSITION_SYSTEM_IDENTITY,
    )
    registry_entries = load_registry(Path(__file__).parents[1] / "specs/obligation_registry.json")
    structural_ids = {"ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY",
                      "DIRECT_PREDECESSOR_HASHES", "STATUS_EVIDENCE",
                      "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
                      "CLAIM_AGGREGATION_RESULT"}
    bridge_ids = {"RELEASE_FIXED_REMOVAL_MAPPING", "CLOSED_PREFIX_REFINEMENT",
                  "REFERENCE_PREFIX_EXTENSION", "HI_BAD_CLOSED_PREFIX_REFLECTION"}
    reference_placeholder_ids = {
        "CERTIFIED_ENVELOPE",
        "CODE_REFERENCE_UPPER_BOUND_MAPPING",
        "REFERENCE_TASKSET",
        "DISCRETE_TICK_EMBEDDING",
        "RELEASE_COUNT",
        "DEMAND_DOMINATION",
        "LO_MODE_RTA",
        "WORST_CASE_START_TIME",
        "CASE1_INTEGER_DOMAIN",
        "CASE2_INTEGER_DOMAIN",
        "ZERO_RELATIVE_START",
        "INHERITED_HI_DOMINATION",
        ALL_TASK_REFERENCE_RTA_ARITHMETIC,
        REFERENCE_TRANSITION_SYSTEM_IDENTITY,
        REFERENCE_MODEL_CONFORMANCE,
        REFERENCE_SEMANTICS_CONTRACT,
        REFERENCE_TASKSET_SCHEDULABLE,
        REFERENCE_HI_SUBSET_SAFETY,
        FINITE_BAD_PREFIX_CONTRADICTION,
        FINAL_CLAIM_COMPOSITION,
        BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION,
        "RELEASE_FIXED_REMOVAL_MAPPING",
        "CLOSED_PREFIX_REFINEMENT",
        "REFERENCE_PREFIX_EXTENSION",
        "HI_BAD_CLOSED_PREFIX_REFLECTION",
    }
    registry_by_id = {str(entry["id"]): entry for entry in registry_entries}
    for obligation_id in topological_order(registry_entries):
        entry = registry_by_id[obligation_id]
        if (entry.get("activation") != "active" or obligation_id in evidence
                or obligation_id in structural_ids or obligation_id in bridge_ids
                or obligation_id in reference_placeholder_ids):
            continue
        builder = SEMANTIC_EVIDENCE_BUILDERS.get(obligation_id)
        if builder is None:
            from formal_toolchain.compiler.semantic_evidence_builders import CandidateEvidence
            built_evidence = CandidateEvidence(
                obligation_id=obligation_id, status="UNRESOLVED", route="UNRESOLVED",
                code="CANDIDATE_EVIDENCE_BUILDER_NOT_IMPLEMENTED", inputs={}, witness={})
        else:
            built_evidence = builder(
                target=target, source_root=inputs["source_root"], artifact_dir=artifact_dir,
                inventory=inventory, preflight=preflight, fixture_check=fixture_check,
                contexts=contexts, evidence=evidence,
            )
        evidence[obligation_id] = built_evidence.to_dict()
    result = {"inputs": inputs, "request": inputs["request"], "inventory": inventory, "target": target,
              "tree": tree, "fixed_data": fixed_data, "actions": actions,
              "domain": domain, "context_body": context_body, "context_hash": context_hash,
              "contexts": contexts,
              "evidence": evidence, "regression_samples": regression_samples}
    if include_reference:
        from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
        from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
        from formal_toolchain.reference.rta_obligations import decompose_rta_obligations
        from formal_toolchain.reference.task_mapping import build_reference_taskset, validate_reference_mapping
        from formal_toolchain.reference.rta_certificate import build_rta_composite

        # compiler 只能输出 untrusted derivation。这里的 envelope view 仅为
        # 生成 reference candidate 所需，不能作为 CERTIFIED_ENVELOPE 信任根。
        if candidate.get("status") != "PASS":
            # candidate envelope 尚未形成时，reference/RTA 没有合法数值输入。
            # 统一输出显式 UNRESOLVED，禁止从空 upper、provenance 或默认值
            # 拼出一套看似完整的 reference taskset。
            unresolved_reference = {"status": "UNRESOLVED", "route": "UNRESOLVED",
                                    "failure": {"code": "CANDIDATE_ENVELOPE_NOT_PASS"}}
            evidence.update({
                "CERTIFIED": unresolved_reference,
                "MAPPING": unresolved_reference,
                "REFERENCE": unresolved_reference,
                "RTA": unresolved_reference,
                "RTA_REPLAY": unresolved_reference,
                "RTA_COMPOSITE": unresolved_reference,
                "RECURRING": unresolved_reference,
                "COROLLARY": unresolved_reference,
                "RELEASE_FIXED_REMOVAL_MAPPING": unresolved_reference,
                "BRIDGE": unresolved_reference,
            })
            return result
        certified = {"status": "CANDIDATE", "schema_version": "candidate_envelope_view_v1",
                     "trust_level": "CANDIDATE_UNVERIFIED", "not_a_certified_envelope": True,
                     "candidate_envelope_hash": sha256_object(candidate),
                     "common_candidate_hash": sha256_object(common),
                     "deployed_candidate_hash": sha256_object(deployed),
                     "lower": dict(candidate.get("lower", {})), "upper": dict(candidate.get("upper", {})),
                     "active_release_budget_upper": dict(candidate.get("active_release_budget_upper", {}))}
        envelope_hash = sha256_object(certified)
        budget_by_task = {}
        for name, row in target.provenance["budget_by_task"].items():
            budget_by_task[name] = {**row, "b_bar": int(certified["upper"][name]),
                                    "certified_envelope_hash": envelope_hash}
        reference = build_reference_taskset(
            target.ordered_tasks, budget_by_task, xf=target.runtime_config.c_amc_sem_lo_degradation_ratio,
            certified_envelope=certified,
            semantic_context_hash=context_hash,
            effective_runtime_config_hash=sha256_object(export_formal_target_config(target)),
            allow_unverified_candidate=True,
        )
        mapping = validate_reference_mapping(
            reference, target.ordered_tasks, budget_by_task=budget_by_task,
            certified_envelope=certified, xf=target.runtime_config.c_amc_sem_lo_degradation_ratio,
            semantic_context_hash=context_hash,
            effective_runtime_config_hash=sha256_object(export_formal_target_config(target)),
        )
        from formal_toolchain.bridge.job_mapping import build_parameterized_release_mapping_certificate
        release_mapping_certificate = build_parameterized_release_mapping_certificate(
            # release-fixed 公式属于 concrete/reference bridge 层，必须绑定
            # bridge context；不能把较早的 semantic context 当成源码桥接边界。
            source_context_hash=bridge_context["hash"])
        # production 与 replay 必须作为一个复合证据输出；任何一侧缺失都不
        # 能被顶层 obligation 当作“只要 production PASS 就够了”。
        rta_composite = build_rta_composite(reference)
        rta = rta_composite["production"]
        replay = rta_composite["replay"]
        reference_obligations = decompose_rta_obligations(
            rta=rta,
            semantic_evidence=evidence,
        )
        evidence.update(reference_obligations)
        recurring = build_recurring_hi_instances(reference, rta_certificate=rta) if rta.get("status") == "PASS" else {"status": rta.get("status"), "failure": "RTA_NOT_PASS"}
        corollary = protected_hi_safety_corollary(recurring) if recurring.get("status") == "PASS" else {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED", "failure": "RECURRING_NOT_PASS"}
        # bridge 证明对象必须由 Phase I-K generator 生成并由 verifier 独立
        # 检查。此函数只导出 verifier 可消费的原始输入，绝不把 transition
        # 数量、hash 或布尔值伪装成 bridge PASS。
        bridge = {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {
                "code": "BRIDGE_PROOF_OBJECT_REQUIRED",
                "message": (
                    "CLOSED_PREFIX_REFINEMENT / REFERENCE_PREFIX_EXTENSION / "
                    "HI_BAD_CLOSED_PREFIX_REFLECTION 必须由 Phase K proof object 验证"
                ),
            },
        }
        evidence.update({"CERTIFIED": certified, "MAPPING": mapping,
                         "RELEASE_FIXED_REMOVAL_MAPPING": release_mapping_certificate,
                         "REFERENCE": {"status": "PASS", "taskset": reference.to_dict()},
                         "RTA": rta, "RTA_REPLAY": replay,
                         "RTA_COMPOSITE": rta_composite, "RECURRING": recurring,
                         "COROLLARY": corollary, "BRIDGE": bridge})
    return result
