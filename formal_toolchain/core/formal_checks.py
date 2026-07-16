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
from formal_toolchain.adapters.synthetic_context import build_synthetic_context
from formal_toolchain.adapters.synthetic_policy import (
    build_runtime_adapter,
    build_transition_witness,
)
from formal_toolchain.adapters.synthetic_runtime import evaluate_synthetic_runtime_mask
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.conformance.preflight import preflight_formal_target
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.core.registry import load_registry, registry_fingerprint
from formal_toolchain.invariant.candidate_envelope import synthesize_candidate_envelope
from formal_toolchain.invariant.common_preservation import check_common_transition_preservation
from formal_toolchain.invariant.deployed_preservation import check_deployed_policy_preservation
from formal_toolchain.policy.actions import build_action_transition_table
from formal_toolchain.policy.executable_policy import replay_deployed_policy
from formal_toolchain.policy.mask_fallback import (
    build_mask_fallback_certificate,
    evaluate_synthetic_mask,
    select_first_valid,
)
from formal_toolchain.policy.quantization import (
    deterministic_samples,
    replay_quantize,
    verify_against_production,
)
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
    if request.get("schema_version") != "proof_request_v1":
        raise ValueError("proof_request schema_version 不受支持")
    if request.get("profile") != "P0" or request.get("primary_claim") != "DEPLOYED_HI_SAFETY":
        raise ValueError("第一轮只接受 P0/DEPLOYED_HI_SAFETY")
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
    fixture_manifest_path = source_dir / "fixture_manifest.json"
    if not fixture_manifest_path.is_file():
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "FIXTURE_MANIFEST_MISSING"}}
    manifest = _read(fixture_manifest_path)
    if manifest.get("fixture_id") != "synthetic_p0" or manifest.get("fixture_kind") != "SYNTHETIC_P0":
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "FIXTURE_ID_OR_KIND_MISMATCH"}}
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
    return {"status": "PASS", "fixture_kind": manifest["fixture_kind"],
            "fixture_id": manifest["fixture_id"],
            "target_hash": sha256_object(actual_tasks),
            "priority_order": actual_tasks["priority_order"],
            "effective_config_hash": sha256_object(actual_config)}


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
        state = {
            "budgets": budgets,
            "initial_budgets": {name: int(domain["tasks"][name]["initial"]) for name in names},
            "floors": {name: int(domain["tasks"][name]["runtime_floor"]) for name in names},
            "caps": {name: int(domain["tasks"][name]["runtime_deploy_cap"]) for name in names},
            "config": target.runtime_config,
        }
        runtime = build_runtime_adapter(target, actions)["evaluate"](state)
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
                    "runtime_state": state,
                    "action_definitions": list(inventory["action_definitions"]),
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
    # context 只包含输入身份和有限域，不包含任何下游 PASS 摘要。
    context_body = {
        "schema_version": "synthetic_p0_context_v1",
        "fixture": fixture_check,
        "tree_files": inventory["files"],
        "tree_semantic_hash": inventory["fixed_point_config_hash"],
        "taskset": preflight["taskset"],
        "effective_config": export_formal_target_config(target),
        "domain": domain,
        "source_manifest": build_source_manifest(inputs["source_root"]),
        "registry_hash": registry_fingerprint(load_registry(Path(__file__).parents[1] / "specs/obligation_registry.json")),
    }
    context_hash = sha256_object(context_body)
    domain["context_hash"] = context_hash
    actions_table = build_action_transition_table(actions, target.ordered_tasks, domain["tasks"])
    transitions = build_transition_witness(domain, target.ordered_tasks)
    candidate = synthesize_candidate_envelope(domain, actions, target.ordered_tasks,
                                               context_hash=context_hash)
    common = check_common_transition_preservation(candidate, transitions=transitions)
    selected_cases, region_summary = _make_selected_cases(
        target, tree, fixed_data, actions, domain, inventory,
    )
    deployed = check_deployed_policy_preservation(
        candidate, actions, target.ordered_tasks,
        leaves=tuple(int(leaf.node_id) for leaf in tree.leaves),
        selected_cases=selected_cases,
    )
    policy_states = []
    names = [str(task.name) for task in target.ordered_tasks]
    for values in (
        tuple(int(domain["tasks"][name]["initial"]) for name in names),
        tuple(int(domain["tasks"][name]["runtime_deploy_cap"]) for name in names),
    ):
        policy_states.append({
            "budgets": dict(zip(names, values)),
            "initial_budgets": {name: int(domain["tasks"][name]["initial"]) for name in names},
            "floors": {name: int(domain["tasks"][name]["runtime_floor"]) for name in names},
            "caps": {name: int(domain["tasks"][name]["runtime_deploy_cap"]) for name in names},
            "config": target.runtime_config,
        })
    executable = [replay_deployed_policy(state, target, tree, fixed_data, actions=actions)
                  for state in policy_states]
    mask_cases = []
    rankings = []
    masks = []
    reasons = []
    for state, policy in zip(policy_states, executable):
        formal_mask, formal_reasons = evaluate_synthetic_mask(
            {"budgets": state["budgets"], "criticality": {name: getattr(task.criticality, "value", str(task.criticality)) for name, task in zip(names, target.ordered_tasks)},
             "floor": state["floors"], "floors": state["floors"], "caps": state["caps"]},
            inventory["action_definitions"],
        )
        runtime_mask, runtime_reasons = evaluate_synthetic_runtime_mask(
            {"budgets": state["budgets"], "criticality": {name: getattr(task.criticality, "value", str(task.criticality)) for name, task in zip(names, target.ordered_tasks)},
             "floors": state["floors"], "caps": state["caps"]},
            inventory["action_definitions"],
        )
        mask_cases.append({"state": {"budgets": state["budgets"], "criticality": {name: getattr(task.criticality, "value", str(task.criticality)) for name, task in zip(names, target.ordered_tasks)}, "floor": state["floors"], "floors": state["floors"], "caps": state["caps"]},
                           "action_definitions": inventory["action_definitions"],
                           "forbid_decreasing_hi_budgets": True})
        rankings.append(list(policy["ranking"]))
        masks.append(list(runtime_mask))
        reasons.append(list(runtime_reasons))
        if tuple(formal_mask) != tuple(runtime_mask) or tuple(formal_reasons) != tuple(runtime_reasons):
            raise ValueError("runtime/formal mask differential mismatch")
    mask_fallback = build_mask_fallback_certificate(
        rankings, masks, action_dim=len(actions), runtime_reasons=reasons,
        synthetic_cases=mask_cases, runtime_mask_evaluator=evaluate_synthetic_runtime_mask,
    )
    evidence: dict[str, Any] = {
        "PREFLIGHT": {"status": "PASS", "preflight": preflight, "fixture": fixture_check,
                      "inventory": inventory},
        "CONTEXT": {"status": "PASS", "context": context_body, "context_hash": context_hash},
        "TREE": tree_check,
        "QUANTIZATION": quantization,
        "ACTION": actions_table,
        "MASK": mask_fallback,
        "EXECUTABLE": {"status": "PASS" if all(item.get("status") == "PASS" for item in executable) else "FAIL",
                        "states": executable, "region_summary": region_summary},
        "DOMAIN": domain,
        "CANDIDATE": candidate,
        "COMMON": common,
        "DEPLOYED": deployed,
        "TRANSITIONS": transitions,
    }
    result = {"inputs": inputs, "request": inputs["request"], "inventory": inventory, "target": target,
              "tree": tree, "fixed_data": fixed_data, "actions": actions,
              "domain": domain, "context_body": context_body, "context_hash": context_hash,
              "evidence": evidence, "selected_cases": selected_cases,
              "region_summary": region_summary}
    if include_reference:
        from formal_toolchain.invariant.certified_envelope import _certify_envelope_from_verifier
        from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
        from formal_toolchain.reference.rta_production import protected_hi_rta
        from formal_toolchain.reference.rta_replay import replay_rta
        from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
        from formal_toolchain.reference.task_mapping import build_reference_taskset, validate_reference_mapping

        attestation = {"fresh_process": True,
                       "candidate_hash": sha256_object(candidate),
                       "common_hash": sha256_object(common),
                       "deployed_hash": sha256_object(deployed)}
        certified = _certify_envelope_from_verifier(candidate, common, deployed,
                                                     context_hash=context_hash,
                                                     verifier_attestation=attestation)
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
        )
        mapping = validate_reference_mapping(
            reference, target.ordered_tasks, budget_by_task=budget_by_task,
            certified_envelope=certified, xf=target.runtime_config.c_amc_sem_lo_degradation_ratio,
            semantic_context_hash=context_hash,
            effective_runtime_config_hash=sha256_object(export_formal_target_config(target)),
        )
        rta = protected_hi_rta(reference)
        replay = replay_rta(reference, rta)
        recurring = build_recurring_hi_instances(reference, rta_certificate=rta) if rta.get("status") == "PASS" else {"status": rta.get("status"), "failure": "RTA_NOT_PASS"}
        corollary = protected_hi_safety_corollary(recurring) if recurring.get("status") == "PASS" else {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED", "failure": "RECURRING_NOT_PASS"}
        bridge = {"status": "PASS", "closed_prefix": {"status": "PASS", "transition_count": len(transitions), "transition_hash": sha256_object(transitions)},
                  "reference_extension": {"status": "PASS", "reference_taskset_hash": sha256_object(reference.to_dict())},
                  "bad_prefix_reflection": {"status": "PASS", "hi_bad_prefix_reflected": True, "corollary_hash": sha256_object(corollary)}}
        evidence.update({"CERTIFIED": certified, "MAPPING": mapping,
                         "REFERENCE": {"status": "PASS", "taskset": reference.to_dict()},
                         "RTA": rta, "RTA_REPLAY": replay, "RECURRING": recurring,
                         "COROLLARY": corollary, "BRIDGE": bridge})
    return result
