"""fresh verifier 从原始 request 重建的输入边界。

这里不读取 compiler 的 candidate_inputs.json 作为事实，也不调用
formal_checks。candidate 只作为待验证对象传入后续 checker。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.adapters.target_factory import FormalTarget, build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.conformance.preflight import preflight_formal_target
from formal_toolchain.conformance.time_domain import build_budget_domain
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.core.contexts import (
    build_bootstrap_context, build_bridge_context, build_bundle_context,
    build_implementation_context, build_invariant_context, build_policy_context,
    build_reference_context_layer, build_semantic_context,
)
from formal_toolchain.core.registry import load_registry, registry_fingerprint
from formal_toolchain.core.hashing import sha256_file, sha256_object


def _proof_safe(value: Any) -> Any:
    """把真实 provider action ratio 转成 canonical decimal string。

    verifier 不接受二进制 float 进入 context/hash；这里仅做确定性编码，
    不改变 inventory 中原始 action 定义的语义值。
    """

    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, Mapping):
        return {str(key): _proof_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_proof_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class VerifierInputs:
    request: Mapping[str, Any]
    workspace: Path
    artifact_dir: Path
    source_root: Path
    target: FormalTarget
    inventory: Mapping[str, Any]
    preflight: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    contexts: Mapping[str, Mapping[str, Any]]


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _workspace_for_request(request_path: Path) -> Path:
    request_path = Path(request_path).resolve()
    if request_path.parent.name != "request":
        raise ValueError("proof_request.json 必须位于 workspace/request 目录")
    return request_path.parent.parent


def load_verifier_inputs(request_path: Path, *, source_root: Path) -> VerifierInputs:
    """仅从 request、tree artifact 和 code root 现场重建 verifier inputs。"""

    from formal_toolchain.adapters.source_manifest import build_source_manifest

    request_path = Path(request_path).resolve()
    request = _read(request_path)
    schema = request.get("schema_version")
    if schema not in {"proof_request_v1", "proof_request_v2"}:
        raise ValueError("proof_request schema_version 不受支持")
    if request.get("profile") != "P0" or request.get("primary_claim", request.get("claim")) != "DEPLOYED_HI_SAFETY":
        raise ValueError("第一轮只接受 P0/DEPLOYED_HI_SAFETY")
    workspace = _workspace_for_request(request_path)
    relative_artifact = str(request["tree_artifact_dir"])
    artifact_dir = (workspace / relative_artifact).resolve()
    if workspace not in artifact_dir.parents:
        raise ValueError("tree artifact 越出 proof workspace")
    recipe = request.get("target_recipe")
    if not isinstance(recipe, Mapping) or not isinstance(recipe.get("factory"), str):
        raise ValueError("request 缺少唯一 target_recipe.factory")
    target = build_target(str(recipe["factory"]), dict(recipe.get("kwargs", {})))
    inventory = inspect_tree_artifact(
        artifact_dir,
        expected_state_dim=len(target.feature_names),
        expected_action_dim=len(target.action_definitions),
        expected_seed=request.get("taskset_seed") if request.get("target_kind") == "REAL_VIPER_SEED" else None,
    )
    preflight = preflight_formal_target(target, artifact_dir)
    source_root = Path(source_root).resolve()
    source_manifest = build_source_manifest(source_root)
    expected_tree_hash = request.get("expected_tree_file_sha256")
    actual_tree_hash = sha256_file(artifact_dir / "integer_tree.json")
    if expected_tree_hash is not None and expected_tree_hash != actual_tree_hash:
        raise ValueError("TREE_REQUEST_HASH_MISMATCH")
    domain = build_budget_domain(target.ordered_tasks, target.provenance.get("budget_by_task"),
                                runtime_config=target.runtime_config)
    config_hash = sha256_object(export_formal_target_config(target))
    registry_hash = registry_fingerprint(load_registry(Path(__file__).parents[1] / "specs/obligation_registry.json"))
    bootstrap = build_bootstrap_context(registry_hash=registry_hash,
                                        profile="P0", claim="DEPLOYED_HI_SAFETY")
    implementation = build_implementation_context(
        bootstrap_context_hash=bootstrap["hash"],
        source_manifest_hash=source_manifest["semantic_hash"], runtime_config_hash=config_hash)
    semantic = build_semantic_context(
        implementation_context_hash=implementation["hash"],
        taskset_fingerprint=sha256_object(preflight["taskset"]),
        effective_runtime_config_hash=config_hash, budget_domain_hash=sha256_object(domain))
    policy = build_policy_context(
        semantic_context_hash=semantic["hash"], tree_inventory_hash=sha256_object(inventory["files"]),
        fixed_point_config_hash=inventory["fixed_point_config_hash"],
        feature_schema_hash=sha256_object(_proof_safe(inventory["feature_names"])),
        action_schema_hash=sha256_object(_proof_safe(inventory["action_definitions"])))
    invariant = build_invariant_context(
        policy_context_hash=policy["hash"], domain_hash=sha256_object(domain),
        action_transition_input_hash=sha256_object({"actions": _proof_safe(inventory["action_definitions"])}))
    reference = build_reference_context_layer(
        invariant_context_hash=invariant["hash"], reference_input_mode="FROZEN_FORMAL_INPUTS")
    bridge = build_bridge_context(reference_context_hash=reference["hash"],
                                  source_manifest_hash=source_manifest["semantic_hash"])
    bundle = build_bundle_context(bridge_context_hash=bridge["hash"],
                                  target_id=request.get("target_id"), claim="DEPLOYED_HI_SAFETY")
    contexts = {"bootstrap_context": bootstrap, "implementation_context": implementation,
                "semantic_context": semantic, "policy_context": policy,
                "invariant_context": invariant, "reference_context": reference,
                "bridge_context": bridge, "bundle_context": bundle}
    return VerifierInputs(
        request=request, workspace=workspace, artifact_dir=artifact_dir,
        source_root=source_root, target=target, inventory=inventory,
        preflight=preflight, source_manifest=source_manifest, contexts=contexts,
    )


def candidate_evidence(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """取出 candidate 的唯一 evidence witness；缺失时返回 None。"""

    witness = candidate.get("witness")
    if not isinstance(witness, Mapping):
        return None
    evidence = witness.get("evidence")
    return evidence if isinstance(evidence, Mapping) else None
