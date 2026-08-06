"""Side-effect-free campaign capability and input audit."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analysis.proof_source_classification import (
    ProofSourceClass,
    classify_proof_source,
    is_blocking_source,
)
from .canonical import file_hash, python_symbol_hash
from .schema import CampaignConfig, MutationClass, MutationManifest
from .mutators.python_binding import bind_symbol
from .coherence import missing_roles


@dataclass(frozen=True)
class PreflightIssue:
    mutation_id: str | None
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "code": self.code,
            "message": self.message,
            "field": self.field,
        }


def audit_campaign(
    config: CampaignConfig,
    *,
    include_disabled: bool = False,
) -> dict[str, Any]:
    """Resolve campaign mutations without creating an experiment workspace.

    Normal execution audits only enabled mutations.  Readiness reviews can set
    ``include_disabled`` to inspect the default-off campaign before enabling it.
    """

    issues: list[PreflightIssue] = []
    if not config.source_root.is_dir():
        issues.append(
            PreflightIssue(None, "SOURCE_ROOT_MISSING", f"source_root 不存在: {config.source_root}")
        )
    enabled = [
        item for item in config.mutations
        if include_disabled or item.enabled
    ]
    ids = {item.mutation_id for item in config.mutations}
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for manifest in enabled:
        mutation_issues = _audit_mutation(manifest, config.source_root)
        if manifest.paired_with:
            if manifest.paired_with not in ids:
                mutation_issues.append(
                    PreflightIssue(
                        manifest.mutation_id,
                        "PAIR_TARGET_MISSING",
                        f"paired_with 不存在: {manifest.paired_with}",
                        "paired_with",
                    )
                )
            elif manifest.paired_with not in seen:
                mutation_issues.append(
                    PreflightIssue(
                        manifest.mutation_id,
                        "PAIR_ORDER_INVALID",
                        "paired_with 必须引用此前已启用的 mutation",
                        "paired_with",
                    )
                )
        issues.extend(mutation_issues)
        resolved.append(
            {
                "mutation_id": manifest.mutation_id,
                "mutation_class": manifest.mutation_class.value,
                "ready": not mutation_issues,
                "issue_codes": [item.code for item in mutation_issues],
            }
        )
        seen.add(manifest.mutation_id)
    return {
        "schema_version": "nonvacuity_preflight_v1",
        "status": "PASS" if not issues else "CAMPAIGN_PREFLIGHT_FAILED",
        "campaign_id": config.campaign_id,
        "audited_mutation_count": len(enabled),
        "enabled_mutation_count": sum(item.enabled for item in config.mutations),
        "include_disabled": include_disabled,
        "capabilities": {
            "jsonschema": find_spec("jsonschema") is not None,
            "z3": find_spec("z3") is not None,
        },
        "resolved_mutations": resolved,
        "issues": [item.to_dict() for item in issues],
    }


def audit_mutation(
    manifest: MutationManifest,
    *,
    source_root: Path,
) -> dict[str, Any]:
    issues = _audit_mutation(manifest, source_root)
    return {
        "schema_version": "nonvacuity_mutation_preflight_v1",
        "status": "PASS" if not issues else "CAMPAIGN_PREFLIGHT_FAILED",
        "mutation_id": manifest.mutation_id,
        "issues": [item.to_dict() for item in issues],
    }


def _audit_mutation(
    manifest: MutationManifest,
    source_root: Path,
) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    mid = manifest.mutation_id
    semantic = manifest.mutation_class not in {
        MutationClass.BASELINE,
        MutationClass.BUNDLE_INTEGRITY,
    }
    if manifest.mutation_class is MutationClass.BUNDLE_INTEGRITY:
        if manifest.reuse_source_bundle is None or not manifest.reuse_source_bundle.is_dir():
            issues.append(
                PreflightIssue(
                    mid,
                    "PROOF_BUNDLE_MISSING",
                    f"reuse_source_bundle 不存在: {manifest.reuse_source_bundle}",
                    "reuse_source_bundle",
                )
            )
    elif (
        not (
            manifest.mutation_class in {MutationClass.ENVELOPE, MutationClass.ENVELOPE_GRADIENT}
            and manifest.metadata.get("dynamic_minimum_slack_selection")
        )
        and (manifest.seed_dir is None or not manifest.seed_dir.is_dir())
    ):
        issues.append(
            PreflightIssue(
                mid,
                "SEED_DIR_MISSING",
                f"seed_dir 不存在: {manifest.seed_dir}",
                "seed_dir",
            )
        )

    parameters = manifest.mutator.get("parameters", {})
    if not isinstance(parameters, Mapping):
        issues.append(PreflightIssue(mid, "PARAMETERS_INVALID", "mutator.parameters 必须为 object"))
        return issues
    placeholders = list(_find_placeholders(parameters))
    for field, value in placeholders:
        issues.append(
            PreflightIssue(mid, "PLACEHOLDER_VALUE", f"仍含占位值: {value!r}", field)
        )

    kind = str(manifest.mutator.get("kind", ""))
    if semantic and not kind:
        issues.append(PreflightIssue(mid, "MUTATOR_KIND_MISSING", "语义 mutation 缺少 kind"))
    if kind in {"dangerous_top1", "tree_ranking"}:
        explicit = parameters.get("leaf_id") is not None and parameters.get("action_id") is not None
        resolver = parameters.get("resolver")
        if not explicit and not parameters.get("leaf_candidates") and not resolver:
            issues.append(
                PreflightIssue(mid, "LEAF_TARGET_UNRESOLVED", "缺少 leaf_id 或 leaf_candidates/resolver")
            )
        if not explicit and not parameters.get("dangerous_actions") and not resolver:
            issues.append(
                PreflightIssue(mid, "ACTION_TARGET_UNRESOLVED", "缺少 action_id 或 dangerous_actions/resolver")
            )
        if explicit and manifest.seed_dir is not None:
            tree_path = manifest.seed_dir / manifest.tree_variant / "integer_tree.json"
            if not tree_path.is_file():
                issues.append(PreflightIssue(mid, "TREE_ARTIFACT_MISSING", str(tree_path), "seed_dir"))
            elif parameters.get("expected_tree_hash") and file_hash(tree_path) != str(parameters["expected_tree_hash"]):
                issues.append(
                    PreflightIssue(mid, "TREE_HASH_MISMATCH", "resolver 绑定的 tree hash 与当前输入不一致", "mutator.parameters.expected_tree_hash")
                )
    if kind in {"python_symbol", "source_overlay"}:
        patches = parameters.get("patches")
        patch_list = patches if isinstance(patches, list) else [parameters]
        if isinstance(patches, list) and not patches:
            issues.append(PreflightIssue(mid, "EMPTY_PATCHES", "source patches 不得为空"))
        source_classes: list[ProofSourceClass] = []
        for index, patch in enumerate(patch_list):
            if not isinstance(patch, Mapping):
                issues.append(PreflightIssue(mid, "PATCH_INVALID", "patch 必须为 object"))
                continue
            issues.extend(_audit_source_patch(mid, source_root, patch, index))
            if patch.get("target_file"):
                source_classes.append(classify_proof_source(str(patch["target_file"])))
        if semantic and source_classes and not any(is_blocking_source(item) for item in source_classes):
            issues.append(
                PreflightIssue(
                    mid,
                    "NONBLOCKING_SEMANTIC_TARGET",
                    "语义负例未命中任何 PPP blocking source",
                    "mutator.parameters",
                )
            )
        for item in source_classes:
            if item is ProofSourceClass.DERIVED_AND_REFRESHED:
                issues.append(
                    PreflightIssue(
                        mid,
                        "DERIVED_TARGET_REFRESHED",
                        "目标会在 freeze 阶段重建，不能作为语义 mutation 源头",
                    )
                )
    if kind == "coherent_source_patch":
        semantic_change_id = parameters.get("semantic_change_id")
        if not isinstance(semantic_change_id, str) or not semantic_change_id:
            issues.append(PreflightIssue(mid, "SEMANTIC_CHANGE_ID_MISSING", "coherent patch 必须声明 semantic_change_id"))
        patches = parameters.get("patches")
        if not isinstance(patches, list) or not patches:
            issues.append(PreflightIssue(mid, "EMPTY_PATCHES", "patches 必须为非空 array"))
        else:
            roles: set[str] = set()
            for index, patch in enumerate(patches):
                if not isinstance(patch, Mapping):
                    issues.append(PreflightIssue(mid, "PATCH_INVALID", "patch 必须为 object"))
                    continue
                role = str(patch.get("role", "")); roles.add(role)
                if role in {"VERIFIER_CHECKER", "AGGREGATOR", "EXPECTED_RESULT_CLASSIFIER"}:
                    issues.append(PreflightIssue(mid, "FORBIDDEN_PATCH_ROLE", f"禁止 patch role: {role}"))
                issues.extend(_audit_coherent_source_patch(mid, source_root, patch, index))
            missing = missing_roles(semantic_change_id, roles)
            if missing:
                issues.append(
                    PreflightIssue(
                        mid,
                        "COHERENT_PATCH_REQUIRED_ROLES_MISSING",
                        f"semantic_change_id={semantic_change_id} 缺少角色: {list(missing)}",
                        "mutator.parameters.patches",
                    )
                )
    if kind in {"json_patch", "action_config", "action_step"}:
        targets = [parameters.get("target_file")]
        targets.extend(
            patch.get("target_file")
            for patch in parameters.get("patches", ())
            if isinstance(patch, Mapping)
        )
        for target in (str(item) for item in targets if item):
            if classify_proof_source(target) is ProofSourceClass.DERIVED_AND_REFRESHED:
                issues.append(
                    PreflightIssue(
                        mid,
                        "DERIVED_TARGET_REFRESHED",
                        f"目标会被 freeze 重建: {target}",
                        "target_file",
                    )
                )
    if kind == "envelope":
        dynamic = bool(manifest.metadata.get("dynamic_minimum_slack_selection"))
        required_fields = ("delta",) if dynamic else ("target_file", "json_pointer", "delta")
        for field in required_fields:
            if parameters.get(field) in (None, ""):
                issues.append(PreflightIssue(mid, "ENVELOPE_TARGET_UNRESOLVED", f"envelope 缺少 {field}", f"mutator.parameters.{field}"))
        if dynamic:
            roots = manifest.metadata.get("bundle_roots")
            if not isinstance(roots, (list, tuple)) or not roots:
                issues.append(PreflightIssue(mid, "D1_BUNDLE_ROOTS_MISSING", "D1 动态选择需要非空 bundle_roots", "metadata.bundle_roots"))
            else:
                for index, root in enumerate(roots):
                    path = Path(str(root))
                    if not path.is_absolute():
                        path = source_root / path
                    if not path.is_dir():
                        issues.append(PreflightIssue(mid, "D1_BUNDLE_ROOT_MISSING", str(path), f"metadata.bundle_roots[{index}]"))
    if kind == "bundle_tamper" and manifest.reuse_source_bundle is not None:
        target = parameters.get("target_file")
        if not isinstance(target, str) or not target:
            issues.append(PreflightIssue(mid, "BUNDLE_TARGET_MISSING", "bundle tamper 缺少 target_file", "mutator.parameters.target_file"))
        else:
            tamper_kind = str(parameters.get("tamper_kind", ""))
            # ``source_file`` tampering runs against an isolated source overlay,
            # not inside the reused request/candidate bundle.  Validate the
            # target against the clean source root; the integrity runner copies
            # it before applying the mutation.
            if tamper_kind == "source_file":
                target_path = source_root / target
            else:
                target_path = manifest.reuse_source_bundle / target
            if tamper_kind != "replace_from" and not target_path.exists():
                issues.append(PreflightIssue(mid, "BUNDLE_TARGET_NOT_FOUND", str(target_path), "mutator.parameters.target_file"))
    mode = str(manifest.activation.get("mode", "")).lower()
    if mode == "symbolic_auto":
        issues.extend(_audit_symbolic_auto(manifest))
    if semantic and "symbolic" in mode and find_spec("z3") is None:
        issues.append(PreflightIssue(mid, "Z3_UNAVAILABLE", "symbolic activation 需要 z3"))
    if "hout" in mode or manifest.metadata.get("hout"):
        hout = manifest.metadata.get("hout")
        if not isinstance(hout, Mapping):
            issues.append(PreflightIssue(mid, "HOUT_CONFIG_MISSING", "HOUT activation 缺少 metadata.hout"))
        else:
            normalized_hout = dict(hout)
            for legacy, phase5 in (
                ("taskset", "taskset_path"),
                ("demand_trace", "demand_trace_path"),
                ("runtime_config", "runtime_config_path"),
            ):
                if normalized_hout.get(legacy) is None and normalized_hout.get(phase5) is not None:
                    normalized_hout[legacy] = normalized_hout[phase5]
            required = {
                "base_command",
                "mutated_command",
                "taskset",
                "scenario_seeds",
                "horizon",
                "controller_release_times",
                "worker_count",
                "runtime_config",
                "random_seed",
            }
            missing = sorted(key for key in required if normalized_hout.get(key) is None)
            if missing:
                issues.append(
                    PreflightIssue(mid, "HOUT_FIELDS_MISSING", f"HOUT 缺少字段: {missing}")
                )
    return issues



_SUPPORTED_SYMBOLIC_AUTO_FORMULAS = {
    "A_MASK_REJECT",
    "B2_NO_FIRST_VALID_DIFFERENCE",
    "B_RAW_TOP1_BREAKS_INVARIANT",
    "B3_ALL_INVALID",
    "B4_GUARD_NECESSITY",
    "C2_ROUNDING_DIFFERENCE",
}


def _audit_symbolic_auto(manifest: MutationManifest) -> list[PreflightIssue]:
    """Validate every binding consumed by ``run_auto_symbolic_activation``.

    The runner indexes these fields directly.  Missing values must therefore
    fail during the read-only preflight rather than after a proof workspace has
    been created.
    """

    mid = manifest.mutation_id
    activation = manifest.activation
    issues: list[PreflightIssue] = []
    binding = activation.get("binding")
    if not isinstance(binding, Mapping):
        issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_BINDING_MISSING", "symbolic_auto 缺少 activation.binding", "activation.binding"))
    else:
        groups = {
            "taskset": ("taskset_path", "taskset"),
            "feature_schema": ("feature_schema_path", "feature_schema"),
            "action_definitions": ("action_definitions_path", "action_definitions"),
        }
        for label, aliases in groups.items():
            if not any(binding.get(key) not in (None, "") for key in aliases):
                issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_BINDING_FIELD_MISSING", f"activation.binding 缺少 {label}", "activation.binding"))
    formula = activation.get("formula_kind")
    if not isinstance(formula, str) or not formula:
        issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_FORMULA_MISSING", "symbolic_auto 缺少 formula_kind", "activation.formula_kind"))
    elif formula not in _SUPPORTED_SYMBOLIC_AUTO_FORMULAS:
        issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_FORMULA_UNSUPPORTED", f"当前 builder 不支持: {formula}", "activation.formula_kind"))
    factory = activation.get("runtime_binding_factory")
    if not isinstance(factory, str) or factory.count(":") != 1 or not all(factory.split(":", 1)):
        issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_RUNTIME_FACTORY_MISSING", "symbolic_auto 缺少 module:object runtime_binding_factory", "activation.runtime_binding_factory"))
    target = manifest.metadata.get("resolved_target")
    required_target = {"tree_path", "tree_sha256", "leaf_id", "action_id"}
    if not isinstance(target, Mapping):
        issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_TARGET_MISSING", "symbolic_auto 缺少 metadata.resolved_target", "metadata.resolved_target"))
    else:
        missing = sorted(key for key in required_target if target.get(key) in (None, ""))
        if missing:
            issues.append(PreflightIssue(mid, "SYMBOLIC_AUTO_TARGET_FIELDS_MISSING", f"resolved_target 缺少字段: {missing}", "metadata.resolved_target"))
    return issues

def _audit_source_patch(
    mutation_id: str,
    source_root: Path,
    patch: Mapping[str, Any],
    index: int,
) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    prefix = f"mutator.parameters.patches[{index}]"
    required = ("target_file", "target_symbol", "before_snippet", "after_snippet")
    missing = [key for key in required if not isinstance(patch.get(key), str) or not patch.get(key)]
    if missing:
        return [
            PreflightIssue(
                mutation_id,
                "SOURCE_PATCH_FIELDS_MISSING",
                f"source patch 缺少字段: {missing}",
                prefix,
            )
        ]
    relative = Path(str(patch["target_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        return [PreflightIssue(mutation_id, "UNSAFE_TARGET_PATH", "target_file 必须是安全相对路径", prefix)]
    path = source_root / relative
    if not path.is_file():
        return [PreflightIssue(mutation_id, "TARGET_FILE_MISSING", f"target_file 不存在: {path}", prefix)]
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        python_symbol_hash(source, str(patch["target_symbol"]))
        expected_count = int(patch.get("occurrence", 1))
        actual_count = source.count(str(patch["before_snippet"]))
        if actual_count != expected_count:
            issues.append(
                PreflightIssue(
                    mutation_id,
                    "SOURCE_SNIPPET_MISMATCH",
                    f"before_snippet 预期 {expected_count} 次，实际 {actual_count} 次",
                    prefix,
                )
            )
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        issues.append(PreflightIssue(mutation_id, "SOURCE_BINDING_INVALID", str(exc), prefix))
    return issues


def _audit_coherent_source_patch(mutation_id: str, source_root: Path, patch: Mapping[str, Any], index: int) -> list[PreflightIssue]:
    prefix = f"mutator.parameters.patches[{index}]"
    required = ("role", "target_file", "target_symbol", "before_ast_hash", "before_snippet", "after_snippet")
    missing = [key for key in required if not isinstance(patch.get(key), str) or not patch.get(key)]
    if missing:
        return [PreflightIssue(mutation_id, "COHERENT_PATCH_FIELDS_MISSING", f"缺少字段: {missing}", prefix)]
    relative = Path(str(patch["target_file"]))
    normalized = relative.as_posix()
    if relative.is_absolute() or ".." in relative.parts or normalized.startswith("formal_toolchain/verifier/"):
        return [PreflightIssue(mutation_id, "VERIFIER_PATCH_FORBIDDEN", normalized, prefix)]
    path = source_root / relative
    if not path.is_file():
        return [PreflightIssue(mutation_id, "TARGET_FILE_MISSING", str(path), prefix)]
    try:
        bound = bind_symbol(path.read_text(encoding="utf-8"), str(patch["target_symbol"]))
        if bound.ast_hash != str(patch["before_ast_hash"]):
            return [PreflightIssue(mutation_id, "BEFORE_AST_HASH_MISMATCH", f"expected={patch['before_ast_hash']} actual={bound.ast_hash}", prefix)]
        if bound.source.count(str(patch["before_snippet"])) != int(patch.get("occurrence", 1)):
            return [PreflightIssue(mutation_id, "SOURCE_SNIPPET_MISMATCH", "snippet 未在绑定 symbol 中唯一命中", prefix)]
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        return [PreflightIssue(mutation_id, "SOURCE_BINDING_INVALID", str(exc), prefix)]
    return []


def _find_placeholders(value: Any, pointer: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _find_placeholders(item, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _find_placeholders(item, f"{pointer}/{index}")
    elif isinstance(value, str):
        upper = value.upper()
        if "REQUIRES_ACTUAL" in upper or "REQUIRED_INPUT_UNAVAILABLE" in upper:
            yield pointer or "/", value


def audit_v2_campaign_path(config_path: Path, *, include_disabled: bool = False) -> dict[str, Any]:
    """Read-only preflight for a sealed v2 RESOLVED campaign.

    Legacy ``load_campaign`` only understands v1.  This adapter translates
    each selected v2 row in memory and applies the exact same mutation audit
    used by execution, without creating an experiment workspace.
    """

    import json

    from .config_io import validate_config_kind, verify_config_hash
    from .v2_runner import _v2_mutation_to_v1

    path = Path(config_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    try:
        validate_config_kind(raw)
    except ValueError as exc:
        return {
            "schema_version": "nonvacuity_campaign_preflight_v2",
            "status": "CAMPAIGN_PREFLIGHT_FAILED",
            "campaign_id": raw.get("campaign_id"),
            "issues": [{"mutation_id": None, "code": "CONFIG_KIND_INVALID", "message": str(exc), "field": "config_kind"}],
        }
    if raw.get("config_kind") != "RESOLVED":
        return {
            "schema_version": "nonvacuity_campaign_preflight_v2",
            "status": "CAMPAIGN_PREFLIGHT_FAILED",
            "campaign_id": raw.get("campaign_id"),
            "issues": [{"mutation_id": None, "code": "CONFIG_NOT_RESOLVED", "message": "v2 preflight 需要 sealed RESOLVED campaign", "field": "config_kind"}],
        }
    try:
        verify_config_hash(raw)
    except ValueError as exc:
        return {
            "schema_version": "nonvacuity_campaign_preflight_v2",
            "status": "CAMPAIGN_PREFLIGHT_FAILED",
            "campaign_id": raw.get("campaign_id"),
            "issues": [{"mutation_id": None, "code": "CONFIG_HASH_INVALID", "message": str(exc), "field": "config_sha256"}],
        }
    source_root = Path(str(raw["source_binding"]["clean_source_root"])).resolve()
    selected = [
        mutation for mutation in raw.get("mutations", [])
        if include_disabled or mutation.get("enabled", False)
    ]
    for mutation in selected:
        mutation_id = str(mutation.get("mutation_id", ""))
        try:
            translated = _v2_mutation_to_v1(mutation, config=raw, base_dir=path.parent)
            manifest = MutationManifest.from_mapping(translated, base_dir=path.parent)
            result = audit_mutation(manifest, source_root=source_root)
            issues.extend(result["issues"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({
                "mutation_id": mutation_id or None,
                "code": "V2_TRANSLATION_INVALID",
                "message": str(exc),
                "field": None,
            })
    return {
        "schema_version": "nonvacuity_campaign_preflight_v2",
        "status": "PASS" if not issues else "CAMPAIGN_PREFLIGHT_FAILED",
        "campaign_id": raw.get("campaign_id"),
        "checked_mutations": [item.get("mutation_id") for item in selected],
        "issues": issues,
    }
