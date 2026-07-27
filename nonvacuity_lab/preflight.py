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
from .canonical import python_symbol_hash
from .schema import CampaignConfig, MutationClass, MutationManifest


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


def audit_campaign(config: CampaignConfig) -> dict[str, Any]:
    """Resolve all enabled mutations without creating an experiment workspace."""

    issues: list[PreflightIssue] = []
    if not config.source_root.is_dir():
        issues.append(
            PreflightIssue(None, "SOURCE_ROOT_MISSING", f"source_root 不存在: {config.source_root}")
        )
    enabled = [item for item in config.mutations if item.enabled]
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
        "enabled_mutation_count": len(enabled),
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
            manifest.mutation_class is MutationClass.ENVELOPE
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
        resolver = parameters.get("resolver")
        if not parameters.get("leaf_candidates") and not resolver:
            issues.append(
                PreflightIssue(mid, "LEAF_TARGET_UNRESOLVED", "leaf_candidates 为空且未声明 resolver")
            )
        if not parameters.get("dangerous_actions") and not resolver:
            issues.append(
                PreflightIssue(mid, "ACTION_TARGET_UNRESOLVED", "dangerous_actions 为空且未声明 resolver")
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
    mode = str(manifest.activation.get("mode", "")).lower()
    if semantic and "symbolic" in mode and find_spec("z3") is None:
        issues.append(PreflightIssue(mid, "Z3_UNAVAILABLE", "symbolic activation 需要 z3"))
    if "hout" in mode or manifest.metadata.get("hout"):
        hout = manifest.metadata.get("hout")
        if not isinstance(hout, Mapping):
            issues.append(PreflightIssue(mid, "HOUT_CONFIG_MISSING", "HOUT activation 缺少 metadata.hout"))
        else:
            required = {
                "base_command",
                "mutated_command",
                "taskset",
                "scenario_seeds",
                "demand_trace",
                "horizon",
                "controller_release_times",
                "worker_count",
                "runtime_config",
                "random_seed",
            }
            missing = sorted(key for key in required if hout.get(key) is None)
            if missing:
                issues.append(
                    PreflightIssue(mid, "HOUT_FIELDS_MISSING", f"HOUT 缺少字段: {missing}")
                )
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
