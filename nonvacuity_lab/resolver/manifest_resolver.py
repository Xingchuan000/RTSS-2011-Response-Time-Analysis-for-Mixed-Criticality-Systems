"""Resolve Phase-3 audit outputs into a still-disabled campaign manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from ..canonical import canonical_json_hash, file_hash
from ..manifest import serializable_manifest
from ..schema import CampaignConfig
from .dangerous_top1 import resolve_dangerous_top1
from .envelope_target import resolve_envelope_target
from .guard_ablation import resolve_guard_ablation


def resolve_campaign(
    config: CampaignConfig,
    *,
    audit_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_root = Path(audit_root).resolve()
    summary = _load_object(audit_root / "audit_summary.json")
    leaf_rows = _load_rows(audit_root / "leaf_audit.json")
    rta_rows = _load_rows(audit_root / "rta_slack.json")
    bundle_rows = _load_rows(audit_root / "bundle_inventory.json")
    guard_catalog = _load_object(
        config.source_root / "nonvacuity_lab" / "catalogs" / "ppp_guard_catalog.json"
    )
    resolved = copy.deepcopy(serializable_manifest(config))
    resolved["enabled"] = False
    resolutions: dict[str, Any] = {}
    unresolved: dict[str, str] = {}

    for mutation in resolved.get("mutations", []):
        mutation["enabled"] = False
        for optional_key in ("seed_dir", "base_seed", "paired_with", "reuse_source_bundle"):
            if mutation.get(optional_key) is None:
                mutation.pop(optional_key, None)
        mutation_id = str(mutation.get("mutation_id", ""))
        mutator = mutation.get("mutator", {})
        parameters = dict(mutator.get("parameters", {})) if isinstance(mutator, Mapping) else {}
        kind = str(mutator.get("kind", "")) if isinstance(mutator, Mapping) else ""
        try:
            if kind in {"dangerous_top1", "tree_ranking"}:
                scoped = _scope_leaf_rows(leaf_rows, mutation)
                target = resolve_dangerous_top1(scoped, require_hout_hit=True)
                parameters.update(
                    {
                        "leaf_id": target.leaf_id,
                        "action_id": target.action_id,
                        "tree_variant": mutation.get("tree_variant", "best_overall"),
                        "expected_tree_hash": target.tree_hash,
                        "activation_source": target.activation_source,
                        "witness_ref": target.witness_ref,
                        "risk_class": target.reason,
                    }
                )
                resolutions[mutation_id] = dict(parameters)
            elif mutation_id.startswith("B4") or (
                mutation.get("mutation_class") == "GUARD_ABLATION"
                and kind in {"source_overlay", "coherent_source_patch"}
            ):
                scoped = _scope_leaf_rows(leaf_rows, mutation)
                target = resolve_guard_ablation(scoped, guard_catalog)
                parameters.update(target)
                resolutions[mutation_id] = target
            elif kind == "envelope":
                target = resolve_envelope_target(
                    rta_rows,
                    seed_root=Path(str(summary["seed_root"])),
                )
                mutation["base_seed"] = target["seed"]
                mutation["seed_dir"] = target["seed_dir"]
                mutation["tree_variant"] = target["tree_variant"]
                parameters.update(
                    {
                        key: value
                        for key, value in target.items()
                        if key not in {"seed", "seed_dir", "tree_variant"}
                    }
                )
                resolutions[mutation_id] = target
            elif mutation_id.startswith("F5"):
                target = _resolve_bundle_target(
                    bundle_rows,
                    mutation,
                    Path(str(summary["proof_bundle_root"])),
                    kind="WITNESS_POINTER",
                )
                parameters.update(
                    {"target_file": target["artifact"], "json_pointer": target["json_pointer"]}
                )
                resolutions[mutation_id] = target
            elif mutation_id.startswith("F6"):
                target = _resolve_bundle_target(
                    bundle_rows,
                    mutation,
                    Path(str(summary["proof_bundle_root"])),
                    kind="OBLIGATION_ARTIFACT",
                )
                parameters.update({"target_file": target["artifact"]})
                resolutions[mutation_id] = target
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unresolved[mutation_id] = str(exc)
        if isinstance(mutator, dict):
            mutator["parameters"] = parameters

    receipt = {
        "schema_version": "nonvacuity_resolver_receipt_v1",
        "campaign_id": config.campaign_id,
        "audit_root": str(audit_root),
        "audit_summary_hash": file_hash(audit_root / "audit_summary.json"),
        "audit_content_hash": summary.get("content_hash"),
        "resolved_targets": resolutions,
        "unresolved_targets": unresolved,
        "resolved_campaign_hash": canonical_json_hash(resolved),
    }
    return resolved, receipt


def _scope_leaf_rows(rows: list[dict[str, Any]], mutation: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = mutation.get("base_seed")
    variant = mutation.get("tree_variant")
    scoped = [
        row
        for row in rows
        if (seed is None or int(row.get("seed", -1)) == int(seed))
        and (variant is None or str(row.get("tree_variant")) == str(variant))
    ]
    if not scoped:
        raise ValueError(f"LEAF_AUDIT_SCOPE_EMPTY:seed={seed}:variant={variant}")
    return scoped


def _resolve_bundle_target(
    rows: list[dict[str, Any]],
    mutation: Mapping[str, Any],
    proof_root: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != kind:
            continue
        try:
            candidates.append(_bundle_relative_target(row, mutation, proof_root))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"{kind}_UNRESOLVED_IN_REUSE_BUNDLE")
    if kind == "WITNESS_POINTER":
        return min(candidates, key=lambda row: (0 if "proof_summary" in row["artifact"] else 1, len(row.get("json_pointer", "")), row["artifact"]))
    return sorted(candidates, key=lambda row: (row.get("obligation_id") is None, row["artifact"]))[0]


def _bundle_relative_target(
    row: Mapping[str, Any],
    mutation: Mapping[str, Any],
    proof_root: Path,
) -> dict[str, Any]:
    reuse = mutation.get("reuse_source_bundle")
    if not reuse:
        raise ValueError("REUSE_SOURCE_BUNDLE_MISSING")
    reuse_path = Path(str(reuse)).resolve()
    artifact_path = proof_root / str(row["artifact"])
    try:
        relative = artifact_path.resolve().relative_to(reuse_path)
    except ValueError as exc:
        raise ValueError(f"BUNDLE_TARGET_OUTSIDE_REUSE_BUNDLE:{artifact_path}") from exc
    return {**dict(row), "artifact": relative.as_posix()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    raw = _load_object(path)
    rows = raw.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"rows 必须为 array: {path}")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _load_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON 顶层必须为 object: {path}")
    return raw
