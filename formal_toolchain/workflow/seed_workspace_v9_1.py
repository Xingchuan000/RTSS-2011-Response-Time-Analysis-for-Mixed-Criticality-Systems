"""Freeze a seed into the single, non-legacy V9.1 proof request surface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.adapters.seed_directory import ALLOWED_VARIANTS
from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.tree_artifact import REQUIRED_FILES
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.core.errors import UnresolvedInputError
from formal_toolchain.core.hashing import sha256_file
from formal_toolchain.v9_1.constants import PRIMARY_CLAIM, PROOF_ROUTE, REQUEST_SCHEMA, SCOPE


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_required_tree_files(source: Path, destination: Path) -> None:
    for name in REQUIRED_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"tree artifact missing or symlinked: {name}")
        shutil.copy2(path, destination / name)


def _strict_target_manifest(seed_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = seed_dir / "formal_target_manifest.json"
    if not path.is_file():
        raise ValueError("V9.1 requires formal_target_manifest.json")
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != "formal_target_manifest_v1":
        raise ValueError("V9.1 only accepts formal_target_manifest_v1")
    required = {"target_id", "target_kind", "authoritative_input_mode"}
    if not required <= set(data):
        raise ValueError("formal_target_manifest_v1 missing target identity fields")
    return path, data


def freeze_seed_workspace_v9_1(
    seed_dir: Path,
    tree_variant: str,
    output_dir: Path,
    *,
    code_root: Path,
    target_recipe: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a V9.1 workspace without route selection or V8 Phase-K artifacts."""

    seed_dir = Path(seed_dir).resolve()
    code_root = Path(code_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not seed_dir.is_dir() or seed_dir.is_symlink():
        raise ValueError("--seed-dir must be an ordinary extracted directory")
    if tree_variant not in ALLOWED_VARIANTS:
        raise ValueError(f"unsupported tree variant: {tree_variant}")
    source_variant = (seed_dir / tree_variant).resolve()
    if not source_variant.is_dir() or source_variant.is_symlink() or seed_dir not in source_variant.parents:
        raise ValueError("tree variant directory is invalid")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError("--out already exists")
        shutil.rmtree(output_dir)

    for relative in ("request/inputs/tree_artifact", "request/inputs/formal_inputs", "candidate", "verified", "logs"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    copied_tree = output_dir / "request/inputs/tree_artifact"
    _copy_required_tree_files(source_variant, copied_tree)

    formal_inputs = seed_dir / "formal_inputs"
    if not formal_inputs.is_dir() or formal_inputs.is_symlink():
        raise UnresolvedInputError("AUTHORITATIVE_TARGET_MISSING", "V9.1 requires authoritative formal_inputs")
    copied_inputs = output_dir / "request/inputs/formal_inputs"
    shutil.rmtree(copied_inputs)
    shutil.copytree(formal_inputs, copied_inputs, symlinks=False)

    manifest_path, target_manifest = _strict_target_manifest(seed_dir)
    shutil.copy2(manifest_path, output_dir / "request/inputs/formal_target_manifest.json")

    recipe_path = Path(target_recipe).resolve() if target_recipe else formal_inputs / "target_recipe.json"
    if not recipe_path.is_file():
        raise ValueError("authoritative target_recipe.json not found")
    recipe = _read_json(recipe_path)
    if not isinstance(recipe, dict) or not isinstance(recipe.get("factory"), str):
        raise ValueError("target_recipe.factory is invalid")
    recipe_kwargs = dict(recipe.get("kwargs", {}))

    metadata = _read_json(source_variant / "metadata.json")
    metadata_seed = metadata.get("taskset_seed") if isinstance(metadata, dict) else None
    declared_seed = target_manifest.get("taskset_seed", metadata_seed)
    if declared_seed is None:
        raise ValueError("V9.1 requires an explicit taskset_seed")
    if metadata_seed is not None and int(metadata_seed) != int(declared_seed):
        raise ValueError("TARGET_SEED_IDENTITY_MISMATCH")

    target = build_target(recipe["factory"], recipe_kwargs)
    effective = export_formal_target_config(target)
    if effective.get("status") != "PASS":
        raise UnresolvedInputError("EFFECTIVE_RUNTIME_CONFIG_REFRESH_FAILED", json.dumps(effective, ensure_ascii=False))
    (copied_inputs / "effective_runtime_config.json").write_text(
        json.dumps(effective, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (copied_inputs / "action_definitions_canonical.json").write_text(
        json.dumps({"schema_version": "action_definitions_canonical_v2_v9_1",
                    "action_definitions": [dict(row) for row in target.action_definitions]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_manifest = build_source_manifest(code_root)
    request = {
        "schema_version": REQUEST_SCHEMA,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": PRIMARY_CLAIM,
        "target_id": target_manifest["target_id"],
        "target_kind": target_manifest["target_kind"],
        "taskset_seed": int(declared_seed),
        "target_recipe": {"factory": recipe["factory"], "kwargs": recipe_kwargs},
        "tree_artifact_dir": "request/inputs/tree_artifact",
        "formal_inputs_dir": "request/inputs/formal_inputs",
        "tree_variant": tree_variant,
        "expected_tree_file_sha256": sha256_file(copied_tree / "integer_tree.json"),
        "source_binding": {
            "source_root_role": "external_argument",
            "source_manifest_semantic_hash": source_manifest["semantic_hash"],
            "required_paths": ["formal_toolchain", "amc_py/rl", "amc_py/viper"],
        },
    }
    request_path = output_dir / "request/proof_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = {name: sha256_file(copied_tree / name) for name in REQUIRED_FILES}
    (output_dir / "request/seed_artifact_inventory.json").write_text(
        json.dumps({"schema_version": "seed_artifact_inventory_v9_1", "tree_variant": tree_variant,
                    "files": inventory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "workspace": output_dir,
        "request": request_path,
        "target_id": target_manifest["target_id"],
        "target_kind": target_manifest["target_kind"],
    }
