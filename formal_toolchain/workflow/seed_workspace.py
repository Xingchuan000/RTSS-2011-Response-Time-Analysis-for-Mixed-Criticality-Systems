"""seed-directory 导入和 request freeze。"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.adapters.tree_artifact import REQUIRED_FILES
from formal_toolchain.core.hashing import sha256_file

from formal_toolchain.adapters.seed_directory import ALLOWED_VARIANTS
from formal_toolchain.core.errors import UnresolvedInputError
from formal_toolchain.core.request_schema import validate_clean_proof_request


def normalize_target_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """把新旧 manifest 规范化为统一 target identity。"""

    if data.get("schema_version") == "formal_target_manifest_v1":
        required = {"target_id", "target_kind", "authoritative_input_mode"}
        if not required <= set(data):
            raise ValueError("formal_target_manifest 缺少 target identity 字段")
        return dict(data)
    if data.get("fixture_id") == "synthetic_p0" and data.get("fixture_kind") == "SYNTHETIC_P0":
        return {"schema_version": "formal_target_manifest_v1", "target_id": "synthetic_p0",
                "target_kind": "SYNTHETIC_P0", "taskset_seed": 0,
                "authoritative_input_mode": "FROZEN_FORMAL_INPUTS",
                "formal_inputs_version": "synthetic_p0_v1"}
    raise ValueError("legacy target manifest 不受支持")


def resolve_target_manifest(seed_dir: Path) -> tuple[Path, dict[str, Any]]:
    """优先使用 formal_target_manifest，旧 synthetic fixture 仅兼容解析。"""

    preferred = Path(seed_dir) / "formal_target_manifest.json"
    legacy = Path(seed_dir) / "fixture_manifest.json"
    path = preferred if preferred.is_file() else legacy
    if not path.is_file():
        raise ValueError("seed 缺少 formal_target_manifest.json")
    return path, normalize_target_manifest(json.loads(path.read_text(encoding="utf-8")))


def workspace_path(seed_dir: Path, output_dir: Path | None = None) -> Path:
    return output_dir or (seed_dir / ".formal_workspace")


def _copy_tree_files(source: Path, destination: Path) -> None:
    """只复制合同内文件，绝不把 HOUT/model/joblib 引入证明输入。"""

    for name in REQUIRED_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"tree artifact 缺少或包含符号链接: {name}")
        shutil.copy2(path, destination / name)


def freeze_seed_workspace(seed_dir: Path, tree_variant: str, output_dir: Path,
                          *, code_root: Path, target_recipe: Path | None = None,
                          overwrite: bool = False,
                          refresh_phase_k_map: bool = False,
                          proof_route: str = "protected_prefix") -> dict[str, Any]:
    """创建 canonical workspace，并输出机器无关的 proof request。"""

    seed_dir = Path(seed_dir).resolve()
    code_root = Path(code_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not seed_dir.is_dir() or seed_dir.is_symlink():
        raise ValueError("--seed-dir 必须是已解压的普通目录")
    if tree_variant not in ALLOWED_VARIANTS:
        raise ValueError(f"不支持的 tree variant: {tree_variant}")
    source_variant = seed_dir / tree_variant
    if not source_variant.is_dir() or source_variant.is_symlink():
        raise ValueError(f"tree variant 目录不存在: {tree_variant}")
    source_variant = source_variant.resolve()
    if seed_dir not in source_variant.parents:
        raise ValueError("tree variant 不得通过符号链接逃逸 seed 目录")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError("--out 已存在；默认拒绝覆盖")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for name in ("request", "candidate", "verified", "logs"):
        (output_dir / name).mkdir()
    copied_tree = output_dir / "request" / "inputs" / "tree_artifact"
    copied_tree.mkdir(parents=True)
    _copy_tree_files(source_variant, copied_tree)

    # formal_inputs 是 canonical target 的唯一文件来源。target.py 通过
    # fixture package 复制进 request 仅作为 provenance，不由 verifier import；
    # factory 仍按 recipe 从 code_root 重新导入，避免复制代码成为第二份实现。
    formal_inputs = seed_dir / "formal_inputs"
    if not formal_inputs.is_dir():
        raise UnresolvedInputError(
            "AUTHORITATIVE_TARGET_MISSING",
            "canonical seed 缺少 authoritative formal_inputs 目录")
    copied_inputs = output_dir / "request" / "inputs" / "formal_inputs"
    shutil.copytree(formal_inputs, copied_inputs, symlinks=False)
    # Phase K case map is a source-derived authoritative binding input.  Reuse
    # the seed copy when it is present and refresh was not requested; otherwise
    # derive it automatically from the current source tree so the caller only
    # needs to provide the seed folder.
    phase_k_case_map = seed_dir / "phase_k_case_map.json"
    existing_phase_k_schema = None
    if phase_k_case_map.is_file():
        try:
            existing_phase_k_schema = json.loads(
                phase_k_case_map.read_text(encoding="utf-8")
            ).get("schema_version")
        except (OSError, ValueError, TypeError):
            existing_phase_k_schema = None
    phase_k_map_refreshed = bool(
        refresh_phase_k_map
        or not phase_k_case_map.is_file()
        or existing_phase_k_schema != "phase_k_transition_path_map_v3_frozen_semantics"
    )
    if phase_k_map_refreshed:
        from formal_toolchain.adapters.source_manifest import build_source_manifest
        from formal_toolchain.bridge.p0_case_manifest import p0_case_manifest_hash
        from formal_toolchain.bridge.runtime_branch_map import (
            PATH_SPECS, _path_row, build_normal_runtime_path_coverage,
        )
        from formal_toolchain.core.hashing import sha256_object
        from formal_toolchain.semantics.frozen_runtime_contract import (
            CONTRACT_VERSION, frozen_contract_manifest,
        )

        source_hash = build_source_manifest(code_root)["semantic_hash"]
        contract_manifest = frozen_contract_manifest(code_root)
        paths = {spec[0]: _path_row(code_root, spec) for spec in PATH_SPECS}
        coverage = build_normal_runtime_path_coverage(code_root)
        if coverage.get("status") != "PASS":
            raise UnresolvedInputError(
                "PHASE_K_MAP_REGENERATION_FAILED",
                f"branch map coverage incomplete: {coverage}",
            )
        generated_map = {
            "schema_version": "phase_k_transition_path_map_v3_frozen_semantics",
            "source_hash": source_hash,
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "formal_semantics_contract_hash": contract_manifest["semantic_hash"],
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "paths": paths,
            "coverage": coverage,
            "case_manifest_hash": p0_case_manifest_hash(),
            "path_map_hash": sha256_object({
                "paths": paths, "coverage": coverage["artifact_hash"],
                "formal_semantics_contract_hash": contract_manifest["semantic_hash"],
            }),
            "generated_during_request_freeze": True,
        }
        (copied_inputs / "phase_k_case_map.json").write_text(
            json.dumps(generated_map, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif phase_k_case_map.is_file():
        shutil.copy2(phase_k_case_map, copied_inputs / phase_k_case_map.name)
    manifest_path, target_manifest = resolve_target_manifest(seed_dir)
    shutil.copy2(manifest_path, output_dir / "request" / "inputs" / "formal_target_manifest.json")
    recipe_path = Path(target_recipe) if target_recipe else formal_inputs / "target_recipe.json"
    if not recipe_path.is_file():
        raise ValueError("未找到 authoritative target recipe")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or not isinstance(recipe.get("factory"), str):
        raise ValueError("target_recipe.factory 非法")
    metadata = json.loads((source_variant / "metadata.json").read_text(encoding="utf-8"))
    metadata_seed = metadata.get("taskset_seed")
    declared_seed = target_manifest.get("taskset_seed", metadata_seed)
    if declared_seed is None:
        declared_seed = 0
    if metadata_seed is not None and int(metadata_seed) != int(declared_seed):
        raise ValueError("TARGET_SEED_IDENTITY_MISMATCH")
    recipe_kwargs = dict(recipe.get("kwargs", {}))

    # The copied formal_inputs are historical inputs from the seed export.  A
    # non-vacuity profile (and even the unified source revision with profile
    # ``off``) must be bound to the effective runtime object constructed by the
    # *current* target factory.  Refresh only the derived inputs in the request
    # workspace; the source seed directory remains immutable.
    from formal_toolchain.adapters.target_factory import build_target
    from formal_toolchain.adapters.runtime_config import export_formal_target_config

    frozen_target = build_target(recipe["factory"], recipe_kwargs)
    effective_config = export_formal_target_config(frozen_target)
    if effective_config.get("status") != "PASS":
        raise UnresolvedInputError(
            "EFFECTIVE_RUNTIME_CONFIG_REFRESH_FAILED",
            json.dumps(effective_config, ensure_ascii=False, sort_keys=True),
        )
    (copied_inputs / "effective_runtime_config.json").write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (copied_inputs / "action_definitions_canonical.json").write_text(
        json.dumps({
            "schema_version": "action_definitions_canonical_v1",
            "action_definitions": [dict(row) for row in frozen_target.action_definitions],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    from formal_toolchain.adapters.source_manifest import build_source_manifest
    from formal_toolchain.semantics.frozen_runtime_contract import (
        CONTRACT_VERSION, frozen_contract_manifest,
    )
    source_manifest = build_source_manifest(code_root)
    contract_manifest = frozen_contract_manifest(code_root)
    from formal_toolchain.routes.config import route_config
    resolved_route = route_config(proof_route)
    request = {
        "schema_version": "proof_request_v3",
        "proof_route": resolved_route.to_dict(),
        "profile": "P0",
        "primary_claim": "DEPLOYED_HI_SAFETY",
        "target_id": target_manifest["target_id"],
        "target_kind": target_manifest["target_kind"],
        "taskset_seed": int(declared_seed),
        "target_recipe": {"factory": recipe["factory"], "kwargs": recipe_kwargs},
        "tree_artifact_dir": "request/inputs/tree_artifact",
        "formal_inputs_dir": "request/inputs/formal_inputs",
        "tree_variant": tree_variant,
        "source_root": ".",
        "expected_tree_file_sha256": sha256_file(copied_tree / "integer_tree.json"),
        "source_binding": {
            "source_root_role": "external_argument",
            "binding_mode": "FROZEN_FORMAL_SEMANTICS",
            "source_manifest_semantic_hash": source_manifest["semantic_hash"],
            "implementation_audit_hash": source_manifest.get("implementation_audit_hash"),
            "required_paths": ["formal_toolchain", "amc_py/rl", "amc_py/viper"],
            "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
        },
        "optional_claims": [],
    }
    validate_clean_proof_request(request)
    (output_dir / "request" / "proof_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = {name: sha256_file(copied_tree / name) for name in REQUIRED_FILES if name != "artifact_manifest.json"}
    (output_dir / "request" / "seed_artifact_inventory.json").write_text(
        json.dumps({"schema_version": "seed_artifact_inventory_v1", "tree_variant": tree_variant,
                    "files": inventory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "request" / "target_recipe.json").write_text(
        json.dumps(request["target_recipe"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostic = {"schema_version": "seed_import_diagnostic_v1", "status": "PASS",
                  "target_id": target_manifest.get("target_id"), "target_kind": target_manifest.get("target_kind"),
                  "tree_variant": tree_variant, "source_root": "code_root",
                  "external_seed_paths_read": [], "hout_used": False,
                  "phase_k_map_refreshed": phase_k_map_refreshed,
                  "formal_semantics_binding_mode": "FROZEN_FORMAL_SEMANTICS",
                  "formal_semantics_contract_version": CONTRACT_VERSION,
                  "formal_semantics_contract_hash": contract_manifest["semantic_hash"],
                  "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
                  "implementation_audit_hash": source_manifest.get("implementation_audit_hash"),
                  "effective_runtime_config_refreshed": True,
                  "action_definitions_canonical_refreshed": True}
    (output_dir / "seed_import_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"workspace": output_dir, "request": output_dir / "request" / "proof_request.json",
            "target_id": target_manifest.get("target_id"), "target_kind": target_manifest.get("target_kind"),
            "phase_k_map_refreshed": phase_k_map_refreshed}
