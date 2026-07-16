"""seed-directory 导入和 request freeze。"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.adapters.tree_artifact import REQUIRED_FILES
from formal_toolchain.core.hashing import sha256_file

from formal_toolchain.adapters.seed_directory import ALLOWED_VARIANTS


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
                          overwrite: bool = False) -> dict[str, Any]:
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
        raise ValueError("canonical seed 缺少 formal_inputs 目录")
    copied_inputs = output_dir / "request" / "inputs" / "formal_inputs"
    shutil.copytree(formal_inputs, copied_inputs, symlinks=False)
    fixture_manifest = seed_dir / "fixture_manifest.json"
    if not fixture_manifest.is_file():
        raise ValueError("canonical seed 缺少 fixture_manifest.json")
    shutil.copy2(fixture_manifest, output_dir / "request" / "inputs" / "fixture_manifest.json")
    recipe_path = Path(target_recipe) if target_recipe else formal_inputs / "target_recipe.json"
    if not recipe_path.is_file():
        raise ValueError("未找到 authoritative target recipe")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or not isinstance(recipe.get("factory"), str):
        raise ValueError("target_recipe.factory 非法")
    metadata = json.loads((source_variant / "metadata.json").read_text(encoding="utf-8"))
    fixture = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    request = {
        "schema_version": "proof_request_v1",
        "profile": "P0",
        "primary_claim": "DEPLOYED_HI_SAFETY",
        "taskset_seed": int(metadata.get("taskset_seed", 0) or 0),
        "fixture_id": fixture.get("fixture_id"),
        "fixture_kind": fixture.get("fixture_kind"),
        "target_recipe": {"factory": recipe["factory"], "kwargs": dict(recipe.get("kwargs", {}))},
        "tree_artifact_dir": "request/inputs/tree_artifact",
        "source_root": ".",
        "expected_tree_file_sha256": sha256_file(copied_tree / "integer_tree.json"),
        "optional_claims": [],
    }
    (output_dir / "request" / "proof_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = {name: sha256_file(copied_tree / name) for name in REQUIRED_FILES if name != "artifact_manifest.json"}
    (output_dir / "request" / "seed_artifact_inventory.json").write_text(
        json.dumps({"schema_version": "seed_artifact_inventory_v1", "tree_variant": tree_variant,
                    "files": inventory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "request" / "target_recipe.json").write_text(
        json.dumps(request["target_recipe"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostic = {"schema_version": "seed_import_diagnostic_v1", "status": "PASS",
                  "fixture_id": fixture.get("fixture_id"), "fixture_kind": fixture.get("fixture_kind"),
                  "tree_variant": tree_variant, "source_root": "code_root",
                  "external_seed_paths_read": [], "hout_used": False}
    (output_dir / "seed_import_diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"workspace": output_dir, "request": output_dir / "request" / "proof_request.json",
            "fixture_id": fixture.get("fixture_id"), "fixture_kind": fixture.get("fixture_kind")}
