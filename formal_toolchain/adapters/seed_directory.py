"""seed 文件夹导入合同和 fail-closed 诊断。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .tree_artifact import REQUIRED_FILES, inspect_tree_artifact

ALLOWED_VARIANTS = frozenset({"best_overall", "best_balanced", "best_performance"})


def resolve_seed_directory(seed_dir: Path, tree_variant: str = "best_overall",
                           output_dir: Path | None = None, *,
                           code_root: Path | None = None,
                           target_recipe: dict[str, Any] | None = None,
                           expected_seed: int | None = None) -> dict[str, Any]:
    """导入 seed 目录并复制 artifact；不会跟随 seed 外部符号链接。"""
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return _diagnostic("SEED_DIRECTORY_REQUIRED", "输入必须是目录")
    if tree_variant not in ALLOWED_VARIANTS:
        return _diagnostic("INVALID_TREE_VARIANT", f"不支持的 tree variant: {tree_variant}")
    source = seed_dir / tree_variant
    try:
        source = source.resolve(strict=True)
    except FileNotFoundError:
        return _diagnostic("TREE_ARTIFACT_MISSING", "tree variant 目录不存在")
    if seed_dir.resolve() not in source.parents:
        return _diagnostic("SYMLINK_OUT_OF_BOUNDS", "artifact 不得通过越界符号链接引用")
    try:
        inventory = inspect_tree_artifact(source)
    except (OSError, ValueError, KeyError) as exc:
        return _diagnostic("ARTIFACT_INVALID", str(exc))
    workspace = Path(output_dir) if output_dir else seed_dir / ".formal_workspace"
    copied = workspace / "tree_artifact"
    copied.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        src = source / name
        if src.is_symlink():
            return _diagnostic("SYMLINK_OUT_OF_BOUNDS", f"禁止 artifact 文件为符号链接: {name}")
        shutil.copy2(src, copied / name)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    declared_seed = metadata.get("taskset_seed") if isinstance(metadata, dict) else None
    if not isinstance(declared_seed, int) or isinstance(declared_seed, bool):
        return _diagnostic("TASKSET_SEED_INVALID", "artifact metadata.taskset_seed 必须是整数")
    if expected_seed is not None and declared_seed != expected_seed:
        return _diagnostic("TASKSET_SEED_MISMATCH", f"artifact metadata seed={declared_seed}，期望 {expected_seed}")
    request = {"schema_version": "proof_request_v2", "profile": "P0",
               "primary_claim": "DEPLOYED_HI_SAFETY", "taskset_seed": declared_seed,
               "target_recipe": target_recipe,
               "tree_artifact_dir": "tree_artifact",
               # canonical request 只能保存 workspace-relative 名称；CLI 参数中的
               # 机器绝对路径不进入 request/hash，真正的 source manifest 由后续
               # Phase D 在独立 context 中生成。
               "source_root": "source_root",
               "expected_tree_file_sha256": inventory["files"]["integer_tree.json"],
               "optional_claims": []}
    (workspace / "proof_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # resolver 只负责导入和记录 recipe；在未完成 target factory 构造、顺序和
    # fingerprint preflight 前，不能把“recipe 存在”升级为 PASS。
    target_status = "UNRESOLVED"
    target_code = "AUTHORITATIVE_TARGET_UNVERIFIED" if target_recipe else "AUTHORITATIVE_TARGET_MISSING"
    diagnostic = {"status": target_status, "code": target_code}
    (workspace / "import_diagnostic.json").write_text(
        json.dumps(diagnostic,
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": target_status, "tree_variant": tree_variant,
            "workspace": workspace.as_posix(), "inventory": inventory,
            "target_resolution": {"status": target_status, "code": target_code}}


def _diagnostic(code: str, message: str) -> dict[str, Any]:
    return {"status": "FAIL", "failure": {"code": code, "message": message,
            "route": "MODEL_CONFORMANCE_FAILED" if code != "ARTIFACT_INVALID" else "PROOF_BUNDLE_INVALID"}}
