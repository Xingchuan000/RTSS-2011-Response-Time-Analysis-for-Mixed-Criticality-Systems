"""F1-F7 mutations applied only to copied integrity workspaces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..canonical import file_hash
from .action_config import set_pointer
from .base import MutationContext, MutationResult, PreflightResult


class BundleTamperMutation:
    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            root = self._root(context)
            kind = str(context.parameters["tamper_kind"])
            supported = {
                "json_pointer",
                "delete_artifact",
                "splice_artifact",
                "replace_from",
                "source_file",
            }
            if kind not in supported:
                raise ValueError(f"不支持的 tamper_kind: {kind}")
            return PreflightResult("PASS", {"root": str(root), "tamper_kind": kind})
        except (KeyError, OSError, ValueError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason")))
        root = self._root(context)
        kind = str(context.parameters["tamper_kind"])
        relative = self._relative(context.parameters["target_file"])
        target = (root / relative).resolve()
        if root.resolve() not in target.parents:
            raise ValueError("tamper target 逃逸工作区")
        before_hash = file_hash(target) if target.is_file() else "MISSING"
        details: dict[str, Any] = {"tamper_kind": kind}
        if kind == "json_pointer":
            data = json.loads(target.read_text(encoding="utf-8"))
            set_pointer(data, str(context.parameters["json_pointer"]), context.parameters["value"])
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif kind == "delete_artifact":
            if not target.is_file():
                raise ValueError(f"待删除 artifact 不存在: {target}")
            target.unlink()
        elif kind in {"splice_artifact", "replace_from"}:
            source = Path(str(context.parameters["source_file"]))
            source = (
                source
                if source.is_absolute()
                else context.source_root / source
            ).resolve()
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"替换来源非法: {source}")
            if kind == "splice_artifact":
                payload = target.read_bytes() + source.read_bytes()
                target.write_bytes(payload)
            else:
                shutil.copy2(source, target)
            details["source_file"] = str(source)
        elif kind == "source_file":
            before = str(context.parameters["before_snippet"])
            after = str(context.parameters["after_snippet"])
            source_text = target.read_text(encoding="utf-8")
            if source_text.count(before) != 1:
                raise ValueError("source tamper snippet 必须恰好匹配一次")
            target.write_text(source_text.replace(before, after, 1), encoding="utf-8")
        after_hash = file_hash(target) if target.is_file() else "DELETED"
        return MutationResult(
            status="PASS" if before_hash != after_hash else "FAIL",
            before_hash=before_hash,
            after_hash=after_hash,
            changed_files=(relative.as_posix(),),
            semantic_change_count=1,
            details=details,
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = result.status == "PASS" and len(result.changed_files) == 1
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())

    @staticmethod
    def _root(context: MutationContext) -> Path:
        root = context.parameters.get("workspace_root")
        if root is None:
            raise ValueError("bundle tamper 需要 isolated workspace_root")
        path = Path(str(root)).resolve()
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"integrity workspace 无效: {path}")
        return path

    @staticmethod
    def _relative(value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("target_file 必须是安全相对路径")
        return path
