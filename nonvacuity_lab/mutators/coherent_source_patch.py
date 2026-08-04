from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Any

from ..canonical import canonical_json_hash, file_hash
from ..schema import FORBIDDEN_PATCH_ROLES, PatchRole, SourcePatchSpec
from .base import MutationContext, MutationResult, PreflightResult
from .python_binding import bind_symbol


FORBIDDEN_PATH_PREFIXES = ("formal_toolchain/verifier/", "formal_toolchain/reporting/")
FORBIDDEN_PATHS = {"formal_toolchain/verifier/checker_catalog.py", "nonvacuity_lab/runners/campaign.py"}


class CoherentSourcePatchMutation:
    def _specs(self, context: MutationContext) -> list[SourcePatchSpec]:
        raw = context.parameters.get("patches")
        if not isinstance(raw, list) or not raw:
            raise ValueError("patches 必须为非空 array")
        return [SourcePatchSpec.from_mapping(item) for item in raw]

    def _target(self, context: MutationContext, relative: str) -> Path:
        if context.source_overlay is None:
            raise ValueError("source_overlay 未创建")
        rel = Path(relative)
        normalized = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("UNSAFE_TARGET_PATH")
        if normalized in FORBIDDEN_PATHS or normalized.startswith(FORBIDDEN_PATH_PREFIXES):
            raise ValueError(f"FORBIDDEN_PATCH_TARGET:{normalized}")
        path = (context.source_overlay / rel).resolve()
        if context.source_overlay.resolve() not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"TARGET_FILE_MISSING:{normalized}")
        return path

    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            specs = self._specs(context)
            if not any(spec.role is PatchRole.DEPLOYED_IMPLEMENTATION for spec in specs):
                raise ValueError("COHERENT_PATCH_MISSING_DEPLOYED_IMPLEMENTATION")
            receipts = []
            for spec in specs:
                path = self._target(context, spec.target_file)
                bound = bind_symbol(path.read_text(encoding="utf-8"), spec.target_symbol)
                if bound.ast_hash != spec.before_ast_hash:
                    raise ValueError(f"BEFORE_AST_HASH_MISMATCH:{spec.target_file}:{spec.target_symbol}")
                if bound.source.count(spec.before_snippet) != spec.occurrence:
                    raise ValueError(f"SOURCE_SNIPPET_MISMATCH:{spec.target_file}")
                receipts.append({"role": spec.role.value, "target_file": spec.target_file, "target_symbol": spec.target_symbol, "before_ast_hash": bound.ast_hash})
            return PreflightResult("PASS", {"patches": receipts})
        except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(preflight.details.get("reason"))
        before, after, receipts = [], [], []
        changed_files, changed_symbols = [], []
        for spec in self._specs(context):
            path = self._target(context, spec.target_file)
            source = path.read_text(encoding="utf-8")
            bound = bind_symbol(source, spec.target_symbol)
            before.append({"file": spec.target_file, "symbol": spec.target_symbol, "ast_hash": bound.ast_hash, "file_hash": file_hash(path)})
            changed_symbol = bound.source.replace(spec.before_snippet, spec.after_snippet, spec.occurrence)
            if changed_symbol == bound.source:
                raise ValueError("PATCH_MADE_NO_CHANGE")
            lines = source.splitlines(keepends=True)
            updated = "".join(lines[:bound.start_line - 1]) + changed_symbol + "".join(lines[bound.end_line:])
            ast.parse(updated, filename=str(path))
            path.write_text(updated, encoding="utf-8")
            after_bound = bind_symbol(updated, spec.target_symbol)
            after.append({"file": spec.target_file, "symbol": spec.target_symbol, "ast_hash": after_bound.ast_hash, "file_hash": file_hash(path)})
            changed_files.append(spec.target_file); changed_symbols.append(spec.target_symbol)
            receipts.append({"role": spec.role.value, "target_file": spec.target_file, "target_symbol": spec.target_symbol, "before_ast_hash": bound.ast_hash, "after_ast_hash": after_bound.ast_hash, "diff": "".join(difflib.unified_diff(bound.source.splitlines(keepends=True), after_bound.source.splitlines(keepends=True)))})
        details = {"semantic_change_id": context.parameters.get("semantic_change_id"), "patches": receipts, "coherent_roles": sorted({item["role"] for item in receipts})}
        return MutationResult("PASS", canonical_json_hash(before), canonical_json_hash(after), tuple(dict.fromkeys(changed_files)), changed_symbols=tuple(changed_symbols), semantic_change_count=1, parser_validation="PASS", details=details)

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = result.status == "PASS" and result.semantic_change_count == 1 and result.before_hash != result.after_hash and bool(result.details.get("semantic_change_id")) and "DEPLOYED_IMPLEMENTATION" in set(result.details.get("coherent_roles", ()))
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())
