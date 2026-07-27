"""Mutation-blind Python source overlay patcher.

It only patches a declared snippet inside a declared symbol in a copied source
tree and verifies that all other top-level symbol AST hashes remain unchanged.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from pathlib import Path

from ..canonical import canonical_json_hash, file_hash, python_symbol_hash
from .base import MutationContext, MutationResult, PreflightResult


class PythonSymbolMutation:
    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            path = self._target(context)
            source = path.read_text(encoding="utf-8")
            symbol = str(context.parameters["target_symbol"])
            python_symbol_hash(source, symbol)
            before = str(context.parameters["before_snippet"])
            occurrence = int(context.parameters.get("occurrence", 1))
            if source.count(before) != occurrence:
                raise ValueError(
                    f"before_snippet 出现次数必须为 {occurrence}，实际 {source.count(before)}"
                )
            return PreflightResult("PASS", {"target": str(path), "symbol": symbol})
        except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(str(preflight.details.get("reason")))
        path = self._target(context)
        before_source = path.read_text(encoding="utf-8")
        before_snippet = str(context.parameters["before_snippet"])
        after_snippet = str(context.parameters["after_snippet"])
        after_source = before_source.replace(before_snippet, after_snippet, 1)
        ast.parse(after_source, filename=str(path))
        symbol = str(context.parameters["target_symbol"])
        before_symbol = python_symbol_hash(before_source, symbol)
        after_symbol = python_symbol_hash(after_source, symbol)
        changed_symbols = _changed_top_level_symbols(before_source, after_source)
        allowed_top = symbol.split(".")[0]
        valid = before_symbol != after_symbol and changed_symbols <= {allowed_top}
        path.write_text(after_source, encoding="utf-8")
        diff_path = context.parameters.get("diff_file")
        if diff_path:
            diff_target = Path(str(diff_path))
            diff_target.parent.mkdir(parents=True, exist_ok=True)
            diff_target.write_text(
                "".join(
                    difflib.unified_diff(
                        before_source.splitlines(keepends=True),
                        after_source.splitlines(keepends=True),
                        fromfile=f"a/{context.parameters['target_file']}",
                        tofile=f"b/{context.parameters['target_file']}",
                    )
                ),
                encoding="utf-8",
            )
        assert context.source_overlay is not None
        return MutationResult(
            status="PASS" if valid else "FAIL",
            before_hash=hashlib.sha256(before_source.encode("utf-8")).hexdigest(),
            after_hash=file_hash(path),
            changed_files=(str(path.relative_to(context.source_overlay)),),
            changed_symbols=(symbol,),
            semantic_change_count=1 if valid else len(changed_symbols),
            parser_validation="PASS",
            rollback_information={"before_symbol_ast_hash": before_symbol},
            details={
                "after_symbol_ast_hash": after_symbol,
                "changed_top_level_symbols": sorted(changed_symbols),
                "mutation_id_in_source": "nonvacuity" in after_snippet.lower(),
            },
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = (
            result.status == "PASS"
            and result.semantic_change_count == 1
            and not result.details.get("mutation_id_in_source")
        )
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())

    @staticmethod
    def _target(context: MutationContext) -> Path:
        if context.source_overlay is None:
            raise ValueError("Python source mutation 需要 source_overlay")
        relative = Path(str(context.parameters["target_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("target_file 必须是安全相对路径")
        path = (context.source_overlay / relative).resolve()
        if context.source_overlay.resolve() not in path.parents:
            raise ValueError("target_file 逃逸 source_overlay")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"target source 不存在或为软链接: {path}")
        return path


class MultiPythonSymbolMutation:
    """Apply one conceptual semantic mutation across mirrored overlay sources."""

    def preflight(self, context: MutationContext) -> PreflightResult:
        patches = context.parameters.get("patches")
        if not isinstance(patches, list) or not patches:
            return PreflightResult("FAIL", {"reason": "source patches 必须为非空 array"})
        receipts = []
        for patch in patches:
            delegated = _delegated_context(context, patch)
            result = PythonSymbolMutation().preflight(delegated)
            receipts.append({"patch": dict(patch), "preflight": dict(result.details)})
            if result.status != "PASS":
                return PreflightResult("FAIL", {"patches": receipts})
        return PreflightResult("PASS", {"patches": receipts})

    def apply(self, context: MutationContext) -> MutationResult:
        preflight = self.preflight(context)
        if preflight.status != "PASS":
            raise ValueError(f"multi-source preflight failed: {preflight.details}")
        assert context.source_overlay is not None
        relative_files = tuple(
            dict.fromkeys(str(patch["target_file"]) for patch in context.parameters["patches"])
        )
        before = [
            {
                "file": relative,
                "hash": file_hash(context.source_overlay / relative),
            }
            for relative in relative_files
        ]
        results = [
            PythonSymbolMutation().apply(_delegated_context(context, patch))
            for patch in context.parameters["patches"]
        ]
        after = [
            {
                "file": relative,
                "hash": file_hash(context.source_overlay / relative),
            }
            for relative in relative_files
        ]
        valid = all(
            result.status == "PASS"
            and not result.details.get("mutation_id_in_source")
            for result in results
        )
        return MutationResult(
            status="PASS" if valid else "FAIL",
            before_hash=canonical_json_hash(before),
            after_hash=canonical_json_hash(after),
            changed_files=relative_files,
            changed_symbols=tuple(result.changed_symbols[0] for result in results),
            semantic_change_count=1,
            parser_validation="PASS" if valid else "FAIL",
            details={
                "conceptual_mutation": context.parameters.get("semantic_group"),
                "patch_results": [result.to_dict() for result in results],
            },
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        valid = (
            result.status == "PASS"
            and result.semantic_change_count == 1
            and result.before_hash != result.after_hash
        )
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())


def _changed_top_level_symbols(before: str, after: str) -> set[str]:
    def hashes(source: str) -> dict[str, str]:
        tree = ast.parse(source)
        return {
            node.name: hashlib.sha256(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
            ).hexdigest()
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    left, right = hashes(before), hashes(after)
    return {name for name in set(left) | set(right) if left.get(name) != right.get(name)}


def _delegated_context(
    context: MutationContext,
    patch: dict[str, object],
) -> MutationContext:
    parameters = dict(patch)
    diff_root = context.parameters.get("diff_dir")
    if diff_root:
        safe_name = str(parameters["target_file"]).replace("/", "__")
        parameters["diff_file"] = str(Path(str(diff_root)) / f"{safe_name}.patch")
    return MutationContext(
        mutation_id=context.mutation_id,
        source_root=context.source_root,
        mutated_seed=context.mutated_seed,
        source_overlay=context.source_overlay,
        parameters=parameters,
    )
