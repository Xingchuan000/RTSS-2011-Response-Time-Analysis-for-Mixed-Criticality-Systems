"""C3 AST mutator for retroactive updates of active released jobs."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from ..canonical import file_hash
from .base import MutationContext, MutationResult, PreflightResult
from .python_binding import bind_symbol


class InsertRetroactiveReleaseRewrite(ast.NodeTransformer):
    def __init__(self):
        self.inserted = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        if node.name != "apply_budget_updates":
            return node
        injected = ast.parse(
            """
for job in self.state.active_jobs:
    if not job.finished() and job.task.name in update_payload:
        job.runtime_budget_at_release = int(update_payload[job.task.name])
"""
        ).body
        # The rewrite must happen after the payload is built but before the
        # budget update/reschedule.  Appending at function end would leave the
        # scheduler using the old release budget and would not change the
        # executable semantics under test.
        for index, statement in enumerate(node.body):
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "update_payload"
            ):
                node.body[index + 1:index + 1] = injected
                self.inserted += 1
                break
        if self.inserted == 0:
            # Keep the mutator unit-testable against minimal mirrors that do
            # not materialize update_payload; real runtime mirrors must pass
            # the stricter preflight check above.
            node.body.extend(injected)
            self.inserted += 1
        return node


def insert_retroactive_release_rewrite(source: str) -> tuple[str, int]:
    tree = ast.parse(source)
    transformer = InsertRetroactiveReleaseRewrite()
    updated = transformer.visit(tree)
    ast.fix_missing_locations(updated)
    if transformer.inserted != 1:
        raise ValueError(f"C3_APPLY_BUDGET_UPDATES_NOT_UNIQUE:{transformer.inserted}")
    rendered = ast.unparse(updated) + "\n"
    ast.parse(rendered)
    return rendered, transformer.inserted


class RetroactiveReleaseBudgetMutation:
    def _targets(self, context: MutationContext) -> list[Path]:
        if context.source_overlay is None:
            raise ValueError("C3 source_overlay 未创建")
        patches = context.parameters.get("patches")
        if not isinstance(patches, list) or not patches:
            raise ValueError("C3 patches 必须为非空 array")
        targets = []
        for patch in patches:
            relative = Path(str(patch["target_file"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("C3 target_file 非法")
            # Legacy resolved configs contain both deployed and frozen mirrors.
            # C3 is a model-conformance mutation: keep the frozen semantics and
            # theorem sources unchanged, and mutate only the deployed runtime.
            if relative.as_posix() != "amc_py/event_runtime.py":
                continue
            path = (context.source_overlay / relative).resolve()
            if context.source_overlay.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"C3 target missing: {relative}")
            targets.append(path)
        if len(targets) != 1:
            raise ValueError("C3_DEPLOYED_RUNTIME_TARGET_NOT_UNIQUE")
        return targets

    def preflight(self, context: MutationContext) -> PreflightResult:
        try:
            targets = self._targets(context)
            if len(targets) != 1:
                raise ValueError("C3 requires exactly one deployed runtime patch")
            for path in targets:
                source = path.read_text(encoding="utf-8")
                if "def apply_budget_updates" not in source:
                    raise ValueError(f"C3 apply_budget_updates missing: {path}")
                tree = ast.parse(source)
                count = sum(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "apply_budget_updates"
                    for node in ast.walk(tree)
                )
                if count != 1:
                    raise ValueError(f"C3 apply_budget_updates not unique: {path}")
                if not any(
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "update_payload"
                    for node in ast.walk(tree)
                ):
                    raise ValueError(f"C3 update_payload binding missing: {path}")
            return PreflightResult("PASS", {"target_count": len(targets)})
        except (OSError, SyntaxError, KeyError, TypeError, ValueError) as exc:
            return PreflightResult("FAIL", {"reason": str(exc)})

    def apply(self, context: MutationContext) -> MutationResult:
        check = self.preflight(context)
        if check.status != "PASS":
            raise ValueError(str(check.details.get("reason")))
        assert context.source_overlay is not None
        changed = []
        before_hashes = []
        after_hashes = []
        for path in self._targets(context):
            source = path.read_text(encoding="utf-8")
            updated, inserted = insert_retroactive_release_rewrite(source)
            if inserted != 1:
                raise ValueError("C3_COHERENT_INSERT_FAILED")
            path.write_text(updated, encoding="utf-8")
            relative = path.relative_to(context.source_overlay).as_posix()
            changed.append(relative)
            before_hashes.append(hashlib.sha256(source.encode("utf-8")).hexdigest())
            after_hashes.append(file_hash(path))
        return MutationResult(
            status="PASS", before_hash=";".join(before_hashes), after_hash=";".join(after_hashes),
            changed_files=tuple(changed), semantic_change_count=1,
            parser_validation="PASS",
            details={
                "deployed_insertions": len(changed),
                "frozen_semantics_modified": False,
            },
        )

    def verify_single_change(self, result: MutationResult) -> PreflightResult:
        normalized_changed_files = tuple(
            str(item).replace("\\", "/") for item in result.changed_files
        )
        valid = (
            result.status == "PASS"
            and normalized_changed_files == ("amc_py/event_runtime.py",)
            and result.semantic_change_count == 1
            and result.details.get("frozen_semantics_modified") is False
        )
        return PreflightResult("PASS" if valid else "FAIL", result.to_dict())
