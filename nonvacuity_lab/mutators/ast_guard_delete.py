from __future__ import annotations

import ast


class MutationError(ValueError):
    pass


class DeleteExactIf(ast.NodeTransformer):
    def __init__(self, expected_test_dump: str):
        self.expected_test_dump = expected_test_dump
        self.deleted = 0

    def visit_If(self, node):
        node = self.generic_visit(node)
        if ast.dump(node.test, include_attributes=False) == self.expected_test_dump:
            self.deleted += 1
            return None
        return node


class DeleteRejectReturn(ast.NodeTransformer):
    """Delete exactly one reject branch selected from current source AST."""

    def __init__(self, reject_reason: str):
        self.reject_reason = reject_reason
        self.deleted = 0

    def visit_If(self, node):
        node = self.generic_visit(node)
        text = ast.unparse(node)
        if self.reject_reason in text and "accepted" in text and "False" in text:
            self.deleted += 1
            return node.orelse or []
        return node


def delete_guard(source: str, expected_test_dump: str) -> str:
    tree = ast.parse(source)
    transformer = DeleteExactIf(expected_test_dump)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    if transformer.deleted != 1:
        raise MutationError(f"expected exactly one guard, deleted={transformer.deleted}")
    return ast.unparse(new_tree) + "\n"


def delete_reject_return(source: str, reject_reason: str) -> str:
    """Remove one matching rejection ``if`` without relying on AST dumps."""
    if not reject_reason:
        raise MutationError("reject_reason must not be empty")
    tree = ast.parse(source)
    transformer = DeleteRejectReturn(reject_reason)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    if transformer.deleted != 1:
        raise MutationError(
            f"GUARD_SELECTOR_NOT_UNIQUE:{reject_reason}:{transformer.deleted}"
        )
    ast.parse(ast.unparse(new_tree))
    return ast.unparse(new_tree) + "\n"
