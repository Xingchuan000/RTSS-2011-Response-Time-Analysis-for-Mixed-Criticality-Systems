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


def delete_guard(source: str, expected_test_dump: str) -> str:
    tree = ast.parse(source)
    transformer = DeleteExactIf(expected_test_dump)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    if transformer.deleted != 1:
        raise MutationError(f"expected exactly one guard, deleted={transformer.deleted}")
    return ast.unparse(new_tree) + "\n"
