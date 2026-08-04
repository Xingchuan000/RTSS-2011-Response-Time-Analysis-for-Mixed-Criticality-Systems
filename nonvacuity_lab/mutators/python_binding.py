from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundSymbol:
    qualified_name: str
    start_line: int
    end_line: int
    ast_hash: str
    source: str


def _hash_node(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()


def bind_symbol(source: str, qualified_name: str) -> BoundSymbol:
    tree = ast.parse(source)
    parts = qualified_name.split(".")
    nodes: list[ast.AST] = list(tree.body)
    target: ast.AST | None = None
    for index, part in enumerate(parts):
        target = next((node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == part), None)
        if target is None:
            raise ValueError(f"SYMBOL_NOT_FOUND:{qualified_name}")
        if index < len(parts) - 1:
            if not isinstance(target, ast.ClassDef):
                raise ValueError(f"SYMBOL_PARENT_NOT_CLASS:{qualified_name}")
            nodes = list(target.body)
    assert target is not None
    lines = source.splitlines(keepends=True)
    start, end = int(target.lineno), int(target.end_lineno)
    return BoundSymbol(qualified_name, start, end, _hash_node(target), "".join(lines[start - 1:end]))
