"""Canonical hashes and narrow diffs used to enforce single mutations."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


DISPLAY_ONLY_KEYS = frozenset({"generated_at", "timestamp", "created_at"})
IGNORED_TREE_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "logs",
        "outputs",
    }
)


def canonicalize_json(value: Any, *, ignore_keys: Iterable[str] = DISPLAY_ONLY_KEYS) -> Any:
    ignored = frozenset(ignore_keys)
    if isinstance(value, dict):
        return {
            str(key): canonicalize_json(item, ignore_keys=ignored)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in ignored
        }
    if isinstance(value, list):
        return [canonicalize_json(item, ignore_keys=ignored) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON 不允许 NaN/Infinity")
        return int(value) if value.is_integer() else float(format(value, ".17g"))
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"hash root 必须为普通目录: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_parts = path.relative_to(root).parts
        if (
            any(part in IGNORED_TREE_PARTS or part.endswith(".egg-info") for part in relative_parts)
            or any(part.startswith(".formal_proof_") for part in relative_parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "sha256": file_hash(path), "size": path.stat().st_size})
    return canonical_json_hash(records)


def json_pointer_diff(before: Any, after: Any, pointer: str = "") -> list[dict[str, Any]]:
    if type(before) is not type(after):
        return [{"pointer": pointer or "/", "before": before, "after": after}]
    if isinstance(before, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{pointer}/{_escape_pointer(str(key))}"
            if key not in before:
                result.append({"pointer": child, "before": None, "after": after[key], "kind": "added"})
            elif key not in after:
                result.append({"pointer": child, "before": before[key], "after": None, "kind": "removed"})
            else:
                result.extend(json_pointer_diff(before[key], after[key], child))
        return result
    if isinstance(before, list):
        result = []
        for index in range(max(len(before), len(after))):
            child = f"{pointer}/{index}"
            if index >= len(before):
                result.append({"pointer": child, "before": None, "after": after[index], "kind": "added"})
            elif index >= len(after):
                result.append({"pointer": child, "before": before[index], "after": None, "kind": "removed"})
            else:
                result.extend(json_pointer_diff(before[index], after[index], child))
        return result
    return [] if before == after else [{"pointer": pointer or "/", "before": before, "after": after}]


def python_symbol_hash(source: str, qualified_symbol: str) -> str:
    tree = ast.parse(source)
    node = _find_symbol(tree, qualified_symbol)
    if node is None:
        raise ValueError(f"未找到 Python symbol: {qualified_symbol}")
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _find_symbol(tree: ast.AST, qualified_symbol: str) -> ast.AST | None:
    parts = qualified_symbol.split(".")
    current: ast.AST = tree
    for part in parts:
        body = getattr(current, "body", ())
        match = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if match is None:
            return None
        current = match
    return current


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
