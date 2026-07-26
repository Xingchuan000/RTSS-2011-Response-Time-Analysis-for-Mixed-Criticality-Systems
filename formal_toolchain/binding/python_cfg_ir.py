"""有限、可执行的 Python CFG path 提取器。

路径从函数体的 statement 顺序递归展开；每个 if 只进入自身真实 body 或
orelse，因而不会把互斥分支的 effect 合并。循环被明确标记为边界，不能被
伪装成一次普通 transition。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class GuardIR:
    test_source: str
    test_ast_hash: str
    polarity: bool
    lineno: int

    def to_dict(self) -> dict[str, Any]:
        return {"test_source": self.test_source, "test_ast_hash": self.test_ast_hash,
                "polarity": self.polarity, "lineno": self.lineno}


@dataclass(frozen=True, slots=True)
class EffectIR:
    kind: str
    target: str | None
    callee: str | None
    source: str
    ast_hash: str
    lineno: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "callee": self.callee,
                "source": self.source, "ast_hash": self.ast_hash, "lineno": self.lineno}


@dataclass(frozen=True, slots=True)
class ExecutablePath:
    entry_function: str
    guards: tuple[GuardIR, ...]
    effects: tuple[EffectIR, ...]
    terminal: str

    @property
    def path_id(self) -> str:
        return sha256_object({"entry_function": self.entry_function,
                              "guards": [x.to_dict() for x in self.guards],
                              "effects": [x.to_dict() for x in self.effects],
                              "terminal": self.terminal})

    def to_dict(self) -> dict[str, Any]:
        return {"entry_function": self.entry_function,
                "guard_ir": [x.to_dict() for x in self.guards],
                "ordered_effect_ir": [x.to_dict() for x in self.effects],
                "terminal": self.terminal, "path_ast_hash": self.path_id}


def _ast_hash(node: ast.AST) -> str:
    return sha256_object(ast.dump(node, include_attributes=False))


def _effect(node: ast.stmt) -> EffectIR:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return EffectIR("ASSIGN", ",".join(ast.unparse(x) for x in targets), None,
                        ast.unparse(node), _ast_hash(node), node.lineno)
    if isinstance(node, ast.AugAssign):
        return EffectIR("AUG_ASSIGN", ast.unparse(node.target), None, ast.unparse(node),
                        _ast_hash(node), node.lineno)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return EffectIR("CALL", None, ast.unparse(node.value.func), ast.unparse(node),
                        _ast_hash(node), node.lineno)
    if isinstance(node, ast.Return):
        return EffectIR("RETURN", None, None, ast.unparse(node), _ast_hash(node), node.lineno)
    if isinstance(node, ast.Raise):
        return EffectIR("RAISE", None, None, ast.unparse(node), _ast_hash(node), node.lineno)
    if isinstance(node, ast.Assert):
        return EffectIR("ASSERT", None, None, ast.unparse(node), _ast_hash(node), node.lineno)
    if isinstance(node, ast.Expr):
        return EffectIR("PURE_EXPR", None, None, ast.unparse(node), _ast_hash(node), node.lineno)
    raise ValueError(f"UNSUPPORTED_EFFECT_AST:{type(node).__name__}:{ast.unparse(node)}")


def _enumerate_block(statements: list[ast.stmt], *, entry_function: str,
                     guards: tuple[GuardIR, ...] = (),
                     effects: tuple[EffectIR, ...] = ()) -> list[ExecutablePath]:
    if not statements:
        return [ExecutablePath(entry_function, guards, effects, "FALLTHROUGH")]
    head, tail = statements[0], statements[1:]
    if isinstance(head, ast.If):
        true_guard = GuardIR(ast.unparse(head.test), _ast_hash(head.test), True, head.lineno)
        false_guard = GuardIR(ast.unparse(head.test), _ast_hash(head.test), False, head.lineno)
        return (_enumerate_block(list(head.body) + tail, entry_function=entry_function,
                                 guards=guards + (true_guard,), effects=effects)
                + _enumerate_block(list(head.orelse) + tail, entry_function=entry_function,
                                   guards=guards + (false_guard,), effects=effects))
    if isinstance(head, (ast.For, ast.While)):
        boundary = EffectIR("LOOP_CALL_BOUNDARY", None, None, ast.unparse(head),
                            _ast_hash(head), head.lineno)
        return _enumerate_block(tail, entry_function=entry_function, guards=guards,
                                effects=effects + (boundary,))
    effect = _effect(head)
    new_effects = effects + (effect,)
    if isinstance(head, (ast.Return, ast.Raise)):
        return [ExecutablePath(entry_function, guards, new_effects, f"RETURN:{head.lineno}")]
    return _enumerate_block(tail, entry_function=entry_function, guards=guards,
                            effects=new_effects)


def find_qualified_function(tree: ast.Module, qualified_function: str) -> ast.FunctionDef | None:
    if "." not in qualified_function:
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == qualified_function]
        return matches[0] if len(matches) == 1 else None
    class_name, function_name = qualified_function.split(".", 1)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        return None
    matches = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == function_name]
    return matches[0] if len(matches) == 1 else None


def enumerate_function_paths(source_root: str | Path, qualified_function: str, *,
                             source_relative: str = "amc_py/event_runtime.py") -> tuple[ExecutablePath, ...]:
    source_path = Path(source_root) / source_relative
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = find_qualified_function(tree, qualified_function)
    if function is None:
        raise ValueError(f"TARGET_FUNCTION_NOT_UNIQUE:{qualified_function}")
    unique = {p.path_id: p for p in _enumerate_block(list(function.body), entry_function=qualified_function)}
    return tuple(unique[key] for key in sorted(unique))


def extract_path_ir(source_root: str | Path, qualified: str, start: int | None = None,
                    end: int | None = None, *,
                    source_relative: str = "amc_py/event_runtime.py") -> dict[str, Any]:
    """兼容旧调用，但正式路径选择不再接受行号范围。"""
    try:
        paths = enumerate_function_paths(source_root, qualified, source_relative=source_relative)
    except (OSError, SyntaxError, ValueError) as exc:
        return {"status": "UNRESOLVED", "failure": str(exc)}
    if start is not None or end is not None:
        return {"status": "UNRESOLVED", "failure": "LINE_RANGE_PATH_BINDING_REMOVED"}
    return {"status": "PASS", "paths": [path.to_dict() for path in paths]}
