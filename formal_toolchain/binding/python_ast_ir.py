"""把受支持 Python 函数转换成可审计的结构化 AST IR。"""

from __future__ import annotations

import ast
from typing import Any

from .supported_ast import validate_supported_ast


def _expr(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, bool, str, float, type(None))):
            raise ValueError("UNSUPPORTED_AST_NODE: 非整数/布尔/字符串常量")
        value = format(node.value, ".17g") if isinstance(node.value, float) else node.value
        return {"kind": "constant", "value": value}
    if isinstance(node, ast.Name):
        return {"kind": "name", "id": node.id, "ctx": type(node.ctx).__name__}
    if isinstance(node, ast.Attribute):
        return {"kind": "attribute", "owner": _expr(node.value), "attr": node.attr}
    if isinstance(node, ast.Starred):
        return {"kind": "starred", "value": _expr(node.value)}
    if isinstance(node, ast.Slice):
        return {"kind": "slice", "lower": None if node.lower is None else _expr(node.lower),
                "upper": None if node.upper is None else _expr(node.upper),
                "step": None if node.step is None else _expr(node.step)}
    if isinstance(node, ast.Subscript):
        return {"kind": "subscript", "value": _expr(node.value), "slice": _expr(node.slice)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return {"kind": type(node).__name__.lower(), "items": [_expr(item) for item in node.elts]}
    if isinstance(node, ast.Set):
        return {"kind": "set", "items": [_expr(item) for item in node.elts]}
    if isinstance(node, ast.Call):
        return {"kind": "call", "function": _expr(node.func), "args": [_expr(item) for item in node.args],
                "keywords": [{"name": item.arg, "value": _expr(item.value)} for item in node.keywords]}
    if isinstance(node, ast.Lambda):
        return {"kind": "lambda", "arguments": [arg.arg for arg in node.args.args], "body": _expr(node.body)}
    if isinstance(node, ast.Dict):
        return {"kind": "dict", "items": [{"key": None if key is None else _expr(key), "value": _expr(value)} for key, value in zip(node.keys, node.values)]}
    if isinstance(node, ast.ListComp):
        return {"kind": "list_comp", "element": _expr(node.elt), "generators": [
            {"target": _expr(gen.target), "iter": _expr(gen.iter),
             "ifs": [_expr(item) for item in gen.ifs], "is_async": bool(gen.is_async)}
            for gen in node.generators]}
    if isinstance(node, ast.DictComp):
        return {"kind": "dict_comp", "key": _expr(node.key), "value": _expr(node.value),
                "generators": [{"target": _expr(gen.target), "iter": _expr(gen.iter),
                                "ifs": [_expr(item) for item in gen.ifs], "is_async": bool(gen.is_async)}
                               for gen in node.generators]}
    if isinstance(node, ast.GeneratorExp):
        return {"kind": "generator_exp", "element": _expr(node.elt),
                "generators": [{"target": _expr(gen.target), "iter": _expr(gen.iter),
                                "ifs": [_expr(item) for item in gen.ifs], "is_async": bool(gen.is_async)}
                               for gen in node.generators]}
    if isinstance(node, ast.JoinedStr):
        return {"kind": "f_string", "source": ast.unparse(node)}
    if isinstance(node, ast.BinOp):
        return {"kind": "binop", "operator": type(node.op).__name__, "left": _expr(node.left), "right": _expr(node.right)}
    if isinstance(node, ast.UnaryOp):
        return {"kind": "unary", "operator": type(node.op).__name__, "operand": _expr(node.operand)}
    if isinstance(node, ast.Compare):
        return {"kind": "compare", "left": _expr(node.left),
                "operators": [type(op).__name__ for op in node.ops],
                "comparators": [_expr(item) for item in node.comparators]}
    if isinstance(node, ast.BoolOp):
        return {"kind": "boolop", "operator": type(node.op).__name__, "values": [_expr(item) for item in node.values]}
    if isinstance(node, ast.IfExp):
        return {"kind": "if_expr", "test": _expr(node.test), "body": _expr(node.body), "orelse": _expr(node.orelse)}
    raise ValueError(f"UNSUPPORTED_AST_NODE: expression {type(node).__name__}")


def _stmt(node: ast.stmt) -> dict[str, Any]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {"kind": "function_def", "name": node.name,
                "arguments": [arg.arg for arg in node.args.args],
                "body": [_stmt(item) for item in node.body], "lineno": node.lineno}
    if isinstance(node, ast.Assign):
        return {"kind": "assign", "targets": [_expr(item) for item in node.targets], "value": _expr(node.value), "lineno": node.lineno}
    if isinstance(node, ast.AnnAssign):
        return {"kind": "assign", "targets": [_expr(node.target)], "value": _expr(node.value), "lineno": node.lineno}
    if isinstance(node, ast.AugAssign):
        return {"kind": "aug_assign", "target": _expr(node.target), "operator": type(node.op).__name__, "value": _expr(node.value), "lineno": node.lineno}
    if isinstance(node, ast.Return):
        return {"kind": "return", "value": None if node.value is None else _expr(node.value), "lineno": node.lineno}
    if isinstance(node, ast.Raise):
        return {"kind": "raise", "value": None if node.exc is None else _expr(node.exc), "lineno": node.lineno}
    if isinstance(node, ast.Assert):
        return {"kind": "assert", "test": _expr(node.test), "message": None if node.msg is None else _expr(node.msg), "lineno": node.lineno}
    if isinstance(node, ast.Nonlocal):
        return {"kind": "nonlocal", "names": list(node.names), "lineno": node.lineno}
    if isinstance(node, ast.If):
        return {"kind": "if", "test": _expr(node.test), "body": [_stmt(x) for x in node.body], "orelse": [_stmt(x) for x in node.orelse], "lineno": node.lineno}
    if isinstance(node, ast.For):
        return {"kind": "for", "target": _expr(node.target), "iter": _expr(node.iter), "body": [_stmt(x) for x in node.body], "lineno": node.lineno}
    if isinstance(node, ast.Expr):
        return {"kind": "expression", "value": _expr(node.value), "lineno": node.lineno}
    if isinstance(node, ast.Import):
        return {
            "kind": "import",
            "names": [{"name": alias.name, "asname": alias.asname} for alias in node.names],
            "lineno": node.lineno,
        }
    if isinstance(node, ast.ImportFrom):
        return {
            "kind": "import_from",
            "module": node.module,
            "level": node.level,
            "names": [{"name": alias.name, "asname": alias.asname} for alias in node.names],
            "lineno": node.lineno,
        }
    if isinstance(node, ast.Continue):
        return {"kind": "continue", "lineno": node.lineno}
    raise ValueError(f"UNSUPPORTED_AST_NODE: statement {type(node).__name__}")


def function_to_ir(source: str, function_name: str) -> dict[str, Any]:
    tree = ast.parse(source)
    # 支持 ``ClassName.method`` 精确定位；不再把类方法误报为“找不到函数”。
    if "." in function_name:
        class_name, method_name = function_name.rsplit(".", 1)
        functions = [method for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
                     for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == method_name]
    else:
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name]
    if len(functions) != 1:
        return {"status": "UNRESOLVED", "failure": {"code": "TARGET_FUNCTION_NOT_UNIQUE", "route": "MODEL_CONFORMANCE_FAILED"}}
    function = functions[0]
    # 只审计目标函数本体；模块 import 和其它未纳入本 obligation 的函数不应
    # 因为存在于同一文件而污染当前绑定结果。
    symbols = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.update(item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
    rejections = validate_supported_ast(function, symbols)
    if rejections:
        return {"status": "UNRESOLVED", "failure": {"code": rejections[0].code,
                "route": "UNRESOLVED", "node_type": rejections[0].node_type,
                "lineno": rejections[0].lineno, "detail": rejections[0].detail}}
    try:
        body = [_stmt(node) for node in function.body]
    except ValueError as exc:
        return {"status": "UNRESOLVED", "failure": {"code": "UNSUPPORTED_AST_NODE",
                "route": "UNRESOLVED", "detail": str(exc)}}
    return {"status": "PASS", "function": function_name, "lineno": function.lineno,
            "arguments": [arg.arg for arg in function.args.args], "body": body}


def object_to_ir(source: str, qualified_name: str) -> dict[str, Any]:
    """E01 公开 API：按模块级或 ClassName.method 定位目标。"""
    return function_to_ir(source, qualified_name)
