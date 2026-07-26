"""实际部署 controller→EventRuntimeEngine.apply_budget_updates 绑定。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from formal_toolchain.binding.python_ast_ir import function_to_ir
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.semantics.frozen_runtime_contract import frozen_event_runtime_path, frozen_runtime_wrapper_path, CONTRACT_VERSION


def bind_controller_runtime(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    wrapper_path = frozen_runtime_wrapper_path(root)
    engine_path = frozen_event_runtime_path(root)
    wrapper = wrapper_path.read_text(encoding="utf-8")
    engine = engine_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper)
    calls = [ast.unparse(node) for node in ast.walk(wrapper_tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "engine.apply_budget_updates"]
    engine_ir = function_to_ir(engine, "EventRuntimeEngine.apply_budget_updates")
    required = ("_advance_time", "apply_updates", "_reschedule")
    engine_tree = ast.parse(engine)
    engine_function = next((item for cls in engine_tree.body
                            if isinstance(cls, ast.ClassDef) and cls.name == "EventRuntimeEngine"
                            for item in cls.body
                            if isinstance(item, ast.FunctionDef) and item.name == "apply_budget_updates"), None)
    engine_text = ast.unparse(engine_function) if engine_function is not None else ""
    missing = [token for token in required if token not in engine_text]
    status = "PASS" if calls and engine_ir.get("status") == "PASS" and not missing else "UNRESOLVED"
    return {"status": status, "schema_version": "controller_binding_v2_frozen_semantics",
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "wrapper_source_hash": sha256_file(wrapper_path),
            "engine_source_hash": sha256_file(engine_path),
            "wrapper_calls": calls, "engine_ir": engine_ir,
            "required_effects": list(required), "missing": missing,
            "binding_hash": sha256_object({"wrapper_calls": calls, "engine_ir": engine_ir,
                                            "required_effects": required})}
