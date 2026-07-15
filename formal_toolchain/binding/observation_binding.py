"""observation extractor 与 tree feature schema 的绑定。"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

from .python_ast_ir import function_to_ir


def _feature_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("feature_names", data) if isinstance(data, dict) else data
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("feature_names.json 必须是字符串列表")
    return values


def _runtime_feature_schema(source_root: Path, ordered_tasks: list[str]) -> list[str]:
    """从 feature_config.py 的字面量常量独立导出运行时 schema。"""
    source = (Path(source_root) / "amc_py/rl/feature_config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    values: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in {"V11_PER_TASK_FEATURE_NAMES", "V11_GLOBAL_FEATURE_NAMES"}:
                value = node.value
                if isinstance(value, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts):
                    values[target.id] = tuple(item.value for item in value.elts)
    if set(values) != {"V11_PER_TASK_FEATURE_NAMES", "V11_GLOBAL_FEATURE_NAMES"}:
        raise ValueError("无法从 feature_config 独立导出 v11 feature schema")
    return [f"T{slot:02d}.{task}.{name}" for slot, task in enumerate(ordered_tasks)
            for name in values["V11_PER_TASK_FEATURE_NAMES"]] + [
                f"G.{name}" for name in values["V11_GLOBAL_FEATURE_NAMES"]]


def bind_observation_runtime(source_root: Path, feature_artifact: Path, *,
                             runtime_feature_names: list[str] | None = None,
                             ordered_tasks: list[str] | None = None,
                             feature_task_order: list[str] | None = None) -> dict[str, Any]:
    path = Path(source_root) / "amc_py/rl/observation.py"
    source = path.read_text(encoding="utf-8")
    function_names = ("build_observation", "build_v11_full_10d_observation",
                      "_build_v11_family_observation")
    functions = {name: function_to_ir(source, name) for name in function_names}
    unresolved = [name for name, ir in functions.items() if ir.get("status") != "PASS"]
    try:
        names = _feature_names(Path(feature_artifact))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "UNRESOLVED", "failure": {"code": "FEATURE_ARTIFACT_INVALID", "route": "MODEL_CONFORMANCE_FAILED", "detail": str(exc)}}
    if unresolved:
        return {"status": "UNRESOLVED", "failure": {"code": "TARGET_FUNCTION_IR_UNRESOLVED", "route": "UNRESOLVED", "functions": unresolved}, "functions": functions}
    if ordered_tasks is None:
        return {"status": "UNRESOLVED", "failure": {"code": "RUNTIME_TASK_ORDER_UNAVAILABLE", "route": "MODEL_CONFORMANCE_FAILED"}}
    try:
        derived_names = _runtime_feature_schema(source_root, list(ordered_tasks))
    except (OSError, SyntaxError, ValueError) as exc:
        return {"status": "UNRESOLVED", "failure": {"code": "RUNTIME_FEATURE_SCHEMA_UNRESOLVED", "route": "UNRESOLVED", "detail": str(exc)}}
    if names != derived_names:
        return {"status": "UNRESOLVED", "failure": {"code": "FEATURE_NAMES_BYTE_MISMATCH", "route": "MODEL_CONFORMANCE_FAILED"}}
    if feature_task_order is None or ordered_tasks != feature_task_order:
        return {"status": "UNRESOLVED", "failure": {"code": "FEATURE_TASK_ORDER_UNVERIFIED", "route": "MODEL_CONFORMANCE_FAILED"}}
    return {"status": "PASS", "extractor_candidates": list(functions), "feature_names": names,
            "feature_count": len(names), "per_task_features": "10d",
            "global_features": "8d", "nan_inf_behavior": "must_be_rejected_or_clipped",
            "functions": functions, "runtime_schema_source": "amc_py/rl/feature_config.py",
            "runtime_feature_names": derived_names}
