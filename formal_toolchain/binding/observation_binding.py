"""Bind tree feature artifacts to the frozen C-AMC-sem/P0 observation schema.

Mutable observation implementations, including q-AMC-specific feature modes,
are recorded only as non-blocking audit inputs.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from .python_ast_ir import function_to_ir
from formal_toolchain.semantics.frozen_runtime_contract import (
    CONTRACT_VERSION,
    frozen_observation_runtime_path,
)


def _feature_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("feature_names", data) if isinstance(data, dict) else data
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("feature_names.json 必须是字符串列表")
    return values


def _literal_tuple_constants(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(source)
    values: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"V11_PER_TASK_FEATURE_NAMES", "V11_GLOBAL_FEATURE_NAMES"}:
            continue
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in value.elts
        ):
            values[target.id] = tuple(str(item.value) for item in value.elts)
    if set(values) != {"V11_PER_TASK_FEATURE_NAMES", "V11_GLOBAL_FEATURE_NAMES"}:
        raise ValueError("冻结 observation schema 缺少字面量特征定义")
    return values["V11_PER_TASK_FEATURE_NAMES"], values["V11_GLOBAL_FEATURE_NAMES"]


def _frozen_feature_schema(source: str, ordered_tasks: list[str]) -> list[str]:
    per_task, global_features = _literal_tuple_constants(source)
    return [
        f"T{slot:02d}.{task}.{name}"
        for slot, task in enumerate(ordered_tasks)
        for name in per_task
    ] + [f"G.{name}" for name in global_features]


def _audit_hashes(source_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for relative in (
        "amc_py/rl/observation.py",
        "amc_py/rl/feature_state.py",
        "amc_py/rl/feature_config.py",
    ):
        path = source_root / relative
        if not path.is_file():
            continue
        payload = path.read_bytes()
        records.append({
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        })
    return {"binding": "NON_BLOCKING_AUDIT_ONLY", "files": records}


def bind_observation_runtime(source_root: Path, feature_artifact: Path, *,
                             runtime_feature_names: list[str] | None = None,
                             ordered_tasks: list[str] | None = None,
                             feature_task_order: list[str] | None = None) -> dict[str, Any]:
    root = Path(source_root)
    frozen_path = frozen_observation_runtime_path(root)
    frozen_source = frozen_path.read_text(encoding="utf-8")
    function_names = (
        "build_observation",
        "build_v11_full_10d_observation",
        "_build_v11_family_observation",
    )
    functions = {name: function_to_ir(frozen_source, name) for name in function_names}
    unresolved = [name for name, ir in functions.items() if ir.get("status") != "PASS"]

    try:
        names = _feature_names(Path(feature_artifact))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "FEATURE_ARTIFACT_INVALID",
                "route": "MODEL_CONFORMANCE_FAILED",
                "detail": str(exc),
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "implementation_audit": _audit_hashes(root),
        }

    if unresolved:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "TARGET_FUNCTION_IR_UNRESOLVED",
                "route": "UNRESOLVED",
                "functions": unresolved,
            },
            "functions": functions,
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "implementation_audit": _audit_hashes(root),
        }

    if ordered_tasks is None:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "RUNTIME_TASK_ORDER_UNAVAILABLE",
                "route": "MODEL_CONFORMANCE_FAILED",
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "implementation_audit": _audit_hashes(root),
        }

    try:
        derived_names = _frozen_feature_schema(frozen_source, list(ordered_tasks))
    except (SyntaxError, ValueError) as exc:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "FROZEN_FEATURE_SCHEMA_UNRESOLVED",
                "route": "UNRESOLVED",
                "detail": str(exc),
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "implementation_audit": _audit_hashes(root),
        }

    if names != derived_names:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "FEATURE_NAMES_BYTE_MISMATCH",
                "route": "MODEL_CONFORMANCE_FAILED",
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "frozen_feature_names": derived_names,
            "artifact_feature_names": names,
            "implementation_audit": _audit_hashes(root),
        }

    if feature_task_order is None or list(ordered_tasks) != list(feature_task_order):
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "FEATURE_TASK_ORDER_UNVERIFIED",
                "route": "MODEL_CONFORMANCE_FAILED",
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "implementation_audit": _audit_hashes(root),
        }

    runtime_schema_matches = runtime_feature_names is None or list(runtime_feature_names) == names
    return {
        "status": "PASS",
        "extractor_candidates": list(functions),
        "feature_names": names,
        "feature_count": len(names),
        "per_task_features": "10d",
        "global_features": "8d",
        "nan_inf_behavior": "must_be_rejected_or_clipped",
        "functions": functions,
        "formal_semantics_contract_version": CONTRACT_VERSION,
        "formal_observation_semantics_source": frozen_path.relative_to(root).as_posix(),
        "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
        "runtime_schema_diagnostic_match": runtime_schema_matches,
        "runtime_schema_diagnostic_blocking": False,
        "implementation_audit": _audit_hashes(root),
    }
