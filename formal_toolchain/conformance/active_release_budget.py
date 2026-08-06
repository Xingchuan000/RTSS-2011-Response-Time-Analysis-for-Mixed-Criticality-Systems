"""Fresh source check for active-release budget snapshot immutability."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file


def _stored_release_budget_attributes(function: ast.FunctionDef) -> list[dict[str, Any]]:
    writes: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr != "runtime_budget_at_release" or not isinstance(node.ctx, ast.Store):
            continue
        writes.append({
            "line": int(getattr(node, "lineno", -1)),
            "column": int(getattr(node, "col_offset", -1)),
            "target": ast.unparse(node),
        })
    return writes


def check_active_release_budget_source_contract(source_root: str | Path) -> dict[str, Any]:
    """Reject controller-time rewrites of a released job's budget snapshot.

    Under P0/C-AMC-sem, ``runtime_budget_at_release`` is fixed when a job is
    released.  Global controller updates may change the budget state used by
    future releases, but ``EventRuntimeEngine.apply_budget_updates`` must not
    write that field on active jobs.
    """

    path = Path(source_root).resolve() / "amc_py" / "event_runtime.py"
    if not path.is_file():
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {"code": "DEPLOYED_EVENT_RUNTIME_SOURCE_MISSING"},
        }
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {
                "code": "DEPLOYED_EVENT_RUNTIME_SOURCE_UNREADABLE",
                "detail": str(exc),
            },
        }
    methods = [
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventRuntimeEngine"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "apply_budget_updates"
    ]
    if len(methods) != 1:
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "failure": {
                "code": "APPLY_BUDGET_UPDATES_BINDING_NOT_UNIQUE",
                "count": len(methods),
            },
        }
    writes = _stored_release_budget_attributes(methods[0])
    witness = {
        "source_file": "amc_py/event_runtime.py",
        "source_sha256": sha256_file(path),
        "target_symbol": "EventRuntimeEngine.apply_budget_updates",
        "runtime_budget_at_release_write_count": len(writes),
        "writes": writes,
        "contract": (
            "controller budget updates affect future releases only; active "
            "runtime_budget_at_release snapshots are immutable"
        ),
    }
    if writes:
        return {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "failure": {
                "code": "ACTIVE_RELEASE_SNAPSHOT_MUTATED",
                "write_count": len(writes),
            },
            "witness": witness,
        }
    return {"status": "PASS", "route": None, "failure": None, "witness": witness}


__all__ = ["check_active_release_budget_source_contract"]
