from __future__ import annotations

import ast
from pathlib import Path


def test_formal_and_runtime_packages_do_not_import_lab():
    root = Path(__file__).resolve().parents[3]
    offenders = []
    for package in ("formal_toolchain", "amc_py"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(name == "nonvacuity_lab" or name.startswith("nonvacuity_lab.") for name in names):
                    offenders.append(str(path.relative_to(root)))
    assert offenders == []
