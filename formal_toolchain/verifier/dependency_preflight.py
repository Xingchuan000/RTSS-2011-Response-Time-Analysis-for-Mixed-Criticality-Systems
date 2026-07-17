"""形式化依赖的快速预检。"""

from __future__ import annotations


def check_formal_dependencies():
    missing = []
    try:
        import z3  # noqa: F401
    except ImportError:
        missing.append("z3-solver")
    return {"status": "PASS" if not missing else "UNRESOLVED", "missing": missing}

