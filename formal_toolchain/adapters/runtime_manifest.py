"""运行时、依赖与 checker 版本 manifest。"""

from __future__ import annotations

import decimal
import importlib.metadata
import locale
import platform
import sys
import sysconfig
import time
from pathlib import Path
from formal_toolchain.core.hashing import sha256_file, sha256_object
from typing import Any


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_runtime_environment_manifest() -> dict[str, Any]:
    """导出会影响解释/序列化环境的值；平台信息仅作为诊断字段。"""
    return {
        "schema_version": "runtime_environment_manifest_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        # 证明对象禁止 JSON float；这里保留精确的 Python repr 字符串。
        "float_info": {"max": repr(sys.float_info.max), "epsilon": repr(sys.float_info.epsilon),
                        "mant_dig": sys.float_info.mant_dig},
        "decimal_context": {"prec": decimal.getcontext().prec,
                             "rounding": decimal.getcontext().rounding},
        "platform_diagnostic": platform.platform(),
        "locale": locale.setlocale(locale.LC_ALL),
        "timezone": time.tzname,
        "source_encoding": "UTF-8",
    }


def build_dependency_manifest() -> dict[str, Any]:
    import sys
    names = ("numpy", "scikit-learn", "z3-solver", "jsonschema", "setuptools")
    pv = sys.version_info
    return {"schema_version": "dependency_manifest_v1",
            "python_version_info": f"{pv.major}.{pv.minor}.{pv.micro}",
            "packages": {name: _version(name) for name in names}}


def _version_constraint_matches(actual: str, constraint: str) -> bool:
    """检查实际版本是否满足 lock 中的精确版本或 PEP 440 约束。

    支持：
    - ``4.13.4.0``：精确版本；
    - ``>=4.13,<5``：兼容版本范围；
    - ``3.11.x``：同一 major/minor 的通配形式。
    """
    if not actual or not constraint:
        return False

    if constraint.endswith(".x"):
        prefix = constraint[:-2]
        actual_parts = actual.split(".")
        prefix_parts = prefix.split(".")
        return actual_parts[: len(prefix_parts)] == prefix_parts

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        has_operator = any(
            token in constraint
            for token in ("<", ">", "=", "!", "~")
        )
        spec_text = constraint if has_operator else f"=={constraint}"
        return Version(actual) in SpecifierSet(spec_text)
    except (ImportError, ValueError):
        # packaging 是 formal 环境的显式依赖；这里只保留测试环境的保守回退。
        return actual == constraint


def _python_lock_matches(actual: str, locked: str) -> bool:
    return _version_constraint_matches(actual, locked)


def build_checker_version_manifest(source_root: Path | None = None) -> dict[str, Any]:
    result = {"schema_version": "checker_version_manifest_v1",
            "formal_toolchain_version": "0.1.0-phase-cde", "python_ast_ir_version": "phase-e-v2",
            "schema_versions": {"common_certificate": "common_certificate_v1",
                                 "p0_profile": "p0_profile_v1"}}
    if source_root is not None:
        root = Path(source_root)
        files = []
        for path in sorted((root / "formal_toolchain").rglob("*.py")):
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
        result["checker_source_files"] = files
        result["checker_build_hash"] = sha256_object(files)
    return result


def check_dependency_policy(
    manifest: dict[str, Any],
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """关键依赖缺失或版本约束不满足时返回 FAIL，不能静默继续。

    proof_dependency_lock.json 可以使用精确版本，也可以使用 PEP 440
    兼容范围。没有 lock 文件时降级为只检查存在性（用于测试环境），
    但正式 verifier 必须提供 lock。
    """
    packages = manifest.get("packages", {})
    if lock is not None:
        expected = lock.get("packages", {})
        mismatches = {}
        for name, expected_version in expected.items():
            actual = packages.get(name)
            if not isinstance(actual, str) or not _version_constraint_matches(
                actual, str(expected_version)
            ):
                mismatches[name] = {
                    "expected": expected_version,
                    "actual": actual,
                }
        python_actual = manifest.get("python_version_info", "")
        python_locked = lock.get("python", "")
        if python_locked and not _python_lock_matches(python_actual, python_locked):
            mismatches["python"] = {"expected": python_locked, "actual": python_actual}
        if mismatches:
            return {
                "status": "FAIL",
                "route": "PROOF_BUNDLE_INVALID",
                "code": "PROOF_DEPENDENCY_LOCK_MISMATCH",
                "mismatches": mismatches,
            }
        return {"status": "PASS", "locked_packages": expected,
                "python_locked": python_locked, "python_actual": python_actual}
    required = ("numpy", "scikit-learn", "z3-solver", "jsonschema", "setuptools")
    missing = [name for name in required if not packages.get(name)]
    if missing:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "DEPENDENCY_LOCK_INCOMPLETE", "missing": missing}
    return {"status": "PASS", "locked_packages": {name: packages.get(name) for name in required}}
