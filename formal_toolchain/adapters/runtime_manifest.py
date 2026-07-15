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
    names = ("numpy", "scikit-learn", "z3-solver", "jsonschema", "setuptools")
    return {"schema_version": "dependency_manifest_v1",
            "packages": {name: _version(name) for name in names}}


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


def check_dependency_policy(manifest: dict[str, Any], required: tuple[str, ...] =
                            ("numpy", "scikit-learn", "z3-solver", "jsonschema", "setuptools")) -> str:
    """关键依赖缺失时只能返回 INVALID/UNRESOLVED，不能静默继续。"""
    packages = manifest.get("packages", {})
    return "PASS" if all(packages.get(name) for name in required) else "UNRESOLVED"
