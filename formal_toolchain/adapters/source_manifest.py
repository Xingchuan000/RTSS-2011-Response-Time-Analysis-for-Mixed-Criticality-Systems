"""生成被证明目标实现的源码 manifest。

manifest 只记录明确列入计划的目标源码及其 SHA-256。缓存、备份、日志和输出
永远不会进入 semantic hash；缺失的目标文件则立即失败，不能静默跳过。
"""

from __future__ import annotations

from pathlib import Path
import ast
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object

TARGET_FILES = (
    "amc_py/event_models.py", "amc_py/event_runtime.py", "amc_py/runtime_models.py",
    "amc_py/runtime_scenarios.py", "amc_py/dqn/experiment.py", "amc_py/rl/env.py",
    "amc_py/rl/actions.py", "amc_py/rl/safety.py", "amc_py/rl/observation.py",
    "amc_py/rl/feature_state.py", "amc_py/rl/feature_config.py",
    "amc_py/viper/fixed_point.py", "amc_py/viper/integer_tree.py",
    "amc_py/viper/tree_policy.py",
)

FORMAL_TARGET_FILES = (
    "formal_toolchain/core/hashing.py", "formal_toolchain/core/registry.py",
    "formal_toolchain/core/contexts.py", "formal_toolchain/verifier/artifact_verifier.py",
    "formal_toolchain/verifier/aggregator.py", "formal_toolchain/verifier/theory_verifier.py",
    "formal_toolchain/binding/action_binding.py", "formal_toolchain/binding/observation_binding.py",
    "formal_toolchain/binding/removal_binding.py", "formal_toolchain/binding/quantization_binding.py",
)


def _import_closure(source_root: Path, initial: tuple[str, ...]) -> list[str]:
    seen = set(initial)
    queue = list(initial)
    while queue:
        relative = queue.pop(0)
        path = source_root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package_parts = relative.split("/")[:-1]
                    base = package_parts[: max(0, len(package_parts) - node.level + 1)]
                    module = ".".join(base + ([module] if module else []))
                modules = [module]
            for module in modules:
                if not (module.startswith("amc_py") or module.startswith("formal_toolchain")):
                    continue
                candidate = source_root / (module.replace(".", "/") + ".py")
                package = source_root / module.replace(".", "/") / "__init__.py"
                resolved = candidate if candidate.is_file() else package
                if not resolved.is_file() and isinstance(node, ast.ImportFrom):
                    # ``from . import sibling`` and ``from ..x import y`` need
                    # symbol-level resolution in addition to module resolution.
                    for alias in node.names:
                        sibling = source_root / (module.replace(".", "/") + "/" + alias.name + ".py")
                        sibling_package = source_root / (module.replace(".", "/") + "/" + alias.name + "/__init__.py")
                        resolved = sibling if sibling.is_file() else sibling_package
                        if resolved.is_file():
                            item = resolved.relative_to(source_root).as_posix()
                            if item not in seen:
                                seen.add(item)
                                queue.append(item)
                    continue
                if resolved.is_file():
                    item = resolved.relative_to(source_root).as_posix()
                    if item not in seen:
                        seen.add(item)
                        queue.append(item)
    return sorted(seen)


def build_source_manifest(source_root: Path, *, include_import_closure: bool = True) -> dict[str, Any]:
    """读取固定目标文件并返回确定性 source tree manifest。"""
    source_root = Path(source_root).resolve()
    # 形式化 checker 文件在正式工程根目录存在时纳入闭包；测试中的最小
    # amc_py 副本没有这些文件时，不把它们误当成目标源码缺失。
    initial = TARGET_FILES + tuple(item for item in FORMAL_TARGET_FILES if (source_root / item).is_file())
    files = _import_closure(source_root, initial) if include_import_closure else list(initial)
    records = []
    for relative in files:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"目标源码缺失或为符号链接: {relative}")
        records.append({"path": relative, "sha256": sha256_file(path),
                        "size": path.stat().st_size})
    manifest = {"schema_version": "source_tree_manifest_v1", "files": records,
                "ignored_patterns": ["__pycache__", ".DS_Store", ".bak_*", "outputs", "logs"]}
    manifest["semantic_hash"] = sha256_object({"files": records})
    return manifest
