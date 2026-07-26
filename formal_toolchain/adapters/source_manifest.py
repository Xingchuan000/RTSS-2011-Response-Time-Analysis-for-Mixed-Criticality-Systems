"""Build the source manifest used by the formal proof pipeline.

The proof semantic hash is intentionally independent from the mutable
experimental runtime.  It binds the frozen C-AMC-sem/P0 model, the formal
checker implementation, and the deployed policy adapter files.  The current
``amc_py`` runtime (including q-AMC) is recorded only in a non-blocking
implementation-audit hash.
"""

from __future__ import annotations

from pathlib import Path
import ast
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.semantics.frozen_runtime_contract import (
    frozen_contract_files,
    is_mutable_runtime_path,
)

# Kept as the public fixed-target list used by tests and packaging.  Unlike the
# previous list, these are stable proof semantics / policy interfaces rather
# than the shared event runtime.
TARGET_FILES = frozen_contract_files() + (
    "amc_py/rl/actions.py",
    "amc_py/rl/safety.py",
    "amc_py/rl/observation.py",
    "amc_py/rl/feature_state.py",
    "amc_py/rl/feature_config.py",
    "amc_py/viper/fixed_point.py",
    "amc_py/viper/integer_tree.py",
    "amc_py/viper/tree_policy.py",
    "formal_toolchain/core/hashing.py",
    "formal_toolchain/core/registry.py",
    "formal_toolchain/core/contexts.py",
)

# Mutable implementation files are useful diagnostics, but changes here do not
# invalidate the frozen proof semantics.
IMPLEMENTATION_AUDIT_FILES = (
    "amc_py/event_models.py", "amc_py/event_runtime.py", "amc_py/runtime_models.py",
    "amc_py/runtime_scenarios.py", "amc_py/dqn/experiment.py", "amc_py/rl/env.py",
    "amc_py/rl/actions.py", "amc_py/rl/safety.py", "amc_py/rl/observation.py",
    "amc_py/rl/feature_state.py", "amc_py/rl/feature_config.py",
    "amc_py/viper/fixed_point.py", "amc_py/viper/integer_tree.py",
    "amc_py/viper/tree_policy.py",
)

FORMAL_TARGET_FILES = (
    "formal_toolchain/verifier/artifact_verifier.py",
    "formal_toolchain/verifier/aggregator.py",
    "formal_toolchain/verifier/theory_verifier.py",
    "formal_toolchain/binding/action_binding.py",
    "formal_toolchain/binding/observation_binding.py",
    "formal_toolchain/binding/removal_binding.py",
    "formal_toolchain/binding/quantization_binding.py",
    "formal_toolchain/bridge/runtime_branch_map.py",
    "formal_toolchain/bridge/transition_compiler.py",
    "formal_toolchain/bridge/transition_cases.py",
    "formal_toolchain/bridge/handler_decomposition.py",
    "formal_toolchain/bridge/state_relation.py",
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
                candidates = []
                if resolved.is_file():
                    candidates.append(resolved)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        sibling = source_root / (module.replace(".", "/") + "/" + alias.name + ".py")
                        sibling_package = source_root / (module.replace(".", "/") + "/" + alias.name + "/__init__.py")
                        if sibling.is_file():
                            candidates.append(sibling)
                        elif sibling_package.is_file():
                            candidates.append(sibling_package)
                for resolved_path in candidates:
                    item = resolved_path.relative_to(source_root).as_posix()
                    if item not in seen:
                        seen.add(item)
                        queue.append(item)
    return sorted(seen)


def _records(source_root: Path, files: list[str] | tuple[str, ...], *, required: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(dict.fromkeys(files)):
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            if required:
                raise FileNotFoundError(f"目标源码缺失或为符号链接: {relative}")
            continue
        records.append({"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size})
    return records


def _all_formal_toolchain_sources(source_root: Path) -> tuple[str, ...]:
    formal_root = source_root / "formal_toolchain"
    if not formal_root.is_dir():
        return ()
    result = []
    for path in formal_root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".json"}:
            continue
        relative = path.relative_to(source_root).as_posix()
        # Proof outputs are instance artifacts; theorem statements/proofs are
        # still included because they live under the source tree as JSON.
        result.append(relative)
    return tuple(sorted(result))


def build_source_manifest(source_root: Path, *, include_import_closure: bool = True) -> dict[str, Any]:
    source_root = Path(source_root).resolve()

    formal_initial = tuple(TARGET_FILES) + tuple(
        item for item in FORMAL_TARGET_FILES if (source_root / item).is_file()
    )
    # In the real repository bind every formal-toolchain source, but never pull
    # mutable runtime modules into the semantic hash through import closure.
    formal_files = set(formal_initial)
    formal_files.update(_all_formal_toolchain_sources(source_root))
    formal_files = {item for item in formal_files if not is_mutable_runtime_path(item)}
    formal_records = _records(source_root, sorted(formal_files), required=True)

    implementation_initial = tuple(
        item for item in IMPLEMENTATION_AUDIT_FILES if (source_root / item).is_file()
    )
    implementation_files = (
        _import_closure(source_root, implementation_initial)
        if include_import_closure and implementation_initial else list(implementation_initial)
    )
    implementation_records = _records(source_root, implementation_files, required=False)

    manifest = {
        "schema_version": "source_tree_manifest_v2_frozen_formal_semantics",
        "binding_mode": "FROZEN_FORMAL_SEMANTICS",
        "files": formal_records,
        "implementation_audit_files": implementation_records,
        "ignored_patterns": ["__pycache__", ".DS_Store", ".bak_*", "outputs", "logs"],
        "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
    }
    manifest["semantic_hash"] = sha256_object({"binding_mode": manifest["binding_mode"], "files": formal_records})
    manifest["implementation_audit_hash"] = sha256_object({"files": implementation_records})
    return manifest
