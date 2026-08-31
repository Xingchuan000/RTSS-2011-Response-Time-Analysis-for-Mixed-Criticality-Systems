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
import json
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
    # The mutable RL environment/actions/observation stack is audited below,
    # but is not part of the blocking C-AMC-sem/P0 semantic hash.
    "amc_py/viper/fixed_point.py",
    "amc_py/viper/integer_tree.py",
    "amc_py/viper/tree_policy.py",
    "formal_toolchain/core/hashing.py",
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
    # V10.1 terminal route. Retired terminal implementations are not part of
    # the PASS DAG or the semantic source root.
    "formal_toolchain/v10_1/verifier.py",
    "formal_toolchain/v10_1/base_refinement.py",
    "formal_toolchain/v10_1/base_section4_1.py",
    "formal_toolchain/v10_1/controller_macro.py",
    "formal_toolchain/v10_1/completion_certificates.py",
    "formal_toolchain/v10_1/carry_in_envelope.py",
    "formal_toolchain/v10_1/feature_transfer.py",
    "formal_toolchain/v10_1/safe_prefix.py",
    "formal_toolchain/v10_1/pcssc.py",
    "formal_toolchain/v10_1/bindings.py",
    "formal_toolchain/v10_1/constants.py",
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


def _active_theory_artifacts(source_root: Path) -> list[str]:
    """Return the non-Python theorem artifacts consumed by V10.1 conformance.

    The import closure already binds the loader/backends/reference Python code.
    These JSON files are equally normative proof dependencies, so bind exactly
    the manifest, declared hashes, active statements, and their proof objects.
    Unused theory files are intentionally not swept into the semantic root.
    """

    theory_root = source_root / "formal_toolchain" / "theory"
    manifest_path = theory_root / "theory_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [
        "formal_toolchain/theory/theory_manifest.json",
        "formal_toolchain/theory/hashes.json",
    ]
    for theorem_id in manifest.get("required_theorems", []):
        statement_rel = f"formal_toolchain/theory/statements/{theorem_id}.json"
        statement = json.loads((source_root / statement_rel).read_text(encoding="utf-8"))
        proof = statement.get("proof_object")
        if not isinstance(proof, dict) or not isinstance(proof.get("path"), str):
            raise ValueError(f"V10_1_THEORY_PROOF_OBJECT_BINDING_INVALID:{theorem_id}")
        proof_path = (theory_root / proof["path"]).resolve()
        if theory_root.resolve() not in proof_path.parents:
            raise ValueError(f"V10_1_THEORY_PROOF_OBJECT_ESCAPES_ROOT:{theorem_id}")
        files.append(statement_rel)
        files.append(proof_path.relative_to(source_root).as_posix())
    return sorted(dict.fromkeys(files))


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


def build_source_manifest(source_root: Path, *, include_import_closure: bool = True) -> dict[str, Any]:
    source_root = Path(source_root).resolve()

    formal_initial = tuple(TARGET_FILES) + tuple(
        item for item in FORMAL_TARGET_FILES if (source_root / item).is_file()
    )
    # Bind the active proof implementation and its transitive Python import
    # closure only.  Retired historical/differential terminals are deliberately
    # outside the V10.1 semantic hash.
    # This is a proof dependency binding, not a repository anti-tamper hash.
    formal_files = (
        _import_closure(source_root, formal_initial)
        if include_import_closure else list(formal_initial)
    )
    formal_files = [item for item in formal_files if not is_mutable_runtime_path(item)]
    formal_files.extend(_active_theory_artifacts(source_root))
    formal_files = sorted(dict.fromkeys(formal_files))
    formal_records = _records(source_root, formal_files, required=True)

    implementation_initial = tuple(
        item for item in IMPLEMENTATION_AUDIT_FILES if (source_root / item).is_file()
    )
    implementation_files = (
        _import_closure(source_root, implementation_initial)
        if include_import_closure and implementation_initial else list(implementation_initial)
    )
    implementation_records = _records(source_root, implementation_files, required=False)

    manifest = {
        "schema_version": "source_tree_manifest_v3_active_proof_closure",
        "binding_mode": "FROZEN_FORMAL_SEMANTICS",
        "files": formal_records,
        "implementation_audit_files": implementation_records,
        "ignored_patterns": ["__pycache__", ".DS_Store", ".bak_*", "outputs", "logs"],
        "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
    }
    manifest["semantic_hash"] = sha256_object({"binding_mode": manifest["binding_mode"], "files": formal_records})
    manifest["implementation_audit_hash"] = sha256_object({"files": implementation_records})
    return manifest
