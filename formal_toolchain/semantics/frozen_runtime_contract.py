"""Frozen C-AMC-sem/P0 semantics used by the formal proof pipeline.

The proof toolchain deliberately binds to these files rather than to the shared
experimental runtime.  Runtime implementations (including q-AMC branches) may
change without invalidating the already-defined C-AMC-sem proof semantics.

The resulting theorem certifies the frozen semantics plus the exported target
parameters.  Conformance of a mutable runtime implementation is a separate,
non-blocking audit concern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from formal_toolchain.core.hashing import sha256_file, sha256_object

FROZEN_EVENT_RUNTIME = "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
FROZEN_EVENT_MODELS = "formal_toolchain/semantics/frozen_c_amc_sem_event_models.py"
FROZEN_RUNTIME_WRAPPER = "formal_toolchain/semantics/frozen_c_amc_sem_runtime_wrapper.py"
FROZEN_ACTION_RUNTIME = "formal_toolchain/semantics/frozen_c_amc_sem_action_runtime.py"
FROZEN_OBSERVATION_RUNTIME = "formal_toolchain/semantics/frozen_c_amc_sem_observation.py"
CONTRACT_VERSION = "c_amc_sem_p0_frozen_runtime_v2"


def frozen_event_runtime_path(source_root: str | Path) -> Path:
    return Path(source_root) / FROZEN_EVENT_RUNTIME


def frozen_event_models_path(source_root: str | Path) -> Path:
    return Path(source_root) / FROZEN_EVENT_MODELS


def frozen_runtime_wrapper_path(source_root: str | Path) -> Path:
    return Path(source_root) / FROZEN_RUNTIME_WRAPPER


def frozen_action_runtime_path(source_root: str | Path) -> Path:
    return Path(source_root) / FROZEN_ACTION_RUNTIME


def frozen_observation_runtime_path(source_root: str | Path) -> Path:
    return Path(source_root) / FROZEN_OBSERVATION_RUNTIME


def frozen_contract_files() -> tuple[str, ...]:
    return (
        FROZEN_EVENT_RUNTIME,
        FROZEN_EVENT_MODELS,
        FROZEN_RUNTIME_WRAPPER,
        FROZEN_ACTION_RUNTIME,
        FROZEN_OBSERVATION_RUNTIME,
    )


def frozen_contract_manifest(source_root: str | Path) -> dict[str, object]:
    root = Path(source_root)
    records = [
        {"path": relative, "sha256": sha256_file(root / relative),
         "size": (root / relative).stat().st_size}
        for relative in frozen_contract_files()
    ]
    return {
        "schema_version": "frozen_runtime_contract_manifest_v2",
        "contract_version": CONTRACT_VERSION,
        "files": records,
        "semantic_hash": sha256_object({
            "contract_version": CONTRACT_VERSION,
            "files": records,
        }),
    }


def is_mutable_runtime_path(relative: str) -> bool:
    normalized = str(relative).replace("\\", "/")
    return (
        normalized == "amc_py/event_runtime.py"
        or normalized == "amc_py/event_models.py"
        or normalized == "amc_py/runtime_models.py"
        or normalized == "amc_py/runtime_scenarios.py"
        or normalized == "amc_py/rl/env.py"
        or normalized == "amc_py/rl/actions.py"
        or normalized == "amc_py/rl/safety.py"
        or normalized == "amc_py/rl/observation.py"
        or normalized == "amc_py/rl/feature_state.py"
        or normalized == "amc_py/rl/feature_config.py"
        or normalized.startswith("amc_py/qamc/")
        or normalized.startswith("tests/test_qamc")
    )


def filter_formal_semantic_files(paths: Iterable[str]) -> tuple[str, ...]:
    """Remove mutable experimental-runtime paths from proof semantic binding."""
    return tuple(sorted({
        str(path).replace("\\", "/")
        for path in paths
        if not is_mutable_runtime_path(str(path))
    }))
