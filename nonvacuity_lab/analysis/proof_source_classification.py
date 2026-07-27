"""Conservative classification of experiment mutation targets.

This module is deliberately lab-only.  It does not import, edit, or alter the
formal toolchain; it merely prevents a campaign from treating an audit-only or
freeze-refreshed file as blocking PPP evidence.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath


class ProofSourceClass(StrEnum):
    BLOCKING_POLICY_SOURCE = "BLOCKING_POLICY_SOURCE"
    BLOCKING_ACTION_SEMANTICS = "BLOCKING_ACTION_SEMANTICS"
    BLOCKING_EVENT_SEMANTICS = "BLOCKING_EVENT_SEMANTICS"
    BLOCKING_MODEL_CONTRACT = "BLOCKING_MODEL_CONTRACT"
    NON_BLOCKING_AUDIT_ONLY = "NON_BLOCKING_AUDIT_ONLY"
    DERIVED_AND_REFRESHED = "DERIVED_AND_REFRESHED"
    UNKNOWN = "UNKNOWN"


_DERIVED_NAMES = {
    "effective_runtime_config.json",
    "action_definitions_canonical.json",
}
_AUDIT_ONLY = {
    "amc_py/rl/env.py",
    "amc_py/event_runtime.py",
    "amc_py/qamc/demand.py",
}


def classify_proof_source(target_file: str) -> ProofSourceClass:
    normalized = PurePosixPath(str(target_file).replace("\\", "/")).as_posix().lstrip("./")
    if PurePosixPath(normalized).name in _DERIVED_NAMES:
        return ProofSourceClass.DERIVED_AND_REFRESHED
    if normalized in _AUDIT_ONLY:
        return ProofSourceClass.NON_BLOCKING_AUDIT_ONLY
    if normalized == "amc_py/viper/tree_policy.py":
        return ProofSourceClass.BLOCKING_POLICY_SOURCE
    if normalized.startswith("formal_toolchain/semantics/"):
        name = PurePosixPath(normalized).name.lower()
        if "event" in name or "mode" in name:
            return ProofSourceClass.BLOCKING_EVENT_SEMANTICS
        if "action" in name or "budget" in name:
            return ProofSourceClass.BLOCKING_ACTION_SEMANTICS
        return ProofSourceClass.BLOCKING_MODEL_CONTRACT
    if normalized.startswith("formal_toolchain/adapters/"):
        return ProofSourceClass.BLOCKING_MODEL_CONTRACT
    if normalized.endswith("/target_recipe.json") or normalized == "formal_inputs/target_recipe.json":
        return ProofSourceClass.BLOCKING_MODEL_CONTRACT
    return ProofSourceClass.UNKNOWN


def is_blocking_source(source_class: ProofSourceClass) -> bool:
    return source_class.value.startswith("BLOCKING_")
