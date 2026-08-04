"""Explicit dispatch plan keeping gradient and integrity paths separate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunKind(StrEnum):
    SEMANTIC_RECOMPILE = "semantic_recompile"
    INTEGRITY_REUSE = "integrity_reuse"
    ENVELOPE_GRADIENT = "envelope_gradient"


@dataclass(frozen=True)
class MutationRunPlan:
    mutation_id: str
    run_kind: RunKind
    run_semantic_path: bool
    run_integrity_path: bool
    run_hout: bool
    require_pair_receipt: bool = False


def build_run_plan(mutation: dict) -> MutationRunPlan:
    mutation_id = str(mutation["mutation_id"])
    mutation_class = str(mutation["mutation_class"])
    if mutation_class in {"ENVELOPE_GRADIENT", "ENVELOPE"}:
        return MutationRunPlan(mutation_id, RunKind.ENVELOPE_GRADIENT, False, False, False)
    if mutation_class.startswith("BUNDLE_") or mutation_class == "SOURCE_BINDING_TAMPER":
        return MutationRunPlan(mutation_id, RunKind.INTEGRITY_REUSE, False, True, False)
    return MutationRunPlan(
        mutation_id, RunKind.SEMANTIC_RECOMPILE, True,
        bool(mutation.get("also_run_old_bundle", False)),
        bool(mutation.get("hout_profile_id")),
        bool(mutation.get("pair_with")),
    )
