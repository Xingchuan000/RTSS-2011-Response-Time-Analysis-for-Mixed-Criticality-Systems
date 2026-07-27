"""Mutation protocol and receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class MutationContext:
    mutation_id: str
    source_root: Path
    mutated_seed: Path | None
    source_overlay: Path | None
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class PreflightResult:
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationResult:
    status: str
    before_hash: str
    after_hash: str
    changed_files: tuple[str, ...]
    changed_pointers: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    semantic_change_count: int = 0
    parser_validation: str = "PASS"
    artifact_manifest_validation: str = "NOT_APPLICABLE"
    rollback_information: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mutation_result_v1",
            "status": self.status,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "changed_files": list(self.changed_files),
            "changed_pointers": list(self.changed_pointers),
            "changed_symbols": list(self.changed_symbols),
            "semantic_change_count": self.semantic_change_count,
            "parser_validation": self.parser_validation,
            "artifact_manifest_validation": self.artifact_manifest_validation,
            "rollback_information": dict(self.rollback_information),
            "details": dict(self.details),
        }


class Mutation(Protocol):
    def preflight(self, context: MutationContext) -> PreflightResult: ...

    def apply(self, context: MutationContext) -> MutationResult: ...

    def verify_single_change(self, result: MutationResult) -> PreflightResult: ...
