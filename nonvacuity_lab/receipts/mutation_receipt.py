from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from ..canonical import file_hash


@dataclass(frozen=True)
class MutationReceipt:
    schema_version: str
    mutation_id: str
    mutation_class: str
    semantic_change_count: int
    changed_files: tuple[str, ...]
    diff_sha256: str | None
    coherence_sha256: str | None
    activation_sha256: str | None
    formal_result_sha256: str | None
    hout_result_sha256: str | None
    artifact_class: str = "NONVACUITY_EXPERIMENT_ONLY"
    eligible_for_deployment_claim: bool = False
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def receipt_hash(path: Path | None) -> str | None:
    return None if path is None or not Path(path).is_file() else file_hash(Path(path))


def write_mutation_receipt(path: Path, receipt: MutationReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mutation_receipt(*, mutation_id: str, mutation_class: str, mutation_result: dict, diff_path: Path | None = None, coherence_path: Path | None = None, activation_path: Path | None = None, formal_result_path: Path | None = None, hout_result_path: Path | None = None, metadata: dict | None = None) -> MutationReceipt:
    return MutationReceipt(
        schema_version="nonvacuity_mutation_receipt_v1",
        mutation_id=mutation_id,
        mutation_class=mutation_class,
        semantic_change_count=int(mutation_result.get("semantic_change_count", 0)),
        changed_files=tuple(str(item) for item in mutation_result.get("changed_files", ())),
        diff_sha256=receipt_hash(diff_path), coherence_sha256=receipt_hash(coherence_path),
        activation_sha256=receipt_hash(activation_path), formal_result_sha256=receipt_hash(formal_result_path),
        hout_result_sha256=receipt_hash(hout_result_path), metadata=dict(metadata or {}),
    )
