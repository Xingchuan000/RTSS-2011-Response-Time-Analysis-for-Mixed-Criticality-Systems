"""两阶段 outer-root 的 immutable leaf set。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True)
class RootLeafSet:
    component_context_hashes: Mapping[str, str]
    certificate_hashes: Mapping[str, str]
    status_evidence_hashes: Mapping[str, str]
    independent_verification_payload_hash: str
    active_obligation_set: tuple[str, ...]
    claim_request: Mapping[str, Any]

    def preimage(self) -> dict[str, Any]:
        return {
            "schema_version": "outer_bundle_root_v3",
            "component_context_hashes": dict(sorted(self.component_context_hashes.items())),
            "verified_obligation_artifact_hashes": dict(sorted(self.certificate_hashes.items())),
            "status_evidence_hashes": dict(sorted(self.status_evidence_hashes.items())),
            "independent_verification_payload_hash": self.independent_verification_payload_hash,
            "active_obligation_set": list(self.active_obligation_set),
            "claim_request": dict(self.claim_request),
        }

    def root(self) -> str:
        return sha256_object(self.preimage())

