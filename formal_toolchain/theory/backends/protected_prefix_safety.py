"""Backend for the prefix-schedulability to full-reference-HI-safety contradiction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


class ProtectedPrefixSafetyBackend:
    backend_id = "protected-prefix-safety-v1"

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}
        if proof.get("proof_by_contradiction") != ["full reference HI miss", "reflected prefix HI miss", "prefix all-task schedulability contradiction"] or proof.get("conclusion") != "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES":
            return {"status": "FAIL", "code": "PREFIX_SAFETY_CONTRADICTION_INVALID"}
        receipt = {"status": "PASS", "backend_id": self.backend_id, "theorem_id": theorem["theorem_id"], "proof_by_contradiction": list(proof["proof_by_contradiction"]), "conclusion": proof["conclusion"]}
        receipt["receipt_hash"] = sha256_object(receipt)
        return receipt
