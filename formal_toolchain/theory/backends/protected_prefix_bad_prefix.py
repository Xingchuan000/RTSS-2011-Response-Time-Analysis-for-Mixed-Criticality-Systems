"""Backend for protected-prefix HI bad-prefix reflection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


class ProtectedPrefixBadPrefixBackend:
    backend_id = "protected-prefix-bad-prefix-v1"

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}
        required = {"same_job_key", "same_absolute_deadline", "same_actual_demand", "same_service_at_deadline", "same_miss_ledger_membership"}
        rows = proof.get("reflection_fields", {})
        if set(rows) != required or not all(rows.values()) or proof.get("global_mode_equality_required") is not False:
            return {"status": "FAIL", "code": "BAD_PREFIX_REFLECTION_FIELDS_INVALID"}
        receipt = {"status": "PASS", "backend_id": self.backend_id, "theorem_id": theorem["theorem_id"], "reflection_fields": dict(rows), "global_mode_equality_required": False}
        receipt["receipt_hash"] = sha256_object(receipt)
        return receipt
