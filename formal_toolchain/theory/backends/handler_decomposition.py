from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object


SOURCE_FILES = (
    "amc_py/event_runtime.py",
    "formal_toolchain/bridge/handler_decomposition.py",
    "formal_toolchain/bridge/transition_cases.py",
    "formal_toolchain/bridge/transition_compiler.py",
    "formal_toolchain/bridge/state_relation.py",
)


class HandlerDecompositionBackend:
    def __init__(self, backend_id: str, required_clauses: tuple[str, ...]):
        self.backend_id = backend_id
        self.required_clauses = required_clauses

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        if proof.get("theorem_id") != theorem.get("theorem_id"):
            return {"status": "FAIL", "code": "THEOREM_ID_MISMATCH"}
        payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumptions = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_statement_hash") != sha256_object(payload) or proof.get("theorem_assumption_hash") != sha256_object(assumptions):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}
        root = Path(__file__).resolve().parents[3]
        bindings = proof.get("source_bindings", {})
        if set(bindings) != set(SOURCE_FILES) or any(bindings[name] != sha256_file(root / name) for name in SOURCE_FILES):
            return {"status": "FAIL", "code": "SOURCE_BINDING_MISMATCH"}
        clauses = proof.get("clauses", {})
        if any(clauses.get(name) != "PASS" for name in self.required_clauses):
            return {"status": "FAIL", "code": "DECOMPOSITION_PROOF_CLAUSES_INCOMPLETE"}
        receipt = {"status": "PASS", "backend_id": self.backend_id, "theorem_statement_hash": proof["theorem_statement_hash"], "theorem_assumption_hash": proof["theorem_assumption_hash"], "source_bindings": bindings, "clauses": clauses}
        receipt["receipt_hash"] = sha256_object(receipt)
        return receipt

