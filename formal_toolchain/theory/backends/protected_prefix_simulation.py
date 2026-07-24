"""Backend for the parameterized protected-prefix simulation receipt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.reference.protected_priority_prefix.observable import observable_schema


class ProtectedPrefixSimulationBackend:
    backend_id = "protected-prefix-simulation-v1"
    REQUIRED_LEMMAS = (
        "TAIL_SERVICE_EXCLUSION", "FINAL_DISPATCH_CORRESPONDENCE",
        "PROTECTED_SERVICE_CORRESPONDENCE", "COMPLETION_REMOVAL_CORRESPONDENCE",
        "DEADLINE_BATCH_CORRESPONDENCE", "ARRIVAL_BATCH_PROJECTION",
        "MODE_TAIL_PHASE_JOIN", "PROTECTED_MACRO_STEP_PRESERVATION",
    )
    SOURCE_FILES = (
        "formal_toolchain/reference/protected_priority_prefix/observable.py",
        "formal_toolchain/reference/protected_priority_prefix/input_projection.py",
        "formal_toolchain/reference/protected_priority_prefix/state_relation.py",
        "formal_toolchain/reference/protected_priority_prefix/macro_step.py",
        "formal_toolchain/reference/protected_priority_prefix/simulation_domain.py",
    )

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}
        if proof.get("quantification") != "forall full execution exists one prefix execution forall natural-number closed boundaries":
            return {"status": "FAIL", "code": "WEAK_SIMULATION_QUANTIFICATION_INVALID"}
        if proof.get("required_lemmas") != list(self.REQUIRED_LEMMAS):
            return {"status": "FAIL", "code": "SIMULATION_LEMMA_SET_INVALID"}
        root = Path(__file__).resolve().parents[3]
        bindings = proof.get("source_bindings", {})
        if set(bindings) != set(self.SOURCE_FILES) or any(bindings[name] != sha256_file(root / name) for name in self.SOURCE_FILES):
            return {"status": "FAIL", "code": "SOURCE_BINDING_MISMATCH"}
        relation_hash = sha256_object(observable_schema())
        if proof.get("protected_observable_schema_hash") != relation_hash:
            return {"status": "FAIL", "code": "PROTECTED_OBSERVABLE_SCHEMA_INVALID"}
        receipt = {"status": "PASS", "backend_id": self.backend_id, "theorem_id": theorem["theorem_id"], "required_lemmas": list(self.REQUIRED_LEMMAS), "source_bindings": dict(bindings), "protected_observable_schema_hash": relation_hash, "quantification": proof["quantification"]}
        receipt["receipt_hash"] = sha256_object(receipt)
        return receipt
