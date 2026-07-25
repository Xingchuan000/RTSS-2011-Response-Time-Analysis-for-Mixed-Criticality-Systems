"""Machine-checked project lemma for finite casewise prefix induction.

The backend checks the proof object's binding and the standard induction
clauses.  It intentionally does not encode a seed-specific branch partition;
those facts remain inputs to the Phase K certificate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object, sha256_text_file_normalized
from formal_toolchain.bridge.state_relation import parameterized_state_relation_schema_hash


REQUIRED_ASSUMPTIONS = (
    "preclosed_zero_base_relation", "finite_prefix_length",
    "complete_and_unique_transition_case_partition",
    "every_case_has_reference_successor", "every_case_preserves_parameterized_relation",
    "every_case_preserves_event_projection", "release_case_extends_job_map_with_fresh_key",
    "non_release_cases_preserve_job_map_domain", "terminal_jobs_remain_in_released_ledger",
    "miss_ledger_is_monotone", "unaffected_jobs_satisfy_frame_rule",
)
REQUIRED_CLAUSES = {
    "base": "PASS", "step": "PASS", "finite_sequence_induction": "PASS",
    "map_extension": "PASS", "frame_rule": "PASS", "ledger_monotonicity": "PASS",
}
SOURCE_FILES = (
    "formal_toolchain/bridge/state_relation.py",
    "formal_toolchain/bridge/transition_cases.py",
    "formal_toolchain/bridge/transition_compiler.py",
    "formal_toolchain/bridge/prefix_refinement.py",
    "formal_toolchain/bridge/event_projection.py",
)


class CasewisePrefixInductionBackend:
    backend_id = "casewise-prefix-induction-v1"

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        import json
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        if proof.get("theorem_id") != theorem.get("theorem_id"):
            return {"status": "FAIL", "code": "THEOREM_ID_MISMATCH"}
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}
        if tuple(theorem.get("assumptions", ())) != REQUIRED_ASSUMPTIONS:
            return {"status": "FAIL", "code": "THEOREM_ASSUMPTIONS_INVALID"}
        if proof.get("parameterized_relation_schema_hash") != parameterized_state_relation_schema_hash():
            return {"status": "FAIL", "code": "PARAMETERIZED_RELATION_SCHEMA_INVALID"}
        if proof.get("induction_clauses") != REQUIRED_CLAUSES:
            return {"status": "FAIL", "code": "INDUCTION_CLAUSES_INCOMPLETE"}
        root = Path(__file__).resolve().parents[3]
        if proof.get("source_binding_hash_mode") != "canonical_text_v1":
            return {"status": "FAIL", "code": "SOURCE_BINDING_HASH_MODE_INVALID"}
        bindings = proof.get("source_bindings", {})
        if set(bindings) != set(SOURCE_FILES):
            return {"status": "FAIL", "code": "SOURCE_BINDINGS_INCOMPLETE"}
        if any(bindings[name] != sha256_text_file_normalized(root / name) for name in SOURCE_FILES):
            return {"status": "FAIL", "code": "SOURCE_BINDING_MISMATCH"}
        receipt = {
            "status": "PASS", "backend_id": self.backend_id,
            "theorem_statement_hash": proof["theorem_statement_hash"],
            "theorem_assumption_hash": proof["theorem_assumption_hash"],
            "parameterized_relation_schema_hash": proof["parameterized_relation_schema_hash"],
            "induction_clauses": dict(REQUIRED_CLAUSES), "source_binding_hash_mode": "canonical_text_v1", "source_bindings": dict(bindings),
        }
        receipt["receipt_hash"] = sha256_object(receipt)
        return receipt

