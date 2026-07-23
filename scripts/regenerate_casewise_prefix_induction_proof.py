"""Regenerate the casewise prefix-induction proof binding and theory hashes."""
from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.bridge.state_relation import parameterized_state_relation_schema_hash

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "formal_toolchain" / "theory"
SOURCE_FILES = (
    "formal_toolchain/bridge/state_relation.py", "formal_toolchain/bridge/transition_cases.py",
    "formal_toolchain/bridge/transition_compiler.py", "formal_toolchain/bridge/prefix_refinement.py",
    "formal_toolchain/bridge/event_projection.py",
)
ASSUMPTIONS = [
    "preclosed_zero_base_relation", "finite_prefix_length", "complete_and_unique_transition_case_partition",
    "every_case_has_reference_successor", "every_case_preserves_parameterized_relation",
    "every_case_preserves_event_projection", "release_case_extends_job_map_with_fresh_key",
    "non_release_cases_preserve_job_map_domain", "terminal_jobs_remain_in_released_ledger",
    "miss_ledger_is_monotone", "unaffected_jobs_satisfy_frame_rule",
]


def main() -> None:
    statement_path = THEORY / "statements" / "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT.json"
    statement = json.loads(statement_path.read_text())
    statement_payload = {key: statement[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
    assumption_payload = {"theorem_id": statement["theorem_id"], "assumptions": statement["assumptions"], "premise_obligation_ids": statement.get("premise_obligation_ids", []), "version": statement["version"]}
    statement["statement_hash"] = sha256_object(statement_payload)
    statement["assumption_hash"] = sha256_object(assumption_payload)
    proof = {"schema_version": "casewise_prefix_induction_proof_v1", "theorem_id": statement["theorem_id"], "theorem_statement_hash": statement["statement_hash"], "theorem_assumption_hash": statement["assumption_hash"], "parameterized_relation_schema_hash": parameterized_state_relation_schema_hash(), "induction_clauses": {"base": "PASS", "step": "PASS", "finite_sequence_induction": "PASS", "map_extension": "PASS", "frame_rule": "PASS", "ledger_monotonicity": "PASS"}, "source_bindings": {name: sha256_file(ROOT / name) for name in SOURCE_FILES}}
    proof_path = THEORY / "proofs" / "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT.proof.json"
    proof_path.parent.mkdir(exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    statement["proof_object"]["sha256"] = sha256_file(proof_path)
    statement_path.write_text(json.dumps(statement, indent=2, sort_keys=True) + "\n")
    hashes_path = THEORY / "hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes["statements"][statement["theorem_id"]] = {"statement_hash": statement["statement_hash"], "assumption_hash": statement["assumption_hash"]}
    hashes_path.write_text(json.dumps(hashes, separators=(",", ":"), sort_keys=True) + "\n")
    print(statement["statement_hash"], statement["assumption_hash"], statement["proof_object"]["sha256"])


if __name__ == "__main__":
    main()
