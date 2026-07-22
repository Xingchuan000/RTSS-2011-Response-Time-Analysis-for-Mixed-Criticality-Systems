from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.hashing import sha256_file
from formal_toolchain.theory.backends.reference_prefix_extension import (
    EXPECTED_CASE_IDS, verify_reference_prefix_extension_math,
)

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "formal_toolchain/theory/proofs/REFERENCE_PREFIX_EXTENSION.proof.json"
STATEMENT_PATH = ROOT / "formal_toolchain/theory/statements/REFERENCE_PREFIX_EXTENSION.json"
REF_STATE_PATH = ROOT / "formal_toolchain/reference/reference_state.py"
EXEC_SEM_PATH = ROOT / "formal_toolchain/reference/executable_semantics.py"


def main() -> None:
    math = verify_reference_prefix_extension_math()
    if math.get("status") != "PASS":
        raise SystemExit(f"reference prefix extension proof generation failed: {math}")
    statement = json.loads(STATEMENT_PATH.read_text(encoding="utf-8"))
    proof = {
        "schema_version": "reference_prefix_extension_proof_v3",
        "theorem_id": "REFERENCE_PREFIX_EXTENSION",
        "case_ids": list(EXPECTED_CASE_IDS),
        "source_bindings": {
            "formal_toolchain/reference/reference_state.py": sha256_file(REF_STATE_PATH),
            "formal_toolchain/reference/executable_semantics.py": sha256_file(EXEC_SEM_PATH),
        },
        "theorem_statement_hash": statement["statement_hash"],
        "theorem_assumption_hash": statement["assumption_hash"],
        "solver_backend": "z3",
        "solver_obligation_receipts": math["obligations"],
        "runtime_checked_contracts": [
            "validate_reference_state",
            "_append_generated_event",
            "_checked_successor",
        ],
    }
    PROOF_PATH.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proof_hash = sha256_file(PROOF_PATH)
    statement["proof_object"] = {
        "path": "proofs/REFERENCE_PREFIX_EXTENSION.proof.json",
        "sha256": proof_hash,
        "backend": "reference-prefix-extension-z3-v3",
    }
    STATEMENT_PATH.write_text(json.dumps(statement, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from formal_toolchain.theory.backends.reference_prefix_extension import ReferencePrefixExtensionBackend
    receipt = ReferencePrefixExtensionBackend().verify(PROOF_PATH, theorem=statement)
    if receipt.get("status") != "PASS":
        raise SystemExit(f"fresh backend verification failed: {receipt}")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
