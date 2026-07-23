from __future__ import annotations

import json
from pathlib import Path

from formal_toolchain.core.hashing import sha256_file
from formal_toolchain.theory.backends.finite_hi_bad_prefix import (
    FiniteHIBadPrefixBackend, current_n6_source_bindings, verify_finite_hi_bad_prefix_math,
)

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "formal_toolchain/theory/proofs/FINITE_HI_BAD_PREFIX_REFLECTION.proof.json"
STATEMENT_PATH = ROOT / "formal_toolchain/theory/statements/FINITE_HI_BAD_PREFIX_REFLECTION.json"


def main() -> None:
    math = verify_finite_hi_bad_prefix_math()
    if math.get("status") != "PASS":
        raise SystemExit(f"N6 proof generation failed: {math}")
    statement = json.loads(STATEMENT_PATH.read_text(encoding="utf-8"))
    proof = {"schema_version": "finite_hi_bad_prefix_reflection_proof_v2",
             "theorem_id": "FINITE_HI_BAD_PREFIX_REFLECTION",
             "theorem_statement_hash": statement["statement_hash"],
             "theorem_assumption_hash": statement["assumption_hash"],
             "source_bindings": current_n6_source_bindings(), "solver_backend": "z3",
             "relation_interface": "n6_closed_prefix_relation_interface_v1",
             "proof_scope": "POINTWISE_RELATION_SPECIALIZATION_OVER_FINITE_CLOSED_PREFIXES",
             "solver_obligation_receipts": math["obligations"]}
    PROOF_PATH.write_text(json.dumps(proof, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statement["proof_object"] = {"path": "proofs/FINITE_HI_BAD_PREFIX_REFLECTION.proof.json",
                                  "sha256": sha256_file(PROOF_PATH), "backend": "finite-hi-bad-prefix-z3-v1"}
    STATEMENT_PATH.write_text(json.dumps(statement, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = FiniteHIBadPrefixBackend().verify(PROOF_PATH, theorem=statement)
    if receipt.get("status") != "PASS":
        raise SystemExit(f"fresh backend verification failed: {receipt}")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
