"""Backend for the parameterized protected-prefix simulation receipt.

This backend requires actual mathematical proof objects, not lemma-name checks
or source-hash bindings.  A proof must contain predecessor receipts, dependency
hashes, base-case/induction receipts, and a verified single-witness quantifier.
"""

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
    REQUIRED_DEPENDENCIES = (
        "full_input_projection_theorem",
        "prefix_execution_existence_theorem",
        "base_case",
        "macro_step_induction",
        "single_witness_compatibility",
    )

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}


        quantified = proof.get("quantification", "")
        if quantified != "forall full execution exists one prefix execution forall natural-number closed boundaries":
            return {"status": "FAIL", "code": "WEAK_SIMULATION_QUANTIFICATION_INVALID"}
        if proof.get("quantifier_order") != "forall-full-exists-one-prefix-forall-boundaries":
            return {"status": "FAIL", "code": "QUANTIFIER_ORDER_UNVERIFIED"}

        if proof.get("required_lemmas") != list(self.REQUIRED_LEMMAS):
            return {"status": "FAIL", "code": "SIMULATION_LEMMA_SET_INVALID"}

        relation_hash = sha256_object(observable_schema())
        if proof.get("protected_observable_schema_hash") != relation_hash:
            return {"status": "FAIL", "code": "PROTECTED_OBSERVABLE_SCHEMA_INVALID"}

        dependencies = proof.get("dependencies", {})
        if not isinstance(dependencies, dict) or set(dependencies) != set(self.REQUIRED_DEPENDENCIES):
            return {"status": "UNRESOLVED",
                    "code": "SIMULATION_DEPENDENCY_RECEIPTS_MISSING",
                    "reason": (
                        "The proof object must contain dependency receipts for: "
                        "full_input_projection_theorem, prefix_execution_existence_theorem, "
                        "base_case, macro_step_induction, single_witness_compatibility. "
                        "Each receipt must be a fresh-verifier-recognised PASS artifact."
                    ),
                    "expected": list(self.REQUIRED_DEPENDENCIES),
                    "actual": sorted(dependencies) if isinstance(dependencies, dict) else []}

        for dep_name, dep_receipt in dependencies.items():
            if not isinstance(dep_receipt, dict) or dep_receipt.get("status") != "PASS":
                return {"status": "UNRESOLVED",
                        "code": f"SIMULATION_DEPENDENCY_{dep_name.upper()}_NOT_PASS",
                        "reason": f"Dependency {dep_name} must be a PASS artifact."}

        single_witness = proof.get("single_witness_compatibility")
        if not isinstance(single_witness, dict) or single_witness.get("status") != "PASS":
            return {"status": "UNRESOLVED",
                    "code": "SINGLE_WITNESS_COMPATIBILITY_UNVERIFIED",
                    "reason": (
                        "The proof must verify that all finite induction prefixes "
                        "originate from the same prefix oracle and the same execution; "
                        "re-selecting a witness per horizon is prohibited."
                    )}

        base_case = dependencies.get("base_case", {})
        induction = dependencies.get("macro_step_induction", {})
        if (isinstance(base_case, dict) and isinstance(induction, dict)
                and base_case.get("relation_schema_hash") != induction.get("relation_schema_hash")):
            return {"status": "FAIL",
                    "code": "BASE_INDUCTION_RELATION_SCHEMA_MISMATCH",
                    "reason": "Base case and induction conclusion must use the same relation schema."}

        full_fp = proof.get("full_taskset_fingerprint")
        prefix_fp = proof.get("prefix_taskset_fingerprint")
        if not isinstance(full_fp, str) or not isinstance(prefix_fp, str):
            return {"status": "UNRESOLVED",
                    "code": "TASKSET_FINGERPRINTS_MISSING",
                    "reason": "The proof must bind full and prefix taskset fingerprints."}

        # All structural checks above are necessary but not sufficient.  The
        # backend still lacks a code-bound parametric proof kernel, so it must
        # not turn self-asserted JSON receipts into a theorem PASS.
        return {
            "status": "UNRESOLVED",
            "code": "PROTECTED_PREFIX_SIMULATION_SOURCE_BOUND_RECEIPTS_REQUIRED",
            "reason": (
                "Static PASS fields and receipt-looking hashes do not prove the "
                "quantified protected-prefix theorem."
            ),
        }
