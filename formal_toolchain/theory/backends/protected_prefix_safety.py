"""Backend for the prefix-schedulability to full-reference-HI-safety contradiction.

The certificate must be constructed from verified predecessor receipts, not from
a three-step text list.  Every ingredient of the contradiction argument must have
an independently verifiable hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


class ProtectedPrefixSafetyBackend:
    backend_id = "protected-prefix-safety-v1"

    REQUIRED_COMPONENTS = (
        "prefix_model_conformance_hash",
        "prefix_all_task_rta_hash",
        "imported_theorem_receipt_hash",
        "weak_simulation_hash",
        "bad_prefix_reflection_hash",
    )
    REQUIRED_CONCLUSION = "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES"
    REQUIRED_PREDECESSORS = (
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
    )

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}


        if proof.get("conclusion") != self.REQUIRED_CONCLUSION:
            return {"status": "FAIL", "code": "PREFIX_SAFETY_CONCLUSION_INVALID"}

        components = proof.get("components", {})
        if not isinstance(components, dict):
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_COMPOSITION_COMPONENTS_MISSING",
                    "reason": (
                        "The proof must contain a components dict with independently "
                        "verifiable receipt hashes for each ingredient of the "
                        "contradiction argument."
                    )}

        missing = set(self.REQUIRED_COMPONENTS) - set(components)
        if missing:
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_COMPOSITION_COMPONENT_HASHES_MISSING",
                    "expected": list(self.REQUIRED_COMPONENTS),
                    "missing": sorted(missing),
                    "reason": (
                        "Each component receipt hash must be provided to construct "
                        "the contradiction chain."
                    )}

        for comp in self.REQUIRED_COMPONENTS:
            value = components.get(comp)
            if not isinstance(value, str) or len(value) != 64:
                return {"status": "UNRESOLVED",
                        "code": f"SAFETY_COMPOSITION_{comp.upper()}_INVALID",
                        "reason": f"Component {comp} must be a 64-char hex receipt hash."}

        if proof.get("contradiction_steps") != [
            "full reference HI miss",
            "reflected prefix HI miss",
            "prefix all-task schedulability contradiction",
        ]:
            return {"status": "FAIL", "code": "PREFIX_SAFETY_CONTRADICTION_INVALID"}

        full_fp = proof.get("full_taskset_fingerprint")
        prefix_fp = proof.get("prefix_taskset_fingerprint")
        if not isinstance(full_fp, str) or not isinstance(prefix_fp, str):
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_TASKSET_FINGERPRINTS_MISSING",
                    "reason": (
                        "The conclusion must bind full and prefix taskset fingerprints "
                        "to prevent conclusion reuse across different task sets."
                    )}

        if proof.get("conclusion_scope") not in ("ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",):
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_CONCLUSION_SCOPE_UNVERIFIED",
                    "reason": (
                        "The conclusion must only claim full-reference HI safety; "
                        "it must NOT extend to tail LO safety."
                    )}

        if proof.get("proof_partition") != ["PP7-A1", "PP7-A2", "PP7-B", "PP8"]:
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_COMPOSITION_PROOF_PARTITION_UNVERIFIED"}
        predecessor_ids = proof.get("predecessor_theorem_ids")
        predecessor_hashes = proof.get("predecessor_receipt_hashes")
        if predecessor_ids != list(self.REQUIRED_PREDECESSORS) or not isinstance(predecessor_hashes, dict):
            return {"status": "UNRESOLVED",
                    "code": "SAFETY_COMPOSITION_PREDECESSOR_RECEIPTS_MISSING"}
        for predecessor in self.REQUIRED_PREDECESSORS:
            value = predecessor_hashes.get(predecessor)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                return {"status": "UNRESOLVED",
                        "code": "SAFETY_COMPOSITION_PREDECESSOR_RECEIPT_INVALID"}

        kernel = proof.get("proof_kernel_receipt")
        kernel_ok = (
            isinstance(kernel, dict)
            and kernel.get("status") == "PASS"
            and kernel.get("theorem_id") == "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX"
            and kernel.get("source_bound") is True
            and kernel.get("contradiction_proved") is True
            and kernel.get("predecessor_receipt_hashes") == predecessor_hashes
        )
        if kernel_ok and proof.get("source_bound") is True:
            return {
                "status": "PASS",
                "backend_id": self.backend_id,
                "theorem_id": theorem.get("theorem_id"),
                "proof_kernel_receipt_hash": sha256_object(kernel),
            }

        # Static fields and receipt-shaped hashes without the source-bound
        # contradiction kernel remain unresolved.
        return {
            "status": "UNRESOLVED",
            "code": "PROTECTED_PREFIX_SAFETY_SOURCE_BOUND_COMPOSITION_REQUIRED",
            "reason": (
                "Static PASS fields and receipt-looking hashes do not prove the "
                "quantified protected-prefix theorem."
            ),
        }
