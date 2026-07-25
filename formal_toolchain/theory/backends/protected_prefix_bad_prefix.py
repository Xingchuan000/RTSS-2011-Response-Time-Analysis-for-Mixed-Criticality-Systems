"""Backend for protected-prefix HI bad-prefix reflection.

The CI must derive every reflection field from the simulation relation; constant
True fields populated without derivation are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


class ProtectedPrefixBadPrefixBackend:
    backend_id = "protected-prefix-bad-prefix-v1"

    REQUIRED_REFLECTION_FIELDS = {
        "job_key", "criticality", "release_time", "absolute_deadline",
        "actual_demand", "service", "completion_state", "miss_ledger",
    }

    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        statement_payload = {key: theorem[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumption_payload = {"theorem_id": theorem["theorem_id"], "assumptions": theorem["assumptions"], "premise_obligation_ids": theorem.get("premise_obligation_ids", []), "version": theorem["version"]}
        if proof.get("theorem_id") != theorem.get("theorem_id") or proof.get("theorem_statement_hash") != sha256_object(statement_payload) or proof.get("theorem_assumption_hash") != sha256_object(assumption_payload):
            return {"status": "FAIL", "code": "THEOREM_HASH_BINDING_INVALID"}


        if proof.get("global_mode_equality_required") is not False:
            return {"status": "FAIL", "code": "GLOBAL_MODE_EQUALITY_IMPROPERLY_REQUIRED"}

        derivation = proof.get("derivation", {})
        if not isinstance(derivation, dict):
            return {"status": "UNRESOLVED",
                    "code": "BAD_PREFIX_REFLECTION_DERIVATION_MISSING",
                    "reason": (
                        "The proof must contain a derivation object that exports each "
                        "reflection field from the simulation relation.  Populating "
                        "constant True values directly in the proof JSON is prohibited."
                    )}

        simulation_receipt = derivation.get("simulation_receipt")
        observable_schema_receipt = derivation.get("observable_schema_receipt")
        deadline_batch_receipt = derivation.get("deadline_batch_receipt")
        if not isinstance(simulation_receipt, dict) or simulation_receipt.get("status") != "PASS":
            return {"status": "UNRESOLVED",
                    "code": "SIMULATION_RECEIPT_NOT_PASS",
                    "reason": "The weak forward simulation receipt must be a verified PASS artifact."}
        if not isinstance(observable_schema_receipt, dict) or observable_schema_receipt.get("status") != "PASS":
            return {"status": "UNRESOLVED",
                    "code": "OBSERVABLE_SCHEMA_RECEIPT_NOT_PASS"}
        if not isinstance(deadline_batch_receipt, dict) or deadline_batch_receipt.get("status") != "PASS":
            return {"status": "UNRESOLVED",
                    "code": "DEADLINE_BATCH_RECEIPT_NOT_PASS"}

        reflection_fields = proof.get("reflection_fields", {})
        if set(reflection_fields) != self.REQUIRED_REFLECTION_FIELDS:
            return {"status": "FAIL", "code": "BAD_PREFIX_REFLECTION_FIELDS_INVALID"}

        field_derivations = derivation.get("field_derivations", {})
        if not isinstance(field_derivations, dict) or set(field_derivations) != self.REQUIRED_REFLECTION_FIELDS:
            return {"status": "UNRESOLVED",
                    "code": "FIELD_DERIVATIONS_MISSING",
                    "reason": (
                        "Each reflection field must be derived from the simulation "
                        "relation with an explicit implication chain, not populated "
                        "as a bare True boolean."
                    ),
                    "expected": sorted(self.REQUIRED_REFLECTION_FIELDS),
                    "actual": sorted(field_derivations) if isinstance(field_derivations, dict) else []}

        for field_name in self.REQUIRED_REFLECTION_FIELDS:
            fd = field_derivations.get(field_name, {})
            if not isinstance(fd, dict):
                return {"status": "UNRESOLVED",
                        "code": f"FIELD_DERIVATION_{field_name.upper()}_MISSING",
                        "reason": f"Field derivation for {field_name} must be a dict with implication steps."}
            if fd.get("derived") is not True:
                return {"status": "UNRESOLVED",
                        "code": f"FIELD_DERIVATION_{field_name.upper()}_NOT_DERIVED",
                        "reason": (
                            f"Field {field_name} has not been derived from the "
                            f"simulation relation; its proof is incomplete."
                        )}
            if not isinstance(fd.get("implication_steps"), list) or len(fd.get("implication_steps", [])) == 0:
                return {"status": "UNRESOLVED",
                        "code": f"FIELD_DERIVATION_{field_name.upper()}_NO_IMPLICATION_STEPS",
                        "reason": f"Field {field_name} must have at least one implication step."}

        earliest = derivation.get("earliest_bad_prefix")
        if not isinstance(earliest, dict) or set(earliest) != {
            "full_hi_job_first_misses_at_deadline", "hi_job_is_protected",
            "deadline_transition_is_observe_only",
            "prefix_incomplete_status_preserved_at_deadline", "construction",
        }:
            return {"status": "UNRESOLVED", "code": "EARLIEST_BAD_PREFIX_DERIVATION_MISSING"}

        # All structural checks above are necessary but not sufficient.  The
        # backend still lacks a code-bound parametric proof kernel, so it must
        # not turn self-asserted JSON receipts into a theorem PASS.
        return {
            "status": "UNRESOLVED",
            "code": "PROTECTED_PREFIX_BAD_PREFIX_SOURCE_BOUND_DERIVATION_REQUIRED",
            "reason": (
                "Static PASS fields and receipt-looking hashes do not prove the "
                "quantified protected-prefix theorem."
            ),
        }
