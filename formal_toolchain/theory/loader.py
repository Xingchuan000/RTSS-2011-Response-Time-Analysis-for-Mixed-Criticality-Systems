"""Minimal source-bound theorem loader for the V10.1 single proof route.

Retired route registries and route-specific theorem backends are absent.  The active V10.1 verifier consumes only the runtime/refinement
lemmas listed in ``RUNTIME_THEOREM_IDS``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file_by_mode, sha256_object
from formal_toolchain.theory.backends.casewise_prefix_induction import CasewisePrefixInductionBackend
from formal_toolchain.theory.backends.finite_hi_bad_prefix import FiniteHIBadPrefixBackend
from formal_toolchain.theory.backends.handler_decomposition import HandlerDecompositionBackend
from formal_toolchain.theory.backends.reference_prefix_extension import ReferencePrefixExtensionBackend


RUNTIME_THEOREM_IDS = (
    "EVENT_HANDLER_MICROSTEP_DECOMPOSITION",
    "ARRIVAL_BATCH_LOOP_DECOMPOSITION",
    "FINITE_RELEASE_FOLD_PRESERVES_RELATION",
    "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",
    "REFERENCE_PREFIX_EXTENSION",
    "FINITE_HI_BAD_PREFIX_REFLECTION",
)

TCB_BACKENDS: dict[str, Any] = {
    "event-handler-decomposition-v1": HandlerDecompositionBackend(
        "event-handler-decomposition-v1",
        ("ast_cfg", "branch_partition", "sequence_composition"),
    ),
    "arrival-batch-decomposition-v1": HandlerDecompositionBackend(
        "arrival-batch-decomposition-v1",
        ("ast_cfg", "finite_fold", "child_cases", "alternative_partition"),
    ),
    "finite-release-fold-v1": HandlerDecompositionBackend(
        "finite-release-fold-v1",
        (
            "base_case", "empty_sequence_case", "head_step", "tail_induction",
            "fresh_extension_composition", "old_domain_frame_composition",
            "ledger_frame_composition",
        ),
    ),
    "casewise-prefix-induction-v1": CasewisePrefixInductionBackend(),
    "reference-prefix-extension-z3-v3": ReferencePrefixExtensionBackend(),
    "finite-hi-bad-prefix-z3-v1": FiniteHIBadPrefixBackend(),
}

N6_MACHINE_PREMISES = (
    "CLOSED_PREFIX_REFINEMENT",
    "REFERENCE_PREFIX_EXTENSION",
    "RELEASE_FIXED_REMOVAL_MAPPING",
    "DEADLINE_OBSERVATION",
    "HI_NONTRUNCATION",
    "EFFECTIVE_EVENT_FRONTIER_RELATION",
    "EARLY_STOP_CONFIGURATION_GATE",
)


def _manifest(theory_dir: Path) -> dict[str, Any]:
    path = theory_dir / "theory_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "v10_1_runtime_theory_manifest_v1":
        raise ValueError("V10_1_RUNTIME_THEORY_MANIFEST_VERSION_INVALID")
    ids = tuple(data.get("required_theorems", ()))
    if ids != RUNTIME_THEOREM_IDS:
        raise ValueError("V10_1_RUNTIME_THEOREM_SET_MISMATCH")
    return data


def load_verified_theory_statement(theory_dir: Path, theorem_id: str) -> dict[str, Any]:
    theory_dir = Path(theory_dir).resolve(strict=True)
    _manifest(theory_dir)
    if theorem_id not in RUNTIME_THEOREM_IDS:
        raise ValueError(f"V10_1_THEOREM_NOT_IN_ACTIVE_RUNTIME_SET:{theorem_id}")

    statement_path = theory_dir / "statements" / f"{theorem_id}.json"
    statement = json.loads(statement_path.read_text(encoding="utf-8"))
    if statement.get("theorem_id") != theorem_id:
        raise ValueError("theory statement id mismatch")

    if theorem_id == "FINITE_HI_BAD_PREFIX_REFLECTION":
        premises = statement.get("premise_obligation_ids")
        if not isinstance(premises, list) or tuple(premises) != N6_MACHINE_PREMISES:
            raise ValueError("FINITE_HI_BAD_PREFIX_MACHINE_PREMISES_INVALID")
        if len(premises) != len(set(premises)):
            raise ValueError("FINITE_HI_BAD_PREFIX_MACHINE_PREMISES_DUPLICATED")

    statement_payload = {
        key: statement[key]
        for key in (
            "theorem_id", "exact_statement", "conclusion", "source_reference",
            "assurance_level", "version",
        )
    }
    assumption_payload = {
        "theorem_id": statement["theorem_id"],
        "assumptions": statement["assumptions"],
        "premise_obligation_ids": statement.get("premise_obligation_ids", []),
        "version": statement["version"],
    }
    if statement["statement_hash"] != sha256_object(statement_payload):
        raise ValueError("theory statement hash mismatch")
    if statement["assumption_hash"] != sha256_object(assumption_payload):
        raise ValueError("theory assumption hash mismatch")

    declared = json.loads((theory_dir / "hashes.json").read_text(encoding="utf-8"))
    if declared.get("schema_version") != "v10_1_runtime_theory_hashes_v1":
        raise ValueError("V10_1_RUNTIME_THEORY_HASH_MANIFEST_VERSION_INVALID")
    expected_hashes = declared.get("statements", {}).get(theorem_id)
    if expected_hashes != {
        "statement_hash": statement["statement_hash"],
        "assumption_hash": statement["assumption_hash"],
    }:
        raise ValueError("theory hashes.json mismatch")

    proof_object = statement.get("proof_object")
    if not isinstance(proof_object, dict):
        raise ValueError(f"V10_1_RUNTIME_THEOREM_PROOF_OBJECT_MISSING:{theorem_id}")
    proof_path = (theory_dir / proof_object["path"]).resolve(strict=True)
    if theory_dir not in proof_path.parents:
        raise ValueError("PROOF_OBJECT_ESCAPES_THEORY_ROOT")
    actual_hash = sha256_file_by_mode(proof_path, proof_object.get("hash_mode", "raw_bytes_v1"))
    if actual_hash != proof_object.get("sha256"):
        raise ValueError(f"THEORY_PROOF_OBJECT_HASH_MISMATCH:{theorem_id}")
    backend_name = str(proof_object.get("backend", ""))
    backend = TCB_BACKENDS.get(backend_name)
    if backend is None:
        raise ValueError(f"V10_1_RUNTIME_THEORY_BACKEND_NOT_AVAILABLE:{backend_name}")
    result = backend.verify(proof_path, theorem=statement)
    if result.get("status") != "PASS":
        raise ValueError(
            f"V10_1_RUNTIME_THEORY_BACKEND_REJECTED:{theorem_id}:"
            f"{result.get('code', 'BACKEND_REJECTED_WITHOUT_CODE')}"
        )
    return statement


def verify_runtime_theory_library(theory_dir: Path) -> dict[str, Any]:
    checked: list[str] = []
    for theorem_id in RUNTIME_THEOREM_IDS:
        try:
            load_verified_theory_statement(theory_dir, theorem_id)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return {
                "status": "FAIL",
                "code": "V10_1_RUNTIME_THEOREM_LOAD_FAILED",
                "theorem_id": theorem_id,
                "message": str(exc),
                "checked_theorems": checked,
            }
        checked.append(theorem_id)
    return {"status": "PASS", "checked_theorems": checked}


__all__ = [
    "RUNTIME_THEOREM_IDS",
    "load_verified_theory_statement",
    "verify_runtime_theory_library",
]
