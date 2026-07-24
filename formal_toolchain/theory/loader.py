from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object


from formal_toolchain.theory.backends.reference_prefix_extension import (
    ReferencePrefixExtensionBackend,
)
from formal_toolchain.theory.backends.finite_hi_bad_prefix import FiniteHIBadPrefixBackend
from formal_toolchain.theory.backends.casewise_prefix_induction import CasewisePrefixInductionBackend
from formal_toolchain.theory.backends.handler_decomposition import HandlerDecompositionBackend
from formal_toolchain.theory.backends.protected_prefix_simulation import ProtectedPrefixSimulationBackend
from formal_toolchain.theory.backends.protected_prefix_bad_prefix import ProtectedPrefixBadPrefixBackend
from formal_toolchain.theory.backends.protected_prefix_safety import ProtectedPrefixSafetyBackend

TCB_BACKENDS: dict[str, Any] = {
    "reference-prefix-extension-z3-v3": ReferencePrefixExtensionBackend(),
    "finite-hi-bad-prefix-z3-v1": FiniteHIBadPrefixBackend(),
    "casewise-prefix-induction-v1": CasewisePrefixInductionBackend(),
    "arrival-batch-decomposition-v1": HandlerDecompositionBackend("arrival-batch-decomposition-v1", ("ast_cfg", "finite_fold", "child_cases", "alternative_partition")),
    "event-handler-decomposition-v1": HandlerDecompositionBackend("event-handler-decomposition-v1", ("ast_cfg", "branch_partition", "sequence_composition")),
    "finite-release-fold-v1": HandlerDecompositionBackend("finite-release-fold-v1", ("base_case", "empty_sequence_case", "head_step", "tail_induction", "fresh_extension_composition", "old_domain_frame_composition", "ledger_frame_composition")),
    "protected-prefix-simulation-v1": ProtectedPrefixSimulationBackend(),
    "protected-prefix-bad-prefix-v1": ProtectedPrefixBadPrefixBackend(),
    "protected-prefix-safety-v1": ProtectedPrefixSafetyBackend(),
}

MACHINE_PREMISES: dict[str, tuple[str, ...]] = {
    "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY": ("REFERENCE_MODEL_CONFORMANCE", "ALL_TASK_REFERENCE_RTA_ARITHMETIC"),
    "REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY": ("REFERENCE_TASKSET_SCHEDULABLE",),
    "FINITE_BAD_PREFIX_CONTRADICTION": ("REFERENCE_HI_SUBSET_SAFETY", "HI_BAD_CLOSED_PREFIX_REFLECTION"),
    "FINITE_HI_BAD_PREFIX_REFLECTION": (
        "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
        "RELEASE_FIXED_REMOVAL_MAPPING", "DEADLINE_OBSERVATION",
        "HI_NONTRUNCATION", "EFFECTIVE_EVENT_FRONTIER_RELATION",
        "EARLY_STOP_CONFIGURATION_GATE",
    ),
    "FINAL_DEPLOYED_HI_SAFETY_COMPOSITION": ("FINITE_BAD_PREFIX_CONTRADICTION",),
    "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION": ("FULL_TO_PREFIX_SIMULATION_DOMAIN",),
    "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION": ("PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",),
    "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX": ("PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION", "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"),
}
THEOREM_REGISTRY_BINDINGS = {
    "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY": "REFERENCE_TASKSET_SCHEDULABLE",
    "REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY": "REFERENCE_HI_SUBSET_SAFETY",
    "FINITE_BAD_PREFIX_CONTRADICTION": "FINITE_BAD_PREFIX_CONTRADICTION",
    "FINITE_HI_BAD_PREFIX_REFLECTION": "HI_BAD_CLOSED_PREFIX_REFLECTION",
    "FINAL_DEPLOYED_HI_SAFETY_COMPOSITION": "FINAL_CLAIM_COMPOSITION",
    "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION_DERIVED",
    "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION": "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
    "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
}
INTEGRITY_ONLY_DEPENDENCIES = {
    "REFERENCE_TASKSET_SCHEDULABLE": {"THEORY_LIBRARY_VERSION"},
}

PROTECTED_PREFIX_THEOREM_IDS = {
    "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
    "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
    "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
}


def _load_manifest(theory_dir: Path) -> dict[str, Any]:
    return json.loads((theory_dir / "theory_manifest.json").read_text(encoding="utf-8"))


def _all_required_theorems(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Extract the union of all required theorem IDs from the manifest."""
    if "common_required_theorems" in manifest:
        common = tuple(manifest.get("common_required_theorems", []))
        route_map = manifest.get("route_required_theorems", {})
        route_theorems: set[str] = set()
        for route_list in route_map.values():
            if isinstance(route_list, list):
                route_theorems.update(route_list)
        return tuple(sorted(set(common) | route_theorems))
    return tuple(manifest.get("required_theorems", []))


def required_theorems_for_route(
    theory_dir: Path,
    route_id: str,
) -> tuple[str, ...]:
    """Return the tuple of required theorem IDs for a specific proof route."""
    theory_dir = theory_dir.resolve(strict=True)
    manifest = _load_manifest(theory_dir)
    if "route_required_theorems" in manifest:
        common = tuple(manifest.get("common_required_theorems", []))
        route_map = manifest.get("route_required_theorems", {})
        route_specific = tuple(route_map.get(route_id, []))
        return common + route_specific
    return tuple(manifest.get("required_theorems", []))


def registry_dependencies_for_theorem(theory_dir: Path, theorem_id: str) -> set[str]:
    if theorem_id in PROTECTED_PREFIX_THEOREM_IDS:
        return set(MACHINE_PREMISES[theorem_id])
    from formal_toolchain.core.registry import load_registry
    registry_entries = load_registry(theory_dir.parent / "specs" / "obligation_registry.json")
    registry_id = THEOREM_REGISTRY_BINDINGS.get(theorem_id)
    if registry_id is None:
        raise ValueError(f"THEOREM_REGISTRY_BINDING_MISSING:{theorem_id}")
    entry = next((row for row in registry_entries if row["id"] == registry_id), None)
    if entry is None:
        raise ValueError(f"THEOREM_REGISTRY_ENTRY_MISSING:{registry_id}")
    return set(entry.get("depends_on", []))


def load_verified_theory_statement(theory_dir: Path, theorem_id: str) -> dict[str, Any]:
    theory_dir = theory_dir.resolve(strict=True)
    manifest = _load_manifest(theory_dir)
    all_required = set(_all_required_theorems(manifest))
    if theorem_id not in all_required:
        raise ValueError(f"theorem is not required by current library: {theorem_id}")

    statement_path = theory_dir / "statements" / f"{theorem_id}.json"
    statement = json.loads(statement_path.read_text(encoding="utf-8"))
    if statement.get("theorem_id") != theorem_id:
        raise ValueError("theory statement id mismatch")
    required_premises = MACHINE_PREMISES.get(theorem_id)
    if required_premises is not None:
        premises = statement.get("premise_obligation_ids")
        if not isinstance(premises, list) or tuple(premises) != required_premises or len(premises) != len(set(premises)):
            raise ValueError(f"THEOREM_MACHINE_PREMISES_INVALID:{theorem_id}")
        from formal_toolchain.core.registry import load_registry
        registry_ids = {row["id"] for row in load_registry(theory_dir.parent / "specs" / "obligation_registry.json")}
        synthetic_route_theorem = theorem_id in PROTECTED_PREFIX_THEOREM_IDS
        if not synthetic_route_theorem and not set(premises) <= registry_ids:
            raise ValueError(f"THEOREM_MACHINE_PREMISE_UNKNOWN:{theorem_id}")
        registry_dependencies = registry_dependencies_for_theorem(theory_dir, theorem_id)
        integrity_only = INTEGRITY_ONLY_DEPENDENCIES.get(
            THEOREM_REGISTRY_BINDINGS[theorem_id], set())
        if registry_dependencies - integrity_only != set(required_premises):
            raise ValueError(f"THEOREM_REGISTRY_PREMISE_MISMATCH:{theorem_id}")
        if registry_dependencies != set(required_premises) | integrity_only:
            raise ValueError(f"THEOREM_REGISTRY_DEPENDENCY_MISMATCH:{theorem_id}")

    statement_payload = {
        key: statement[key]
        for key in (
            "theorem_id",
            "exact_statement",
            "conclusion",
            "source_reference",
            "assurance_level",
            "version",
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

    declared = json.loads(
        (theory_dir / "hashes.json").read_text(encoding="utf-8")
    )["statements"].get(theorem_id)
    if declared != {
        "statement_hash": statement["statement_hash"],
        "assumption_hash": statement["assumption_hash"],
    }:
        raise ValueError("theory hashes.json mismatch")

    proof_object = statement.get("proof_object")
    if proof_object:
        proof_path = (theory_dir / proof_object["path"]).resolve(strict=True)
        if theory_dir not in proof_path.parents:
            raise ValueError("PROOF_OBJECT_ESCAPES_THEORY_ROOT")
        if sha256_file(proof_path) != proof_object["sha256"]:
            raise ValueError("theory proof object hash mismatch")
        backend_name = proof_object.get("backend", "")
        backend = TCB_BACKENDS.get(backend_name)
        if backend is not None:
            result = backend.verify(proof_path, theorem=statement)
            if result.get("status") != "PASS":
                raise ValueError(f"PROOF_OBJECT_BACKEND_REJECTED:{theorem_id}")
        elif backend_name:
            raise ValueError(f"PROOF_BACKEND_NOT_AVAILABLE:{backend_name}")
    else:
        assurance = statement.get("assurance_level", "DECLARED_AXIOM_TCB")
        if assurance in ("MACHINE_CHECKED_PROJECT_LEMMA",):
            raise ValueError(
                f"THEOREM_MARKS_MACHINE_CHECKED_BUT_NO_PROOF_OBJECT:{theorem_id}"
            )

    return statement


def verify_theory_library(theory_dir: Path) -> dict[str, Any]:
    """验证整个 theory library 中所有 required theorems。"""
    theory_dir = theory_dir.resolve(strict=True)
    manifest_path = theory_dir / "theory_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "code": "THEORY_MANIFEST_MISSING"}
    manifest = _load_manifest(theory_dir)
    for theorem_id in _all_required_theorems(manifest):
        try:
            load_verified_theory_statement(theory_dir, theorem_id)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return {
                "status": "FAIL",
                "code": "THEOREM_LOAD_FAILED",
                "theorem_id": theorem_id,
                "message": str(exc),
            }
    return {"status": "PASS", "checked_theorems": list(_all_required_theorems(manifest))}


def verify_theory_library_for_route(
    theory_dir: Path,
    route_id: str,
) -> dict[str, Any]:
    """验证指定 route 所需的全部 theorem。

    规则：
    - strict_full 不验证 PP theorem backend。
    - protected_prefix 必须验证 PP theorem backend，未实现返回 UNRESOLVED。
    """
    theory_dir = theory_dir.resolve(strict=True)
    manifest_path = theory_dir / "theory_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "code": "THEORY_MANIFEST_MISSING"}

    manifest = _load_manifest(theory_dir)
    if "route_required_theorems" not in manifest:
        return verify_theory_library(theory_dir)

    theorem_ids = list(required_theorems_for_route(theory_dir, route_id))
    checked: list[str] = []
    for theorem_id in theorem_ids:
        try:
            load_verified_theory_statement(theory_dir, theorem_id)
            checked.append(theorem_id)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            if theorem_id in PROTECTED_PREFIX_THEOREM_IDS:
                return {
                    "status": "UNRESOLVED",
                    "code": "PROTECTED_PREFIX_THEOREM_UNRESOLVED",
                    "theorem_id": theorem_id,
                    "route_id": route_id,
                    "message": str(exc),
                    "checked_theorems": checked,
                }
            return {
                "status": "FAIL",
                "code": "THEOREM_LOAD_FAILED",
                "theorem_id": theorem_id,
                "message": str(exc),
            }

    return {
        "status": "PASS",
        "route_id": route_id,
        "checked_theorems": checked,
    }
