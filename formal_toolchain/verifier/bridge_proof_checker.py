from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.transition_cases import REQUIRED_P0_CASE_IDS
from formal_toolchain.bridge.state_relation import parameterized_state_relation_schema_hash, validate_n6_relation_interface
from formal_toolchain.bridge.prefix_refinement import CLOSED_PREFIX_REFINEMENT_WITNESS_SCHEMA_VERSION


_THEORY_HASHES = json.loads(
    (Path(__file__).parents[1] / "theory/hashes.json").read_text(encoding="utf-8")
)["statements"]


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_hash_map(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _is_hash(item) for key, item in value.items()
    )


def _verified_closed_prefix_witness(
    *, candidate_witness: Mapping[str, Any], receipt_hash: str, replay_hash: str,
) -> dict[str, Any]:
    """Return the validated N6-facing interface of a fresh N5 certificate."""

    relation_interface = candidate_witness.get("n6_relation_interface", {})
    validate_n6_relation_interface(relation_interface)
    transition_system_id = candidate_witness.get("reference_transition_system_id")
    if (
        transition_system_id != "FIXED_EXECUTABLE_REFERENCE_P0_V3"
        or relation_interface.get("reference_transition_system_id")
        != transition_system_id
    ):
        raise ValueError("CLOSED_PREFIX_REFERENCE_TRANSITION_SYSTEM_ID_MISMATCH")
    return {
        "fresh_theorem_receipt_hash": receipt_hash,
        "fresh_source_replay_hash": replay_hash,
        "reference_transition_system_id": transition_system_id,
        "n6_relation_interface": dict(relation_interface),
        "parameterized_relation_schema_hash": candidate_witness.get(
            "parameterized_relation_schema_hash"
        ),
        "pointwise_closed_prefix_relation": candidate_witness.get(
            "pointwise_closed_prefix_relation"
        ),
    }


def _base(candidate: Mapping[str, Any], obligation_id: str,
          bridge_context_hash: str) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_PROOF_OBJECT_MISSING"}
    if candidate.get("obligation_id") != obligation_id:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_PROOF_OBJECT_ID_MISMATCH"}
    if not verify_obligation_certificate(candidate):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "BRIDGE_PROOF_OBJECT_HASH_INVALID"}
    if candidate.get("certificate_context_hash") != bridge_context_hash:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "BRIDGE_CONTEXT_HASH_MISMATCH"}
    if not _verify_hash_map(candidate.get("direct_predecessor_hashes")):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "BRIDGE_PREDECESSOR_HASHES_INVALID"}
    return None


def _verify_cases(candidate: Mapping[str, Any], obligation_id: str,
                  bridge_context_hash: str, *, raw_inputs: Any = None,
                  reference_taskset: Mapping[str, Any] | None = None,
                  certified_envelope: Mapping[str, Any] | None = None,
                  contexts: Mapping[str, Mapping[str, Any]] | None = None,
                  predecessors: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    failure = _base(candidate, obligation_id, bridge_context_hash)
    if failure:
        return failure
    witness = candidate.get("witness", {})
    if not isinstance(witness, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_WITNESS_MISSING"}
    if obligation_id == "CLOSED_PREFIX_REFINEMENT":
        return _verify_universal_closed_prefix(candidate, bridge_context_hash, raw_inputs=raw_inputs,
                                               reference_taskset=reference_taskset, certified_envelope=certified_envelope)
    if witness.get("hi_bad_prefix_reflected") is True:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "LEGACY_BOOLEAN_BAD_PREFIX_WITNESS_REJECTED"}
    case_ids = witness.get("case_ids")
    if obligation_id == "CLOSED_PREFIX_REFINEMENT":
        if (not isinstance(case_ids, list) or not case_ids
                or len(case_ids) != len(set(case_ids))
                or set(case_ids) != set(REQUIRED_P0_CASE_IDS)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "TRANSITION_CASE_COVERAGE_INCOMPLETE"}
        coverage = witness.get("coverage")
        if (not isinstance(coverage, Mapping) or coverage.get("status") != "PASS"
                or coverage.get("missing") or coverage.get("unknown")
                or coverage.get("duplicate") or coverage.get("unresolved_cases")):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "TRANSITION_CASE_COVERAGE_INCOMPLETE"}
        inputs = candidate.get("inputs", {})
        if (candidate.get("schema_version") != "closed_prefix_refinement_v1"
                or inputs.get("source_branch_count") != len(case_ids)
                or not _is_hash(candidate.get("source_hash"))
                or not _is_hash(inputs.get("branch_map_hash"))
                or not _is_hash(inputs.get("release_mapping_hash"))
                or not {"base_relation", "same_timestamp", "positive_time",
                        "controller_postclosure", "event_projection", "release_mapping"}
                   <= set(candidate.get("direct_predecessor_hashes", {}))):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "BRIDGE_SOURCE_OR_BRANCH_BINDING_MISSING"}
        theorem = _THEORY_HASHES.get("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT", {})
        if (candidate.get("theorem_hash") != theorem.get("statement_hash")
                or inputs.get("theorem_hash") != theorem.get("statement_hash")):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "BRIDGE_THEOREM_HASH_MISMATCH"}
    elif obligation_id == "REFERENCE_PREFIX_EXTENSION":
        witness_schema = witness.get("schema_version", "")
        if witness_schema in ("reference_prefix_extension_v2", "reference_prefix_extension_v3"):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_LEGACY_SCHEMA_REJECTED"}
        if witness_schema != "reference_prefix_extension_v4":
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "PREFIX_EXTENSION_UNKNOWN_SCHEMA"}
        expected_cases = {
            "SAME_TIMESTAMP_CLOSURE",
            "READY_SERVICE_OR_EARLIER_BOUNDARY",
            "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
        }
        if set(witness.get("case_ids", [])) != expected_cases:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_CASE_IDS_MISMATCH"}
        if not _is_hash(witness.get("backend_receipt_hash")):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_BACKEND_RECEIPT_MISSING"}
        expected_predecessors = {
            "REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER",
        }
        if set(candidate.get("direct_predecessor_hashes", {})) != expected_predecessors:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_PREDECESSOR_SET_MISMATCH"}
        if len(candidate.get("direct_predecessor_hashes", {})) != 3:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_PREDECESSOR_COUNT_MISMATCH"}
        inputs = candidate.get("inputs", {})
        theorem = _THEORY_HASHES.get("REFERENCE_PREFIX_EXTENSION", {})
        if (inputs.get("theorem_statement_hash") != theorem.get("statement_hash")
                or inputs.get("theorem_assumption_hash") != theorem.get("assumption_hash")):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_THEOREM_HASH_MISMATCH"}
        for field in ("reference_taskset_fingerprint", "theorem_proof_object_hash",
                      "reference_state_source_hash", "executable_semantics_source_hash"):
            if not _is_hash(inputs.get(field, "")):
                return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                        "code": f"PREFIX_EXTENSION_{field.upper()}_MISSING"}
    if obligation_id == "HI_BAD_CLOSED_PREFIX_REFLECTION":
        required = {"job_key", "release_time", "deadline", "service", "miss_time"}
        if not required <= set(witness.get("required_quantities", [])):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                    "code": "BAD_PREFIX_QUANTIFIED_WITNESS_INCOMPLETE"}
        theorem = _THEORY_HASHES.get("FINITE_HI_BAD_PREFIX_REFLECTION", {})
        theorem_input = candidate.get("inputs", {}).get("theorem", {})
        witness_theorem = witness.get("theorem", {})
        if (theorem_input.get("statement_hash") != theorem.get("statement_hash")
                or (theorem_input.get("assumption_hash", witness_theorem.get("assumption_hash"))
                    != theorem.get("assumption_hash"))
                or not _is_hash(candidate.get("inputs", {}).get("state_relation_schema"))
                or not {"closed_prefix", "prefix_extension", "deadline_observation",
                        "hi_nontruncation", "event_projection",
                        "release_mapping"} <= set(candidate.get("direct_predecessor_hashes", {}))):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "BAD_PREFIX_THEOREM_OR_SCHEMA_MISMATCH"}
    if raw_inputs is None or not isinstance(reference_taskset, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_REPLAY_INPUTS_MISSING"}
    from formal_toolchain.verifier.bridge_replay import (
        BridgeReplayInputs, compare_candidate_replay, replay_all_transition_cases,
    )
    case_map_path = Path(raw_inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PHASE_K_CASE_MAP_MISSING"}
    case_manifest = json.loads(case_map_path.read_text(encoding="utf-8"))
    replay = replay_all_transition_cases(BridgeReplayInputs(
        source_root=Path(raw_inputs.source_root),
        source_manifest_hash=str(raw_inputs.source_manifest.get("semantic_hash", "")),
        case_manifest=case_manifest, reference_taskset=reference_taskset,
        certified_envelope=dict(certified_envelope or {}),
        semantic_context_hash=str(raw_inputs.contexts["semantic_context"]["hash"]),
        reference_context_hash=str(raw_inputs.contexts["reference_context"]["hash"]),
        bridge_context_hash=bridge_context_hash,
        runtime_config=raw_inputs.target.runtime_config,
    ))
    if replay.get("status") != "PASS":
        return replay
    if obligation_id == "CLOSED_PREFIX_REFINEMENT":
        consistency = compare_candidate_replay(candidate, replay)
        if consistency.get("status") != "PASS":
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "BRIDGE_CANDIDATE_REPLAY_MISMATCH", "witness": consistency}
    theorem_hash = candidate.get("inputs", {}).get("theorem_hash")
    if obligation_id == "CLOSED_PREFIX_REFINEMENT" and not _is_hash(theorem_hash):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_THEOREM_HASH_MISSING"}
    fresh_witness: dict[str, Any] = {
        "certificate_hash": sha256_object(dict(candidate)),
        "fresh_source_replay_hash": sha256_object(replay),
    }
    if obligation_id == "CLOSED_PREFIX_REFINEMENT":
        fresh_witness["case_count"] = len(case_ids or [])
        fresh_witness["fresh_replay"] = replay
    else:
        fresh_witness["reused_closed_prefix_case_replay"] = True
    return {"status": "PASS", "route": None, "code": None, "witness": fresh_witness}


def _verify_universal_closed_prefix(candidate: Mapping[str, Any], bridge_context_hash: str,
                                    *, raw_inputs: Any = None,
                                    reference_taskset: Mapping[str, Any] | None = None,
                                    certified_envelope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    witness = candidate.get("witness", {})
    witness_schema = witness.get("schema_version")
    if witness_schema == "closed_prefix_refinement_v1":
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_LEGACY_SCHEMA_REJECTED"}
    if witness_schema != CLOSED_PREFIX_REFINEMENT_WITNESS_SCHEMA_VERSION:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "CLOSED_PREFIX_UNKNOWN_SCHEMA"}
    if "model_bounds_hash" in candidate or "job_slots" in witness:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_SLOT_BASED_WITNESS_REJECTED"}
    if witness.get("parameterized_relation_schema_hash") != parameterized_state_relation_schema_hash():
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_PARAMETERIZED_SCHEMA_MISSING"}
    if witness.get("pointwise_closed_prefix_relation") is not True:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_UNIVERSAL_PROOF_MISSING"}
    receipt_hash = witness.get("theorem_proof_receipt_hash")
    if not _is_hash(receipt_hash):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_UNIVERSAL_PROOF_MISSING"}
    try:
        from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement
        theory_dir = Path(__file__).resolve().parents[1] / "theory"
        theorem = load_verified_theory_statement(theory_dir, "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")
        backend = TCB_BACKENDS[theorem["proof_object"]["backend"]]
        receipt = backend.verify(theory_dir / theorem["proof_object"]["path"], theorem=theorem)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "CLOSED_PREFIX_THEOREM_BACKEND_FAILED", "failure": str(exc)}
    if receipt.get("status") != "PASS" or receipt.get("receipt_hash") != receipt_hash:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CLOSED_PREFIX_THEOREM_RECEIPT_MISMATCH"}
    try:
        validate_n6_relation_interface(witness.get("n6_relation_interface", {}))
    except ValueError as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": str(exc)}
    cases = witness.get("transition_case_certificates")
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_P0_CASE_IDS):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "CLOSED_PREFIX_CASE_CONTRACTS_MISSING"}
    from formal_toolchain.bridge.transition_cases import EXPECTED_MAP_UPDATE_KIND
    required = ("created_key_fresh_proved", "released_ledger_contract_proved", "terminal_ledger_contract_proved", "miss_ledger_contract_proved", "unaffected_job_frame_proved", "effective_frontier_contract_proved")
    for item in cases:
        row = item.get("witness", {}) if isinstance(item, Mapping) else {}
        expected_kind = EXPECTED_MAP_UPDATE_KIND.get(row.get("case_id"), "UNCHANGED")
        checks = tuple(k for k in required if k != "created_key_fresh_proved" or expected_kind == "EXTEND_WITH_FRESH_RELEASE")
        if (row.get("parameterized_contract_status") != "PASS"
                or row.get("parameterized_relation_schema_hash") != parameterized_state_relation_schema_hash()
                or any(row.get(k) is not True for k in checks)
                or not isinstance(row.get("evidence_hashes"), (list, tuple)) or not row.get("evidence_hashes")
                or not isinstance(row.get("local_footprint_hash"), str) or len(row.get("local_footprint_hash")) != 64
                or row.get("map_update_kind") != expected_kind
                or (expected_kind == "EXTEND_WITH_FINITE_RELEASE_BATCH" and len(row.get("batch_decomposition_receipt_hash", "")) != 64)):
            return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "CLOSED_PREFIX_CASE_CONTRACTS_MISSING"}
    if raw_inputs is None or not isinstance(reference_taskset, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "BRIDGE_REPLAY_INPUTS_MISSING"}
    # The transition replay remains a fresh source check; it cannot be replaced
    # by candidate's universal boolean or by a bounded model hash.
    from formal_toolchain.verifier.bridge_replay import BridgeReplayInputs, replay_all_transition_cases, compare_candidate_replay
    case_map_path = Path(raw_inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PHASE_K_CASE_MAP_MISSING"}
    replay = replay_all_transition_cases(BridgeReplayInputs(source_root=Path(raw_inputs.source_root), source_manifest_hash=str(raw_inputs.source_manifest.get("semantic_hash", "")), case_manifest=json.loads(case_map_path.read_text(encoding="utf-8")), reference_taskset=reference_taskset, certified_envelope=dict(certified_envelope or {}), semantic_context_hash=str(raw_inputs.contexts["semantic_context"]["hash"]), reference_context_hash=str(raw_inputs.contexts["reference_context"]["hash"]), bridge_context_hash=bridge_context_hash, runtime_config=raw_inputs.target.runtime_config))
    if replay.get("status") != "PASS":
        return replay
    if witness.get("handler_decomposition_hash") != replay.get("handler_decomposition_hash"):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "HANDLER_DECOMPOSITION_REPLAY_MISMATCH"}
    consistency = compare_candidate_replay(candidate, replay)
    if consistency.get("status") != "PASS":
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "BRIDGE_CANDIDATE_REPLAY_MISMATCH", "witness": consistency}
    # Preserve the small, validated semantic interface consumed by N6.  The
    # fresh verifier used to replace the whole closed-prefix witness with two
    # audit hashes; downstream N6 rebuilding then received a PASS predecessor
    # that no longer exposed its transition-system identity or relation
    # interface.
    try:
        verified_witness = _verified_closed_prefix_witness(
            candidate_witness=witness,
            receipt_hash=receipt["receipt_hash"],
            replay_hash=sha256_object(replay),
        )
    except ValueError as exc:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": str(exc),
        }
    return {
        "status": "PASS",
        "route": None,
        "code": None,
        "witness": verified_witness,
    }


def verify_closed_prefix_proof_object(*, candidate: Mapping[str, Any],
                                      bridge_context_hash: str, **kwargs: Any) -> dict[str, Any]:
    return _verify_cases(candidate, "CLOSED_PREFIX_REFINEMENT", bridge_context_hash, **kwargs)


def verify_prefix_extension_proof_object(*, candidate: Mapping[str, Any],
                                         bridge_context_hash: str, **kwargs: Any) -> dict[str, Any]:
    failure = _base(candidate, "REFERENCE_PREFIX_EXTENSION", bridge_context_hash)
    if failure:
        return failure
    theorem = _THEORY_HASHES.get("REFERENCE_PREFIX_EXTENSION", {})
    candidate_witness = candidate.get("witness")
    if not isinstance(candidate_witness, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "PREFIX_EXTENSION_WITNESS_MISSING"}

    sv = candidate_witness.get("schema_version", "")
    if sv in ("reference_prefix_extension_v2", "reference_prefix_extension_v3"):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "PREFIX_EXTENSION_LEGACY_SCHEMA_REJECTED"}
    if sv == "reference_prefix_extension_v4":
        expected_cases = {
            "SAME_TIMESTAMP_CLOSURE",
            "READY_SERVICE_OR_EARLIER_BOUNDARY",
            "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
        }
        if set(candidate_witness.get("case_ids", [])) != expected_cases:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_CASE_IDS_MISMATCH"}
        if not _is_hash(candidate_witness.get("backend_receipt_hash")):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_BACKEND_RECEIPT_MISSING"}
        expected_predecessors = {
            "REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER",
        }
        if set(candidate.get("direct_predecessor_hashes", {})) != expected_predecessors:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_PREDECESSOR_SET_MISMATCH"}
        if len(candidate.get("direct_predecessor_hashes", {})) != 3:
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_PREDECESSOR_COUNT_MISMATCH"}
        inputs = candidate.get("inputs", {})
        for field in ("theorem_statement_hash", "theorem_assumption_hash",
                      "theorem_proof_object_hash", "reference_taskset_fingerprint",
                      "reference_state_source_hash", "executable_semantics_source_hash"):
            if not _is_hash(inputs.get(field, "")):
                return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                        "code": f"PREFIX_EXTENSION_{field.upper()}_MISSING"}
        if (inputs.get("theorem_statement_hash") != theorem.get("statement_hash")
                or inputs.get("theorem_assumption_hash") != theorem.get("assumption_hash")):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_THEOREM_HASH_MISMATCH"}

        from formal_toolchain.core.hashing import sha256_file
        ref_state_path = Path(__file__).resolve().parents[1] / "reference" / "reference_state.py"
        exec_sem_path = Path(__file__).resolve().parents[1] / "reference" / "executable_semantics.py"
        if inputs.get("reference_state_source_hash") != sha256_file(ref_state_path):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_REFERENCE_STATE_SOURCE_MISMATCH"}
        if inputs.get("executable_semantics_source_hash") != sha256_file(exec_sem_path):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_EXECUTABLE_SEMANTICS_SOURCE_MISMATCH"}

    fresh_witness = {
        "certificate_hash": sha256_object(dict(candidate)),
        "schema_version": sv or "unknown",
        "has_transition_cases": sv == "reference_prefix_extension_v4",
        "reused_closed_prefix_case_replay": sv != "reference_prefix_extension_v4",
    }
    return {"status": "PASS", "route": None, "code": None, "witness": fresh_witness}


def verify_bad_prefix_proof_object(*, candidate: Mapping[str, Any],
                                   bridge_context_hash: str,
                                   contexts: Mapping[str, Mapping[str, Any]],
                                   predecessors: Mapping[str, Mapping[str, Any]],
                                   **kwargs: Any) -> dict[str, Any]:
    failure = _base(candidate, "HI_BAD_CLOSED_PREFIX_REFLECTION", bridge_context_hash)
    if failure:
        return failure
    from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement
    from formal_toolchain.bridge.bad_prefix import build_hi_bad_prefix_reflection_certificate

    theory_dir = Path(__file__).resolve().parents[1] / "theory"
    try:
        theorem = load_verified_theory_statement(theory_dir, "FINITE_HI_BAD_PREFIX_REFLECTION")
        proof_path = theory_dir / theorem["proof_object"]["path"]
        backend = TCB_BACKENDS[theorem["proof_object"]["backend"]]
        receipt = backend.verify(proof_path, theorem=theorem)
        if receipt.get("status") != "PASS":
            return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "N6_THEOREM_BACKEND_REJECTED"}
        rebuilt = build_hi_bad_prefix_reflection_certificate(
            verified_predecessors=predecessors, contexts=contexts,
            context_hash=bridge_context_hash, theorem_statement=theorem,
            theorem_proof_receipt=receipt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "N6_FRESH_REBUILD_FAILED", "failure": str(exc)}
    if candidate.get("obligation_status") == "PASS" and candidate != rebuilt:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "N6_REPLAY_MISMATCH"}
    return {"status": "PASS", "route": None, "code": None, "witness": dict(rebuilt.get("witness", {}))}


def verify_prefix_extension_proof_object(*, candidate: Mapping[str, Any],
                                         bridge_context_hash: str,
                                         contexts: Mapping[str, Mapping[str, Any]],
                                         predecessors: Mapping[str, Mapping[str, Any]],
                                         reference_taskset: Mapping[str, Any],
                                         **_: Any) -> dict[str, Any]:
    """Freshly rebuild and compare the complete prefix-extension certificate."""
    failure = _base(candidate, "REFERENCE_PREFIX_EXTENSION", bridge_context_hash)
    if failure:
        return failure
    from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement
    from formal_toolchain.bridge.prefix_extension import build_parameterized_prefix_extension_certificate
    theory_dir = Path(__file__).resolve().parents[1] / "theory"
    theorem = load_verified_theory_statement(theory_dir, "REFERENCE_PREFIX_EXTENSION")
    if candidate.get("witness", {}).get("schema_version") in {"reference_prefix_extension_v2", "reference_prefix_extension_v3"}:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "PREFIX_EXTENSION_LEGACY_SCHEMA_REJECTED"}
    backend = TCB_BACKENDS.get(theorem.get("proof_object", {}).get("backend"))
    if backend is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "REFERENCE_PREFIX_BACKEND_MISSING"}
    receipt = backend.verify(theory_dir / theorem["proof_object"]["path"], theorem=theorem)
    if receipt.get("status") != "PASS":
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "REFERENCE_PREFIX_BACKEND_REJECTED", "backend_result": receipt}
    expected_ids = {"REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER"}
    if set(predecessors) != expected_ids:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "PREFIX_EXTENSION_PREDECESSOR_SET_MISMATCH"}
    try:
        rebuilt = build_parameterized_prefix_extension_certificate(
            reference_taskset=reference_taskset,
            reference_taskset_certificate=predecessors["REFERENCE_TASKSET"],
            time_progress_certificate=predecessors["TIME_PROGRESS"],
            event_order_certificate=predecessors["EFFECTIVE_EVENT_ORDER"],
            contexts=contexts, context_hash=bridge_context_hash,
            theorem_statement=theorem, theorem_proof_receipt=receipt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_PREFIX_REBUILD_FAILED", "failure": str(exc)}
    if not verify_obligation_certificate(rebuilt):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_PREFIX_REBUILD_INVALID"}
    if candidate.get("obligation_status") == "PASS" and candidate != rebuilt:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "REFERENCE_PREFIX_REPLAY_MISMATCH"}
    verified_witness = dict(rebuilt.get("witness", {}))
    # N6 consumes the verified prefix-extension predecessor, not the transient
    # rebuilt object.  Keep the verified reference-taskset identity in the
    # witness because the generic fresh certificate envelope intentionally
    # replaces semantic inputs with audit metadata.
    verified_witness["reference_taskset_fingerprint"] = rebuilt.get(
        "inputs", {}
    ).get("reference_taskset_fingerprint")
    return {
        "status": "PASS",
        "route": None,
        "code": None,
        "witness": verified_witness,
    }
