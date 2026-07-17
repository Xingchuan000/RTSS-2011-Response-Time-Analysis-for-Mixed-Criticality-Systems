"""Phase I-K proof object 的 verifier-side checker。

checker 只消费 candidate proof object，并对其 schema、context、源码 hash、
case 集合和上游 hash 做检查；它不调用 compiler 的 proof generator。
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.transition_cases import REQUIRED_P0_CASE_IDS


_THEORY_HASHES = json.loads(
    (Path(__file__).parents[1] / "theory/hashes.json").read_text(encoding="utf-8")
)["statements"]


def _is_hash(value: Any) -> bool:
    """判断 proof object 中的 hash 是否为规范的小写 SHA-256。"""

    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_hash_map(value: Any) -> bool:
    """验证 direct predecessor/hash map 的结构，不接受空字符串占位。"""

    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _is_hash(item) for key, item in value.items()
    )


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
                  certified_envelope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    failure = _base(candidate, obligation_id, bridge_context_hash)
    if failure:
        return failure
    witness = candidate.get("witness", {})
    if not isinstance(witness, Mapping):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "BRIDGE_WITNESS_MISSING"}
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
        theorem = _THEORY_HASHES.get("REFERENCE_PREFIX_EXTENSION", {})
        theorem_input = candidate.get("inputs", {}).get("theorem", {})
        if (candidate.get("inputs", {}).get("theorem_id") != obligation_id
                or theorem_input.get("theorem_id") != obligation_id
                or theorem_input.get("statement_hash") != theorem.get("statement_hash")
                or theorem_input.get("assumption_hash") != theorem.get("assumption_hash")
                or not {"event_order", "time_progress"}
                   <= set(candidate.get("direct_predecessor_hashes", {}))):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "PREFIX_EXTENSION_THEOREM_MISMATCH"}
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
                        "hi_nontruncation", "event_projection", "protected_hi",
                        "release_mapping"} <= set(candidate.get("direct_predecessor_hashes", {}))):
            return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                    "code": "BAD_PREFIX_THEOREM_OR_SCHEMA_MISMATCH"}
    # 结构字段通过后，必须消费当前源码的 fresh GuardIR/EffectIR/Z3 replay。
    # 没有原始 verifier inputs 时保持 unresolved，不把 candidate case ID 当
    # 成 transition 证明。
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


def verify_closed_prefix_proof_object(*, candidate: Mapping[str, Any],
                                      bridge_context_hash: str, **kwargs: Any) -> dict[str, Any]:
    return _verify_cases(candidate, "CLOSED_PREFIX_REFINEMENT", bridge_context_hash, **kwargs)


def verify_prefix_extension_proof_object(*, candidate: Mapping[str, Any],
                                         bridge_context_hash: str, **kwargs: Any) -> dict[str, Any]:
    return _verify_cases(candidate, "REFERENCE_PREFIX_EXTENSION", bridge_context_hash, **kwargs)


def verify_bad_prefix_proof_object(*, candidate: Mapping[str, Any],
                                   bridge_context_hash: str, **kwargs: Any) -> dict[str, Any]:
    return _verify_cases(candidate, "HI_BAD_CLOSED_PREFIX_REFLECTION", bridge_context_hash, **kwargs)
