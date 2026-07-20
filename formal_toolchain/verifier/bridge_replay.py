"""verifier-side 的源码 GuardIR/EffectIR/Z3 逐 case replay。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True)
class BridgeReplayInputs:
    source_root: Path
    source_manifest_hash: str
    case_manifest: Mapping[str, Any]
    reference_taskset: Mapping[str, Any]
    certified_envelope: Mapping[str, Any]
    semantic_context_hash: str
    reference_context_hash: str
    bridge_context_hash: str
    runtime_config: Any | None = None


def _unresolved(code: str, **witness: Any) -> dict[str, Any]:
    return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": code, "witness": witness}


def rebuild_runtime_branch_map(inputs: BridgeReplayInputs) -> dict[str, Any]:
    """从当前源码重新解析 branch map，并验证 case map 的 source hash。"""
    from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map
    expected = inputs.case_manifest
    if expected.get("source_hash") != inputs.source_manifest_hash:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID", "code": "CASE_MAP_SOURCE_HASH_MISMATCH"}
    return build_runtime_branch_map(inputs.source_root, source_hash=inputs.source_manifest_hash, path_map=expected)


def replay_all_transition_cases(inputs: BridgeReplayInputs) -> dict[str, Any]:
    """每次 verifier 调用都重新编译 EffectIR 并逐 case 调用 Z3。"""
    try:
        import z3  # noqa: F401
    except ImportError:
        return _unresolved("FORMAL_DEPENDENCY_MISSING", missing=["z3-solver"])
    branch_map = rebuild_runtime_branch_map(inputs)
    if branch_map.get("status") != "PASS":
        return {"status": branch_map.get("status", "UNRESOLVED"), "route": "PROOF_BUNDLE_INVALID",
                "code": "FRESH_BRANCH_MAP_FAILED", "witness": branch_map}
    from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
    from formal_toolchain.bridge.transition_compiler import compile_and_prove_all_transition_cases
    bounds = derive_p0_model_bounds(inputs.reference_taskset)
    compiled = compile_and_prove_all_transition_cases(
        branch_map, bridge_context_hash=inputs.bridge_context_hash, bounds=bounds,
        runtime_config=inputs.runtime_config)
    if compiled.get("status") != "PASS":
        return {"status": compiled.get("status", "UNRESOLVED"), "route": "UNRESOLVED",
                "code": "FRESH_TRANSITION_REPLAY_FAILED", "witness": compiled}
    results = []
    for proof in compiled.get("proofs", []):
        row = proof.to_dict() if hasattr(proof, "to_dict") else dict(proof)
        results.append({"case_id": row.get("case_id"), "source_branch_id": row.get("source_branch_id"),
                        "formula_hash": sha256_object({key: row.get(key) for key in (
                            "precondition_formula", "concrete_delta", "projected_reference_delta", "preservation_formula")}),
                        "effect_ir_hash": next((item.get("effect_ir_hash") for item in branch_map.get("paths", [])
                                                 if item.get("path_id") == row.get("source_branch_id")), None),
                        "concrete_feasibility": row.get("concrete_feasibility"),
                        "reference_totality": row.get("reference_totality"),
                        "relation_preservation": row.get("relation_preservation"),
                        "z3_proof_result": row.get("z3_proof_result")})
    if any(row.get("z3_proof_result") != "PASS" for row in results):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_Z3_CASE_NOT_PASS", "cases": results}
    return {"status": "PASS", "route": None, "code": None,
            "source_manifest_hash": inputs.source_manifest_hash,
            "branch_map_hash": sha256_object(branch_map), "cases": results,
            "case_count": len(results), "solver": {"name": "z3", "timeout_ms": 30000}}


def compare_candidate_replay(candidate: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """比较 candidate 中每个 case 的 fresh formula/EffectIR 结果。

    candidate 的总状态不能作为信任根，但其逐 case 证明材料仍必须和
    verifier 基于当前源码重新生成的材料逐项一致；只比较 case ID 会
    允许同一组 case ID 搭配另一套公式或 EffectIR 混入证明包。
    """
    witness = candidate.get("witness", {})
    claimed = witness.get("transition_case_certificates") if isinstance(witness, Mapping) else None
    fresh = replay.get("cases")
    if not isinstance(claimed, list) or not isinstance(fresh, list) or len(claimed) != len(fresh):
        return {"status": "FAIL", "code": "BRIDGE_REPLAY_CASE_SET_MISMATCH"}
    claimed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in claimed:
        if not isinstance(row, Mapping) or not isinstance(row.get("inputs"), Mapping):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_CASE_SCHEMA_INVALID"}
        case_id = str(row["inputs"].get("case_id"))
        if case_id in claimed_by_id:
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_DUPLICATE_CASE_ID", "case_id": case_id}
        claimed_by_id[case_id] = row
    fresh_by_id: dict[str, Mapping[str, Any]] = {}
    for row in fresh:
        if not isinstance(row, Mapping):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_FRESH_CASE_SCHEMA_INVALID"}
        case_id = str(row.get("case_id"))
        if case_id in fresh_by_id:
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_FRESH_DUPLICATE_CASE_ID", "case_id": case_id}
        fresh_by_id[case_id] = row
    claimed_ids = set(claimed_by_id)
    fresh_ids = set(fresh_by_id)
    if claimed_ids != fresh_ids:
        return {"status": "FAIL", "code": "BRIDGE_REPLAY_CASE_ID_MISMATCH", "claimed": sorted(claimed_ids), "fresh": sorted(fresh_ids)}
    for case_id, fresh_row in fresh_by_id.items():
        claimed_row = claimed_by_id[case_id]
        claimed_inputs = claimed_row["inputs"]
        claimed_witness = claimed_row.get("witness")
        if not isinstance(claimed_witness, Mapping):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_CASE_SCHEMA_INVALID", "case_id": case_id}
        # fresh replay 与 candidate 使用同一组四项证明公式字段计算摘要，
        # 从而验证 candidate 没有只复制 case ID 而替换实际证明内容。
        claimed_formula_hash = sha256_object({key: claimed_witness.get(key) for key in (
            "precondition_formula", "concrete_delta", "projected_reference_delta", "preservation_formula")})
        declared_formula_hash = claimed_inputs.get("formula_hash", claimed_row.get("formula_hash", claimed_formula_hash))
        declared_effect_ir_hash = claimed_inputs.get("effect_ir_hash", claimed_witness.get("effect_ir_hash"))
        if declared_formula_hash != fresh_row.get("formula_hash"):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_FORMULA_HASH_MISMATCH", "case_id": case_id,
                    "claimed": declared_formula_hash, "fresh": fresh_row.get("formula_hash")}
        if declared_effect_ir_hash != fresh_row.get("effect_ir_hash"):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_EFFECT_IR_HASH_MISMATCH", "case_id": case_id,
                    "claimed": declared_effect_ir_hash, "fresh": fresh_row.get("effect_ir_hash")}
    return {"status": "PASS", "case_count": len(fresh)}
