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


def _handler_decomposition_replay_inputs(
    compiled: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return the same semantic proof representation used by candidate build.

    ``compile_bridge`` builds the handler-decomposition certificate from
    ``compiled["proofs"]`` (raw semantic proof rows).  Fresh replay must use
    that exact representation as well.  Feeding certificate envelopes instead
    changes packaging-only fields such as ``artifact_hash`` and produces a
    false HANDLER_DECOMPOSITION_REPLAY_MISMATCH even when every formula and
    transition contract is identical.
    """
    proofs = compiled.get("proofs")
    if not isinstance(proofs, list) or not proofs:
        return []
    rows: list[Mapping[str, Any]] = []
    for item in proofs:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            return []
        rows.append(item)
    return rows


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
    # Composite arrival cases depend on the finite-fold certificate.  Replay
    # primitives first, build that certificate from the fresh primitive
    # certificates, then compile the composite cases with the bound receipt.
    compiled = compile_and_prove_all_transition_cases(
        branch_map, bridge_context_hash=inputs.bridge_context_hash, bounds=bounds,
        runtime_config=inputs.runtime_config, defer_composites=True)
    if not compiled.get("proof_certificates"):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "FRESH_PRIMITIVE_REPLAY_FAILED", "witness": compiled}
    from formal_toolchain.bridge.handler_decomposition import build_arrival_batch_decomposition_certificate
    primitive_certificates = [
        item for item in compiled.get("proof_certificates", [])
        if item.get("inputs", {}).get("case_id") in {
            "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE",
        }
    ]
    arrival_decomposition = build_arrival_batch_decomposition_certificate(
        source_root=inputs.source_root,
        branch_map=branch_map,
        transition_case_certificates=primitive_certificates,
        context_hash=inputs.bridge_context_hash,
    )
    if arrival_decomposition.get("status") != "PASS":
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "FRESH_ARRIVAL_FOLD_REPLAY_FAILED",
                "witness": arrival_decomposition}
    batch_decomposition_certificates = {
        path["path_id"]: dict(arrival_decomposition)
        for path in branch_map.get("paths", [])
        if path.get("case_id") in {
            "ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0",
        }
    }
    compiled = compile_and_prove_all_transition_cases(
        branch_map, bridge_context_hash=inputs.bridge_context_hash, bounds=bounds,
        runtime_config=inputs.runtime_config,
        batch_decomposition_certificates=batch_decomposition_certificates)
    if compiled.get("status") != "PASS":
        return {"status": compiled.get("status", "UNRESOLVED"), "route": "UNRESOLVED",
                "code": "FRESH_TRANSITION_REPLAY_FAILED", "witness": compiled}
    results = []
    for proof in compiled.get("proofs", []):
        row = proof.to_dict() if hasattr(proof, "to_dict") else dict(proof)
        results.append({"case_id": row.get("case_id"), "source_branch_id": row.get("source_branch_id"),
                        "formula_hash": sha256_object({key: row.get(key) for key in (
                            "precondition_formula", "concrete_delta", "projected_reference_delta", "relation_preservation_formula")}),
                        "effect_ir_hash": next((item.get("effect_ir_hash") for item in branch_map.get("paths", [])
                                                 if item.get("path_id") == row.get("source_branch_id")), None),
                        "concrete_feasibility": row.get("concrete_feasibility"),
                        "reference_totality": row.get("reference_totality"),
                        "relation_preservation": row.get("relation_preservation"),
                        "z3_proof_result": row.get("z3_proof_result"),
                        "parameterized_contract_status": row.get("parameterized_contract_status"),
                        "parameterized_contract_hash": sha256_object({key: row.get(key) for key in (
                            "parameterized_relation_schema_hash", "local_footprint_hash", "map_update_kind",
                            "modified_components", "semantic_effect_kinds", "evidence_hashes",
                            "created_key_fresh_proved", "released_ledger_contract_proved",
                            "terminal_ledger_contract_proved", "miss_ledger_contract_proved",
                            "unaffected_job_frame_proved", "effective_frontier_contract_proved",
                            "batch_decomposition_receipt_hash")})})
    if any(row.get("z3_proof_result") != "PASS" for row in results):
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_Z3_CASE_NOT_PASS", "cases": results}
    from formal_toolchain.bridge.handler_decomposition import (
        HANDLER_DECOMPOSITION_SCHEMA_VERSION,
        build_handler_decomposition_certificate,
    )
    handler_inputs = _handler_decomposition_replay_inputs(compiled)
    if not handler_inputs:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "FRESH_HANDLER_PROOF_ROWS_MISSING"}
    decomposition = build_handler_decomposition_certificate(
        inputs.source_root, context_hash=inputs.bridge_context_hash,
        transition_case_certificates=handler_inputs)
    if decomposition.get("status") != "PASS" or decomposition.get("schema_version") != HANDLER_DECOMPOSITION_SCHEMA_VERSION:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_HANDLER_DECOMPOSITION_FAILED", "handler_decomposition": decomposition}
    return {"status": "PASS", "route": None, "code": None,
            "source_manifest_hash": inputs.source_manifest_hash,
            "branch_map_hash": sha256_object(branch_map), "cases": results,
            "case_count": len(results), "solver": {"name": "z3", "timeout_ms": 30000},
            "handler_decomposition": decomposition,
            "handler_decomposition_hash": decomposition.get("artifact_hash")}


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
            "precondition_formula", "concrete_delta", "projected_reference_delta", "relation_preservation_formula")})
        declared_formula_hash = claimed_inputs.get("formula_hash", claimed_row.get("formula_hash", claimed_formula_hash))
        declared_effect_ir_hash = claimed_inputs.get("effect_ir_hash", claimed_witness.get("effect_ir_hash"))
        if declared_formula_hash != fresh_row.get("formula_hash"):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_FORMULA_HASH_MISMATCH", "case_id": case_id,
                    "claimed": declared_formula_hash, "fresh": fresh_row.get("formula_hash")}
        if declared_effect_ir_hash != fresh_row.get("effect_ir_hash"):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_EFFECT_IR_HASH_MISMATCH", "case_id": case_id,
                    "claimed": declared_effect_ir_hash, "fresh": fresh_row.get("effect_ir_hash")}
        for field in ("parameterized_contract_status",):
            if claimed_witness.get(field) != fresh_row.get(field):
                return {"status": "FAIL", "code": "BRIDGE_REPLAY_PARAMETERIZED_CONTRACT_MISMATCH", "case_id": case_id}
        claimed_contract_hash = claimed_inputs.get("parameterized_contract_hash")
        if claimed_contract_hash is None:
            claimed_contract_hash = sha256_object({key: claimed_witness.get(key) for key in (
                "parameterized_relation_schema_hash", "local_footprint_hash", "map_update_kind",
                "modified_components", "semantic_effect_kinds", "evidence_hashes",
                "created_key_fresh_proved", "released_ledger_contract_proved",
                "terminal_ledger_contract_proved", "miss_ledger_contract_proved",
                "unaffected_job_frame_proved", "effective_frontier_contract_proved",
                "batch_decomposition_receipt_hash")})
        if claimed_contract_hash != fresh_row.get("parameterized_contract_hash"):
            return {"status": "FAIL", "code": "BRIDGE_REPLAY_PARAMETERIZED_CONTRACT_MISMATCH", "case_id": case_id}
    return {"status": "PASS", "case_count": len(fresh)}
