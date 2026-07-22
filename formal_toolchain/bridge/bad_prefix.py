"""Phase K 参数化 HI bad-prefix reflection certificate builder。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate


def _theory(theorem_id: str) -> dict[str, str]:
    return json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"][theorem_id]


def build_hi_bad_prefix_reflection_certificate(*, closed_prefix_certificate: Mapping[str, Any],
                                               prefix_extension_certificate: Mapping[str, Any],
                                               release_mapping_certificate: Mapping[str, Any],
                                               deadline_observation_certificate: Mapping[str, Any],
                                               hi_nontruncation_certificate: Mapping[str, Any],
                                               effective_frontier_certificate: Mapping[str, Any],
                                               early_stop_gate_certificate: Mapping[str, Any],
                                               state_relation_schema: str,
                                               context_hash: str,
                                               theorem_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = (
        closed_prefix_certificate,
        prefix_extension_certificate,
        release_mapping_certificate,
        deadline_observation_certificate,
        hi_nontruncation_certificate,
        effective_frontier_certificate,
        early_stop_gate_certificate,
    )
    if any(not verify_obligation_certificate(item) or item.get("obligation_status") != "PASS"
           or item.get("certificate_context_hash") != context_hash for item in required):
        raise ValueError("bad-prefix reflection 前置证书无效")
    theorem = theorem_manifest or _theory("FINITE_HI_BAD_PREFIX_REFLECTION")
    if theorem.get("theorem_id") not in (None, "FINITE_HI_BAD_PREFIX_REFLECTION"):
        raise ValueError("bad-prefix theorem manifest 不匹配")
    # 这里绑定的是 first-miss 的实际关系公式，而不是几个可由调用方任意
    # 填写的布尔 witness。closed-prefix 保证两个状态处于同一个合法前缀，
    # prefix-extension 保证 reference 侧仍可继续；以下公式明确要求 miss
    # 的 job identity、release/deadline、service 和同刻事件投影同时相等。
    # 这是证书中记录的关系契约，变量均通过谓词参数显式声明；它不是
    # 调用方可以随意填写的几个布尔字段，也不冒充已经执行过的独立 SMT
    # 查询。closed-prefix、event-projection 和 deadline-observation 前置证书
    # 分别提供该契约所需的合法前缀、同刻事件投影和 first-miss 数据来源。
    miss_relation_formula = (
        "forall job_key, release_time, deadline, service, miss_time: "
        "StateRelationAtFirstMiss(job_key, release_time, deadline, service, miss_time) "
        "implies (ConcreteHIMiss(job_key, release_time, deadline, service, miss_time) "
        "iff ReferenceHIMiss(job_key, release_time, deadline, service, miss_time))"
    )
    predecessors = {
        "CLOSED_PREFIX_REFINEMENT": closed_prefix_certificate["artifact_hash"],
        "REFERENCE_PREFIX_EXTENSION": prefix_extension_certificate["artifact_hash"],
        "RELEASE_FIXED_REMOVAL_MAPPING": release_mapping_certificate["artifact_hash"],
        "DEADLINE_OBSERVATION": deadline_observation_certificate["artifact_hash"],
        "HI_NONTRUNCATION": hi_nontruncation_certificate["artifact_hash"],
        "EFFECTIVE_EVENT_FRONTIER_RELATION": effective_frontier_certificate["artifact_hash"],
        "EARLY_STOP_CONFIGURATION_GATE": early_stop_gate_certificate["artifact_hash"],
    }
    return obligation_certificate(
        obligation_id="HI_BAD_CLOSED_PREFIX_REFLECTION", status="PASS", context_hash=context_hash,
        inputs={"theorem": theorem, "state_relation_schema": state_relation_schema},
        witness={"formula_language": "first_order_contract_v1",
                 "first_miss": "earliest PreClosed(t)",
                 "miss_relation_formula": miss_relation_formula,
                 "required_quantities": ["job_key", "release_time", "deadline", "service", "miss_time"],
                 "theorem": theorem}, direct_predecessor_hashes=predecessors,
        checker_id=__name__, checker_version="phase-k-v1")
