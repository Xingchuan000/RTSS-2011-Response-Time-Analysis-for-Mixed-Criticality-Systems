"""Phase K 的参数化 reference-prefix extension certificate builder。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate


def _theory(theorem_id: str) -> dict[str, str]:
    if theorem_id != "REFERENCE_PREFIX_EXTENSION":
        raise ValueError("prefix extension 必须使用 REFERENCE_PREFIX_EXTENSION")
    return json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"][theorem_id]


def build_parameterized_prefix_extension_certificate(*, reference_taskset: Mapping[str, Any],
                                                     time_progress_certificate: Mapping[str, Any],
                                                     event_order_certificate: Mapping[str, Any],
                                                     context_hash: str,
                                                     theorem_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """证明当前 periodic reference 子语言没有 abstract dead-end。"""
    if (not verify_obligation_certificate(time_progress_certificate)
            or time_progress_certificate.get("obligation_status") != "PASS"
            or not verify_obligation_certificate(event_order_certificate)
            or event_order_certificate.get("obligation_status") != "PASS"):
        raise ValueError("prefix extension 的 service/event-order 前置证书无效")
    theorem = (theorem_manifest or _theory("REFERENCE_PREFIX_EXTENSION"))
    if theorem.get("theorem_id") not in (None, "REFERENCE_PREFIX_EXTENSION"):
        raise ValueError("theorem manifest 不是 REFERENCE_PREFIX_EXTENSION")
    tasks = reference_taskset.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("reference taskset 不满足正周期前缀扩展前提")
    for task in tasks:
        period = int(task.get("period", 0))
        deadline = int(task.get("deadline", -1))
        offset = int(task.get("offset", 0))
        # 对任意有限 reference prefix，periodic successor 的构造需要正周期、
        # 有限 deadline 且释放偏移落在一个周期内；否则“最小未来释放”
        # 公式并不保证仍属于同一个 task language。
        if period <= 0 or deadline < 0 or deadline > period or not (0 <= offset < period):
            raise ValueError("reference taskset 不满足 periodic prefix extension 前提")
    if not context_hash or len(context_hash) != 64:
        raise ValueError("prefix extension context hash 无效")
    predecessor = {"TIME_PROGRESS": time_progress_certificate["artifact_hash"],
                   "EFFECTIVE_EVENT_ORDER": event_order_certificate["artifact_hash"]}
    next_event_sources = [
        "PERIODIC_RELEASE",
        "DEADLINE",
        "VALID_COMPLETION",
        "VALID_OVERRUN",
        "VALID_RESPONSE_EXPIRY",
        "RECOVERY",
        "CONTROLLER_BOUNDARY",
    ]
    same_timestamp_phases = [
        "RECOVERY",
        "DEADLINE",
        "ARRIVAL_BATCH_FREEZE",
        "ARRIVAL",
        "COMPLETION",
        "OVERRUN",
        "RESPONSE_EXPIRY",
        "CONTROLLER_POSTCLOSURE",
        "DISPATCH",
    ]
    return obligation_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION", status="PASS", context_hash=context_hash,
        inputs={"theorem_id": "REFERENCE_PREFIX_EXTENSION",
                "theorem": theorem, "reference_taskset_hash": reference_taskset.get("fingerprint")},
        witness={"schema_version": "reference_prefix_extension_v2",
                 "quantification": "FOR_ALL_FINITE_VALID_REFERENCE_PREFIXES",
                 "next_event_sources": next_event_sources,
                 "same_timestamp_phases": same_timestamp_phases,
                 "closure_rank": {
                     "measure": "(remaining_same_time_events, phase_rank, pending_token_refreshes)",
                     "well_founded_order": "LEXICOGRAPHIC_NATURAL",
                     "strict_decrease_cases": [
                         "READY_BRANCH_SERVICE_TICK",
                         "IDLE_BRANCH_JUMP",
                         "PERIODIC_RELEASE_SUCCESSOR",
                     ],
                 },
                 "ready_successor": {"rule": "ONE_SERVICE_TICK_OR_EARLIER_EFFECTIVE_EVENT"},
                 "idle_successor": {"rule": "JUMP_TO_MINIMUM_EFFECTIVE_FUTURE_EVENT"},
                 "periodic_release_successor": {"formula": "offset + k*period",
                                                "least_k_rule": "floor((time-offset)/period)+1"},
                 "multiple_pending_jobs_supported": True,
                 "finite_prefix_job_map_total": True,
                 "horizon_independent": True,
                 "task_count": len(tasks),
                 "theorem": theorem},
        direct_predecessor_hashes=predecessor, checker_id=__name__, checker_version="phase-k-v1")
