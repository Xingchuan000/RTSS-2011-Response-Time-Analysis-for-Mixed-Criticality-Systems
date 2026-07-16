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
    predecessor = {"time_progress": time_progress_certificate["artifact_hash"],
                   "event_order": event_order_certificate["artifact_hash"]}
    return obligation_certificate(
        obligation_id="REFERENCE_PREFIX_EXTENSION", status="PASS", context_hash=context_hash,
        inputs={"theorem_id": "REFERENCE_PREFIX_EXTENSION",
                "theorem": theorem, "reference_taskset_hash": reference_taskset.get("fingerprint")},
        witness={"ready_branch": "ONE_SERVICE_TICK", "empty_ready_branch": "JUMP_TO_NEXT_EVENT",
                 "periodic_release_language": "release = offset + k * period",
                 "least_future_release_rule": "min{offset+k*period | offset+k*period > time}",
                 "successor_conditions": ["period > 0", "0 <= offset < period", "0 <= deadline <= period"],
                 "task_count": len(tasks),
                 "theorem": theorem},
        direct_predecessor_hashes=predecessor, checker_id=__name__, checker_version="phase-k-v1")
