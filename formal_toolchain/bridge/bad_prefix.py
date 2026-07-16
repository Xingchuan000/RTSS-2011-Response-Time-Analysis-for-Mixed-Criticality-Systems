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
                                               deadline_observation_certificate: Mapping[str, Any],
                                               hi_nontruncation_certificate: Mapping[str, Any],
                                               event_projection_certificate: Mapping[str, Any],
                                               state_relation_schema: str,
                                               context_hash: str,
                                               theorem_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = (closed_prefix_certificate, prefix_extension_certificate,
                deadline_observation_certificate, hi_nontruncation_certificate,
                event_projection_certificate)
    if any(not verify_obligation_certificate(item) or item.get("obligation_status") != "PASS"
           or item.get("certificate_context_hash") != context_hash for item in required):
        raise ValueError("bad-prefix reflection 前置证书无效")
    theorem = theorem_manifest or _theory("FINITE_HI_BAD_PREFIX_REFLECTION")
    predecessors = {
        "closed_prefix": closed_prefix_certificate["artifact_hash"],
        "prefix_extension": prefix_extension_certificate["artifact_hash"],
        "deadline_observation": deadline_observation_certificate["artifact_hash"],
        "hi_nontruncation": hi_nontruncation_certificate["artifact_hash"],
        "event_projection": event_projection_certificate["artifact_hash"],
    }
    return obligation_certificate(
        obligation_id="HI_BAD_PREFIX_REFLECTION", status="PASS", context_hash=context_hash,
        inputs={"theorem": theorem, "state_relation_schema": state_relation_schema},
        witness={"first_miss": "earliest PreClosed(t)", "miss_monotonic": True,
                 "same_hi_job_key": True, "event_projection_preserves_hi_timing": True,
                 "theorem": theorem}, direct_predecessor_hashes=predecessors,
        checker_id=__name__, checker_version="phase-k-v1")
