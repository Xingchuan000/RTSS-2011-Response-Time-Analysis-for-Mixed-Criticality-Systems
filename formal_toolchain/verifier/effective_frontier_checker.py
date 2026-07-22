from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.core.hashing import sha256_object


def verify_effective_event_frontier_relation(
    *,
    candidate_certificate: Mapping[str, Any],
    raw_inputs: Any,
    verified_predecessors: Mapping[str, Mapping[str, Any]],
    expected_context_hash: str,
    fresh_runtime_snapshot: Any,
    fresh_reference_snapshot: Any | None = None,
    **_: Any,
) -> dict[str, Any]:
    if fresh_runtime_snapshot is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_RUNTIME_SNAPSHOT_MISSING"}
    rebuilt_frontier = effective_frontier(
        fresh_runtime_snapshot.queue_snapshot,
        fresh_runtime_snapshot,
    )
    witness = {
        "schema_version": "effective_event_frontier_relation_v1",
        "frontier_hash": sha256_object(rebuilt_frontier),
        "event_count": len(rebuilt_frontier),
    }
    if candidate_certificate.get("obligation_status") == "PASS" and candidate_certificate.get("witness") != witness:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "EFFECTIVE_EVENT_FRONTIER_REPLAY_MISMATCH",
        }
    return {"status": "PASS", "route": None, "code": None, "witness": witness}
