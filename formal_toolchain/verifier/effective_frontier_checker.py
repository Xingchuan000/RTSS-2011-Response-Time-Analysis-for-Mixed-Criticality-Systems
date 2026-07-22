from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.bridge.logical_events import LogicalEvent
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
    concrete_frontier = effective_frontier(
        fresh_runtime_snapshot.queue_snapshot,
        fresh_runtime_snapshot,
    )
    reference_frontier = None
    if fresh_reference_snapshot is not None:
        ref_frontier_raw = getattr(fresh_reference_snapshot, "frontier", None)
        if ref_frontier_raw is not None:
            reference_frontier = tuple(sorted(
                e if isinstance(e, LogicalEvent) else LogicalEvent(
                    time=int(getattr(e, "time", 0)),
                    phase_rank=int(getattr(e, "phase_rank", 0)),
                    kind=getattr(e, "kind", LogicalEventKind.SVC),
                )
                for e in ref_frontier_raw
            ))
    c_keys = tuple((e.time, e.phase_rank, e.kind.value) for e in concrete_frontier)
    r_keys = None
    if reference_frontier is not None:
        r_keys = tuple((e.time, e.phase_rank, e.kind.value) for e in reference_frontier)
    frontier_match = (c_keys == r_keys) if r_keys is not None else None
    witness = {
        "schema_version": "effective_event_frontier_relation_v1",
        "concrete_frontier_hash": sha256_object(concrete_frontier),
        "concrete_event_count": len(concrete_frontier),
        "reference_frontier_count": len(reference_frontier) if reference_frontier is not None else None,
        "frontier_match": frontier_match,
    }
    if frontier_match is False:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "EFFECTIVE_EVENT_FRONTIER_RELATION_MISMATCH",
            "witness": witness,
        }
    if candidate_certificate.get("obligation_status") == "PASS" and candidate_certificate.get("witness", {}).get("frontier_hash") != sha256_object(concrete_frontier):
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "EFFECTIVE_EVENT_FRONTIER_REPLAY_MISMATCH",
            "witness": witness,
        }
    return {"status": "PASS", "route": None, "code": None, "witness": witness}
