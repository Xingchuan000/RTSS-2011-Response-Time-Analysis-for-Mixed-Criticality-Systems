from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.bridge.effective_event_frontier import effective_frontier
from formal_toolchain.bridge.logical_events import LogicalEvent, LogicalEventKind
from formal_toolchain.core.hashing import sha256_object


def _job_key(value: Any) -> tuple[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("LOGICAL_EVENT_JOB_KEY_INVALID")
    return (str(value[0]), int(value[1]))


def _logical_event(value: Any) -> LogicalEvent:
    if isinstance(value, LogicalEvent):
        return value
    if isinstance(value, Mapping):
        get = value.get
    else:
        get = lambda name, default=None: getattr(value, name, default)
    raw_kind = get("kind", LogicalEventKind.SVC)
    kind = raw_kind if isinstance(raw_kind, LogicalEventKind) else LogicalEventKind(str(raw_kind))
    raw_batch = get("batch_jobs", ()) or ()
    return LogicalEvent(
        time=int(get("time", 0)),
        phase_rank=int(get("phase_rank", 0)),
        kind=kind,
        job_key=_job_key(get("job_key", None)),
        batch_jobs=tuple(_job_key(item) for item in raw_batch),
        fifo_rank=int(get("fifo_rank", 0)),
    )


def _frontier_payload(frontier: tuple[LogicalEvent, ...]) -> list[dict[str, Any]]:
    """Canonical JSON payload used by both replay comparison and hashing."""

    return [
        {
            "time": int(event.time),
            "phase_rank": int(event.phase_rank),
            "kind": event.kind.value,
            "job_key": list(event.job_key) if event.job_key is not None else None,
            "batch_jobs": [list(job_key) for job_key in event.batch_jobs],
            "fifo_rank": int(event.fifo_rank),
        }
        for event in frontier
    ]


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
    if fresh_reference_snapshot is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_RUNTIME_SNAPSHOT_MISSING"}

    concrete_frontier = tuple(sorted(effective_frontier(
        fresh_runtime_snapshot.queue_snapshot,
        fresh_runtime_snapshot,
    )))
    ref_frontier_raw = getattr(fresh_reference_snapshot, "frontier", None)
    if ref_frontier_raw is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "FRESH_REFERENCE_FRONTIER_MISSING"}
    reference_frontier = tuple(sorted(_logical_event(event) for event in ref_frontier_raw))

    concrete_payload = _frontier_payload(concrete_frontier)
    reference_payload = _frontier_payload(reference_frontier)
    concrete_hash = sha256_object(concrete_payload)
    reference_hash = sha256_object(reference_payload)
    frontier_match = concrete_payload == reference_payload
    first_mismatch = None
    if not frontier_match:
        limit = min(len(concrete_payload), len(reference_payload))
        index = next((i for i in range(limit)
                      if concrete_payload[i] != reference_payload[i]), limit)
        first_mismatch = {
            "index": index,
            "concrete": concrete_payload[index] if index < len(concrete_payload) else None,
            "reference": reference_payload[index] if index < len(reference_payload) else None,
        }
    witness = {
        "schema_version": "effective_event_frontier_relation_v2",
        "concrete_frontier_hash": concrete_hash,
        "reference_frontier_hash": reference_hash,
        "concrete_event_count": len(concrete_frontier),
        "reference_frontier_count": len(reference_frontier),
        "frontier_match": frontier_match,
        "first_mismatch": first_mismatch,
    }
    if not frontier_match:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "EFFECTIVE_EVENT_FRONTIER_RELATION_MISMATCH",
            "witness": witness,
        }

    candidate_witness = candidate_certificate.get("witness", {})
    candidate_hash = candidate_witness.get("frontier_hash")
    if candidate_certificate.get("obligation_status") == "PASS" and candidate_hash != concrete_hash:
        return {
            "status": "FAIL",
            "route": "PROOF_BUNDLE_INVALID",
            "code": "EFFECTIVE_EVENT_FRONTIER_REPLAY_MISMATCH",
            "witness": {**witness, "candidate_frontier_hash": candidate_hash},
        }
    return {"status": "PASS", "route": None, "code": None, "witness": witness}
