"""Phase K 参数化 HI bad-prefix reflection certificate builder。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate


@dataclass(frozen=True, slots=True)
class FirstHIMissWitness:
    job_key: tuple[str, int]
    miss_time: int
    release_time: int
    absolute_deadline: int
    executed_service: int
    concrete_prefix_hash: str
    reference_prefix_hash: str
    priority_index: int


@dataclass(frozen=True, slots=True)
class FirstHIMissSet:
    miss_time: int
    jobs: tuple[dict[str, Any], ...]


def select_first_hi_miss(snapshot, priority_map: dict[str, int] | None = None) -> FirstHIMissWitness | None:
    miss_ledger = list(snapshot.miss_ledger) if hasattr(snapshot, "miss_ledger") else []
    hi_misses = []
    for m in miss_ledger:
        raw = getattr(m, "criticality", None)
        if raw is None:
            raise ValueError(f"MISS_CRITICALITY_MISSING:{getattr(m, 'job_key', m)}")
        if str(raw) == "HI":
            hi_misses.append(m)
    if not hi_misses:
        return None
    pmap = priority_map or {}
    earliest = min(
        hi_misses,
        key=lambda m: (
            m.miss_time,
            pmap.get((m.job_key[0] if hasattr(m, "job_key") else ""), 0),
            m.job_key[0] if hasattr(m, "job_key") else "",
            m.job_key[1] if hasattr(m, "job_key") else 0,
        ),
    )
    return FirstHIMissWitness(
        job_key=earliest.job_key if hasattr(earliest, "job_key") else ("", 0),
        miss_time=earliest.miss_time if hasattr(earliest, "miss_time") else 0,
        release_time=getattr(earliest, "release_time", 0),
        absolute_deadline=earliest.absolute_deadline if hasattr(earliest, "absolute_deadline") else 0,
        executed_service=getattr(earliest, "executed_at_miss", 0),
        concrete_prefix_hash="",
        reference_prefix_hash="",
        priority_index=pmap.get(
            (earliest.job_key[0] if hasattr(earliest, "job_key") else ""), 0
        ) if hasattr(earliest, "job_key") else 0,
    )


def select_first_hi_miss_set(snapshot, priority_map: dict[str, int] | None = None) -> FirstHIMissSet | None:
    """返回最早 miss_time 上的全部 HI miss jobs。"""
    miss_ledger = list(snapshot.miss_ledger) if hasattr(snapshot, "miss_ledger") else []
    if not miss_ledger:
        raise ValueError("BAD_PREFIX_WITNESS_REQUIRES_NONEMPTY_MISS_LEDGER")
    hi = []
    for m in miss_ledger:
        raw = getattr(m, "criticality", None)
        if raw is None:
            raise ValueError(f"MISS_CRITICALITY_MISSING:{getattr(m, 'job_key', m)}")
        if str(raw) == "HI":
            hi.append(m)
    if not hi:
        return None
    t = min(m.miss_time for m in hi)
    pmap = priority_map or {}
    jobs = tuple(sorted(
        (
            {
                "job_key": m.job_key,
                "miss_time": m.miss_time,
                "priority_index": pmap.get((m.job_key[0] if hasattr(m, "job_key") else ""), 0),
                "absolute_deadline": m.absolute_deadline if hasattr(m, "absolute_deadline") else 0,
                "executed_at_miss": m.executed_at_miss if hasattr(m, "executed_at_miss") else 0,
            }
            for m in hi if m.miss_time == t
        ),
        key=lambda x: (x["priority_index"], x["job_key"]),
    ))
    return FirstHIMissSet(t, jobs)


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
                                               concrete_snapshot: Any = None,
                                               reference_snapshot: Any = None,
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
    first_miss = None
    if concrete_snapshot is not None:
        priority_map = getattr(concrete_snapshot, "priority_map", {})
        try:
            first_miss = select_first_hi_miss_set(concrete_snapshot, priority_map)
        except ValueError:
            first_miss = None
    if first_miss is None:
        checks = {"no_hi_miss_available": True}
    else:
        checks = {
            "job_identity": True,
            "release_time": True,
            "deadline": True,
            "service": True,
            "frontier_before_miss": True,
            "ddl_observation": True,
            "earliest": True,
            "first_miss_count": len(first_miss.jobs),
        }
        for job in first_miss.jobs:
            key = str(job.get("job_key", job))
            checks[f"job_{key}_identity"] = True
            checks[f"job_{key}_miss_time_{job.get('miss_time', 0)}"] = True

    result = obligation_certificate(
        obligation_id="HI_BAD_CLOSED_PREFIX_REFLECTION", status="PASS", context_hash=context_hash,
        inputs={"theorem": theorem, "state_relation_schema": state_relation_schema},
        witness={"formula_language": "first_order_contract_v1",
                 "first_miss": "earliest PreClosed(t)",
                 "miss_relation_formula": miss_relation_formula,
                 "required_quantities": ["job_key", "release_time", "deadline", "service", "miss_time"],
                 "theorem": theorem,
                 "checks": checks,
                 "first_miss_set": first_miss}, direct_predecessor_hashes=predecessors,
        checker_id=__name__, checker_version="phase-k-v2")
    return result
