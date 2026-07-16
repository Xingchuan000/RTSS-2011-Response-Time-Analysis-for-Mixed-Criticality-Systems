"""Phase F05：deadline observation、removal 和 HI non-truncation 合同。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def check_deadline_removal_contract(tasks: Sequence[Any], *, runtime_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not tasks:
        raise ValueError("taskset 不能为空")
    if runtime_evidence is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "DEADLINE_REMOVAL_RUNTIME_EVIDENCE_MISSING"}}
    trace = runtime_evidence.get("trace")
    records = runtime_evidence.get("job_records")
    if (not isinstance(trace, Sequence) or not trace or
        not isinstance(records, Mapping) or not records or
        not all(event in trace for event in ("job_completion", "deadline_miss", "budget_overrun"))):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "DEADLINE_REMOVAL_EVIDENCE_INCOMPLETE"}}
    flat = [record for group in records.values() if isinstance(group, Mapping)
            for record in group.values() if isinstance(record, Mapping)]
    observe_records = [record for group_name, group in records.items()
                       if group_name == "deadline_observe_only" and isinstance(group, Mapping)
                       for record in group.values() if isinstance(record, Mapping)]
    hi_records = [record for record in flat if record.get("criticality") == "HI"]
    deadline_observe_only = "deadline_miss" in trace and bool(observe_records) and all(
        record.get("dropped") is False and record.get("completion_time") is None for record in observe_records)
    hi_bounded = bool(hi_records) and all(record.get("executed_time", 0) <= record.get("hi_budget", 0)
                                          for record in hi_records)
    lo_plus_one = any(record.get("executed_time") == record.get("release_budget", -1) + 1
                      for record in flat if record.get("release_budget") is not None)
    completed_or_explicitly_cancelled = all(
        record.get("criticality") != "HI" or not record.get("dropped") or
        record.get("completion_time") is not None or record.get("cancellation_reason") is not None
        for record in flat)
    if not (deadline_observe_only and completed_or_explicitly_cancelled and lo_plus_one and hi_bounded):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "DEADLINE_REMOVAL_DERIVED_CONTRACT_FAILED"}}
    return {"status": "PASS", "schema_version": "deadline_removal_contract_v1",
            "deadline_is_observation_only": True, "hi_removed_only_on_completion": True,
            "lo_primary_max_service": "release_budget+1", "task_count": len(tasks)}
