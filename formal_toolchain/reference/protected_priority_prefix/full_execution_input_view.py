"""Release-ledger input view for one arbitrary full execution parameter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from formal_toolchain.core.hashing import sha256_object
from .input_oracle import FullJobInput


class FullExecutionReleaseLedger(Protocol):
    """Release ledger of one selected full execution parameter ``xi``."""

    execution_id: str

    def record_for(self, task_name: str, release_index: int) -> FullJobInput: ...


class ReleaseLedgerView(FullExecutionReleaseLedger, Protocol):
    def record_for(self, task_name: str, release_index: int) -> FullJobInput: ...


@dataclass(frozen=True, slots=True)
class FullExecutionInputView:
    """A fixed view of the release ledger of one full execution.

    It deliberately has no WCET fallback.  Missing records are proof failures,
    because the view is required to represent the selected execution itself.
    """
    execution_id: str
    records: Mapping[tuple[str, int], FullJobInput]
    periodic_history_verified: bool
    release_fixed_verified: bool

    def input_for(self, task_name: str, release_index: int) -> FullJobInput:
        key = (task_name, release_index)
        if key not in self.records:
            raise ValueError(f"FULL_EXECUTION_RELEASE_RECORD_MISSING:{task_name}:{release_index}")
        return self.records[key]

    def record_for(self, task_name: str, release_index: int) -> FullJobInput:
        """Protocol spelling used by theorem routes."""
        return self.input_for(task_name, release_index)

    def oracle_fingerprint(self) -> str:
        return sha256_object({
            "execution_id": self.execution_id,
            "records": sorted((key, value.release_time, value.actual_demand, value.hi_class)
                               for key, value in self.records.items()),
            "periodic_history_verified": self.periodic_history_verified,
            "release_fixed_verified": self.release_fixed_verified,
        })


def build_symbolic_full_execution_input_theorem(
    *, full_taskset: Any, conformance_witness: Mapping[str, Any]
) -> dict[str, Any]:
    """Generate the universal release-ledger contract from conformance facts.

    This is a theorem receipt, not a concrete demand stream.  It therefore
    never calls ``input_for`` and cannot silently substitute WCET values.
    """
    rows = {
        row.get("condition_id"): row
        for row in conformance_witness.get("condition_results", ())
        if isinstance(row, Mapping)
    }
    periodic = rows.get("FINITE_INDEPENDENT_PERIODIC_SUBLANGUAGE", {}).get("passed") is True
    release_fixed = rows.get("RELEASE_FIXED_DEMAND_DOMINATION", {}).get("passed") is True
    prefix_extensible = rows.get("REFERENCE_PREFIX_EXTENSIBILITY", {}).get("passed") is True
    standard_initial = rows.get("STANDARD_EMPTY_LO_INITIALIZATION", {}).get("passed") is True
    transition_identity = rows.get("REFERENCE_TRANSITION_SYSTEM_IDENTITY", {}).get("passed") is True
    tasks = tuple(getattr(full_taskset, "tasks", ()))
    static_domain = bool(tasks) and all(
        int(task.period) > 0 and 0 <= int(task.offset) < int(task.period)
        and 0 < int(task.deadline) <= int(task.period)
        and int(task.c_lo) > 0 and int(task.c_hi) > 0
        for task in tasks
    )
    # Infinite recurring ledgers require more than periodic parameters: the
    # reference semantics must start from its standard state and every finite
    # prefix must extend while time progresses.  Demand domination supplies the
    # task-type bounds copied by the protected projection.
    ok = (periodic and release_fixed and prefix_extensible
          and standard_initial and transition_identity and static_domain)
    fp = full_taskset.to_dict()["fingerprint"] if hasattr(full_taskset, "to_dict") else None
    payload = {
        "theorem_id": "FULL_REFERENCE_RECURRING_INPUT_ORACLE",
        "status": "PASS" if ok else "UNRESOLVED",
        "reference_taskset_fingerprint": fp,
        "forall_full_reference_executions": ok,
        "unique_release_record_for_every_job_key": ok,
        "release_fixed_actual_demand": ok,
        "infinite_recurring_domain": ok,
        "demand_not_regenerated_from_wcet": ok,
        "demand_contract_complete": ok,
        "lo_demand_le_reference_c_lo": ok,
        "normal_hi_demand_le_reference_c_lo": ok,
        "abnormal_hi_demand_in_lo_hi_interval": ok,
        "periodic_history_from_conformance": periodic,
        "release_fixed_from_conformance": release_fixed,
        "prefix_extensibility_from_conformance": prefix_extensible,
        "standard_initial_from_conformance": standard_initial,
        "transition_identity_from_conformance": transition_identity,
        "finite_instance_data_used": False,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def build_full_execution_input_view(
    *, execution_id: str, records: Mapping[tuple[str, int], FullJobInput],
    periodic_history_verified: bool, release_fixed_verified: bool,
) -> FullExecutionInputView:
    """Build only from release-ledger records supplied by the fresh verifier."""
    normalized = {}
    for key, record in records.items():
        if tuple(record.job_key) != tuple(key):
            raise ValueError("FULL_EXECUTION_INPUT_VIEW_KEY_MISMATCH")
        normalized[tuple(key)] = record
    return FullExecutionInputView(
        execution_id=execution_id, records=normalized,
        periodic_history_verified=periodic_history_verified,
        release_fixed_verified=release_fixed_verified,
    )
