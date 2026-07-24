"""Release-ledger input view for one arbitrary full execution parameter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from formal_toolchain.core.hashing import sha256_object
from .input_oracle import FullJobInput


class ReleaseLedgerView(Protocol):
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

    def oracle_fingerprint(self) -> str:
        return sha256_object({
            "execution_id": self.execution_id,
            "records": sorted((key, value.release_time, value.actual_demand, value.hi_class)
                               for key, value in self.records.items()),
            "periodic_history_verified": self.periodic_history_verified,
            "release_fixed_verified": self.release_fixed_verified,
        })


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
