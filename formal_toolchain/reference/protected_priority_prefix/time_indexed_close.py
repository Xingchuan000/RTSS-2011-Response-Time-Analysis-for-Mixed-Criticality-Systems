"""Common integer-time closed observations for protected-prefix simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class TimeIndexedClosedObservation:
    time: int
    source_boundary_before: int
    source_boundary_after: int
    is_inserted_idle_stutter: bool
    protected_observable: Mapping[str, Any]


def _observable(state: Any, projector: Any) -> Mapping[str, Any]:
    value = projector(state) if projector is not None else state
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        value = asdict(value)
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if k != "time"}
    return {"value": value}


def close_at(
    execution: Sequence[Any],
    t: int,
    *,
    observable_projector: Any = None,
) -> TimeIndexedClosedObservation:
    """Return the unique integer-time observation induced by one execution.

    A jump is expanded by changing only the conceptual time coordinate.  No
    active job, service, completion, or miss-ledger field is synthesized.
    """
    if isinstance(t, bool) or not isinstance(t, int) or t < 0 or not execution:
        raise ValueError("TIME_INDEXED_CLOSE_ARGUMENT_INVALID")
    ordered = sorted(execution, key=lambda state: int(state.time))
    if t < int(ordered[0].time):
        raise ValueError("TIME_INDEXED_CLOSE_BEFORE_EXECUTION")
    before = ordered[0]
    after = ordered[-1]
    for state in ordered:
        if int(state.time) <= t:
            before = state
        if int(state.time) >= t:
            after = state
            break
    source_before = int(before.time)
    source_after = int(after.time)
    if source_before == t:
        chosen = before
        inserted = False
    else:
        chosen = replace(before, time=t) if hasattr(before, "time") else before
        inserted = source_after > source_before
    return TimeIndexedClosedObservation(
        time=t,
        source_boundary_before=source_before,
        source_boundary_after=source_after,
        is_inserted_idle_stutter=inserted,
        protected_observable=_observable(chosen, observable_projector),
    )


def verify_time_indexed_domain(
    execution: Sequence[Any], *, observable_projector: Any = None
) -> dict[str, Any]:
    if not execution:
        return {"status": "UNRESOLVED", "code": "CLOSE_AT_EXECUTION_MISSING"}
    times = [int(state.time) for state in execution]
    if times != sorted(set(times)):
        return {"status": "FAIL", "code": "EXECUTION_BOUNDARY_TIMES_NOT_STRICT"}
    observations = [close_at(execution, t, observable_projector=observable_projector)
                    for t in range(times[0], times[-1] + 1)]
    return {
        "status": "PASS",
        "all_integer_times_observable": True,
        "time_indexed_closed_observation_defined": True,
        "observation_count": len(observations),
        "execution_time_fingerprint": sha256_object(times),
    }
