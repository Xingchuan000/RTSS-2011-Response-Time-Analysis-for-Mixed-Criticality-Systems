"""Finite diagnostics and theorem-side contracts for integer-time CloseAt.

A finite execution sample can validate the *definition* of ``CloseAt`` on the
sampled horizon.  It cannot prove the quantified idle-jump theorem used by the
Protected Priority Prefix simulation.  The universal theorem must be supplied
by a source-bound local transition proof kernel.
"""

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


def _strict_boundary_times(execution: Sequence[Any]) -> list[int]:
    if not execution:
        raise ValueError("CLOSE_AT_EXECUTION_MISSING")
    times = [int(state.time) for state in execution]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError("EXECUTION_BOUNDARY_TIMES_NOT_STRICT")
    return times


def close_at(
    execution: Sequence[Any],
    t: int,
    *,
    observable_projector: Any = None,
) -> TimeIndexedClosedObservation:
    """Evaluate ``CloseAt`` on the closed finite horizon supplied.

    The function inserts conceptual observations only between two *existing*
    closed boundaries.  It deliberately rejects extrapolation beyond the last
    boundary: repeating the final state forever would silently assume complete
    execution existence and time divergence, which are separate obligations.
    """
    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ValueError("TIME_INDEXED_CLOSE_ARGUMENT_INVALID")
    times = _strict_boundary_times(execution)
    if t < times[0]:
        raise ValueError("TIME_INDEXED_CLOSE_BEFORE_EXECUTION")
    if t > times[-1]:
        raise ValueError("TIME_INDEXED_CLOSE_AFTER_FINITE_HORIZON")

    before = execution[0]
    after = execution[-1]
    for state in execution:
        state_time = int(state.time)
        if state_time <= t:
            before = state
        if state_time >= t:
            after = state
            break

    source_before = int(before.time)
    source_after = int(after.time)
    if source_before == t:
        chosen = before
        inserted = False
    else:
        # This is a finite diagnostic definition only.  The local idle-jump
        # theorem must separately prove that changing only time is legitimate.
        chosen = replace(before, time=t) if hasattr(before, "time") else before
        inserted = source_before < t < source_after

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
    """Check the finite-horizon ``CloseAt`` definition.

    ``PASS`` here means that the supplied finite sample is internally
    well-formed.  ``parameterized`` is intentionally false and the receipt is
    not sufficient for the universal idle-jump theorem.
    """
    try:
        times = _strict_boundary_times(execution)
    except ValueError as exc:
        return {"status": "FAIL", "code": str(exc), "parameterized": False}

    observations = [
        close_at(execution, t, observable_projector=observable_projector)
        for t in range(times[0], times[-1] + 1)
    ]
    return {
        "status": "PASS",
        "scope": "FINITE_EXECUTION_DIAGNOSTIC",
        "parameterized": False,
        "finite_horizon_only": True,
        "all_integer_times_within_finite_horizon_observable": True,
        "time_indexed_closed_observation_definition_exercised": True,
        "observation_count": len(observations),
        "execution_time_fingerprint": sha256_object(times),
    }
