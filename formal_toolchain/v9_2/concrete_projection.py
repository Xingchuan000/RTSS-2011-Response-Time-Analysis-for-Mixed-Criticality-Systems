"""Projection of concrete C-AMC-sem timestamp records to kernel phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .timestamp_trace import TimestampSemanticRecord


@dataclass(frozen=True, slots=True)
class ProjectedKernelStep:
    time: int
    phases: tuple[str, ...]
    p0_effect: tuple[Any, ...]
    p1_effect: Any
    p2_effect: tuple[Any, ...]
    p3_effect: tuple[Any, ...]
    p4_effect: Any
    p5_effect: tuple[Any, Any]
    p6_effect: Any
    p7_effect: tuple[Any, int]


def preliminary_dispatch_is_stutter(record: TimestampSemanticRecord) -> bool:
    """The pre-controller reschedule must consume no service and no time."""

    dispatch = record.preliminary_dispatch
    if dispatch is None:
        return True
    if isinstance(dispatch, dict):
        return int(dispatch.get("service_quantum", 0)) == 0 and int(dispatch.get("time_delta", 0)) == 0
    return getattr(dispatch, "service_quantum", 0) == 0 and getattr(dispatch, "time_delta", 0) == 0


def project_timestamp_record(record: TimestampSemanticRecord) -> ProjectedKernelStep:
    if tuple(record.phase_order) != ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        raise ValueError("EVENT_ORDER_CONFORMANCE_FAILED")
    if not preliminary_dispatch_is_stutter(record):
        raise ValueError("EVENT_ORDER_CONFORMANCE_FAILED")
    if record.service_quantum not in {0, 1}:
        raise ValueError("SERVICE_QUANTUM_CONFORMANCE_FAILED")
    return ProjectedKernelStep(
        time=int(record.time), phases=record.phase_order,
        p0_effect=record.settled_completions, p1_effect=record.recovery_before_deadline,
        p2_effect=record.deadline_observations, p3_effect=record.frozen_arrivals,
        p4_effect=record.mode_switch,
        p5_effect=(record.controller_observation, record.controller_action),
        p6_effect=record.final_dispatch, p7_effect=(record.final_dispatch, record.service_quantum),
    )


def project_prefix(records: Iterable[TimestampSemanticRecord]) -> tuple[ProjectedKernelStep, ...]:
    return tuple(project_timestamp_record(record) for record in records)


__all__ = ["ProjectedKernelStep", "preliminary_dispatch_is_stutter", "project_prefix",
           "project_timestamp_record"]
