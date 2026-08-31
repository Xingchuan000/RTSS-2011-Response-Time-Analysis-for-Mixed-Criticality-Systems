from __future__ import annotations

from scripts.diagnose_pcssc_fixed_witness import (
    _diagnostic_signal,
    _iterate_fixed_witness,
)


def test_fixed_witness_iteration_finds_postfix_without_switching_case() -> None:
    values = {5: 9, 9: 12, 12: 12}
    result = _iterate_fixed_witness(
        initial_horizon=5,
        deadline=20,
        evaluator=lambda R: (values[R], {"fixed": True}),
    )
    assert result.status == "POSTFIX_FOUND"
    assert result.response_bound == 12
    assert list(result.path) == [
        {"R": 5, "W": 9},
        {"R": 9, "W": 12},
        {"R": 12, "W": 12},
    ]


def test_fixed_witness_iteration_stops_when_workload_crosses_deadline() -> None:
    result = _iterate_fixed_witness(
        initial_horizon=5,
        deadline=20,
        evaluator=lambda R: (21, {"fixed": True}),
    )
    assert result.status == "EXCEEDS_DEADLINE"
    assert result.response_bound is None
    assert list(result.path) == [{"R": 5, "W": 21}]


def test_signal_marks_pointwise_conservatism_only_when_every_fixed_case_closes() -> None:
    assert _diagnostic_signal(
        original_status="UNRESOLVED",
        statuses=["POSTFIX_FOUND", "POSTFIX_FOUND"],
    ) == "STRONG_OUTER_WITNESS_SWITCHING_CONSERVATISM_SIGNAL"


def test_signal_preserves_fixed_case_failure() -> None:
    assert _diagnostic_signal(
        original_status="UNRESOLVED",
        statuses=["POSTFIX_FOUND", "EXCEEDS_DEADLINE"],
    ) == "FIXED_OUTER_WITNESS_STILL_EXCEEDS_DEADLINE"
