from __future__ import annotations

from pathlib import Path

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.registry import build_claim_closure, load_registry
from formal_toolchain.reference import rta_production
from formal_toolchain.reference.rta_production import (
    all_task_reference_rta,
    analyze_reference_task,
    worst_case_start,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset
from formal_toolchain.verifier.artifact_verifier import verify_certificate


CTX = "0" * 64


def _task(name: str, *, priority: int, period: int = 10, deadline: int = 10,
          c_lo: int = 1, criticality: str = "LO") -> ReferenceTask:
    return ReferenceTask(
        name=name,
        period=period,
        deadline=deadline,
        c_lo=c_lo,
        c_hi=max(1, c_lo if criticality == "HI" else min(c_lo, 1)),
        criticality=criticality,
        priority_index=priority,
        code_c_lo=max(1, c_lo),
        code_c_hi=max(1, c_lo),
        degraded_cost=1 if criticality == "LO" else None,
    )


def test_worst_case_start_fails_before_trace_growth_when_hp_utilization_is_one() -> None:
    higher = (
        _task("h0", priority=0, c_lo=6),
        _task("h1", priority=1, c_lo=4),
    )
    result = worst_case_start(_task("target", priority=2), higher)
    assert result["status"] == "FAIL"
    assert result["failure"] == "HIGHER_PRIORITY_LO_UTILIZATION_NOT_BELOW_ONE"
    assert result["trace"] == []
    assert result["utilization_numerator"] == result["utilization_denominator"]


def test_lo_failure_does_not_enter_worst_case_start(monkeypatch) -> None:
    target = _task("bad", priority=0, period=10, deadline=10, c_lo=11)

    def forbidden(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError("worst_case_start must not run after LO RTA failure")

    monkeypatch.setattr(rta_production, "worst_case_start", forbidden)
    result = analyze_reference_task(target, ())
    assert result["status"] == "FAIL"
    assert result["start"]["status"] == "NOT_APPLICABLE"
    assert result["case1"] == []
    assert result["case2"] == []


def test_all_task_rta_stops_before_switch_domain_materialization_after_lo_failure() -> None:
    taskset = ReferenceTaskset(
        tasks=(
            _task("t0", priority=0, c_lo=1),
            _task("t1", priority=1, c_lo=10),
        ),
        source_context_hash=CTX,
    )
    result = all_task_reference_rta(taskset)
    assert result["status"] == "FAIL"
    assert all(row["start"]["status"] == "NOT_APPLICABLE" for row in result["tasks"])
    assert all(row["case1"] == [] and row["case2"] == [] for row in result["tasks"])


def test_relative_policy_schema_refs_resolve() -> None:
    cert = obligation_certificate(
        obligation_id="ACTION_TRANSITION",
        status="PASS",
        context_hash=CTX,
        inputs={"candidate": True},
        witness={"result": "PASS"},
        evidence=[{"candidate": True}],
        checker_id="test",
        checker_version="1",
    )
    result = verify_certificate(cert, schema_name="policy.schema.json")
    assert result["status"] == "PASS"


def test_conditional_authorization_gate_is_not_unconditionally_in_claim_closure() -> None:
    registry_path = Path(__file__).resolve().parents[3] / "formal_toolchain/specs/obligation_registry.json"
    closure = build_claim_closure(load_registry(registry_path), "DEPLOYED_HI_SAFETY")
    assert "EARLY_STOP_CLOSURE_COMPLETION" not in closure.authorization
    assert "EARLY_STOP_CONFIGURATION_GATE" in closure.mathematical
