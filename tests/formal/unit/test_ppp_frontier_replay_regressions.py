from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from formal_toolchain.bridge.logical_events import LogicalEvent, LogicalEventKind, PHASE_RANK
from formal_toolchain.bridge.prefix_extension import _receipt_is_valid
from formal_toolchain.bridge.state_relation import P0ReferenceState
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.registry import build_claim_closure, load_registry
from formal_toolchain.reference.runtime_snapshot import build_p0_reference_runtime_snapshot
from formal_toolchain.routes.registry import resolve_registry
from formal_toolchain.routes.resolver import resolve_route
from formal_toolchain.theory.loader import load_verified_theory_statement
from formal_toolchain.verifier.bootstrap_checks import build_interface_coverage_report
from formal_toolchain.verifier.checker_catalog import VERIFIER_CHECKERS
from formal_toolchain.verifier.effective_frontier_checker import (
    verify_effective_event_frontier_relation,
)
from formal_toolchain.verifier.recompute import (
    _fresh_reference_prefix_backend,
    _fresh_reference_prefix_extension_object,
)


@dataclass(frozen=True)
class _RawEvent:
    time: int
    event_type: str
    task_name: str | None
    release_index: int | None
    token: int | None = None
    fifo_rank: int = 0


class _RuntimeSnapshot:
    def __init__(self, queue_snapshot, active_job_keys):
        self.queue_snapshot = tuple(queue_snapshot)
        self.active_job_keys = tuple(active_job_keys)

    def completion_token(self, _key):
        return None

    def overrun_token(self, _key):
        return None

    def response_token(self, _key):
        return None


def _run_frontier_checker(reference_frontier):
    runtime = _RuntimeSnapshot(
        (_RawEvent(10, "DEADLINE_CHECK", "tau", 0, fifo_rank=4),),
        (("tau", 0),),
    )
    return verify_effective_event_frontier_relation(
        candidate_certificate={"obligation_status": "UNRESOLVED"},
        raw_inputs=SimpleNamespace(),
        verified_predecessors={},
        expected_context_hash="0" * 64,
        fresh_runtime_snapshot=runtime,
        fresh_reference_snapshot=SimpleNamespace(frontier=reference_frontier),
    )


def test_frontier_checker_hashes_logical_events_as_canonical_json():
    frontier = (
        LogicalEvent(
            time=10,
            phase_rank=PHASE_RANK[LogicalEventKind.DDL],
            kind=LogicalEventKind.DDL,
            job_key=("tau", 0),
            fifo_rank=4,
        ),
    )
    result = _run_frontier_checker(frontier)
    assert result["status"] == "PASS"
    assert result["witness"]["frontier_match"] is True
    assert result["witness"]["concrete_frontier_hash"] == result["witness"]["reference_frontier_hash"]


def test_frontier_checker_compares_job_identity_not_only_phase_and_kind():
    frontier = (
        LogicalEvent(
            time=10,
            phase_rank=PHASE_RANK[LogicalEventKind.DDL],
            kind=LogicalEventKind.DDL,
            job_key=("different", 0),
            fifo_rank=4,
        ),
    )
    result = _run_frontier_checker(frontier)
    assert result["status"] == "FAIL"
    assert result["code"] == "EFFECTIVE_EVENT_FRONTIER_RELATION_MISMATCH"
    assert result["witness"]["first_mismatch"]["index"] == 0


def test_p0_reference_snapshot_preserves_paired_frontier_and_mode():
    frontier = (
        LogicalEvent(5, PHASE_RANK[LogicalEventKind.REM], LogicalEventKind.REM, ("tau", 0)),
    )
    state = P0ReferenceState(
        time=0,
        mode="HI",
        ready_jobs=(("tau", 0),),
        running_job=("tau", 0),
        effective_event_frontier=frontier,
    )
    snapshot = build_p0_reference_runtime_snapshot(state)
    assert snapshot.mode == "HI"
    assert snapshot.running == ("tau", 0)
    assert snapshot.frontier == frontier


def test_prefix_extension_receipt_uses_declared_hash_mode():
    theory_root = Path(__file__).resolve().parents[3] / "formal_toolchain" / "theory"
    theorem = load_verified_theory_statement(theory_root, "REFERENCE_PREFIX_EXTENSION")
    _statement, receipt, error = _fresh_reference_prefix_backend()
    assert error is None
    proof_path = theory_root / theorem["proof_object"]["path"]
    assert theorem["proof_object"]["hash_mode"] == "canonical_json_v1"
    assert _receipt_is_valid(receipt, theorem, proof_path)


def test_reference_prefix_extension_builder_has_no_transition_identity_cycle(monkeypatch):
    context_hash = "1" * 64
    contexts = {
        "bridge_context": {"hash": context_hash},
        "reference_context": {"hash": "2" * 64},
        "semantic_context": {"hash": "3" * 64},
    }
    inputs = SimpleNamespace(contexts=contexts)
    fresh_reference = SimpleNamespace(to_dict=lambda: {"fingerprint": "fp", "tasks": [{"name": "t"}]})

    def cert(oid, layer):
        return obligation_certificate(
            obligation_id=oid,
            status="PASS",
            context_hash=contexts[layer]["hash"],
            inputs={},
            witness={"reference_taskset": fresh_reference.to_dict()} if oid == "REFERENCE_TASKSET" else {},
            checker_id="test",
            checker_version="test",
        )

    predecessors = {
        "REFERENCE_TASKSET": cert("REFERENCE_TASKSET", "reference_context"),
        "TIME_PROGRESS": cert("TIME_PROGRESS", "semantic_context"),
        "EFFECTIVE_EVENT_ORDER": cert("EFFECTIVE_EVENT_ORDER", "semantic_context"),
    }
    monkeypatch.setattr(
        "formal_toolchain.verifier.recompute._fresh_reference_prefix_backend",
        lambda: ({"theorem_id": "REFERENCE_PREFIX_EXTENSION"}, {"status": "PASS"}, None),
    )
    monkeypatch.setattr(
        "formal_toolchain.bridge.prefix_extension.build_parameterized_prefix_extension_certificate",
        lambda **_kwargs: {"obligation_id": "REFERENCE_PREFIX_EXTENSION", "obligation_status": "PASS"},
    )
    obj, error = _fresh_reference_prefix_extension_object(
        inputs=inputs,
        fresh_certificates=predecessors,
        fresh_reference=fresh_reference,
    )
    assert error is None
    assert obj["obligation_status"] == "PASS"


def test_interface_coverage_passes_for_global_and_resolved_ppp_registry():
    spec_root = Path(__file__).resolve().parents[3] / "formal_toolchain" / "specs"
    global_registry = load_registry(spec_root / "obligation_registry.json")
    global_report = build_interface_coverage_report(
        registry=global_registry,
        spec_root=spec_root,
        checker_catalog=VERIFIER_CHECKERS,
        structural_ids=set(),
    )
    assert global_report["status"] == "PASS"

    resolved = list(resolve_registry("protected_prefix").entries)
    strategy = resolve_route("protected_prefix")
    combined = {**VERIFIER_CHECKERS, **strategy.checker_catalog()}
    resolved_ids = {row["id"] for row in resolved}
    closure = build_claim_closure(resolved, "DEPLOYED_HI_SAFETY")
    resolved_report = build_interface_coverage_report(
        registry=resolved,
        spec_root=spec_root,
        checker_catalog={key: value for key, value in combined.items() if key in resolved_ids},
        structural_ids=set(closure.structural),
    )
    assert resolved_report["status"] == "PASS"
