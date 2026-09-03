import inspect
import sys
import types

import pytest

# The production formal environment installs z3-solver.  These tests exercise
# only arithmetic case-domain/terminal logic, so temporarily provide a placeholder
# in lightweight CI environments where z3 is intentionally absent.
_fake_z3_added = "z3" not in sys.modules
if _fake_z3_added:
    sys.modules["z3"] = types.ModuleType("z3")

from formal_toolchain.v10_1 import pcssc
from formal_toolchain.v10_1.constants import (
    FRAMEWORK_REVISION,
    TARGET_PROVED_PCSSC_CASE_CONSISTENT,
)
from formal_toolchain.v10_1.kernel.symbolic_state import BoundModel, TaskBound

if _fake_z3_added:
    sys.modules.pop("z3", None)


def _model() -> tuple[BoundModel, TaskBound]:
    lo = TaskBound(
        name="lo0", priority=0, period=7, deadline=7, criticality="LO",
        c_lo=2, c_hi=2, initial_budget=2, degraded_cost=1,
        actual_demand_min=1, actual_demand_max=2,
    )
    target = TaskBound(
        name="hi0", priority=1, period=12, deadline=12, criticality="HI",
        c_lo=5, c_hi=8, initial_budget=5,
        actual_demand_min=1, actual_demand_max=8,
    )
    return BoundModel(tasks=(lo, target), agent_period=5), target


def test_v10_12_deadline_canonical_domain_is_stable_complete_and_unique():
    model, target = _model()
    hp = (model.tasks[0],)
    domain, partition_receipts, domain_hash = pcssc._deadline_canonical_case_domain(
        model, target, hp
    )

    assert FRAMEWORK_REVISION == "V10.16_ADAPTIVE_PHASE_BLOCK_PCSSC"
    assert domain
    assert len({case.id for case in domain}) == len(domain)
    assert all(case.canonical_deadline == target.deadline for case in domain)
    assert len(domain_hash) == 64

    for receipt in partition_receipts:
        cells = receipt["switch_cells"]
        assert receipt["status"] == "PASS"
        assert receipt["constructed_once_at_deadline"] is True
        assert receipt["complete"] is True
        assert cells[0] == "LO_SWITCH[0,0]"


def test_case_consistent_terminal_aggregates_per_case_completion_bounds(monkeypatch):
    model, target = _model()
    hp = (model.tasks[0],)

    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda model, target, path, horizon: {
            "obligation_id": f"CONTROLLER_PATH_PREFIX_COVERAGE::{target.name},R={horizon}",
            "status": "PASS",
        },
    )

    def fake_workload(cache, model, target, hp_tasks, path, protected, response, *, horizon, theta, switch, classification):
        # Each fixed class has its own constant completion demand.  No common
        # postfix theorem is needed; the formal terminal aggregates max R_case.
        demand = 5 + (theta % 3) + (2 if classification == "ABNORMAL" else 0)
        return demand, {
            "theta": theta,
            "switch_profile": switch.id,
            "target_classification": classification,
        }

    monkeypatch.setattr(pcssc, "_workload_case", fake_workload)
    def fake_phase_block(*args, **kwargs):
        case = args[-1]
        demand = 5 + (case.theta % 3) + (2 if case.target_classification == "ABNORMAL" else 0)
        cert = {
            **case.as_dict(),
            "status": "PASS",
            "R": demand,
            "W": demand,
            "candidate_path": [{"block_id": "M1_A0", "status": "PASS"}],
            "controller_prefix_receipt": {"obligation_id": "PREFIX", "status": "PASS"},
            "case_theorem_basis": "V10_16_ADAPTIVE_PHASE_BLOCK",
            "phase_block_joint_period": 1,
            "phase_block_leaf_count": 1,
            "phase_block_leaf_digest_sha256": "a" * 64,
            "worst_phase_block": "M1_A0",
            "phase_block_receipts": [],
            "global_q_enumerated": False,
        }
        return cert, cert["candidate_path"], None

    monkeypatch.setattr(pcssc, "_phase_block_postfix_search_v10_16", fake_phase_block)
    bound, tested, receipts, failure = pcssc._case_consistent_postfix_search(
        model, target, hp, object(), set(), {}
    )

    assert failure is None
    assert bound is not None and 0 < bound <= target.deadline
    assert tested and all(row["status"] == "PASS" for row in tested)
    ids = {row["obligation_id"] for row in receipts}
    assert f"ALL_CASES_POSTFIX_COVERED::{target.name}" in ids
    assert any(value.startswith(f"CASE_CONSISTENT_RESPONSE_CERTIFICATE::{target.name},") for value in ids)
    assert any(value.startswith("PCSSC_REFINED_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_16::") for value in ids)


def test_case_consistent_terminal_fails_closed_if_one_fixed_case_has_no_postfix(monkeypatch):
    model, target = _model()
    hp = (model.tasks[0],)

    monkeypatch.setattr(
        pcssc,
        "_controller_prefix_coverage_receipt",
        lambda model, target, path, horizon: {"obligation_id": "PREFIX", "status": "PASS"},
    )

    def fake_workload(cache, model, target, hp_tasks, path, protected, response, *, horizon, theta, switch, classification):
        if switch.kind == "PRE_HI" and classification == "ABNORMAL":
            return target.deadline + 1, {"bad": True}
        return 1, {"bad": False}

    monkeypatch.setattr(pcssc, "_workload_case", fake_workload)
    monkeypatch.setattr(
        pcssc,
        "_phase_block_postfix_search_v10_16",
        lambda *args, **kwargs: (
            None,
            [],
            "PHASE_BLOCK_REFINEMENT_INSUFFICIENT:test",
        ),
    )
    bound, tested, receipts, failure = pcssc._case_consistent_postfix_search(
        model, target, hp, object(), set(), {}
    )

    assert bound is None
    assert failure is not None and failure.startswith("CASE_CONSISTENT_PCSSC_UNRESOLVED:")
    assert any(row["status"] == "UNRESOLVED" for row in tested)
    assert not any(
        row["obligation_id"].startswith("ALL_CASES_POSTFIX_COVERED::")
        for row in receipts
    )


def test_case_terminal_never_forges_global_postfix_and_has_distinct_route():
    source = inspect.getsource(pcssc._case_consistent_postfix_search)
    assert "\"obligation_id\": f\"POSTFIX_RESPONSE_CERTIFICATE::" not in source
    assert "CASE_CONSISTENT_RESPONSE_CERTIFICATE::" in source
    assert TARGET_PROVED_PCSSC_CASE_CONSISTENT == "TARGET_PROVED_BY_PCSSC_CASE_CONSISTENT"
