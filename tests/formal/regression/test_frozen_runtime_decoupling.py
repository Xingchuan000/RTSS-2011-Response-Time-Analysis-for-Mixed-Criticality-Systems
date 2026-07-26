from types import SimpleNamespace

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.runtime_snapshot import build_p0_reference_runtime_snapshot
from formal_toolchain.semantics.frozen_preclosed_state import build_frozen_preclosed_bundle
from formal_toolchain.verifier.effective_frontier_checker import (
    _frontier_payload,
    verify_effective_event_frontier_relation,
)


class _Scenario:
    def actual_cost_for(self, task, release_index):
        assert release_index == 0
        return {"lo": 3, "hi": 7}[task.name]


def _target():
    tasks = (
        SimpleNamespace(
            name="lo", criticality="LO", period=10, deadline=10,
            c_lo=3, initial_runtime_budget=2,
        ),
        SimpleNamespace(
            name="hi", criticality="HI", period=20, deadline=20,
            c_lo=5, c_hi=8, initial_runtime_budget=5,
        ),
    )
    config = SimpleNamespace(
        semantics="C_AMC_SEM",
        c_amc_sem_primary_on_switch_time=True,
        c_amc_sem_lo_degradation_ratio=0.5,
        end_time=100,
    )
    return SimpleNamespace(
        ordered_tasks=tasks,
        scenario=_Scenario(),
        runtime_config=config,
    )


def _reference():
    return {
        "tasks": [
            {"name": "lo", "priority_index": 0, "initial_runtime_budget": 2, "degraded_cost": 2},
            {"name": "hi", "priority_index": 1, "initial_runtime_budget": 5},
        ]
    }


def test_frozen_preclosed_pair_does_not_require_event_runtime_engine():
    concrete, reference, snapshot = build_frozen_preclosed_bundle(_target(), _reference())
    assert concrete.mode == reference.mode == snapshot.mode == "HI"
    assert concrete.running_job == reference.running_job == ("lo", 0)
    assert concrete.effective_event_frontier == reference.effective_event_frontier
    assert all("q_amc" not in str(item).lower() for item in snapshot.queue_snapshot)


def test_frozen_frontier_fresh_replay_passes():
    concrete, reference, snapshot = build_frozen_preclosed_bundle(_target(), _reference())
    frontier_hash = sha256_object(
        _frontier_payload(tuple(sorted(concrete.effective_event_frontier)))
    )
    result = verify_effective_event_frontier_relation(
        candidate_certificate={
            "obligation_status": "PASS",
            "witness": {"frontier_hash": frontier_hash},
        },
        raw_inputs=None,
        verified_predecessors={},
        expected_context_hash="",
        fresh_runtime_snapshot=snapshot,
        fresh_reference_snapshot=build_p0_reference_runtime_snapshot(reference),
    )
    assert result["status"] == "PASS"
